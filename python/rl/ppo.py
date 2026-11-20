"""Actor-critic (PPO) fine-tuning of the v2 window model on real surface-growth
episodes (plan/actor_critic.md).

Iteration:
  1. rollout: W worker processes each grow one small surface (1x1, 10.86 Å) with
     the current policy (window_generation_v2.py surface ... --top_k 0 --record),
     substrate = last frame of a random interface MD training run
  2. update: PPO on every sampled candidate (accepted or not)

Policy  = masked softmax of logits / T (T = 0.8) over the guide's allowed set,
          exactly as sampled (no top-k), one action per sampled token.
Reward  per candidate (window completion):
          immediate  -1 per defect and -0.5 per dead end of the raw model output
          future     accepted: -blame (1 if it placed the culprit of a later
                     backtrack, 0.5 if ambiguous / substrate) - W_FINAL x defects
                     among its final atoms at the end or when it is undone;
                     rejected: 0; clean but not
                     picked: critic estimate (bootstrap)
Critic  = value head on the (detached) final hidden state, two outputs per token
          (expected immediate, expected future reward); baseline V = sum.
Advantage A_t = R - V(prefix before t), normalised; PPO clip 0.2; KL to the frozen
start model (k3 estimator) beta; small entropy bonus.

usage:
  python/rl/ppo.py loop --init CK --tokenizer TOK --run NAME [--iters 30 --workers 8]
  python/rl/ppo.py check EP.pkl --checkpoint CK      # log-prob consistency test
"""
import sys, os, json, glob, time, pickle, argparse, subprocess
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from train import load_checkpoint, AtomicGPT, GPTConfig

# repository root (two levels above this file); all paths below are relative to it
ROOT = os.path.relpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
T_POL = 0.8
W_DEF, W_DEAD = 1.0, 0.5
W_FINAL = 2.0     # per defect left in an accepted window's final atoms (end, or when undone); --w_final


# ------------------------------------------------------------------ data
def load_episodes(files):
    samples, metas = [], []
    for f in files:
        try:
            d = pickle.load(open(f, "rb"))
        except Exception as e:                          # killed / crashed worker
            print(f"skip {f}: {e}"); continue
        metas.append(d["meta"])
        samples += d["samples"]
    return samples, metas


def immediate_reward(s):
    rb = s["raw_bad"]
    return -W_DEF * sum(v for k, v in rb.items() if k != "dead_end") - W_DEAD * rb.get("dead_end", 0)


def future_target(s):
    """(target, kind): kind 'obs' observed, 'zero' rejected, 'boot' clean but not picked"""
    if s["chosen"]:
        f = s["future"] or {}
        return -f.get("blame", 0.0) - W_FINAL * (f.get("final_defects", 0) + f.get("undone_defects", 0)), "obs"
    if s["raw_bad"]:
        return 0.0, "zero"
    return 0.0, "boot"


class Batch:
    """padded ids + flat action arrays for a list of samples"""
    def __init__(self, samples, device):
        B = len(samples)
        T = max(len(s["ids"]) for s in samples)
        ids = np.zeros((B, T), np.int64)
        rows, pos, tok, lp, al = [], [], [], [], []
        for b, s in enumerate(samples):
            ids[b, :len(s["ids"])] = s["ids"]
            for p, t, a, l in s["act"]:
                rows.append(b); pos.append(p); tok.append(t); lp.append(l); al.append(a)
        L = max(len(a) for a in al)
        A = np.full((len(al), L), -1, np.int64)
        for k, a in enumerate(al):
            A[k, :len(a)] = a
        self.ids = torch.from_numpy(ids).to(device)
        self.rows = torch.tensor(rows, device=device)
        self.pos = torch.tensor(pos, device=device)
        self.tok = torch.tensor(tok, device=device)
        self.lp_old = torch.tensor(lp, dtype=torch.float32, device=device)
        self.allowed = torch.from_numpy(A).to(device)
        # index of the action's token inside its allowed list
        self.k = (self.allowed == self.tok[:, None]).float().argmax(1)
        self.sample_of_act = self.rows
        # last action of every sample (bootstrap position)
        last = {}
        for k, b in enumerate(rows):
            last[b] = k
        self.last_act = torch.tensor([last[b] for b in range(B)], device=device)


def masked_logp(model, batch, need_entropy=False):
    """log-prob of each action under the masked policy, (+ entropy), and the
    hidden state at the predicting position."""
    h = model.hidden(batch.ids)                                     # (B, T, C)
    hp = h[batch.rows, batch.pos - 1]                               # (N, C)
    logits = model.lm_head(hp) / T_POL                              # (N, V)
    valid = batch.allowed >= 0
    sub = logits.gather(1, batch.allowed.clamp(min=0)).masked_fill(~valid, float("-inf"))
    lsm = sub.log_softmax(-1)
    lp = lsm.gather(1, batch.k[:, None])[:, 0]
    ent = None
    if need_entropy:
        p = lsm.exp()
        ent = -(p * lsm.masked_fill(~valid, 0)).sum(-1)
    return lp, ent, hp


class ValueHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Linear(256, 2))

    def forward(self, h):
        return self.net(h)


# ------------------------------------------------------------------ update
def update(model, ref, vhead, opt, samples, args, device, log):
    model.eval(); ref.eval()                     # no dropout: log-probs must match sampling
    rng = np.random.default_rng(args.seed)
    imm = np.array([immediate_reward(s) for s in samples], np.float32)
    ft = [future_target(s) for s in samples]
    fut = np.array([f[0] for f in ft], np.float32)
    kind = np.array([f[1] for f in ft])

    # pass 1 (no grad): reference log-probs, old values, bootstrap
    ref_lp, v_old, chunks = [], [], []
    idx = np.arange(len(samples))
    for a in range(0, len(idx), args.mb):
        sl = idx[a:a + args.mb]
        bt = Batch([samples[i] for i in sl], device)
        with torch.no_grad():
            lr_, _, _ = masked_logp(ref, bt)
            _, _, hp = masked_logp(model, bt)
            q = vhead(hp)
        ref_lp.append(lr_.cpu()); v_old.append(q.cpu())
        # bootstrap: future estimate at the last action of clean-but-not-picked samples
        qb = q[bt.last_act, 1].cpu().numpy()
        for j, i in enumerate(sl):
            if kind[i] == "boot":
                fut[i] = qb[j]
        chunks.append(sl)
    R = imm + fut
    # advantages per action token
    adv_list = []
    for sl, q in zip(chunks, v_old):
        rows = np.concatenate([[j] * len(samples[i]["act"]) for j, i in enumerate(sl)])
        adv_list.append(torch.from_numpy(R[sl][rows]) - q.sum(1))
    adv_all = torch.cat(adv_list)
    mu, sd = adv_all.mean(), adv_all.std() + 1e-6
    stats = dict(pg=[], vl=[], kl=[], ent=[], clip=[], ratio_dev=[])
    for ep in range(args.epochs):
        order = rng.permutation(len(chunks))
        for c in order:
            sl = chunks[c]
            bt = Batch([samples[i] for i in sl], device)
            rows = np.concatenate([[j] * len(samples[i]["act"]) for j, i in enumerate(sl)])
            A = ((adv_list[c] - mu) / sd).to(device)
            t_imm = torch.from_numpy(imm[sl][rows]).to(device)
            t_fut = torch.from_numpy(fut[sl][rows]).to(device)
            fut_mask = torch.from_numpy((kind[sl] != "boot")[rows]).to(device).float()
            lp, ent, hp = masked_logp(model, bt, need_entropy=True)
            q = vhead(hp.detach())
            ratio = (lp - bt.lp_old).exp()
            pg = -torch.min(ratio * A, ratio.clamp(1 - args.clip, 1 + args.clip) * A).mean()
            d = ref_lp[c].to(device) - lp
            kl = (d.exp() - d - 1).mean()
            vl = ((q[:, 0] - t_imm) ** 2).mean() + (((q[:, 1] - t_fut) ** 2) * fut_mask).sum() / fut_mask.sum().clamp(min=1)
            loss = pg + args.beta * kl - args.ent * ent.mean() + vl
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(vhead.parameters()), 1.0)
            opt.step()
            stats["pg"].append(pg.item()); stats["vl"].append(vl.item()); stats["kl"].append(kl.item())
            stats["ent"].append(ent.mean().item())
            stats["clip"].append(((ratio - 1).abs() > args.clip).float().mean().item())
            if ep == 0:
                stats["ratio_dev"].append((ratio - 1).abs().max().item())
    # explained variance of the old values
    vt = torch.cat([q.sum(1) for q in v_old]); rt = adv_all + vt
    ev = 1 - (rt - vt).var() / rt.var().clamp(min=1e-6)
    out = {k: float(np.mean(v)) for k, v in stats.items() if v}
    out["ratio_dev_first_mb"] = stats["ratio_dev"][0] if stats["ratio_dev"] else None
    out.update(R_mean=float(R.mean()), imm_mean=float(imm.mean()), fut_obs_mean=float(fut[kind == "obs"].mean())
               if (kind == "obs").any() else None, value_explained_var=float(ev))
    return out


# ------------------------------------------------------------------ rollout
def substrate_frames():
    vr = set(json.load(open("data/sio2_iface_win8_z/sequences.json"))["val_runs"])
    fs = sorted(glob.glob("data/interface/structures/SiO2Si/*.extxyz"))
    return [f for f in fs if not any(os.path.basename(f).startswith(v) for v in vr)]


def rollout(ck, tok, out_dir, n_workers, seed, args):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    subs = substrate_frames()
    procs = []
    for w in range(n_workers):
        sub = subs[int(rng.integers(len(subs)))]
        cmd = ["venv/bin/python", "-u", "python/window_generation_v2.py", "surface",
               "--checkpoint", ck, "--tokenizer", tok, "--force_bonds", "--gap_mask", "--relax", "--snap",
               "--n_cand", "8", "--n_relax", "2", "--deadend", "--seed", str(seed * 100 + w),
               "--substrate_frame", f"{sub}@-1", "--lateral", str(args.lateral),
               "--max_backtracks", str(args.max_backtracks), "--max_jump", "6", "--top_k", "0",
               "--record", f"{out_dir}/ep{w}.pkl", "--out", f"{out_dir}/ep{w}.extxyz"]
        env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        procs.append(subprocess.Popen(cmd, stdout=open(f"{out_dir}/ep{w}.log", "w"),
                                      stderr=subprocess.STDOUT, env=env))
    t0 = time.time()
    for p in procs:
        try:
            p.wait(timeout=max(1, args.episode_timeout - (time.time() - t0)))
        except subprocess.TimeoutExpired:
            p.kill()
    return sorted(glob.glob(f"{out_dir}/ep*.pkl"))


def episode_stats(samples, metas):
    n = len(samples)
    clean = sum(1 for s in samples if not s["raw_bad"])
    dead = sum(1 for s in samples if "dead_end" in s["raw_bad"])
    chosen = [s for s in samples if s["chosen"]]
    return dict(episodes=len(metas), samples=n, raw_clean_pct=100 * clean / max(n, 1),
                deadend_pct=100 * dead / max(n, 1),
                accepted_repaired_pct=100 * sum(s["repaired"] for s in chosen) / max(len(chosen), 1),
                blamed_accepts=sum(1 for s in chosen if s["future"] and s["future"]["blame"] > 0),
                windows_done=float(np.mean([m.get("n_done", 0) for m in metas])) if metas else 0,
                windows_total=float(np.mean([m.get("n_windows", 0) for m in metas])) if metas else 0,
                backtracks=float(np.mean([m.get("n_back", 0) for m in metas])) if metas else 0,
                interior_defects=float(np.mean([m.get("interior_defects", 0) for m in metas])) if metas else 0,
                samples_per_window=n / max(sum(m.get("n_done", 0) for m in metas), 1))


def save_ck(model, vhead, path, it):
    torch.save({"step": it, "val_loss": 0.0, "model_state": model.state_dict(),
                "config": vars(model.config) if not hasattr(model.config, "__dataclass_fields__")
                else __import__("dataclasses").asdict(model.config),
                "value_state": vhead.state_dict()}, path)


def cmd_loop(a):
    global W_FINAL
    W_FINAL = a.w_final
    dev = "cuda"
    run_dir, ck_dir = f"data/rl/{a.run}", f"checkpoints/rl/{a.run}"
    os.makedirs(run_dir, exist_ok=True); os.makedirs(ck_dir, exist_ok=True)
    model, _ = load_checkpoint(a.init, dev)
    ref, _ = load_checkpoint(a.init, dev)
    for p in ref.parameters():
        p.requires_grad_(False)
    vhead = ValueHead(model.config.d_model).to(dev)
    start = 0
    prev = sorted(glob.glob(f"{ck_dir}/iter_*.pt"))
    if prev:                                       # resume
        ck = torch.load(prev[-1], map_location=dev)
        model.load_state_dict(ck["model_state"]); vhead.load_state_dict(ck["value_state"])
        start = ck["step"] + 1
        print(f"resumed from {prev[-1]}")
    else:
        save_ck(model, vhead, f"{ck_dir}/iter_000.pt", 0)
        start = 0
    opt = torch.optim.AdamW([{"params": model.parameters(), "lr": a.lr},
                             {"params": vhead.parameters(), "lr": a.lr_value}], weight_decay=0.0)
    import copy as _copy
    crit = _copy.copy(a); crit.epochs = a.critic_epochs        # settings for critic-only updates
    if start == 0 and a.critic_pretrain:
        # offline critic warm start on already recorded episodes (policy frozen)
        files = sorted(set(sum((glob.glob(g) for g in a.critic_pretrain), [])))
        samples, metas = load_episodes(files)
        for p in model.parameters():
            p.requires_grad_(False)
        t0 = time.time()
        pre = _copy.copy(a); pre.epochs = a.pretrain_epochs
        up = update(model, ref, vhead, opt, samples, pre, dev, None)
        ev0 = up["value_explained_var"]
        a0 = _copy.copy(a); a0.epochs = 0
        ev1 = update(model, ref, vhead, opt, samples, a0, dev, None)["value_explained_var"]
        for p in model.parameters():
            p.requires_grad_(True)
        rec = dict(iter=-1, critic_pretrain=len(files), samples=len(samples), epochs=a.pretrain_epochs,
                   explained_var_before=ev0, explained_var_after=ev1, value_loss=up.get("vl"),
                   seconds=round(time.time() - t0))
        print(json.dumps(rec), flush=True)
        with open(f"{run_dir}/metrics.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        save_ck(model, vhead, f"{ck_dir}/iter_000.pt", 0)
    for it in range(start, a.iters):
        ck = f"{ck_dir}/iter_{it:03d}.pt"
        if not os.path.exists(ck):
            save_ck(model, vhead, ck, it)
        t0 = time.time()
        files = rollout(ck, a.tokenizer, f"{run_dir}/iter_{it:03d}", a.workers, a.seed + it, a)
        samples, metas = load_episodes(files)
        st = episode_stats(samples, metas)
        t1 = time.time()
        if not samples:
            print(f"iter {it}: no samples"); continue
        a.seed_it = it
        if it < a.value_warmup:                     # critic warm-up: value head only
            for p in model.parameters():
                p.requires_grad_(False)
        up = update(model, ref, vhead, opt, samples, crit if it < a.value_warmup else a, dev, None)
        for p in model.parameters():
            p.requires_grad_(True)
        save_ck(model, vhead, f"{ck_dir}/iter_{it + 1:03d}.pt", it + 1)
        rec = dict(iter=it, rollout_s=round(t1 - t0), update_s=round(time.time() - t1),
                   critic_only=it < a.value_warmup, **st, **up)
        rec["undone_defects"] = sum(s["future"].get("undone_defects", 0) for s in samples
                                    if s["chosen"] and s["future"])
        print(json.dumps(rec), flush=True)
        with open(f"{run_dir}/metrics.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")


def cmd_check(a):
    """recomputed masked log-probs (teacher forcing, full forward) vs the ones
    recorded at sampling time (KV cache, CPU sampling)"""
    model, _ = load_checkpoint(a.checkpoint, "cuda"); model.eval()
    samples, metas = load_episodes(a.files)
    print(f"{len(samples)} samples, {sum(len(s['act']) for s in samples)} actions; meta {metas}")
    print(episode_stats(samples, metas))
    worst = 0.0
    for k in range(0, min(len(samples), 256), 32):
        bt = Batch(samples[k:k + 32], "cuda")
        with torch.no_grad():
            lp, _, _ = masked_logp(model, bt)
        worst = max(worst, float((lp - bt.lp_old).abs().max()))
    print(f"max |log p recomputed - log p sampled| = {worst:.2e}")
    R = [immediate_reward(s) + future_target(s)[0] for s in samples]
    print(f"reward: mean {np.mean(R):.2f}, kinds {dict(zip(*np.unique([future_target(s)[1] for s in samples], return_counts=True)))}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    l = sub.add_parser("loop")
    l.add_argument("--init", required=True)
    l.add_argument("--tokenizer", required=True)
    l.add_argument("--run", required=True)
    l.add_argument("--iters", type=int, default=30)
    l.add_argument("--workers", type=int, default=8)
    l.add_argument("--lateral", type=float, default=10.86)
    l.add_argument("--max_backtracks", type=int, default=40)
    l.add_argument("--episode_timeout", type=float, default=2400)
    l.add_argument("--lr", type=float, default=1e-5)
    l.add_argument("--lr_value", type=float, default=1e-3)
    l.add_argument("--beta", type=float, default=0.02)
    l.add_argument("--ent", type=float, default=0.001)
    l.add_argument("--clip", type=float, default=0.2)
    l.add_argument("--epochs", type=int, default=2)
    l.add_argument("--mb", type=int, default=32)
    l.add_argument("--value_warmup", type=int, default=1, help="first iterations: critic only")
    l.add_argument("--seed", type=int, default=0)
    l.add_argument("--w_final", type=float, default=W_FINAL)
    l.add_argument("--critic_epochs", type=int, default=4, help="epochs of critic-only updates")
    l.add_argument("--pretrain_epochs", type=int, default=10)
    l.add_argument("--critic_pretrain", nargs="*", default=[],
                   help="episode .pkl globs for an offline critic warm start before iteration 0")
    c = sub.add_parser("check")
    c.add_argument("files", nargs="+")
    c.add_argument("--checkpoint", required=True)
    a = ap.parse_args()
    os.chdir(ROOT)                                   # relative paths from the repository root
    {"loop": cmd_loop, "check": cmd_check}[a.cmd](a)
