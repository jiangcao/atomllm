"""Step 0 of plan/actor_critic.md: rejection-sampling fine-tuning (RFT).

sample: K guided completions per MD-crop prompt (growth settings: force_bonds,
        gap mask, T 0.8, top-k 50), judged by the fair-completion rule; clean
        completions (judged atoms > 0, no defect) are written as training
        sequences. Also reports per-shape clean / defect rates, so with
        --split val it is the evaluation (before / after fine-tuning).
build:  merge sample shards + replay of the original training sequences into
        a sequences.json for train.py --init_checkpoint.

usage:
  python/rl/rft.py sample --checkpoint CK --tokenizer TOK --split train \
      --n_prompts 2000 --k 8 --seed 0 --out data/rft/shard0.json
  python/rl/rft.py build --shards data/rft/shard*.json --base data/sio2_iface_win8_z/sequences.json \
      --replay 1.0 --max_per_prompt 4 --out data/rft/sequences.json
"""
import sys, os, json, time, glob, argparse
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import window_generation_v2 as wg2
from window_tokenizer_v2 import WindowZTokenizer
from rl.prompts import load_frames, sample_prompt, SHAPES
from rl.judge import judge

DATA = ["data/sio2_all/dataset.extxyz", "data/interface/dataset.extxyz"]


def cmd_sample(a):
    from train import load_checkpoint
    tok = WindowZTokenizer.load(a.tokenizer)
    model, _ = load_checkpoint(a.checkpoint, a.device); model.eval()
    wg2.FORCE_BONDS = True
    if a.gap_mask:
        wg2.GAP = {("Si", "O"): (2.10, 2.55), ("Si", "Si"): (2.60, 2.85)}
    frames, bad = load_frames(a.data, a.val_runs, a.split)
    print(f"{len(frames)} {a.split} frames", flush=True)
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    key = lambda lv: "x".join(map(str, lv))
    st = {key(s): dict(prompts=0, samples=0, failed=0, clean=0, prompts_with_clean=0,
                       judged=0, defective=0) for s in SHAPES}
    kept, t0, n = [], time.time(), 0
    while n < a.n_prompts:
        batch = []
        while len(batch) < a.prompts_per_batch and n + len(batch) < a.n_prompts:
            pr = sample_prompt(frames, bad, tok, rng)
            if pr is not None:
                batch.append(pr)
        guides = [pr.guide(tok) for pr in batch for _ in range(a.k)]
        res = wg2.z_generate_batch(model, guides, a.temperature, a.top_k, 20, a.device)
        for p, pr in enumerate(batch):
            s = st[key(pr.levels)]
            s["prompts"] += 1
            any_clean = False
            for j in range(a.k):
                g = guides[p * a.k + j]
                ok, info = res[p * a.k + j]
                s["samples"] += 1
                if not ok:
                    s["failed"] += 1
                    continue
                s_new, p_new = g.new_atoms()
                nj, defs = judge(pr, s_new, p_new)
                s["judged"] += nj; s["defective"] += len(defs)
                if nj > 0 and not defs:
                    s["clean"] += 1; any_clean = True
                    kept.append({"prompt": n + p, "levels": pr.levels, "label": pr.label,
                                 "ids": info["ids"] + [tok.eos_id]})
            s["prompts_with_clean"] += any_clean
        n += len(batch)
        tot = {k: sum(v[k] for v in st.values()) for k in ("samples", "clean")}
        print(f"{n}/{a.n_prompts} prompts, {tot['clean']}/{tot['samples']} clean samples "
              f"[{time.time() - t0:.0f}s]", flush=True)
    print("shape        prompts  samples  failed  clean%  prompts_w_clean%  defective_atoms%")
    for k, v in st.items():
        if v["prompts"]:
            print(f"{k:12s} {v['prompts']:7d}  {v['samples']:7d}  {v['failed']:6d}  "
                  f"{100 * v['clean'] / v['samples']:5.1f}  {100 * v['prompts_with_clean'] / v['prompts']:15.1f}  "
                  f"{100 * v['defective'] / max(v['judged'], 1):15.1f}")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump({"args": vars(a), "stats": st, "kept": kept}, open(a.out, "w"))
        print(f"{len(kept)} clean sequences -> {a.out}")


def cmd_build(a):
    rng = np.random.default_rng(a.seed)
    seqs, per = [], {}
    for f in sorted(set(sum((glob.glob(s) for s in a.shards), []))):
        for r in json.load(open(f))["kept"]:
            k = (f, r["prompt"])
            if per.get(k, 0) < a.max_per_prompt:
                per[k] = per.get(k, 0) + 1
                seqs.append(r["ids"])
    base = json.load(open(a.base))
    n_rep = int(a.replay * len(seqs))
    idx = rng.choice(len(base["sequences"]), size=min(n_rep, len(base["sequences"])), replace=False)
    out = seqs + [base["sequences"][i] for i in idx]
    rng.shuffle(out)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"vocab_size": base["vocab_size"], "sequences": out,
               "val_sequences": base["val_sequences"], "val_runs": base["val_runs"]}, open(a.out, "w"))
    print(f"{len(seqs)} RFT sequences ({len(per)} prompts) + {len(idx)} replay -> {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--tokenizer", required=True)
    s.add_argument("--data", nargs="+", default=DATA)
    s.add_argument("--val_runs", default="data/sio2_iface_win8_z/sequences.json")
    s.add_argument("--split", default="train", choices=["train", "val"])
    s.add_argument("--n_prompts", type=int, default=2000)
    s.add_argument("--k", type=int, default=8)
    s.add_argument("--prompts_per_batch", type=int, default=16)
    s.add_argument("--temperature", type=float, default=0.8)
    s.add_argument("--top_k", type=int, default=50)
    s.add_argument("--no_gap_mask", dest="gap_mask", action="store_false")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--device", default="cuda")
    s.add_argument("--out", default=None)
    b = sub.add_parser("build")
    b.add_argument("--shards", nargs="+", required=True)
    b.add_argument("--base", required=True)
    b.add_argument("--replay", type=float, default=1.0)
    b.add_argument("--max_per_prompt", type=int, default=4)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--out", required=True)
    a = ap.parse_args()
    {"sample": cmd_sample, "build": cmd_build}[a.cmd](a)
