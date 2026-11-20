"""
Generation with the internal-coordinate window model (window_tokenizer_v2).

Reuses window_generation's completion test and surface growth unchanged; only
the guide and the sampling loop differ (patched into that module):

  ZWindowGuide : WindowGuide (clash / overcoordination / O-O masks, fill box,
                 env atoms) + ZState (queue, focus, frames) for the v2 grammar
  z_generate   : sampling loop for
                   TYPE R TH PH | [NEXT] | [JUMP] TYPE X Y Z | [EOS]
     * TYPE  : only elements still to place; [JUMP] only when the queue is
               empty; [EOS] forced once all fill atoms are placed
     * TH    : bins where no PH gives a valid position are masked
     * PH    : positions outside the fill box or creating a defect are masked
     * [NEXT]: with --force_bonds, refused while the focus is undercoordinated
               (counting env atoms) and a bonding partner can still be put in
               the fill box next to it
     If no valid continuation exists the atom is resampled; after
     max_atom_retries the focus is popped (or the sample aborted for a jump).

Usage (from the repo root):
    python python/window_generation_v2.py complete \
        --checkpoint checkpoints/sio2_win8_z/checkpoint_best_*.pt \
        --tokenizer data/sio2_win8_z/tokenizer.json \
        --sequences data/sio2_win8_z/sequences.json [--force_bonds]
    python python/window_generation_v2.py surface ... (as window_generation.py)
"""

import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

import window_generation as wg
from window_tokenizer_v2 import WindowZTokenizer, ZState
from guided_generation import _pair, BOND_CUTOFFS, MIN_DIST, MAX_SI_CN, MAX_HF_CN
# no cutoff / minimum distance / gap reaches further than this (Å)
R_NEAR = max(list(BOND_CUTOFFS.values()) + list(MIN_DIST.values()) + [2.85]) + 0.05

FORCE_BONDS = False
# (1) near-miss mask: no new atom at a distance that is neither a bond nor
# clearly non-bonded from an existing atom (the band just beyond the bond
# cutoff, where most rejected atoms had their partner). None = off.
GAP = None   # {("Si","O"): (2.10, 2.55), ("Si","Si"): (2.60, 2.85)}
# (A) complete env atoms first: placed atoms near the box but outside the
# model's window (not in its context, so it cannot bond to them by itself)
# that lack bonds get a partner by a directed [JUMP] at the start of the
# window: element fixed, X/Y/Z restricted to bonding positions next to them
COMPLETE_ENV = False
NEED = {"Si": ("O", 4), "O": ("Si", 2), "Hf": ("O", 6)}
TYPICAL_R = {("Si", "O"): 1.64, ("Hf", "O"): 2.1}


class ZWindowGuide(wg.WindowGuide):
    def __init__(self, tok, levels, counts, ctx_sym, ctx_local, env_sym, env_local,
                 fill_lo, fill_hi, **kw):
        super().__init__(tok, levels, counts, ctx_sym, ctx_local, env_sym, env_local,
                         fill_lo, fill_hi, **kw)
        self.fill_lo, self.fill_hi = np.asarray(fill_lo, float), np.asarray(fill_hi, float)
        self.ctx_sym = list(ctx_sym)
        self.ctx_pos = tok.snap_abs(np.asarray(ctx_local, float).reshape(-1, 3))
        self.n_model_ctx = len(self.ctx_sym)
        self.type_of = {v: k for k, v in self.type_ids.items()}
        ids = lambda p, n: torch.tensor([tok.tid(f"{p}_{k}") for k in range(n)])
        c = tok.config
        self.r_ids, self.th_ids, self.ph_ids = ids("R", c["n_r"]), ids("TH", c["n_th"]), ids("PH", c["n_ph"])
        self.zs = ZState(self.ctx_sym, self.ctx_pos)

    def reset(self):
        super().reset()
        self.zs = ZState(self.ctx_sym, self.ctx_pos)

    # guide index of ZState atom i (guide = ctx + env + fill, zs = ctx + fill)
    def gidx(self, i):
        return i if i < self.n_model_ctx else self.n_ctx + (i - self.n_model_ctx)

    def in_box(self, p):
        return np.all((p >= self.fill_lo) & (p < self.fill_hi), axis=-1)

    def bad(self, elem, p):
        """(m,3) local Å -> bool (m,) invalid. Same result as bad_full, but
        only atoms within R_NEAR of the candidates' bounding box are checked."""
        if self.period is not None:
            return self.bad_full(elem, p)
        out = ~self.in_box(p)
        ok = ~out
        if not ok.any() or not self.sym:
            return out
        q = p[ok]
        P = self.frac * self.L
        rn = max([R_NEAR] + [hi + 0.05 for _, hi in GAP.values()]) if GAP else R_NEAR
        near = np.where(np.all((P >= q.min(0) - rn) & (P <= q.max(0) + rn), axis=1))[0]
        if not len(near):
            return out
        P = P[near]
        sym = np.asarray(self.sym)[near]
        n_siO, n_cat, n_o, has_hf = (self.n_siO[near], self.n_cat[near], self.n_o[near],
                                     self.has_hf[near])
        diff = q[:, None, :] - P[None]
        D = np.sqrt(np.einsum("mnk,mnk->mn", diff, diff))
        mind = np.array([_pair(MIN_DIST, elem, x) for x in sym])
        cut = np.array([_pair(BOND_CUTOFFS, elem, x) for x in sym])
        bad = (D < mind).any(1)
        B = D < cut
        is_si, is_o, is_hf = sym == "Si", sym == "O", sym == "Hf"
        if elem == "Si":
            bad |= (B & (is_si | is_o)).sum(1) > MAX_SI_CN
            bad |= (B & is_si & (n_siO + 1 > MAX_SI_CN)).any(1)
            bad |= (B & is_o & (n_cat + 1 > np.where(has_hf, 4, 2))).any(1)
        elif elem == "O":
            bad |= (B & is_o).any(1)
            bad |= (B & (is_si | is_hf)).sum(1) > np.where((B & is_hf).any(1), 4, 2)
            bad |= (B & is_si & (n_siO + 1 > MAX_SI_CN)).any(1)
            bad |= (B & is_hf & (n_o + 1 > MAX_HF_CN)).any(1)
        elif elem == "Hf":
            bad |= (B & is_o).sum(1) > MAX_HF_CN
            bad |= (B & is_o & (n_cat + 1 > 4)).any(1)
        if GAP:
            for (x, y), (lo, hi) in GAP.items():
                if elem in (x, y):
                    other = y if elem == x else x
                    bad |= ((D >= lo) & (D < hi) & (sym == other)[None]).any(1)
        out[np.where(ok)[0][bad]] = True
        return out

    def bad_full(self, elem, p):
        """reference implementation (all atoms)."""
        out = ~self.in_box(p)
        ok = ~out
        if ok.any():
            out[ok] = self._bad_positions(elem, p[ok] / self.L)
        ok = ~out
        if GAP and ok.any() and self.sym:
            D = self._dists(p[ok] / self.L)
            sym = np.array(self.sym)
            g = np.zeros(ok.sum(), bool)
            for (a, b), (lo, hi) in GAP.items():
                if elem in (a, b):
                    other = b if elem == a else a
                    g |= ((D >= lo) & (D < hi) & (sym == other)[None]).any(1)
            out[np.where(ok)[0][g]] = True
        return out

    def place(self, elem, p, parent=-1):
        self.add_atom(elem, np.asarray(p) / self.L)
        self.zs.add(elem, p, parent)

    def coordination(self, i):
        g, s = self.gidx(i), self.zs.sym[i]
        return {"Si": self.n_siO, "O": self.n_cat, "Hf": self.n_o}[s][g]

    def focus_needs_bond(self):
        """Focus undercoordinated and a partner fits in the fill box?"""
        f = self.zs.focus
        s = self.zs.sym[f]
        if s not in NEED:
            return False
        e, cn = NEED[s]
        if self.remaining.get(e, 0) <= 0 or self.coordination(f) >= cn:
            return False
        fr = self.zs.frame(f)
        r0 = TYPICAL_R.get((s, e), TYPICAL_R.get((e, s), 1.64))
        th = np.linspace(0, np.pi, 33)[1:-1]
        ph = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        T, P = np.meshgrid(th, ph, indexing="ij")
        for r in (r0 - 0.05, r0, r0 + 0.05):
            cand = ZState.cartesian(fr, r, T.ravel(), P.ravel())
            if (~self.bad(e, cand)).any():
                return True
        return False

    def _coord_g(self, g):
        s = self.sym[g]
        return {"Si": self.n_siO, "O": self.n_cat, "Hf": self.n_o}[s][g]

    def needy_env(self, reach=2.10):
        """env atoms within reach of the fill box that lack bonds, nearest
        to the box first -> [(guide index, partner element)]"""
        out = []
        for g in range(self.n_model_ctx, self.n_ctx):
            s = self.sym[g]
            if s not in NEED:
                continue
            e, cn = NEED[s]
            if self._coord_g(g) >= cn:
                continue
            p = self.frac[g] * self.L
            gap = np.linalg.norm(np.maximum(0, np.maximum(self.fill_lo - p, p - self.fill_hi)))
            if gap <= reach:
                out.append((gap, g, e))
        return [(g, e) for _, g, e in sorted(out)]

    def directed_bins(self, g, e, r=(1.52, 1.72)):
        """(m,3) bin indices of valid positions for a new e bonded to atom g"""
        tok = self.tok
        c = tok.pos_of_bin(np.arange(self.n_bins))
        p = self.frac[g] * self.L
        ax = [np.where(self.allowed[k] & (np.abs(c - p[k]) <= r[1]))[0] for k in range(3)]
        if any(len(a) == 0 for a in ax):
            return np.zeros((0, 3), int)
        B = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
        P = c[B]
        d = np.linalg.norm(P - p, axis=1)
        keep = (d >= r[0]) & (d <= r[1])
        B, P = B[keep], P[keep]
        if len(B):
            B = B[~self.bad(e, P)]
        return B

    def coord_ok(self, g):
        s = self.sym[g]
        return s not in NEED or self._coord_g(g) >= NEED[s][1]

    def new_atoms(self):
        return self.zs.sym[self.n_model_ctx:], self.zs.pos[self.n_model_ctx:]


def _sample(logits, allowed_ids, temperature, top_k, mask=None):
    sub = logits[allowed_ids] / temperature
    if mask is not None:
        sub = sub.masked_fill(mask, float("-inf"))
    if torch.isinf(sub).all():
        return None
    if top_k:
        k = min(top_k, int(torch.isfinite(sub).sum()))
        kth = torch.topk(sub, k)[0][-1]
        sub = sub.masked_fill(sub < kth, float("-inf"))
    return int(torch.multinomial(F.softmax(sub, -1), 1))


RECORD_ACTIONS = False   # RL: record every sampled choice (allowed set + log-prob)


def _sample_rec(guide, pos, logits, allowed_ids, temperature, top_k, mask=None):
    """_sample + record (token, allowed token ids, log-prob) at sequence position
    pos. The log-prob is that of the masked softmax at this temperature, exact
    for top_k None (the RL policy)."""
    k = _sample(logits, allowed_ids, temperature, top_k, mask)
    if k is not None and RECORD_ACTIONS:
        ok = torch.ones(len(allowed_ids), dtype=torch.bool) if mask is None else ~mask.cpu()
        al = allowed_ids.cpu()[ok]
        lsm = (logits.cpu()[al] / temperature).log_softmax(-1)
        tok = int(allowed_ids[k])
        guide.act[pos] = (tok, al.numpy().astype(np.int16), float(lsm[int((al == tok).nonzero()[0, 0])]))
    return k


def z_generate_co(model, guide: ZWindowGuide, temperature=0.8, top_k=50,
               max_atom_retries=20, device="cpu"):
    guide.reset()
    guide.act = {}
    tok, zs = guide.tok, guide.zs
    ids = guide.prompt_ids()
    max_len = model.config.max_seq_len
    dev = lambda t: t.to(device)
    r_ids, th_ids, ph_ids = dev(guide.r_ids), dev(guide.th_ids), dev(guide.ph_ids)
    xyz_ids = [dev(guide.axis_ids[a]) for a in "XYZ"]
    n_retry_total, retries = 0, 0
    seeds = guide.needy_env() if COMPLETE_ENV else []

    while True:
        if sum(guide.remaining.values()) == 0:
            act = [(p_, t_, al, lp) for p_, (t_, al, lp) in sorted(guide.act.items())
                   if p_ < len(ids) and ids[p_] == t_] if RECORD_ACTIONS else None
            return True, {"retries": n_retry_total, "seeds": len(guide.needy_env()) if COMPLETE_ENV else 0,
                          "ids": list(ids), "act": act}
        if seeds:
            g, e = seeds[0]
            B = (guide.directed_bins(g, e)
                 if guide.remaining.get(e, 0) > 0 and not guide.coord_ok(g) else np.zeros((0, 3), int))
            if not len(B) or len(ids) + 6 >= max_len:
                seeds.pop(0)
                continue
            seq = ids + [tok.jump_id, guide.type_ids[e]]
            for ax in range(3):
                lg = (yield seq)
                m = torch.ones(len(xyz_ids[ax]), dtype=torch.bool, device=device)
                m[torch.from_numpy(np.unique(B[:, ax])).to(device)] = False
                kk = _sample_rec(guide, len(seq), lg, xyz_ids[ax], temperature, top_k, m)
                B = B[B[:, ax] == kk]
                seq.append(int(xyz_ids[ax][kk]))
            ids = seq
            guide.place(e, tok.pos_of_bin(B[0]))
            if guide.coord_ok(g):
                seeds.pop(0)
            continue
        if len(ids) + 5 >= max_len:
            return None, {"reason": "max_seq_len", "retries": n_retry_total}
        zs = guide.zs
        f = zs.focus
        lg = (yield ids)
        # ---- slot 0: TYPE / NEXT / JUMP ----
        opts = []
        if f is not None:
            opts += [guide.type_ids[e] for e, n in guide.remaining.items() if n > 0]
            if not (FORCE_BONDS and retries == 0 and guide.focus_needs_bond()):
                opts.append(tok.next_id)
        else:
            opts.append(tok.jump_id)
        opts_t = torch.tensor(opts, device=device)
        k = _sample_rec(guide, len(ids), lg, opts_t, temperature, top_k)
        first = opts[k]
        if first == tok.next_id:
            ids.append(first); zs.next(); retries = 0
            continue

        if first == tok.jump_id:
            seq = ids + [first]
            lg = (yield seq)
            tids = torch.tensor([guide.type_ids[e] for e, n in guide.remaining.items() if n > 0],
                                device=device)
            t = int(tids[_sample_rec(guide, len(seq), lg, tids, temperature, top_k)])
            elem = guide.type_of[t]
            seq.append(t)
            b = []
            for ax in range(3):
                lg = (yield seq)
                allowed = torch.from_numpy(guide.allowed[ax]).to(device)
                m = ~allowed
                if ax == 2:
                    cx, cy = tok.pos_of_bin(np.array(b))
                    cz = tok.pos_of_bin(np.arange(tok.n_bins))
                    cand = np.column_stack([np.full_like(cz, cx), np.full_like(cz, cy), cz])
                    m = m | torch.from_numpy(guide.bad(elem, cand)).to(device)
                kk = _sample_rec(guide, len(seq), lg, xyz_ids[ax], temperature, top_k, m)
                if kk is None:
                    break
                b.append(kk); seq.append(int(xyz_ids[ax][kk]))
            if len(b) < 3:
                retries += 1; n_retry_total += 1
                if retries > max_atom_retries:
                    return None, {"reason": "jump failed", "retries": n_retry_total}
                continue
            ids = seq
            guide.place(elem, tok.pos_of_bin(np.array(b)))
            retries = 0
            continue

        # ---- child of focus: TYPE R TH PH ----
        elem = guide.type_of[first]
        fr = zs.frame(f)
        seq = ids + [first]
        kr = _sample_rec(guide, len(seq), (yield seq), r_ids, temperature, top_k)
        seq.append(int(r_ids[kr]))
        r = tok.r_val(kr)
        T, P = np.meshgrid(tok.th_val(np.arange(len(th_ids))),
                           tok.ph_val(np.arange(len(ph_ids))), indexing="ij")
        cand = ZState.cartesian(fr, r, T.ravel(), P.ravel())
        badg = guide.bad(elem, cand).reshape(T.shape)
        th_mask = torch.from_numpy(badg.all(1)).to(device)
        kt = _sample_rec(guide, len(seq), (yield seq), th_ids, temperature, top_k, th_mask)
        if kt is not None:
            seq.append(int(th_ids[kt]))
            kp = _sample_rec(guide, len(seq), (yield seq), ph_ids, temperature, top_k,
                             torch.from_numpy(badg[kt]).to(device))
        if kt is None or kp is None:
            retries += 1; n_retry_total += 1
            if retries > max_atom_retries:      # give up on this focus
                ids.append(tok.next_id); zs.next(); retries = 0
            continue
        seq.append(int(ph_ids[kp]))
        ids = seq
        guide.place(elem, cand.reshape(*T.shape, 3)[kt, kp], parent=f)
        retries = 0




@torch.no_grad()
def z_generate(model, guide: ZWindowGuide, temperature=0.8, top_k=50,
               max_atom_retries=20, device="cpu"):
    return z_generate_batch(model, [guide], temperature, top_k, max_atom_retries, device)[0]


@torch.no_grad()
def z_generate_batch(model, guides, temperature=0.8, top_k=50, max_atom_retries=20,
                     device="cpu"):
    """(5) sample one window per guide, all in lock-step: every step the
    pending sequences go through the model as one right-padded batch (causal
    attention: the logits at each sequence's last token are exact).
    Returns [(out, info)] per guide."""
    # the guided sampling (masks, top-k, multinomial) runs on the CPU: one
    # device->host copy of the batch logits per step instead of many small
    # synchronising GPU ops per row
    gens = [z_generate_co(model, g, temperature, top_k, max_atom_retries, "cpu") for g in guides]
    res, pend = [None] * len(gens), {}
    for k, g in enumerate(gens):
        try:
            pend[k] = next(g)
        except StopIteration as e:
            res[k] = e.value
    if USE_KV_CACHE:
        return _drive_cached(model, gens, res, pend, device)
    while pend:
        ks = list(pend)
        T = max(len(pend[k]) for k in ks)
        x = np.zeros((len(ks), T), dtype=np.int64)
        for r, k in enumerate(ks):
            x[r, :len(pend[k])] = pend[k]
        logits = model(torch.from_numpy(x).to(device))[0]
        last = torch.tensor([len(pend[k]) - 1 for k in ks], device=device)
        lg = logits[torch.arange(len(ks), device=device), last].float().cpu()
        for r, k in enumerate(ks):
            try:
                pend[k] = gens[k].send(lg[r])
            except StopIteration as e:
                res[k] = e.value
                del pend[k]
    return res


USE_KV_CACHE = True   # --no_kv_cache: recompute the full sequence every step


def _drive_cached(model, gens, res, pend, device):
    """Lock-step driver with a key-value cache. Each row keeps the tokens it
    has cached; a request that extends them feeds only the new tokens, one
    that diverges (retry / discarded branch) rewinds to the common prefix."""
    cache = model.alloc_cache(len(gens), device)
    ctoks = [[] for _ in gens]
    while pend:
        ks = list(pend)
        starts, chunks = [], []
        for k in ks:
            s, c = pend[k], ctoks[k]
            if len(c) < len(s) and s[:len(c)] == c:
                p = len(c)
            else:                                     # rewind
                p, m = 0, min(len(c), len(s) - 1)
                while p < m and s[p] == c[p]:
                    p += 1
            p = min(p, len(s) - 1)                    # always feed >= 1 token
            starts.append(p); chunks.append(s[p:])
            ctoks[k] = list(s)
        N = max(len(c) for c in chunks)
        x = np.zeros((len(ks), N), dtype=np.int64)
        for r, c in enumerate(chunks):
            x[r, :len(c)] = c
        meta = torch.tensor([ks, starts, [len(c) for c in chunks]], dtype=torch.long).to(device)
        lg = model.forward_cached(torch.from_numpy(x).to(device), meta[0], meta[1], meta[2],
                                  cache).float().cpu()
        for r, k in enumerate(ks):
            try:
                pend[k] = gens[k].send(lg[r])
            except StopIteration as e:
                res[k] = e.value
                del pend[k]
    return res


# patch into the v1 driver
wg.WindowGuide = ZWindowGuide
wg.guided_generate = z_generate
wg.guided_generate_batch = z_generate_batch
wg.WindowTokenizer = WindowZTokenizer

if __name__ == "__main__":
    if "--force_bonds" in sys.argv:
        sys.argv.remove("--force_bonds")
        FORCE_BONDS = True
    if "--complete_env" in sys.argv:
        sys.argv.remove("--complete_env")
        COMPLETE_ENV = True
    TOP_K_CLI = 50
    if "--top_k" in sys.argv:                      # 0 -> no top-k (exact RL policy)
        k_ = sys.argv.index("--top_k"); TOP_K_CLI = int(sys.argv[k_ + 1]) or None
        del sys.argv[k_:k_ + 2]
    RECORD_FILE = None
    if "--record" in sys.argv:                     # RL episode recording (python/rl/recorder.py)
        k_ = sys.argv.index("--record"); RECORD_FILE = sys.argv[k_ + 1]
        del sys.argv[k_:k_ + 2]
        RECORD_ACTIONS = True
        from rl.recorder import Recorder
        wg.RECORDER = Recorder()
    if "--no_kv_cache" in sys.argv:
        sys.argv.remove("--no_kv_cache")
        USE_KV_CACHE = False
    if "--gap_mask" in sys.argv:
        sys.argv.remove("--gap_mask")
        GAP = {("Si", "O"): (2.10, 2.55), ("Si", "Si"): (2.60, 2.85)}
    src = open(wg.__file__).read()
    main = src[src.index('if __name__ == "__main__":'):]
    main = main.replace('if __name__ == "__main__":', "if True:")
    print(f"window v2 generation, force_bonds={FORCE_BONDS}, gap_mask={GAP}, "
          f"complete_env={COMPLETE_ENV}", flush=True)
    if TOP_K_CLI != 50:
        import functools
        wg.guided_generate_batch = functools.partial(z_generate_batch, top_k=TOP_K_CLI)
    exec(compile(main, wg.__file__, "exec"), vars(wg))
    if RECORD_FILE:
        wg.RECORDER.save(RECORD_FILE)
