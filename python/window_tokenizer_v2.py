"""
Window tokenizer v2: fill atoms in internal (Z-matrix-like) coordinates,
grown breadth-first over the bond graph.

Same open (non-periodic) window, header and known region as window_tokenizer
(v1). Only the fill section changes:

    [BOS] COMP Q KX KY KZ N_Si N_O N_Hf
    (TYPE X Y Z)*                        context, absolute bins, growth order
    [GEN]
    ( TYPE R TH PH                       child of the current focus atom
    | [NEXT]                             focus done -> pop next from queue
    | [JUMP] TYPE X Y Z )*               atom not reachable by bonds (absolute)
    [EOS]

Queue / focus (identical rule in encoder, decoder and generation guide):
  * initial queue: context atoms that are "open", i.e. have fewer than their
    ideal number of bonds (Si 4, O 2, Hf 7) among the context atoms, in
    context order. The first queue entry is the focus.
  * every placed fill atom is appended to the queue.
  * [NEXT] pops the focus. When the queue is empty and fill atoms remain, the
    encoder emits [JUMP] for the unplaced fill atom nearest to any placed atom
    (nearest to the window centre if nothing is placed); it joins the queue.
  * [EOS] right after the last fill atom.

Children of a focus are its bonded (true graph), still unplaced fill atoms,
emitted one at a time: next = smallest PH in the current frame.

Frame for a child of focus f (all from reconstructed positions, so the
decoder rebuilds exactly the encoder's frame):
  ref1 = parent of f (the atom it was emitted from) if any, else the nearest
         placed atom != f;  u = unit(ref1 - f)
  ref2 = nearest placed atom != f, ref1 not (anti)parallel to u; w = ref2 - f
         (window x / y axis if none)
  e1 = unit(w - (w.u) u),  e2 = u x e1
  child = f + R (cos TH u + sin TH (cos PH e1 + sin PH e2))
TH is the angle to the parent bond (O-Si-O ~109 deg, Si-O-Si ~145 deg).

The encoder measures R/TH/PH of the TRUE child against the RECONSTRUCTED
focus and frame, so quantisation error stays per atom (no drift along
chains). A child whose R falls outside the R range is emitted as [JUMP].

Usage:
    python window_tokenizer_v2.py --dataset data/sio2/dataset.extxyz \
        --window 8 --crops_per_frame 32 --out_dir data/sio2_win8_z
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from structure_defects import BOND_CUTOFFS, find_defects
from window_tokenizer import (AXES, HEADER_LEN, WINDOW_CONFIG, WindowTokenizer,
                              composition_of, growth_order, random_crop,
                              random_levels)

WINDOW_Z_CONFIG = dict(
    WINDOW_CONFIG,
    format="window_v2",
    r_min=1.2, r_max=2.6, n_r=140,     # 0.01 Å
    n_th=128,                          # 1.4 deg
    n_ph=256,                          # 1.4 deg
)
IDEAL_CN = {"Si": 4, "O": 2, "Hf": 7}
# per-atom defect kinds for unlikelihood targets (bit flags)
UNDER, OVER = 1, 2
UNDER_LABELS = {"Si_dangling", "O_nonbridging", "Hf_undercoord"}


def defect_kinds(atoms) -> np.ndarray:
    """Per atom: 0 clean, UNDER (missing bonds) | OVER (extra / O-O bonds)."""
    k = np.zeros(len(atoms), int)
    for i, lab in find_defects(atoms):
        k[i] |= UNDER if lab in UNDER_LABELS else OVER
    return k


def _cut(a, b):
    return BOND_CUTOFFS.get((a, b), BOND_CUTOFFS.get((b, a), 0.0))


def bond_graph(sym, pos) -> list[set]:
    """Non-periodic bond graph with the find_defects rules (Si..Si only a bond
    if both Si have < 4 O neighbours)."""
    sym = np.asarray(sym)
    n = len(sym)
    nb = [set() for _ in range(n)]
    if n < 2:
        return nb
    D = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    C = np.array([[_cut(a, b) for b in sym] for a in sym])
    B = (D < C) & ~np.eye(n, dtype=bool)
    n_o = (B & (sym == "O")[None]).sum(1)
    si = sym == "Si"
    edge = si[:, None] & si[None] & ((n_o[:, None] >= 4) | (n_o[None] >= 4))
    B &= ~edge
    for i, j in zip(*np.nonzero(B)):
        nb[i].add(int(j))
    return nb


class ZState:
    """Placed atoms (reconstructed positions), parents, queue, focus. Drives
    encoding, decoding and the generation guide identically."""

    def __init__(self, ctx_sym, ctx_pos):
        self.sym = list(ctx_sym)
        self.pos = np.asarray(ctx_pos, float).reshape(-1, 3).copy()
        self.parent = [-1] * len(self.sym)
        self.n_ctx = len(self.sym)
        nb = bond_graph(self.sym, self.pos)
        self.queue = [i for i in range(self.n_ctx)
                      if len(nb[i]) < IDEAL_CN.get(self.sym[i], 0)]

    @property
    def focus(self):
        return self.queue[0] if self.queue else None

    def next(self):
        self.queue.pop(0)

    def _nearest(self, f, exclude, u=None):
        if len(self.pos) == 0:
            return None
        d = np.linalg.norm(self.pos - self.pos[f], axis=1)
        d[list(exclude)] = np.inf
        if u is not None:
            v = self.pos - self.pos[f]
            nv = np.linalg.norm(v, axis=1)
            cos = np.abs(v @ u) / np.maximum(nv, 1e-9)
            d[cos > 0.95] = np.inf
        k = int(np.argmin(d))
        return k if np.isfinite(d[k]) else None

    def frame(self, f):
        """-> (origin, u, e1, e2) for children of atom f."""
        o = self.pos[f]
        r1 = self.parent[f] if self.parent[f] >= 0 else self._nearest(f, {f})
        if r1 is None:
            u = np.array([0.0, 0.0, 1.0])
        else:
            u = self.pos[r1] - o
            u /= np.linalg.norm(u)
        r2 = self._nearest(f, {f} | ({r1} if r1 is not None else set()), u)
        if r2 is not None:
            w = self.pos[r2] - o
        else:
            w = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.95 else np.array([0.0, 1.0, 0.0])
        e1 = w - (w @ u) * u
        e1 /= np.linalg.norm(e1)
        return o, u, e1, np.cross(u, e1)

    @staticmethod
    def internal(frame, p):
        o, u, e1, e2 = frame
        v = p - o
        r = np.linalg.norm(v)
        th = np.arccos(np.clip(v @ u / r, -1, 1))
        ph = np.arctan2(v @ e2, v @ e1) % (2 * np.pi)
        return r, th, ph

    @staticmethod
    def cartesian(frame, r, th, ph):
        o, u, e1, e2 = frame
        r, th, ph = np.broadcast_arrays(np.asarray(r, float), np.asarray(th, float),
                                        np.asarray(ph, float))
        d = (np.cos(th)[..., None] * u
             + np.sin(th)[..., None] * (np.cos(ph)[..., None] * e1
                                         + np.sin(ph)[..., None] * e2))
        return o + r[..., None] * d

    def add(self, s, p, parent=-1):
        self.sym.append(s)
        self.pos = np.vstack([self.pos, p])
        self.parent.append(parent)
        self.queue.append(len(self.sym) - 1)
        return len(self.sym) - 1


class WindowZTokenizer(WindowTokenizer):
    def __init__(self, config: dict = None):
        super().__init__(config or WINDOW_Z_CONFIG)
        c = self.config
        extra = ["[NEXT]", "[JUMP]"]
        extra += [f"R_{k}" for k in range(c["n_r"])]
        extra += [f"TH_{k}" for k in range(c["n_th"])]
        extra += [f"PH_{k}" for k in range(c["n_ph"])]
        for t in extra:
            self.token2id[t] = len(self.token2id)
        self.id2token = {i: t for t, i in self.token2id.items()}

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f)["config"])

    @property
    def next_id(self): return self.token2id["[NEXT]"]
    @property
    def jump_id(self): return self.token2id["[JUMP]"]

    # -------- internal-coordinate bins --------
    def r_bin(self, r):
        c = self.config
        return int(np.floor((r - c["r_min"]) / (c["r_max"] - c["r_min"]) * c["n_r"]))

    def r_val(self, k):
        c = self.config
        return c["r_min"] + (np.asarray(k) + 0.5) * (c["r_max"] - c["r_min"]) / c["n_r"]

    def th_bin(self, th):
        n = self.config["n_th"]
        return int(np.clip(np.floor(th / np.pi * n), 0, n - 1))

    def th_val(self, k):
        return (np.asarray(k) + 0.5) * np.pi / self.config["n_th"]

    def ph_bin(self, ph):
        n = self.config["n_ph"]
        return int(np.floor(ph / (2 * np.pi) * n)) % n

    def ph_val(self, k):
        return (np.asarray(k) + 0.5) * 2 * np.pi / self.config["n_ph"]

    def quantise_child(self, frame, p):
        """True position -> (bins) or None if R out of range."""
        r, th, ph = ZState.internal(frame, p)
        kr = self.r_bin(r)
        if not 0 <= kr < self.config["n_r"]:
            return None
        return kr, self.th_bin(th), self.ph_bin(ph)

    def recon_child(self, frame, kr, kt, kp):
        return ZState.cartesian(frame, self.r_val(kr), self.th_val(kt), self.ph_val(kp))

    def snap_abs(self, p):
        return self.pos_of_bin(self.bin_of(np.asarray(p)))

    # -------- encode --------
    def encode_window(self, sym, local, levels, comp, clean, return_recon=False,
                      kinds=None):
        """kinds (per atom, defect_kinds codes): if given, also return the
        unlikelihood targets: positions in ids of
          * the [NEXT] that closes an UNDER atom (ctx or fill): the decision
            to leave it with missing bonds
          * the angle tokens (TH, PH) or [JUMP] X Y Z of every defective fill
            atom: the place where it was put (R and TYPE are not blamed)."""
        sym = np.asarray(sym)
        local = np.asarray(local, float)
        known = self.is_known(local, levels)
        order = growth_order(local, known, np.full(3, self.W / 2))
        ctx = [i for i in order if known[i]]
        fill = [i for i in order if not known[i]]
        counts = {e: int((sym[fill] == e).sum()) for e in self.config["elements"]}
        ids = self.header(comp, clean, levels, counts)
        ids += self.atom_ids(sym[ctx], local[ctx]) + [self.gen_id]

        nb = bond_graph(sym, local)                  # true topology
        st = ZState(sym[ctx], self.snap_abs(local[ctx]))
        state_of = {g: k for k, g in enumerate(ctx)}  # original idx -> state idx
        orig_of = list(ctx)
        unplaced = set(fill)
        ul = []
        bad = (lambda g: kinds is not None and kinds[g])
        while unplaced:
            f = st.focus
            if f is None:                            # jump
                placed = np.array(orig_of, int)
                if len(placed):
                    cand = sorted(unplaced)
                    d = np.linalg.norm(local[cand][:, None] - local[placed][None], axis=-1).min(1)
                else:
                    cand = sorted(unplaced)
                    d = np.linalg.norm(local[cand] - self.W / 2, axis=1)
                g = cand[int(np.argmin(d))]
                b = self.bin_of(local[g])
                if bad(g):
                    ul += [len(ids) + 2, len(ids) + 3, len(ids) + 4]
                ids += [self.jump_id, self.tid(f"TYPE_{sym[g]}")]
                ids += [self.tid(f"{a}_{int(v)}") for a, v in zip(AXES, b)]
                state_of[g] = st.add(sym[g], self.pos_of_bin(b))
                orig_of.append(g)
                unplaced.discard(g)
                continue
            kids = [g for g in nb[orig_of[f]] if g in unplaced]
            if not kids:
                if bad(orig_of[f]) and kinds[orig_of[f]] & UNDER:
                    ul.append(len(ids))
                ids.append(self.next_id)
                st.next()
                continue
            fr = st.frame(f)
            # next child: smallest PH in the current frame
            phs = [ZState.internal(fr, local[g])[2] for g in kids]
            g = kids[int(np.argmin(phs))]
            q = self.quantise_child(fr, local[g])
            if q is None:                            # R out of range -> jump
                b = self.bin_of(local[g])
                if bad(g):
                    ul += [len(ids) + 2, len(ids) + 3, len(ids) + 4]
                ids += [self.jump_id, self.tid(f"TYPE_{sym[g]}")]
                ids += [self.tid(f"{a}_{int(v)}") for a, v in zip(AXES, b)]
                state_of[g] = st.add(sym[g], self.pos_of_bin(b))
            else:
                kr, kt, kp = q
                if bad(g):
                    ul += [len(ids) + 2, len(ids) + 3]
                ids += [self.tid(f"TYPE_{sym[g]}"), self.tid(f"R_{kr}"),
                        self.tid(f"TH_{kt}"), self.tid(f"PH_{kp}")]
                state_of[g] = st.add(sym[g], self.recon_child(fr, kr, kt, kp), parent=f)
            orig_of.append(g)
            unplaced.discard(g)
        ids.append(self.eos_id)
        if return_recon:
            return ids, st, orig_of
        if kinds is not None:
            return ids, ul
        return ids

    # -------- decode --------
    def decode(self, ids):
        """-> symbols, local positions (Å), index of the first fill atom."""
        toks = [self.id2token.get(i, "[UNK]") for i in ids[HEADER_LEN:]]
        k = 0
        sym, pos = [], []
        while k < len(toks) and toks[k] != "[GEN]":
            if toks[k].startswith("TYPE_") and k + 3 < len(toks):
                sym.append(toks[k][5:])
                pos.append([self.pos_of_bin(int(toks[k + 1 + j].split("_")[1])) for j in range(3)])
                k += 4
            else:
                k += 1
        st = ZState(sym, np.array(pos).reshape(-1, 3))
        k += 1
        while k < len(toks):
            t = toks[k]
            if t == "[EOS]":
                break
            if t == "[NEXT]":
                if st.queue:
                    st.next()
                k += 1
            elif t == "[JUMP]" and k + 4 < len(toks):
                b = [int(toks[k + 2 + j].split("_")[1]) for j in range(3)]
                st.add(toks[k + 1][5:], self.pos_of_bin(np.array(b)))
                k += 5
            elif t.startswith("TYPE_") and k + 3 < len(toks) and st.focus is not None:
                kr, kt, kp = (int(toks[k + 1 + j].split("_")[1]) for j in range(3))
                f = st.focus
                st.add(t[5:], self.recon_child(st.frame(f), kr, kt, kp), parent=f)
                k += 4
            else:
                k += 1
        return list(st.sym), st.pos, st.n_ctx


def build_dataset(tok, frames, crops_per_frame, val_frac, max_len, seed=0,
                  keep_val_runs=None, val_prefix=None):
    """crops_per_frame: int or list (one per frame). keep_val_runs: fixed
    held-out runs (e.g. from an earlier dataset); then val_frac applies only
    to runs whose label starts with val_prefix (new data)."""
    rng = np.random.default_rng(seed)
    runs = sorted({a.info.get("label", str(i)) for i, a in enumerate(frames)})
    if keep_val_runs is None:
        val_runs = set(rng.choice(runs, max(1, int(round(val_frac * len(runs)))), replace=False))
    else:
        val_runs = set(keep_val_runs)
        new = [r for r in runs if val_prefix and r.startswith(val_prefix)]
        if new:
            val_runs |= set(rng.choice(new, max(1, int(round(val_frac * len(new)))), replace=False))
    if isinstance(crops_per_frame, int):
        crops_per_frame = [crops_per_frame] * len(frames)
    out = {"train": [], "val": []}
    ul_out = {"train": [], "val": []}       # unlikelihood target positions per sequence
    stats = defaultdict(int)
    for i, atoms in enumerate(tqdm(frames, desc="Cropping")):
        split = "val" if atoms.info.get("label", str(i)) in val_runs else "train"
        defective = defect_kinds(atoms)
        comp = composition_of(atoms)
        for _ in range(crops_per_frame[i]):
            sym, local, dfl = random_crop(atoms, tok.W, rng, defective)
            if len(sym) == 0:
                continue
            levels = random_levels(rng, tok.config["n_known_levels"])
            ids, ul = tok.encode_window(sym, local, levels, comp, clean=not dfl.any(),
                                        kinds=dfl)
            if len(ids) > max_len:
                stats["too_long"] += 1
                continue
            out[split].append(ids)
            ul_out[split].append(ul)
            stats["ul_targets"] += len(ul)
            stats["ul_next"] += sum(ids[p] == tok.next_id for p in ul)
            stats["clean" if not dfl.any() else "defect"] += 1
            stats["jump"] += ids.count(tok.jump_id)
            stats["next"] += ids.count(tok.next_id)
    return out, ul_out, dict(stats), sorted(val_runs)


if __name__ == "__main__":
    from ase.io import read
    p = argparse.ArgumentParser(description="Window crops, internal-coordinate fill")
    p.add_argument("--dataset", nargs="+", default=["data/sio2/dataset.extxyz"],
                   help="extxyz files, optionally FILE:CROPS_PER_FRAME")
    p.add_argument("--window", type=float, default=8.0)
    p.add_argument("--crops_per_frame", type=int, default=32)
    p.add_argument("--val_frac", type=float, default=0.05, help="fraction of runs")
    p.add_argument("--keep_val_runs", default=None,
                   help="sequences.json whose val_runs stay held out")
    p.add_argument("--val_prefix", default=None,
                   help="with --keep_val_runs: hold out val_frac of runs with this label prefix")
    p.add_argument("--max_len", type=int, default=520)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    tok = WindowZTokenizer(dict(WINDOW_Z_CONFIG, window=args.window))
    frames, cpf = [], []
    for spec in args.dataset:
        path, _, n = spec.partition(":")
        fr = read(path, index=":")
        frames += fr
        cpf += [int(n) if n else args.crops_per_frame] * len(fr)
        print(f"{path}: {len(fr)} frames, {cpf[-1]} crops/frame")
    keep = json.load(open(args.keep_val_runs))["val_runs"] if args.keep_val_runs else None
    print(f"{len(frames)} frames, window {args.window} Å, vocab {tok.vocab_size}")
    data, ul, stats, val_runs = build_dataset(tok, frames, cpf, args.val_frac, args.max_len,
                                          args.seed, keep_val_runs=keep,
                                          val_prefix=args.val_prefix)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok.save(out / "tokenizer.json")
    lens = [len(s) for s in data["train"]]
    with open(out / "sequences.json", "w") as f:
        json.dump({"vocab_size": tok.vocab_size, "sequences": data["train"],
                   "val_sequences": data["val"], "val_runs": val_runs,
                   # unlikelihood targets (train.py --ul_weight): token positions of
                   # defect decisions; scored with the label slot set to q_clean_id
                   "ul_positions": ul["train"], "val_ul_positions": ul["val"],
                   "q_pos": 2, "q_clean_id": tok.tid("Q_clean")}, f)
    print(f"train {len(data['train'])}, val {len(data['val'])} "
          f"({len(val_runs)} held-out runs); {stats}")
    print(f"seq len min {min(lens)} mean {np.mean(lens):.0f} max {max(lens)}")
