"""
Non-periodic window tokenizer: the model sees an open box of side W (Å), never
a periodic cell. Amorphous structures are built by sliding this window.

Sequence layout:
    [BOS] COMP_<c> Q_<q> KX_<i> KY_<j> KZ_<k> N_Si_<n> N_O_<n> N_Hf_<n>
          (TYPE X Y Z)* [GEN] (TYPE X Y Z)* [EOS]
          `-- context --'      `-- fill --'

  * X/Y/Z_<b>: window-local position bin, x = (b + 0.5) / n_bins * W, no wrap.
  * Known region: an atom is context iff x < tx or y < ty or z < tz, with
    t = level / n_known_levels * W (level 0 = nothing known along that axis).
    This is the shape a raster scan in +x, +y, +z leaves in each window.
    Everything else in the window is the fill region, generated after [GEN].
  * N_<el>_<n>: number of fill atoms per element (lets the guide fix counts).
  * Q_clean iff no atom in the window is defective, judged in the full
    structure the window was cut from (atoms cut at the faces are not defects).
  * Context and fill are each in cluster-growing order (plain distances); the
    fill continues the growth from the context, as at generation time.

Training windows are random crops (random origin, axis permutation and
reflections) of periodic MD cells. W must be smaller than the cell, so a crop
never contains two periodic images of one atom.

Usage:
    python window_tokenizer.py --dataset data/sio2/dataset.extxyz --window 8 \
        --crops_per_frame 32 --out_dir data/sio2_win8
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read
from tqdm import tqdm

from structure_defects import find_defects

WINDOW_CONFIG = {
    "format": "window_v1",
    "window": 8.0,            # Å, side of the cubic window
    "n_bins": 256,            # position bins per axis
    "n_known_levels": 16,     # resolution of the known-region thresholds
    "max_fill_count": 160,    # per element
    "elements": ["Si", "O", "Hf"],
    "compositions": ["SiO2", "HfO2", "HfSiO4", "mixed"],
}
HEADER_LEN = 9    # BOS COMP Q KX KY KZ N_Si N_O N_Hf
AXES = "XYZ"


class WindowTokenizer:
    def __init__(self, config: dict = None):
        self.config = dict(config or WINDOW_CONFIG)
        c = self.config
        toks = ["[PAD]", "[BOS]", "[EOS]", "[UNK]", "[GEN]", "Q_clean", "Q_defect"]
        toks += [f"COMP_{n}" for n in c["compositions"]]
        toks += [f"K{a}_{k}" for a in AXES for k in range(c["n_known_levels"])]
        toks += [f"N_{e}_{n}" for e in c["elements"] for n in range(c["max_fill_count"] + 1)]
        toks += [f"TYPE_{e}" for e in c["elements"]]
        toks += [f"{a}_{b}" for a in AXES for b in range(c["n_bins"])]
        self.token2id = {t: i for i, t in enumerate(toks)}
        self.id2token = {i: t for t, i in self.token2id.items()}
        self.W = c["window"]
        self.n_bins = c["n_bins"]

    # -------- basics --------
    @property
    def vocab_size(self): return len(self.token2id)
    @property
    def pad_id(self): return 0
    @property
    def bos_id(self): return self.token2id["[BOS]"]
    @property
    def eos_id(self): return self.token2id["[EOS]"]
    @property
    def gen_id(self): return self.token2id["[GEN]"]

    def tid(self, tok: str) -> int:
        return self.token2id[tok]

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"config": self.config, "token2id": self.token2id}, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f)["config"])

    def thresholds(self, levels) -> np.ndarray:
        return np.asarray(levels) / self.config["n_known_levels"] * self.W

    def is_known(self, local: np.ndarray, levels) -> np.ndarray:
        t = self.thresholds(levels)
        return ((local < t) & (np.asarray(levels) > 0)).any(axis=1)

    def bin_of(self, x: np.ndarray) -> np.ndarray:
        return np.clip(np.floor(x / self.W * self.n_bins).astype(int), 0, self.n_bins - 1)

    def pos_of_bin(self, b) -> np.ndarray:
        return (np.asarray(b) + 0.5) / self.n_bins * self.W

    # -------- encode / decode --------
    def header(self, comp: str, clean: bool, levels, fill_counts: dict) -> list[int]:
        ids = [self.bos_id, self.tid(f"COMP_{comp}"),
               self.tid("Q_clean" if clean else "Q_defect")]
        ids += [self.tid(f"K{a}_{int(k)}") for a, k in zip(AXES, levels)]
        mx = self.config["max_fill_count"]
        ids += [self.tid(f"N_{e}_{min(fill_counts.get(e, 0), mx)}")
                for e in self.config["elements"]]
        assert len(ids) == HEADER_LEN
        return ids

    def atom_ids(self, sym, local) -> list[int]:
        out = []
        bins = self.bin_of(local)
        for s, b in zip(sym, bins):
            out.append(self.tid(f"TYPE_{s}"))
            out += [self.tid(f"{a}_{int(v)}") for a, v in zip(AXES, b)]
        return out

    def encode_window(self, sym, local, levels, comp, clean) -> list[int]:
        sym = np.asarray(sym)
        known = self.is_known(local, levels)
        order = growth_order(local, known, np.full(3, self.W / 2))
        ctx = [i for i in order if known[i]]
        fill = [i for i in order if not known[i]]
        counts = {e: int((sym[fill] == e).sum()) for e in self.config["elements"]}
        return (self.header(comp, clean, levels, counts)
                + self.atom_ids(sym[ctx], local[ctx]) + [self.gen_id]
                + self.atom_ids(sym[fill], local[fill]) + [self.eos_id])

    def decode(self, ids) -> tuple[list, np.ndarray, int]:
        """-> symbols, local positions (Å), index of the first fill atom."""
        toks = [self.id2token.get(i, "[UNK]") for i in ids[HEADER_LEN:]]
        sym, pos, n_ctx = [], [], None
        k = 0
        while k < len(toks):
            t = toks[k]
            if t == "[GEN]":
                n_ctx = len(sym); k += 1; continue
            if t == "[EOS]":
                break
            if t.startswith("TYPE_") and k + 3 < len(toks):
                sym.append(t[5:])
                pos.append([self.pos_of_bin(int(toks[k + 1 + j].split("_")[1]))
                            for j in range(3)])
                k += 4; continue
            k += 1
        return sym, np.array(pos).reshape(-1, 3), (len(sym) if n_ctx is None else n_ctx)


def growth_order(pos: np.ndarray, known: np.ndarray, centre: np.ndarray) -> list[int]:
    """Cluster growing with plain distances: the known atoms first (from the
    one nearest the window centre), then the rest, each step adding the
    nearest unplaced atom of the current group."""
    n = len(pos)
    if n == 0:
        return []
    order, placed = [], np.zeros(n, bool)
    min_d = np.full(n, np.inf)
    for group in (known, ~known):
        idx = np.where(group)[0]
        if len(idx) == 0:
            continue
        if not order:
            start = idx[np.argmin(np.linalg.norm(pos[idx] - centre, axis=1))]
            order.append(int(start)); placed[start] = True
            min_d = np.linalg.norm(pos - pos[start], axis=1)
        while True:
            cand = np.where(group & ~placed)[0]
            if len(cand) == 0:
                break
            nxt = int(cand[np.argmin(min_d[cand])])
            order.append(nxt); placed[nxt] = True
            min_d = np.minimum(min_d, np.linalg.norm(pos - pos[nxt], axis=1))
    return order


# ============================================================
# CROPS FROM PERIODIC CELLS
# ============================================================

def random_crop(atoms: Atoms, W: float, rng: np.random.Generator,
                defective: np.ndarray):
    """Random non-periodic window of side W from an orthorhombic periodic cell
    (random origin, axis permutation, reflections). Returns symbols, local
    positions, per-atom defect flags."""
    L = np.diag(np.asarray(atoms.get_cell()))
    assert (L > W).all(), f"window {W} Å must be smaller than cell {L}"
    frac = atoms.get_scaled_positions(wrap=True)
    perm = rng.permutation(3)
    frac, L = frac[:, perm], L[perm]
    flip = rng.random(3) < 0.5
    frac = np.where(flip, 1.0 - frac, frac)
    origin = rng.random(3)
    local = ((frac - origin) % 1.0) * L
    inside = (local < W).all(axis=1)
    sym = np.array(atoms.get_chemical_symbols())[inside]
    return sym, local[inside], defective[inside]


def random_levels(rng: np.random.Generator, n_levels: int) -> list[int]:
    """Known-region thresholds as a raster scan produces them: each axis is
    known below a threshold or not at all; sometimes nothing is known."""
    if rng.random() < 0.1:
        return [0, 0, 0]
    lv = [int(rng.integers(1, n_levels)) if rng.random() < 0.6 else 0 for _ in range(3)]
    if not any(lv):
        lv[int(rng.integers(3))] = int(rng.integers(1, n_levels))
    return lv


def composition_of(atoms: Atoms) -> str:
    s = set(atoms.get_chemical_symbols())
    return {frozenset({"Si", "O"}): "SiO2", frozenset({"Hf", "O"}): "HfO2",
            frozenset({"Hf", "Si", "O"}): "HfSiO4"}.get(frozenset(s), "mixed")


def build_dataset(tok: WindowTokenizer, frames: list, crops_per_frame: int,
                  val_frac: float, max_len: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    runs = sorted({a.info.get("label", str(i)) for i, a in enumerate(frames)})
    val_runs = set(rng.choice(runs, max(1, int(round(val_frac * len(runs)))),
                              replace=False))
    out = {"train": [], "val": []}
    stats = defaultdict(int)
    for i, atoms in enumerate(tqdm(frames, desc="Cropping")):
        split = "val" if atoms.info.get("label", str(i)) in val_runs else "train"
        defective = np.zeros(len(atoms), bool)
        for j, _ in find_defects(atoms):
            defective[j] = True
        comp = composition_of(atoms)
        for _ in range(crops_per_frame):
            sym, local, dfl = random_crop(atoms, tok.W, rng, defective)
            if len(sym) == 0:
                continue
            levels = random_levels(rng, tok.config["n_known_levels"])
            ids = tok.encode_window(sym, local, levels, comp, clean=not dfl.any())
            if len(ids) > max_len:
                stats["too_long"] += 1
                continue
            out[split].append(ids)
            stats["clean" if not dfl.any() else "defect"] += 1
    return out, dict(stats), sorted(val_runs)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Non-periodic window crops for training")
    p.add_argument("--dataset", default="data/sio2/dataset.extxyz")
    p.add_argument("--window", type=float, default=8.0)
    p.add_argument("--n_bins", type=int, default=256)
    p.add_argument("--crops_per_frame", type=int, default=32)
    p.add_argument("--val_frac", type=float, default=0.05, help="fraction of runs")
    p.add_argument("--max_len", type=int, default=520)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = dict(WINDOW_CONFIG, window=args.window, n_bins=args.n_bins)
    tok = WindowTokenizer(cfg)
    frames = read(args.dataset, index=":")
    print(f"{len(frames)} frames, window {args.window} Å, vocab {tok.vocab_size}")
    data, stats, val_runs = build_dataset(tok, frames, args.crops_per_frame,
                                          args.val_frac, args.max_len, args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok.save(out / "tokenizer.json")
    lens = [len(s) for s in data["train"]]
    with open(out / "sequences.json", "w") as f:
        json.dump({"vocab_size": tok.vocab_size, "sequences": data["train"],
                   "val_sequences": data["val"], "val_runs": val_runs}, f)
    print(f"train {len(data['train'])}, val {len(data['val'])} "
          f"({len(val_runs)} held-out runs); {stats}")
    print(f"seq len min {min(lens)} mean {np.mean(lens):.0f} max {max(lens)}")
