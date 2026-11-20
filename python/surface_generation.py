"""
Grow amorphous SiO2 on a Si(001) surface with a moving context window.

The model only knows periodic cubes of ~9-11 Å, so a large structure is built
window by window. The structure is periodic in x, y with the window's lateral
size (2a = 10.85 Å for Si); the window slides up along z:

    z ↑  ┌──────────┐ z0 + L
         │ margin   │   kept empty (the model thinks z wraps to the bottom)
         │ fill     │   new atoms: z in [front - below, z0 + L - margin]
         ├──────────┤ front = top of the atoms placed so far
         │ context  │   atoms already placed, given to the model as prefix
         └──────────┘ z0 = front - context

Each window is prompted as
    [BOS] COMP_SiO2 Q_clean L L L  <context atoms, cluster-growing order>
and the model continues with new atoms under DefectGuide masks (grammar,
stoichiometry, clashes, overcoordination, O-O). Distances are minimum-image in
x, y only and include atoms just below the window. After a window, atoms that
no later window can reach are final and must be defect-free; otherwise the
window is resampled (and after too many failures the previous window too).

Usage (repo root):
    venv/bin/python python/surface_generation.py \
        --checkpoint checkpoints/sio2/checkpoint_final_*.pt \
        --tokenizer data/sio2/tokenizer.json --oxide_thickness 20 --device cuda
"""

import argparse
import time

import numpy as np
import torch
from ase import Atoms
from ase.build import bulk
from ase.io import write

from atomic_tokenizer import AtomicTokenizer
from guided_generation import DefectGuide, guided_generate
from structure_defects import BOND_CUTOFFS, find_defects
from train import load_checkpoint

SI_A = 5.431
SIO2_UNITS_PER_A3 = 2.2 / (60.084 / 0.60221)   # 2.2 g/cm^3 -> SiO2 units / Å^3
MAX_BOND = max(BOND_CUTOFFS.values())


# ============================================================
# STRUCTURE STATE (global Cartesian, periodic in x, y)
# ============================================================

def si001_slab(L: float, n_cells_z: int) -> tuple[list, np.ndarray, np.ndarray]:
    """Unreconstructed Si(001) slab, lateral 2a x 2a scaled to L. Returns
    symbols, positions, and a 'fixed' mask for the bottom two layers."""
    s = bulk("Si", "diamond", a=SI_A, cubic=True).repeat((2, 2, n_cells_z))
    pos = s.get_positions()
    pos[:, :2] *= L / (2 * SI_A)
    pos[:, 2] -= pos[:, 2].min()
    layers = np.unique(np.round(pos[:, 2], 3))
    fixed = pos[:, 2] <= layers[1] + 1e-3
    return list(s.get_chemical_symbols()), pos, fixed


def to_atoms(sym, pos, L, vacuum=10.0) -> Atoms:
    zmax = pos[:, 2].max() if len(pos) else 0.0
    return Atoms(sym, positions=pos, cell=[L, L, zmax + vacuum],
                 pbc=(True, True, False))


def defect_atoms(sym, pos, L, check: np.ndarray) -> dict:
    """Defects among atoms with check=True (others still see all neighbours)."""
    out = {}
    for i, label in find_defects(to_atoms(sym, pos, L)):
        if check[i]:
            out[label] = out.get(label, 0) + 1
    return out


# ============================================================
# WINDOW GUIDE
# ============================================================

class WindowGuide(DefectGuide):
    """DefectGuide whose state starts from the context atoms of one window,
    with non-periodic z and a z range for new atoms."""

    def __init__(self, tokenizer, L, counts, ctx_sym, ctx_frac, n_tokenized,
                 z_lo, z_hi, penalty=float("inf")):
        super().__init__(tokenizer, "SiO2", sum(counts.values()),
                         cell_length=L, penalty=penalty)
        self.counts = dict(counts)
        self.z_lo, self.z_hi = z_lo, z_hi            # fractional, window frame
        self.n_tok = n_tokenized                     # first n are in the prompt
        # Preload context (incl. atoms below the window) to get coordinations
        super().reset()
        for s, f in zip(ctx_sym, ctx_frac):
            self.remaining[s] = self.remaining.get(s, 0) + 1
            self.add_atom(s, np.asarray(f))
        self.n_ctx = len(ctx_sym)
        self.remaining = dict(self.counts)
        self._snap = {k: getattr(self, k).copy() for k in
                      ("frac", "n_siO", "n_cat", "has_hf", "n_o")}
        self._snap_sym = list(self.sym)
        self._ctx_ids = []
        for s, f in zip(ctx_sym[:n_tokenized], ctx_frac[:n_tokenized]):
            self._ctx_ids.append(self.vocab.encode_token(f"TYPE_{s}"))
            for a, x in zip("XYZ", f):
                b = min(max(int(np.floor(x * self.n_bins)), 0), self.n_bins - 1)
                self._ctx_ids.append(self.vocab.encode_token(f"{a}_{b}"))

    def reset(self):
        if not hasattr(self, "_snap"):           # during super().__init__
            return super().reset()
        for k, v in self._snap.items():
            setattr(self, k, v.copy())
        self.sym = list(self._snap_sym)
        self.remaining = dict(self.counts)

    def prompt_ids(self):
        return super().prompt_ids() + self._ctx_ids

    def _dists(self, cand_frac):
        d = cand_frac[:, None, :] - self.frac[None, :, :]
        d[..., :2] -= np.round(d[..., :2])            # periodic in x, y only
        return np.linalg.norm(d * self.L, axis=-1)

    def _bad_positions(self, elem, cand_frac):
        bad = super()._bad_positions(elem, cand_frac)
        z = cand_frac[:, 2]
        return bad | (z < self.z_lo) | (z > self.z_hi)

    def new_atoms(self):
        return self.sym[self.n_ctx:], self.frac[self.n_ctx:]


def window_order(frac: np.ndarray, L: float) -> list[int]:
    """Cluster-growing order (as in the tokenizer) with non-periodic z."""
    n = len(frac)
    if n == 0:
        return []

    def dist(f, others):
        d = others - f
        d[:, :2] -= np.round(d[:, :2])
        return np.linalg.norm(d * L, axis=1)

    start = int(np.argmin(dist(np.full(3, 0.5), frac)))
    order, placed = [start], np.zeros(n, bool)
    placed[start] = True
    min_d = dist(frac[start], frac)
    for _ in range(n - 1):
        nxt = int(np.argmin(np.where(placed, np.inf, min_d)))
        order.append(nxt)
        placed[nxt] = True
        min_d = np.minimum(min_d, dist(frac[nxt], frac))
    return order


# ============================================================
# GROWTH
# ============================================================

def build_window(tokenizer, sym, pos, L, args):
    front = pos[:, 2].max()
    z0 = front - args.context
    in_win = (pos[:, 2] >= z0) & (pos[:, 2] < z0 + L)
    below = (pos[:, 2] >= z0 - MAX_BOND) & (pos[:, 2] < z0)

    win_idx = np.where(in_win)[0]
    frac = pos[win_idx] / L
    frac[:, 2] = (pos[win_idx, 2] - z0) / L
    frac[:, :2] %= 1.0
    order = window_order(frac, L)
    win_idx, frac = win_idx[order], frac[order]

    bel_idx = np.where(below)[0]
    bfrac = pos[bel_idx] / L
    bfrac[:, 2] = (pos[bel_idx, 2] - z0) / L
    bfrac[:, :2] %= 1.0

    z_lo = (front - args.below_front - z0) / L
    z_hi = (L - args.margin) / L
    n_su = int(round((z_hi - (front - z0) / L) * L ** 3 * SIO2_UNITS_PER_A3))
    counts = {"Si": n_su, "O": 2 * n_su}

    guide = WindowGuide(tokenizer, L, counts,
                        [sym[i] for i in win_idx] + [sym[i] for i in bel_idx],
                        np.vstack([frac, bfrac]), len(win_idx), z_lo, z_hi,
                        penalty=args.penalty)
    return guide, z0, front


def grow(model, tokenizer, args):
    L = tokenizer.config["cell_min"] + round(
        (2 * SI_A - tokenizer.config["cell_min"]) / tokenizer.config["cell_resolution"]
    ) * tokenizer.config["cell_resolution"]
    sym, pos, fixed = si001_slab(L, args.substrate_cells)
    substrate_top = pos[:, 2].max()
    target_top = substrate_top + args.oxide_thickness
    print(f"Si(001) substrate: {len(sym)} atoms, lateral {L:.2f} Å "
          f"(strain {L / (2 * SI_A) - 1:+.2%}), top at z={substrate_top:.2f} Å")
    print(f"Window {L:.2f} Å: context {args.context} Å, margin {args.margin} Å, "
          f"target oxide {args.oxide_thickness} Å")

    history = []          # (n_atoms before window) for backtracking
    n_tries_total, n_back = 0, 0
    fail_streak = 0
    while pos[:, 2].max() < target_top:
        guide, z0, front = build_window(tokenizer, sym, pos, L, args)
        k = len(history) + 1
        n_new = sum(guide.counts.values())
        print(f"\nwindow {k}: z0={z0:.2f} front={front:.2f} Å, "
              f"{guide.n_tok} context atoms in prompt, +{n_new} new "
              f"(Si {guide.counts['Si']}, O {guide.counts['O']})")

        ok = False
        for t in range(1, args.max_tries + 1):
            n_tries_total += 1
            t0 = time.time()
            atoms, info = guided_generate(model, guide, temperature=args.temperature,
                                          device=args.device)
            if atoms is None:
                print(f"  try {t}: aborted ({info['reason']}) {time.time()-t0:.1f}s")
                continue
            s_new, f_new = guide.new_atoms()
            p_new = f_new * L
            p_new[:, 2] = z0 + f_new[:, 2] * L
            sym2 = sym + list(s_new)
            pos2 = np.vstack([pos, p_new])
            fixed2 = np.concatenate([fixed, np.zeros(len(s_new), bool)])
            # Atoms no later window can bond to are final
            final_z = pos2[:, 2].max() - args.below_front - MAX_BOND
            check = (~fixed2) & (pos2[:, 2] < final_z)
            d = defect_atoms(sym2, pos2, L, check)
            if d:
                print(f"  try {t}: rejected, final-layer defects {d} "
                      f"{time.time()-t0:.1f}s")
                continue
            print(f"  try {t}: accepted, front -> {pos2[:, 2].max():.2f} Å "
                  f"{time.time()-t0:.1f}s")
            history.append(len(sym))
            sym, pos, fixed = sym2, pos2, fixed2
            ok, fail_streak = True, 0
            break

        if not ok:
            fail_streak += 1
            if not history or n_back >= args.max_backtracks:
                print(f"\nGave up at window {k} after {n_tries_total} tries "
                      f"({n_back} backtracks)")
                break
            n = history.pop()
            sym, pos, fixed = sym[:n], pos[:n], fixed[:n]
            n_back += 1
            print(f"  backtracking: dropped window {k - 1}")

    atoms = to_atoms(sym, pos, L)
    atoms.set_tags(fixed.astype(int))      # 1 = fixed bottom Si layers
    print(f"\nGrown {len(atoms)} atoms ({atoms.get_chemical_formula()}), "
          f"{len(history)} windows, {n_tries_total} tries, {n_back} backtracks")
    return atoms, substrate_top


def report(atoms, label, surface_skin=3.0):
    pos = atoms.get_positions()
    top = pos[:, 2].max()
    interior = (atoms.get_tags() == 0) & (pos[:, 2] < top - surface_skin)
    surface = pos[:, 2] >= top - surface_skin
    L = atoms.cell[0, 0]
    sym = atoms.get_chemical_symbols()
    print(f"[{label}] interior defects: {defect_atoms(sym, pos, L, interior) or 'none'}; "
          f"top {surface_skin} Å surface: {defect_atoms(sym, pos, L, surface) or 'none'}")


def relax(atoms, device):
    from ase.constraints import FixAtoms
    from ase.optimize import FIRE
    from mace.calculators import mace_mp
    atoms = atoms.copy()
    atoms.set_constraint(FixAtoms(mask=atoms.get_tags() == 1))
    atoms.calc = mace_mp(model="medium", device=device, default_dtype="float64")
    FIRE(atoms, logfile=None).run(fmax=0.02, steps=1000)
    atoms.info["energy_per_atom"] = atoms.get_potential_energy() / len(atoms)
    atoms.calc = None
    return atoms


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", default="data/surface/SiO2_on_Si001.extxyz")
    p.add_argument("--device", default="cuda")
    p.add_argument("--oxide_thickness", type=float, default=20.0, help="Å")
    p.add_argument("--substrate_cells", type=int, default=2,
                   help="Si(001) slab thickness in conventional cells (a = 5.431 Å)")
    p.add_argument("--context", type=float, default=4.5,
                   help="Å of placed structure below the front shown to the model")
    p.add_argument("--margin", type=float, default=2.0,
                   help="Å at the window top kept empty (z wraps in the model)")
    p.add_argument("--below_front", type=float, default=1.0,
                   help="Å below the front where new atoms may still go")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--penalty", type=float, default=float("inf"))
    p.add_argument("--max_tries", type=int, default=30, help="per window")
    p.add_argument("--max_backtracks", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--relax", action="store_true",
                   help="MACE-relax the result (bottom Si layers fixed)")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    tokenizer = AtomicTokenizer.load(args.tokenizer)
    model, _ = load_checkpoint(args.checkpoint, args.device)
    model.eval()

    atoms, _ = grow(model, tokenizer, args)
    report(atoms, "as generated")
    frames = [atoms]
    if args.relax:
        r = relax(atoms, args.device)
        report(r, "MACE relaxed")
        frames.append(r)

    from pathlib import Path
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write(args.out, frames)
    print(f"Saved to {args.out} ({len(frames)} frame(s): as generated"
          f"{', relaxed' if args.relax else ''})")
