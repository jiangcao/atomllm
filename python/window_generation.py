"""
Generation with the non-periodic window model (window_tokenizer.py).

Nothing is periodic: the window is an open box, and the generated structure
has open boundaries. A large structure is built by a raster scan of fill
boxes (x fastest, then y, then z); each window = its fill box extended by the
context depth c on the low side of each axis, so the known region is exactly
"x < c or y < c or z < c" as in training.

WindowGuide reuses DefectGuide's masks (clashes, overcoordination, O-O) with
plain distances to every placed atom near the window (also outside it), and
restricts new atoms to the fill box. Undercoordination is checked once an
atom is out of reach of every remaining fill box; a defect there rejects the
window (backtracking after repeated failures).

Modes:
  complete : refill the fill region of crops of held-out MD frames, put the
             result back into the full periodic cell and count defects there
             (the in-distribution quality test)
  surface  : grow SiO2 on an open Si(001) patch
"""

import argparse
import copy
import json
import time

import numpy as np
import torch
from ase import Atoms
from ase.build import bulk
from ase.io import read, write

from guided_generation import DefectGuide, guided_generate
from structure_defects import BOND_CUTOFFS, find_defects
from train import load_checkpoint
from window_tokenizer import WindowTokenizer, growth_order

MAX_BOND = max(BOND_CUTOFFS.values())
SI_A = 5.431
SIO2_UNITS_PER_A3 = 2.2 / (60.084 / 0.60221)
RECORDER = None   # RL episode recorder (python/rl/recorder.py), set by window_generation_v2 --record


class _VocabView:
    def __init__(self, tok): self.tok = tok
    def decode_id(self, i): return self.tok.id2token[i]


class WindowGuide(DefectGuide):
    """
    ctx_sym/ctx_local : atoms given to the model (window-local Å, known region)
    env_sym/env_local : other placed atoms near the fill box (any position,
                        e.g. outside the window) - for the masks only
    fill_lo/fill_hi   : window-local box where new atoms may go
    """

    def __init__(self, tok: WindowTokenizer, levels, counts: dict,
                 ctx_sym, ctx_local, env_sym, env_local, fill_lo, fill_hi,
                 comp="SiO2", penalty=float("inf"), period=None):
        """period: true periodicity (Å, per axis) of the structure the window
        sits in - only for the completion test; None = open boundaries."""
        self.tok, self.vocab = tok, _VocabView(tok)
        self.period = None if period is None else np.asarray(period) / tok.W
        self.n_bins, self.L = tok.n_bins, tok.W
        self.cell = np.diag([tok.W] * 3)
        self.penalty = penalty
        self.type_ids = {e: tok.tid(f"TYPE_{e}") for e in tok.config["elements"]}
        self.id2elem = {v: k for k, v in self.type_ids.items()}
        self.axis_ids = {a: torch.tensor([tok.tid(f"{a}_{b}") for b in range(self.n_bins)])
                         for a in "XYZ"}
        self.eos = tok.eos_id
        self.counts = {e: int(counts.get(e, 0)) for e in tok.config["elements"]}
        self.levels, self.comp = list(levels), comp
        centres = tok.pos_of_bin(np.arange(self.n_bins))
        self.allowed = [(centres >= fill_lo[k]) & (centres < fill_hi[k]) for k in range(3)]

        DefectGuide.reset(self)
        for s, p in list(zip(ctx_sym, ctx_local)) + list(zip(env_sym, env_local)):
            self.remaining[s] = self.remaining.get(s, 0) + 1
            self.add_atom(s, np.asarray(p) / self.L)
        self.n_ctx = len(self.sym)
        self._snap = (list(self.sym), self.frac.copy(), self.n_siO.copy(),
                      self.n_cat.copy(), self.has_hf.copy(), self.n_o.copy())
        self.remaining = dict(self.counts)
        self._prompt = (tok.header(comp, True, self.levels, self.counts)
                        + tok.atom_ids(ctx_sym, np.asarray(ctx_local).reshape(-1, 3))
                        + [tok.gen_id])

    def reset(self):
        if not hasattr(self, "_snap"):
            return DefectGuide.reset(self)
        s = self._snap
        self.sym, self.frac = list(s[0]), s[1].copy()
        self.n_siO, self.n_cat, self.has_hf, self.n_o = (s[2].copy(), s[3].copy(),
                                                         s[4].copy(), s[5].copy())
        self.remaining = dict(self.counts)

    def prompt_ids(self):
        return list(self._prompt)

    def _dists(self, cand_frac):
        d = cand_frac[:, None, :] - self.frac[None, :, :]
        if self.period is not None:
            d -= np.round(d / self.period) * self.period
        return np.linalg.norm(d * self.L, axis=-1)

    def mask(self, logits, group):
        out = super().mask(logits, group)
        slot = len(group)
        if slot in (1, 2, 3):
            ids = self.axis_ids["XYZ"[slot - 1]].to(logits.device)
            bad = torch.from_numpy(~self.allowed[slot - 1]).to(logits.device)
            out[ids[bad]] = float("-inf")
        return out

    def to_atoms(self):
        return True

    def new_atoms(self):
        return self.sym[self.n_ctx:], self.frac[self.n_ctx:] * self.L


# ============================================================
# COMPLETION TEST (in distribution, held-out runs)
# ============================================================

def crop_with_transform(atoms, W, rng):
    """Random crop (as window_tokenizer.random_crop) that also returns the
    transform back to cell coordinates and the permuted cell lengths."""
    L = np.diag(np.asarray(atoms.get_cell()))
    frac = atoms.get_scaled_positions(wrap=True)
    perm, flip, origin = rng.permutation(3), rng.random(3) < 0.5, rng.random(3)
    f = frac[:, perm]
    f = np.where(flip, 1.0 - f, f)
    Lp = L[perm]
    local = ((f - origin) % 1.0) * Lp
    inside = np.where((local < W).all(1))[0]

    def to_cell(loc):
        g = (loc / Lp + origin) % 1.0
        g = np.where(flip, 1.0 - g, g)
        out = np.empty_like(g)
        out[:, perm] = g
        return out @ np.asarray(atoms.get_cell())
    return inside, local, to_cell, Lp


def completion_test(model, tok, frames, levels_list, n_per, temperature, device, seed=0):
    """Refill the fill region of clean crops of held-out frames and count
    defects of the window atoms in the full periodic cell. The model sees an
    open window; the guide knows the cell's true periodicity."""
    rng = np.random.default_rng(seed)
    W = tok.W
    bad_atoms = [{i for i, _ in find_defects(a)} for a in frames]
    for levels in levels_list:
        res = {"clean": 0, "defective": 0, "aborted": 0}
        dsum, t0, done = {}, time.time(), 0
        t = tok.thresholds(levels) * (np.asarray(levels) > 0)
        while done < n_per:
            fi = int(rng.integers(len(frames)))
            atoms = frames[fi]
            inside, local, to_cell, Lp = crop_with_transform(atoms, W, rng)
            if bad_atoms[fi] & set(inside.tolist()):
                continue                       # original crop not clean
            known = tok.is_known(local[inside], levels)
            ctx_i, fill_i = inside[known], inside[~known]
            if len(fill_i) == 0:
                continue
            done += 1
            if len(ctx_i):
                ctx_i = ctx_i[growth_order(local[ctx_i], np.ones(len(ctx_i), bool),
                                           np.full(3, W / 2))]
            sym = np.array(atoms.get_chemical_symbols())
            env_i = np.setdiff1d(np.arange(len(atoms)), inside)
            counts = {e: int((sym[fill_i] == e).sum()) for e in tok.config["elements"]}
            guide = WindowGuide(tok, levels, counts, sym[ctx_i], local[ctx_i],
                                sym[env_i], local[env_i], t, np.full(3, W), period=Lp)
            out, info = guided_generate(model, guide, temperature=temperature, device=device)
            if out is None:
                res["aborted"] += 1
                continue
            s_new, p_new = guide.new_atoms()
            keep = np.setdiff1d(np.arange(len(atoms)), fill_i)
            new = Atoms(list(sym[keep]) + list(s_new),
                        positions=np.vstack([atoms.get_positions()[keep], to_cell(p_new)]),
                        cell=atoms.get_cell(), pbc=True)
            win = set(np.searchsorted(keep, ctx_i).tolist()) | set(range(len(keep), len(new)))
            d = {}
            for i, lab in find_defects(new):
                if i in win:
                    d[lab] = d.get(lab, 0) + 1
            res["clean" if not d else "defective"] += 1
            for k, v in d.items():
                dsum[k] = dsum.get(k, 0) + v
        fill = "x".join(f"{W - x:.1f}" for x in t)
        print(f"levels {levels} (fill box {fill} Å): {res['clean']}/{n_per} clean, "
              f"{res['defective']} defective, {res['aborted']} aborted; "
              f"defects {dsum}  [{time.time() - t0:.0f}s]", flush=True)


# ============================================================
# SURFACE GROWTH (open boundaries everywhere)
# ============================================================

def si001_block(Lx, Ly, n_cells_z):
    nx, ny = int(np.ceil(Lx / SI_A)), int(np.ceil(Ly / SI_A))
    s = bulk("Si", "diamond", a=SI_A, cubic=True).repeat((nx, ny, n_cells_z))
    pos = s.get_positions()
    pos[:, 2] -= pos[:, 2].max()            # top layer at z = 0
    return list(s.get_chemical_symbols()), pos, nx * SI_A, ny * SI_A


def substrate_from_frame(spec, lateral, keep_above=1.0):
    """Relaxed Si substrate from a Si/SiO2 interface MD frame
    (batch_melt_quench SiO2Si: periodic Si(001) slab / oxide sandwich with a
    per-atom 'substrate' array). spec = FILE or FILE@INDEX. The oxide is
    removed, the slab is unwrapped along z, its TOP interface becomes the
    growth surface (highest substrate atom at z = 0) and it is repeated
    laterally to cover `lateral` Å (whole MD cells)."""
    path, _, idx = spec.partition("@")
    a = read(path, index=int(idx) if idx else -1)
    sub = a.arrays["substrate"].astype(bool)
    L = np.diag(a.cell.array)
    # keep only Si still bonded into the slab: Si-Si connected (bond cutoff) to the
    # always-fixed core and with >= 2 Si neighbours; substrate Si pulled into
    # the oxide during MD (oxidised, 0-1 Si neighbours) are dropped
    from ase.neighborlist import neighbor_list
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    si = a[sub]
    i, j = neighbor_list("ij", si, {("Si", "Si"): BOND_CUTOFFS[("Si", "Si")]})
    n_nb = np.bincount(i, minlength=len(si))
    ok = n_nb >= 2
    keep_edges = ok[i] & ok[j]
    g = coo_matrix((np.ones(keep_edges.sum()), (i[keep_edges], j[keep_edges])), shape=(len(si),) * 2)
    _, lab = connected_components(g, directed=False)
    core = si.arrays["fixed_always"].astype(bool)
    keep = ok & np.isin(lab, np.unique(lab[core]))
    n_drop = int((~keep).sum())
    # unwrap z with the slab: the largest empty gap between slab Si along z is
    # the oxide; everything above the cut belongs below the slab bottom
    zs = np.sort(si.get_positions(wrap=True)[keep][:, 2])
    gaps = np.diff(np.concatenate([zs, [zs[0] + L[2]]]))
    k = np.argmax(gaps)
    z_cut = zs[k] + gaps[k] / 2
    P = a.get_positions(wrap=True)
    zu = np.where(P[:, 2] > z_cut, P[:, 2] - L[2], P[:, 2])
    slab_idx = np.where(sub)[0][keep]
    top, bot = zu[slab_idx].max(), zu[slab_idx].min()
    # the relaxed slab plus the interface layer bonded into it: every atom of
    # the frame between the slab bottom and keep_above Å over the top slab Si
    # (interfacial O / suboxide Si fill the pits of the rough surface); the
    # bottom interface's oxide lies below `bot` and is dropped
    sel = np.zeros(len(a), bool); sel[slab_idx] = True
    sel |= (zu >= bot) & (zu <= top + keep_above)
    sym1 = np.array(a.get_chemical_symbols())[sel]
    p = P[sel].copy(); p[:, 2] = zu[sel]
    p[:, 2] -= p[:, 2].max()                   # highest kept atom at z = 0
    n = max(1, int(round(lateral / L[0])))
    pos = np.vstack([p + np.array([i * L[0], j * L[1], 0]) for i in range(n) for j in range(n)])
    sym = list(sym1) * (n * n)
    n_o = int((sym1 == "O").sum())
    print(f"substrate from {spec}: {len(slab_idx)} slab Si per cell ({n_drop} oxidised/detached "
          f"dropped) + {int(sel.sum()) - len(slab_idx)} interface atoms ({n_o} O) up to "
          f"{keep_above:.1f} Å above the slab; {n}x{n} cells; frame clean={a.info.get('defect_free')}")
    return sym, pos, n * L[0], n * L[1]


# ------------------------------------------------------------
# relax-then-judge and dead-end detection for surface growth
# ------------------------------------------------------------

def _fib_sphere(n=256):
    k = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * k / n)
    th = np.pi * (1 + 5 ** 0.5) * k
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


_SPHERE = _fib_sphere()
# dead_ends clearance thresholds (Å) for a hypothetical new partner:
# new O vs O / vs saturated Si, new Si vs Si / vs saturated O, and vs an
# undercoordinated atom of the other element (it could bond instead)
DEADEND_THR = {"O_O": 2.3, "O_Si": 2.2, "Si_Si": 2.6, "Si_O": 2.2, "under": 1.45, "r": 1.62}


def missing_bonds(sym, pos, idx):
    """Missing bonds of atoms idx, with the rules of structure_defects
    (coordination: Si needs 4 partners from O, Si-Si bonds and accepted
    stretched Si-Si contacts; O needs 2 Si). Negative = overcoordinated.
    Evaluated on the open cluster within 6 Å of the atoms."""
    from structure_defects import coordination
    sym = np.asarray(sym)
    idx = np.asarray(idx, int)
    out = np.zeros(len(idx), int)
    if not len(idx):
        return out
    d = np.linalg.norm(pos[:, None] - pos[idx][None], axis=-1).min(1)
    cl = np.where(d < 6.0)[0]
    _, _, counts = coordination(Atoms(list(sym[cl]), positions=pos[cl], pbc=False))
    where = {int(g): k for k, g in enumerate(cl)}
    for r, i in enumerate(idx):
        c = counts[where[int(i)]]
        if sym[i] == "Si":
            out[r] = 4 - (c["O"] + c["Si"])
        elif sym[i] == "O":
            out[r] = 2 - (c["Si"] + c["Hf"])
    return out


def dead_ends(sym, pos, idx, open_lo, open_hi, need):
    """Atoms of idx (missing need[k] > 0 bonds) for which no partner can be
    placed any more: no point at the Si-O bond length (1.62 Å) that lies in a
    not-yet-filled box (open_lo/open_hi) and keeps clear of existing atoms.
    Undercoordinated atoms may be approached (they could bond too)."""
    sym = np.asarray(sym)
    dead = []
    if not len(open_lo):
        return [i for i, m in zip(idx, need) if m > 0]
    for i, m in zip(idx, need):
        if m <= 0:
            continue
        cand_el = "O" if sym[i] == "Si" else "Si"
        q = pos[i] + DEADEND_THR["r"] * _SPHERE
        inbox = np.zeros(len(q), bool)
        for a in range(0, len(open_lo), 64):
            lo, hi = open_lo[a:a + 64], open_hi[a:a + 64]
            inbox |= np.any(np.all((q[:, None] >= lo[None]) & (q[:, None] < hi[None]), axis=2), axis=1)
        if not inbox.any():
            dead.append(i)
            continue
        q = q[inbox]
        near = np.where(np.linalg.norm(pos - pos[i], axis=1) < 5.0)[0]
        near = near[near != i]
        d = np.linalg.norm(q[:, None] - pos[near][None], axis=-1)
        se = sym[near]
        if cand_el == "O":     # new O: clear of O, and of saturated Si (else overcoordination)
            thr = np.where(se == "O", DEADEND_THR["O_O"], DEADEND_THR["O_Si"])
        else:                  # new Si: clear of Si, and of saturated O
            thr = np.where(se == "Si", DEADEND_THR["Si_Si"], DEADEND_THR["Si_O"])
        # partners that are themselves undercoordinated may bond to the new atom
        under = missing_bonds(sym, pos, near) > 0
        thr = np.where(under & (se != cand_el), DEADEND_THR["under"], thr)
        if not np.all(d >= thr[None], axis=1).any():
            dead.append(i)
    return dead


def local_relax(calc, sym, pos, movable, frozen_shell=6.0, fmax=0.1, steps=100,
                springs=(), spring_rt=1.70, spring_k=10.0):
    """MACE FIRE relaxation of atoms `movable`, all other atoms within
    frozen_shell Å fixed (open cluster). springs: atom pairs (full indices)
    held together by a Hookean restraint beyond spring_rt Å. Returns new full
    position array."""
    from ase.constraints import FixAtoms, Hookean
    from ase.optimize import FIRE
    movable = np.asarray(movable, int)
    d = np.linalg.norm(pos[:, None] - pos[movable][None], axis=-1).min(1)
    cl = np.where(d < frozen_shell)[0]
    mov_in = np.isin(cl, movable)
    a = Atoms([sym[i] for i in cl], positions=pos[cl], pbc=False)
    where = {int(g): k for k, g in enumerate(cl)}
    cons = [FixAtoms(indices=np.where(~mov_in)[0])]
    cons += [Hookean(where[i], where[j], k=spring_k, rt=spring_rt)
             for i, j in springs if i in where and j in where]
    a.set_constraint(cons)
    a.calc = calc
    FIRE(a, logfile=None, maxstep=0.1).run(fmax=fmax, steps=steps)
    out = pos.copy()
    out[cl] = a.get_positions()
    return out


def snap_near_misses(sym, pos, atoms_idx, movable, r0=1.62, r_max=2.70):
    """(2) For each undercoordinated atom of atoms_idx, find the nearest
    undercoordinated partner of the other element (Si<->O) at a near-miss
    distance (cutoff .. r_max) and move the pair (the movable ones) to r0.
    Returns (new positions, [(i, j)] pairs)."""
    sym = np.asarray(sym)
    pos = pos.copy()
    mov = np.zeros(len(sym), bool); mov[np.asarray(movable, int)] = True
    pairs, used = [], set()
    for i in atoms_idx:
        if i in used or sym[i] not in ("Si", "O") or missing_bonds(sym, pos, [i])[0] <= 0:
            continue
        e = "O" if sym[i] == "Si" else "Si"
        d = np.linalg.norm(pos - pos[i], axis=1)
        cand = np.where((sym == e) & (d >= BOND_CUTOFFS[("Si", "O")]) & (d < r_max))[0]
        cand = [j for j in cand[np.argsort(d[cand])] if j not in used and (mov[i] or mov[j])]
        cand = [j for j in cand if missing_bonds(sym, pos, [j])[0] > 0]
        if not cand:
            continue
        j = cand[0]
        v = pos[j] - pos[i]
        dist = np.linalg.norm(v)
        shift = (dist - r0) * v / dist
        if mov[i] and mov[j]:
            pos[i] += shift / 2; pos[j] -= shift / 2
        elif mov[i]:
            pos[i] += shift
        else:
            pos[j] -= shift
        pairs.append((int(i), int(j))); used |= {int(i), int(j)}
    return pos, pairs


def grow_surface(model, tok, args):
    W = tok.W
    lev = args.context_level
    c = lev / tok.config["n_known_levels"] * W
    s = W - c
    if args.substrate_frame:
        sym, pos, Lx, Ly = substrate_from_frame(args.substrate_frame, args.lateral,
                                                 args.substrate_keep_above)
    else:
        sym, pos, Lx, Ly = si001_block(args.lateral, args.lateral, args.substrate_cells)
    z_bot = pos[:, 2].min()
    n_sub = len(sym)
    z0, z1 = 0.3, args.oxide_thickness
    sx = sy = s
    Lp = 0.0
    if args.continue_from:
        # continue a smaller grown patch: it occupies substrate tile (0,0) =
        # the first n_ps substrate atoms; keep its (relaxed) substrate and its
        # oxide, and fill only the rest. Lateral boxes are shrunk so that the
        # old patch edge Lp lies on the box grid.
        prev = read(args.continue_from)
        pt = prev.get_tags() == 1
        n_ps = int(pt.sum())
        assert pt[:n_ps].all(), "previous run: substrate atoms first"
        n_t = max(1, int(round(Lx / (np.ptp(prev.positions[:n_ps, 0]) + 2.0))))
        n_t = int(round(np.sqrt(len(sym) / n_ps))) if len(sym) % n_ps == 0 else n_t
        Lp = Lx / n_t
        # align the previous (vacuum-centred) frame with this one: most
        # frequent same-element displacement between its substrate and tile (0,0)
        cur = pos[:len(sym) // (n_t * n_t)]
        csym = np.asarray(sym[:len(cur)]); psym = np.asarray(prev.get_chemical_symbols()[:n_ps])
        dv = (prev.positions[:n_ps, None] - cur[None])[psym[:, None] == csym[None]]
        keys, cnt = np.unique(np.round(dv / 0.05).astype(int), axis=0, return_counts=True)
        shift = keys[np.argmax(cnt)] * 0.05
        shift = np.median(dv[np.all(np.abs(dv - shift) < 0.1, axis=1)], axis=0)
        pp = prev.positions - shift
        # substrate = the previous run's own (relaxed) substrate, tiled; the
        # frame extraction may keep other atoms (e.g. after a cutoff change)
        sym = list(psym) * (n_t * n_t)
        pos = np.vstack([pp[:n_ps] + np.array([i * Lp, j * Lp, 0]) for i in range(n_t) for j in range(n_t)])
        n_sub = len(sym)
        z_bot = pos[:, 2].min()
        m = int(np.ceil(Lp / s - 1e-6))
        sx = sy = Lp / m
        sym = list(sym) + list(prev.get_chemical_symbols()[n_ps:])
        pos = np.vstack([pos, pp[n_ps:]])
        print(f"continuing {args.continue_from}: {len(prev) - n_ps} oxide atoms on tile (0,0) "
              f"{Lp:.2f} x {Lp:.2f} Å (shift {np.round(shift, 2)}); lateral boxes {sx:.2f} Å", flush=True)
    n_base = len(sym)          # substrate + continued oxide: given, never regenerated
    nx, ny, nz = int(np.ceil(Lx / sx - 1e-6)), int(np.ceil(Ly / sy - 1e-6)), int(np.ceil((z1 - z0) / s))
    cells = [(i, j, k) for k in range(nz) for j in range(ny) for i in range(nx)
             if not ((i + 1) * sx <= Lp + 1e-6 and (j + 1) * sy <= Lp + 1e-6)]

    def box(cell):
        i, j, k = cell
        lo = np.array([i * sx, j * sy, z0 + k * s])
        hi = np.minimum(lo + [sx, sy, s], [Lx, Ly, z1])
        return lo, hi

    # boundary skin: lateral faces, top surface, substrate bottom - not checked
    def interior(p):
        return ((p[:, 0] > MAX_BOND) & (p[:, 0] < Lx - MAX_BOND)
                & (p[:, 1] > MAX_BOND) & (p[:, 1] < Ly - MAX_BOND)
                & (p[:, 2] > z_bot + 2 * MAX_BOND) & (p[:, 2] < z1 - MAX_BOND))

    print(f"Open Si(001) patch {Lx:.2f} x {Ly:.2f} Å, {n_sub} Si; oxide to z={z1} Å; "
          f"window {W} Å, context {c:.1f} Å, fill boxes {s:.1f} Å: "
          f"{nx}x{ny}x{nz} -> {len(cells)} windows", flush=True)

    # an atom is final (checked) once no remaining fill box can still bond to
    # it. Growth places Si and O and the guide masks overcoordination, so a
    # missing bond can only be supplied as a Si-O bond: reach = Si-O cutoff
    # (2.10 Å). A larger reach (MAX_BOND = Hf-Si 3.0 Å, or Si-Si 2.45 Å)
    # checks atoms at boxes that cannot fix them -> endless rejections.
    reach = BOND_CUTOFFS[("Si", "O")]
    rng = np.random.default_rng(args.seed)
    # substrate atoms no fill box can bond to are given as they are (final, never checked)
    los = np.array([box(cc)[0] for cc in cells]); his = np.array([box(cc)[1] for cc in cells])
    dmin0 = np.full(len(sym), np.inf)
    for a0 in range(0, len(los), 64):
        gap = np.maximum(0, np.maximum(los[None, a0:a0 + 64] - pos[:, None], pos[:, None] - his[None, a0:a0 + 64]))
        dmin0 = np.minimum(dmin0, np.linalg.norm(gap, axis=-1).min(1))
    history, final = [], dmin0 > reach
    calc = None
    if args.relax:
        from mace.calculators import mace_mp
        calc = mace_mp(model="medium", device=args.device, default_dtype="float32")
    stats = {"raw_ok": 0, "relax_rescued": 0, "rejected": 0, "deadend_rejects": 0}
    traj = []

    def snap(event, win, blo, bhi):
        if args.traj:
            f = Atoms(sym, positions=pos, pbc=False)
            f.set_tags((np.arange(len(sym)) < n_sub).astype(int))
            f.info.update(event=event, window=int(win), n_windows=len(cells), tries=n_tries,
                          backtracks=n_back, box_lo=" ".join(f"{v:.3f}" for v in blo),
                          box_hi=" ".join(f"{v:.3f}" for v in bhi))
            traj.append(f)
    # substrate atoms that already lack a bond and cannot get a partner even
    # with every box open are given defects of the substrate: final, never checked
    sub_open = np.where(~final & (np.arange(len(sym)) < n_base) & interior(pos))[0]
    need0 = missing_bonds(sym, pos, sub_open)
    given = dead_ends(sym, pos, sub_open, los, his, need0)
    final[given] = True
    print(f"substrate: {int((need0 > 0).sum())} open atoms lack bonds, "
          f"{len(given)} cannot be completed (given substrate defects)", flush=True)
    o_debt = 0.0
    n = 0
    n_back, n_tries = 0, 0
    last_culprit = None
    while n < len(cells):
        lo, hi = box(cells[n])
        w0 = lo - c
        local = pos - w0
        in_win = np.all((local >= 0) & (local < W), axis=1)
        # at the low lateral domain edge nothing is known along that axis
        levels = [lev if (k == 2 or lo[k] > 0) else 0 for k in range(3)]
        fill_lo = np.where(np.asarray(levels) > 0, c, lo - w0)
        fill_hi = hi - w0
        known = in_win & np.any((local < c) & (np.asarray(levels) > 0), axis=1)
        ctx = np.where(known)[0]
        if len(ctx):
            ctx = ctx[growth_order(local[ctx], np.ones(len(ctx), bool), np.full(3, W / 2))]
        near = np.all((local > fill_lo - 2 * MAX_BOND) & (local < fill_hi + 2 * MAX_BOND), axis=1)
        env = np.where(near & ~known)[0]

        vol = np.prod(hi - lo)
        x = vol * SIO2_UNITS_PER_A3
        n_si = int(np.floor(x) + (rng.random() < x - np.floor(x)))
        n_o = int(round(2 * n_si + o_debt))
        o_debt += 2 * n_si - n_o
        counts = {"Si": n_si, "O": max(n_o, 0)}
        guide = WindowGuide(tok, levels, counts,
                            [sym[i] for i in ctx], local[ctx],
                            [sym[i] for i in env], local[env], fill_lo, fill_hi)

        ok = False
        blame = {}       # substrate / earlier-window atom -> tries it made fail
        pre_dead = set()
        if args.deadend:
            # open atoms near this window that are dead ends already before it
            # (all boxes from this one on still open): not this window's fault
            cur_lo = np.array([box(cc)[0] for cc in cells[n:]]); cur_hi = np.array([box(cc)[1] for cc in cells[n:]])
            near0 = np.all((pos > lo - 2 * MAX_BOND) & (pos < hi + 2 * MAX_BOND), axis=1)
            c0 = np.where(near0 & ~final & interior(pos))[0]
            pre_dead = set(dead_ends(sym, pos, c0, cur_lo, cur_hi, missing_bonds(sym, pos, c0)))
        # defective atoms before this window (given defects stay given)
        pre_bad = {i for i, _ in find_defects(Atoms(sym, positions=pos, pbc=False))}
        rem = cells[n + 1:]
        open_lo = np.array([box(cc)[0] for cc in rem]).reshape(-1, 3)
        open_hi = np.array([box(cc)[1] for cc in rem]).reshape(-1, 3)
        lo_w, hi_w = lo - 2 * MAX_BOND, hi + 2 * MAX_BOND

        def finality(p2):
            """atoms out of reach of all remaining fill boxes are final"""
            if not rem:
                return np.ones(len(p2), bool)
            dmin = np.full(len(p2), np.inf)
            for a in range(0, len(open_lo), 64):
                gap = np.maximum(0, np.maximum(open_lo[None, a:a + 64] - p2[:, None],
                                               p2[:, None] - open_hi[None, a:a + 64]))
                dmin = np.minimum(dmin, np.linalg.norm(gap, axis=-1).min(1))
            return dmin > reach

        def evaluate(sym2, p2, now_final, final2):
            """defects of atoms becoming final + dead ends of open atoms near the window
            + already-final atoms that were clean and whose surroundings changed
            (placed or relaxed atoms within 3 Å): a relaxation must not damage
            atoms judged earlier"""
            check = now_final & ~final2 & interior(p2)
            moved = np.linalg.norm(p2[:len(pos)] - pos, axis=1) > 0.02
            touched = np.concatenate([moved, np.ones(len(p2) - len(pos), bool)])
            if touched.any():
                fi = np.where(final2 & interior(p2))[0]
                fi = fi[~np.isin(fi, list(pre_bad))]
                if len(fi):
                    dt = np.linalg.norm(p2[fi][:, None] - p2[touched][None], axis=-1).min(1)
                    check[fi[dt < 3.0]] = True
            bad_atoms = {}
            if check.any():
                a2 = Atoms(sym2, positions=p2, pbc=False)
                for i, lab in find_defects(a2):
                    if check[i]:
                        bad_atoms[i] = lab
            if args.deadend:
                near_w = np.all((p2 > lo_w) & (p2 < hi_w), axis=1)
                cand = np.where(near_w & ~now_final & ~final2 & interior(p2))[0]
                need = missing_bonds(sym2, p2, cand)
                for i in dead_ends(sym2, p2, cand, open_lo, open_hi, need):
                    if i not in pre_dead:
                        bad_atoms.setdefault(i, "dead_end")
            return bad_atoms

        def open_need(sym2, p2, now_final, final2):
            """(5) score: bonds still missing on open atoms around the window
            (fewer = easier for the next windows)"""
            near_w = np.all((p2 > lo_w) & (p2 < hi_w), axis=1)
            cand = np.where(near_w & ~now_final & ~final2)[0]
            return int(np.clip(missing_bonds(sym2, p2, cand), 0, None).sum())

        K = max(1, args.n_cand)
        batched = K > 1 and "guided_generate_batch" in globals()
        n_rounds = max(1, int(np.ceil(args.max_tries / K))) if batched else args.max_tries
        for t in range(1, n_rounds + 1):
            if batched:
                guides = [guide] + [copy.deepcopy(guide) for _ in range(K - 1)]
                outs = guided_generate_batch(model, guides, temperature=args.temperature,
                                             device=args.device)
            else:
                guides = [guide]
                outs = [guided_generate(model, guide, temperature=args.temperature,
                                        device=args.device)]
            n_tries += len(guides)
            stats["samples"] = stats.get("samples", 0) + len(guides)
            cands = []
            for gk, (g, (out, info)) in enumerate(zip(guides, outs)):
                if out is None:
                    continue
                s_new, p_new = g.new_atoms()
                sym2 = sym + list(s_new)
                pos2 = np.vstack([pos, p_new + w0]) if len(s_new) else pos
                final2 = np.concatenate([final, np.zeros(len(s_new), bool)])
                now_final = finality(pos2)
                c_ = dict(sym2=sym2, pos2=pos2, final2=final2, now_final=now_final,
                          new_idx=np.arange(len(sym), len(sym2)), relaxed=False, g=gk)
                # --one_go: accept whatever the model samples (no checks, no backtracking)
                c_["bad"] = {} if args.one_go else evaluate(sym2, pos2, now_final, final2)
                c_["raw_bad"] = dict(c_["bad"])
                cands.append(c_)
            if not cands:
                continue
            clean = [c_ for c_ in cands if not c_["bad"]]
            stats["raw_clean"] = stats.get("raw_clean", 0) + len(clean)
            chosen = None
            if clean:
                chosen = min(clean, key=lambda c_: open_need(c_["sym2"], c_["pos2"],
                                                             c_["now_final"], c_["final2"]))
                stats["raw_ok"] += 1
            elif calc is not None:
                # (2) repair the most promising candidates: snap near-miss
                # pairs to bond length, relax with springs on them, then free
                for c_ in sorted(cands, key=lambda c_: len(c_["bad"]))[:args.n_relax]:
                    sym2, pos2, final2, new_idx = c_["sym2"], c_["pos2"], c_["final2"], c_["new_idx"]
                    dn = np.linalg.norm(pos2[:, None] - pos2[new_idx][None], axis=-1).min(1) \
                        if len(new_idx) else np.full(len(sym2), np.inf)
                    # open atoms within 4 Å of the new atoms move: oxide atoms
                    # and the substrate's top layers / interface layer (z > -3 Å)
                    movable = np.where((dn < 4.0) & ~final2
                                       & ((np.arange(len(sym2)) >= n_sub) | (pos2[:, 2] > -3.0)))[0]
                    movable = np.union1d(movable, new_idx)
                    pairs = []
                    if args.snap:
                        pos2, pairs = snap_near_misses(sym2, pos2, list(c_["bad"]), movable)
                        movable = np.union1d(movable, [k for p_ in pairs for k in p_])
                        if pairs:
                            pos2 = local_relax(calc, sym2, pos2, movable, springs=pairs, steps=60)
                    pos2 = local_relax(calc, sym2, pos2, movable)
                    now_final = finality(pos2)
                    c_.update(pos2=pos2, now_final=now_final, relaxed=True, n_snap=len(pairs),
                              bad=evaluate(sym2, pos2, now_final, final2))
                    stats["snapped_pairs"] = stats.get("snapped_pairs", 0) + len(pairs)
                    if not c_["bad"]:
                        chosen = c_
                        stats["relax_rescued"] += 1
                        break
            if RECORDER is not None:
                RECORDER.on_try(n, t, guides, outs, cands, chosen)
            # blame: pre-existing atoms that made candidates fail
            for c_ in cands:
                for i in c_["bad"]:
                    if i < len(sym):
                        blame[i] = blame.get(i, 0) + 1
            if chosen is None:
                stats["rejected"] += 1
                best = min(cands, key=lambda c_: len(c_["bad"]))
                bad = {}
                for i, lab in best["bad"].items():
                    bad[lab] = bad.get(lab, 0) + 1
                if "dead_end" in bad:
                    stats["deadend_rejects"] += 1
                if args.verbose:
                    p2, s2 = best["pos2"], best["sym2"]
                    for i, lab in best["bad"].items():
                        d = np.linalg.norm(p2 - p2[i], axis=1)
                        nb = [(str(s2[j]), round(float(d[j]), 2)) for j in np.argsort(d)[1:6]]
                        where = ("substrate" if i < n_sub else "continued patch" if i < n_base else
                                 "earlier window" if i < len(sym) else "this window")
                        print(f"    {lab} atom {i} ({where}) at {np.round(p2[i], 2)}; nearest {nb}")
                    print(f"  window {n + 1} try {t}: {len(cands)} samples, 0 clean; best rejected {bad}"
                          + (" (after relax)" if best["relaxed"] else ""), flush=True)
                continue
            if args.verbose:
                print(f"  window {n + 1} try {t}: accepted ({len(clean)}/{len(cands)} clean raw"
                      + (", repaired" if chosen["relaxed"] else "") + ")", flush=True)
            history.append((len(sym), final.copy(), o_debt, pos.copy()))
            if RECORDER is not None:
                RECORDER.on_accept(len(sym), len(chosen["sym2"]))
            sym, pos = chosen["sym2"], chosen["pos2"]
            final = chosen["final2"] | chosen["now_final"]
            ok = True
            break

        if ok:
            snap("accept", n + 1, lo, hi)
            n += 1
            if n % 10 == 0 or n == len(cells):
                print(f"window {n}/{len(cells)} done, {len(sym) - n_sub} oxide atoms, "
                      f"{n_tries} tries, {n_back} backtracks; {stats}", flush=True)
            continue
        if not history or n_back >= args.max_backtracks:
            print(f"Gave up at window {n + 1}/{len(cells)} ({n_tries} tries, {n_back} backtracks)")
            break
        # an atom that made most tries fail cannot be fixed here: undo windows
        # back to (and including) the window responsible for it -
        #   oxide atom     -> the window that placed it
        #   substrate atom -> the earliest window whose box could bond to it
        n_fail = n + 1
        culprit = max(blame, key=blame.get) if blame else None
        if culprit is None or blame[culprit] < args.max_tries // 2:
            culprit = None
        n_pop = 1
        if culprit is not None:
            if culprit < n_base:
                gaps = [np.linalg.norm(np.maximum(0, np.maximum(box(cells[j])[0] - pos[culprit],
                                                               pos[culprit] - box(cells[j])[1])))
                        for j in range(len(history))]
                tw = next((j for j, g in enumerate(gaps) if g <= reach), len(history) - 1)
            else:
                tw = max(j for j, h in enumerate(history) if h[0] <= culprit)
            n_pop = len(history) - tw
        # never more than max_jump windows at once (a deep jump discards much
        # good work) - unless the same culprit failed again after a capped jump
        if not (culprit is not None and culprit == last_culprit):
            n_pop = min(n_pop, args.max_jump)
        last_culprit = culprit
        if RECORDER is not None:
            # defects among already-final interior atoms of the windows about to be undone
            fi = final & interior(pos)
            RECORDER.before_backtrack([i for i, _ in find_defects(Atoms(sym, positions=pos, pbc=False)) if fi[i]])
        for _ in range(max(1, n_pop)):
            k, final, o_debt, pos_saved = history.pop()
            if RECORDER is not None:
                RECORDER.on_pop()
            n -= 1
            if not history:
                break
        sym, pos = sym[:k], pos_saved      # positions before relaxations of undone windows
        n_back += 1
        if RECORDER is not None:
            RECORDER.on_backtrack(culprit, n_base)
        why = "" if culprit is None else (f" (given atom {culprit})" if culprit < n_base
                                           else f" (atom {culprit} placed there)")
        print(f"  window {n_fail} failed {args.max_tries}x: backtracking to window {n + 1}{why}",
              flush=True)
        snap("backtrack", n + 1, *box(cells[n]))

    if args.traj:
        write(args.traj, traj)
        print(f"growth trajectory: {len(traj)} frames -> {args.traj}")
    atoms = Atoms(sym, positions=pos, pbc=False)
    atoms.center(vacuum=10.0)
    tags = np.zeros(len(atoms), int); tags[:n_sub] = 1
    atoms.set_tags(tags)
    shift = atoms.get_positions()[0] - pos[0]
    # growth frame (for post-processing, e.g. anneal + interior judging)
    atoms.info.update(shift=" ".join(f"{v:.6f}" for v in shift), Lx=float(Lx), Ly=float(Ly),
                      z_bot=float(z_bot), z1=float(z1), n_sub=int(n_sub))
    inner = interior(atoms.get_positions() - shift)
    d_in, d_out = {}, {}
    for i, lab in find_defects(atoms):
        dd = d_in if inner[i] else d_out
        dd[lab] = dd.get(lab, 0) + 1
    if RECORDER is not None:
        RECORDER.on_end([i for i, _ in find_defects(atoms) if inner[i]], n, len(cells), n_back)
    print(f"\nGrown {len(atoms)} atoms ({atoms.get_chemical_formula()}), "
          f"{n}/{len(cells)} windows, {n_tries} tries, {n_back} backtracks")
    print(f"  interior defects: {d_in or 'none'}")
    print(f"  boundary (faces/top/bottom skin) defects: {d_out or 'none'}")
    return atoms


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["complete", "surface"])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=0)
    # complete
    p.add_argument("--dataset", default="data/sio2/dataset.extxyz")
    p.add_argument("--sequences", default=None, help="sequences.json with val_runs")
    p.add_argument("--n_per", type=int, default=20)
    # surface
    p.add_argument("--lateral", type=float, default=16.0, help="Å, patch size")
    p.add_argument("--substrate_cells", type=int, default=2)
    p.add_argument("--substrate_frame", default=None,
                   help="FILE[@INDEX]: relaxed Si slab from a SiO2Si interface MD frame "
                        "(top interface, repeated laterally) instead of ideal Si(001)")
    p.add_argument("--continue_from", default=None,
                   help="surface: grown structure of a smaller patch (substrate tile 0,0) to extend")
    p.add_argument("--n_cand", type=int, default=1,
                   help="surface: candidates sampled per try as one GPU batch (best clean one kept)")
    p.add_argument("--n_relax", type=int, default=1,
                   help="surface: candidates repaired/relaxed per try when none is clean")
    p.add_argument("--snap", action="store_true",
                   help="surface: snap near-miss Si-O pairs to bond length before relaxing")
    p.add_argument("--deadend_thr", default=None,
                   help="surface: dead-end thresholds, e.g. O_O=2.3,O_Si=2.2,Si_Si=2.6,Si_O=2.2,under=1.45")
    p.add_argument("--traj", default=None,
                   help="surface: write a snapshot after every accepted window / backtrack")
    p.add_argument("--relax", action="store_true",
                   help="surface: MACE-relax new + open nearby atoms before judging a window")
    p.add_argument("--one_go", action="store_true",
                   help="accept the first sample of every window: no defect checks, no backtracking")
    p.add_argument("--deadend", action="store_true",
                   help="surface: reject a window at once if an open atom can no longer get a partner")
    p.add_argument("--substrate_keep_above", type=float, default=1.0,
                   help="Å of the MD interface layer kept above the top slab Si")
    p.add_argument("--oxide_thickness", type=float, default=12.0)
    p.add_argument("--context_level", type=int, default=8)
    p.add_argument("--max_tries", type=int, default=30)
    p.add_argument("--max_backtracks", type=int, default=20)
    p.add_argument("--max_jump", type=int, default=6, help="max windows undone per backtrack")
    p.add_argument("--out", default="data/surface/window_SiO2_on_Si001.extxyz")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    if getattr(args, "deadend_thr", None):
        for kv in args.deadend_thr.split(","):
            k, v = kv.split("=")
            DEADEND_THR[k.strip()] = float(v)

    torch.manual_seed(args.seed)
    tok = WindowTokenizer.load(args.tokenizer)
    model, _ = load_checkpoint(args.checkpoint, args.device)
    model.eval()

    if args.mode == "complete":
        frames = read(args.dataset, index=":")
        if args.sequences:
            with open(args.sequences) as f:
                val_runs = set(json.load(f)["val_runs"])
            frames = [a for a in frames if a.info.get("label") in val_runs]
        print(f"{len(frames)} held-out frames")
        n_lv = tok.config["n_known_levels"]
        h = n_lv // 2
        completion_test(model, tok, frames,
                        [[0, 0, h], [h, h, 0], [h, h, h], [0, 0, 0]],
                        args.n_per, args.temperature, args.device, args.seed)
    else:
        atoms = grow_surface(model, tok, args)
        from pathlib import Path
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        write(args.out, atoms)
        print(f"Saved to {args.out}")
