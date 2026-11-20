"""
Coordination-based defect detection for Si / SiO2 / HfO2 / HfSiO4 structures.

Used to keep interface/bulk traps out of training data and out of generated
structures. A structure is "clean" when every atom has its ideal bonding:

    Si : exactly 4 neighbours from {Si, O}   (Si0 ... Si4+ suboxide states are
         all fine, so a Si/SiO2 interface with Si-Si + Si-O bonds is allowed).
         A Si-Si contact counts as a bond only when both Si have < 4 O
         neighbours: two fully oxidised Si that share an edge (two-membered
         ring, Si..Si ~2.3-2.45 A with MACE-MP) are not bonded and are not a
         defect.
    O  : bonded only to cations; 2 Si if all-Si, 2..4 cations if any Hf
    Hf : 6..8 O neighbours

Flagged defects (common interface / oxide trap precursors):
    Si_dangling      Si with < 4 neighbours  (Pb center at interface, E' in oxide)
    Si_overcoord     Si with > 4 neighbours
    O_nonbridging    O with < 2 cation neighbours (NBOHC, "missing Si" / broken bond)
    O_overcoord      O with too many cation neighbours
    O_peroxy         O-O bond (peroxy linkage)
    Hf_undercoord    Hf with < 6 O
    Hf_overcoord     Hf with > 8 O

Usage:
    python structure_defects.py file.extxyz [more.extxyz ...]
"""

from collections import Counter

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list

# Bond cutoffs (Å), roughly the first RDF minimum for each pair
BOND_CUTOFFS = {
    ("Si", "Si"): 2.60,   # Si-Si bond ~2.35, stretched to 2.45-2.6 at Si/SiO2 interfaces
                          # (MACE-relaxed interface MD, 2026-11-20); edge-sharing Si..Si
                          # handled in find_defects
    ("Si", "O"):  2.10,
    ("O", "O"):   1.70,
    ("Hf", "O"):  2.70,
    ("Hf", "Si"): 3.00,
}

CATIONS = {"Si", "Hf"}


def _cutoff_dict():
    d = {}
    for (a, b), r in BOND_CUTOFFS.items():
        d[(a, b)] = r
        d[(b, a)] = r
    return d


# Si-Si bonds shorter than this always count; stretched Si-Si contacts between
# SI_SI_STRICT and the Si-Si bond cutoff only complete a Si up to 4 partners,
# and only between two Si that both have < 4 partners: a stretched contact
# can fill a dangling bond but never makes a Si overcoordinated (2026-11-20)
SI_SI_STRICT = 2.45


def coordination(atoms: Atoms):
    """Per atom: (n_strict, n_eff, counts) with counts[i] = Counter of
    bonded neighbour elements (strict bonds + accepted stretched Si-Si).
    Si: O (< Si-O cutoff) + Si-Si (< SI_SI_STRICT) + accepted stretched Si-Si;
    Si..Si next to a fully oxidised Si is edge sharing, not a bond."""
    symbols = np.array(atoms.get_chemical_symbols())
    n = len(atoms)
    counts = [Counter() for _ in range(n)]
    present = set(symbols.tolist())
    cutoffs = {k: v for k, v in _cutoff_dict().items()
               if k[0] in present and k[1] in present}
    if not cutoffs:
        return np.zeros(n, int), np.zeros(n, int), counts
    i_idx, j_idx, d = neighbor_list("ijd", atoms, cutoffs)
    n_o = np.zeros(n, int)
    np.add.at(n_o, i_idx[symbols[j_idx] == "O"], 1)
    sisi = (symbols[i_idx] == "Si") & (symbols[j_idx] == "Si")
    edge = sisi & ((n_o[i_idx] >= 4) | (n_o[j_idx] >= 4))
    stretched = sisi & ~edge & (d >= SI_SI_STRICT)
    strict = ~edge & ~stretched
    for i, j in zip(i_idx[strict], j_idx[strict]):
        counts[i][symbols[j]] += 1
    n_strict = np.array([sum(c.values()) for c in counts], int)
    # stretched Si-Si: both partners below 4, nearest first, fill up to 4
    order = np.argsort(d[stretched])
    si, sj = i_idx[stretched][order], j_idx[stretched][order]
    n_eff = n_strict.copy()
    for i, j in zip(si, sj):
        if i < j and n_eff[i] < 4 and n_eff[j] < 4:
            n_eff[i] += 1; n_eff[j] += 1
            counts[i]["Si"] += 1; counts[j]["Si"] += 1
    return n_strict, n_eff, counts


def find_defects(atoms: Atoms) -> list[tuple[int, str]]:
    """Return a list of (atom_index, defect_label) for every defective atom."""
    symbols = atoms.get_chemical_symbols()
    _, _, counts = coordination(atoms)
    defects = []
    for i, (s, c) in enumerate(zip(symbols, counts)):
        if s == "Si":
            cn = c["Si"] + c["O"]
            if cn < 4:
                defects.append((i, "Si_dangling"))
            elif cn > 4:
                defects.append((i, "Si_overcoord"))
        elif s == "O":
            if c["O"] > 0:
                defects.append((i, "O_peroxy"))
            n_cat = c["Si"] + c["Hf"]
            max_cat = 4 if c["Hf"] > 0 else 2
            if n_cat < 2:
                defects.append((i, "O_nonbridging"))
            elif n_cat > max_cat:
                defects.append((i, "O_overcoord"))
        elif s == "Hf":
            if c["O"] < 6:
                defects.append((i, "Hf_undercoord"))
            elif c["O"] > 8:
                defects.append((i, "Hf_overcoord"))
    return defects


def defect_summary(atoms: Atoms) -> Counter:
    return Counter(label for _, label in find_defects(atoms))


def is_defect_free(atoms: Atoms, max_defects: int = 0) -> bool:
    return len(find_defects(atoms)) <= max_defects


if __name__ == "__main__":
    import argparse
    from ase.io import read

    p = argparse.ArgumentParser(description="Report coordination defects")
    p.add_argument("files", nargs="+")
    args = p.parse_args()

    for f in args.files:
        frames = read(f, index=":")
        n_clean = sum(is_defect_free(a) for a in frames)
        total = Counter()
        for a in frames:
            total += defect_summary(a)
        print(f"{f}: {n_clean}/{len(frames)} defect-free frames; "
              f"defects: {dict(total)}")
