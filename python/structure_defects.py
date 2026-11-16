"""
Coordination-based defect detection for Si / SiO2 / HfO2 / HfSiO4 structures.

Used to keep interface/bulk traps out of training data and out of generated
structures. A structure is "clean" when every atom has its ideal bonding:

    Si : exactly 4 neighbours from {Si, O}   (Si0 ... Si4+ suboxide states are
         all fine, so a Si/SiO2 interface with Si-Si + Si-O bonds is allowed)
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
    ("Si", "Si"): 2.45,   # Si-Si bond ~2.35; excludes edge-sharing Si..Si (~2.5)
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


def find_defects(atoms: Atoms) -> list[tuple[int, str]]:
    """Return a list of (atom_index, defect_label) for every defective atom."""
    symbols = atoms.get_chemical_symbols()
    present = set(symbols)
    cutoffs = {k: v for k, v in _cutoff_dict().items()
               if k[0] in present and k[1] in present}
    if not cutoffs:
        return []
    i_idx, j_idx = neighbor_list("ij", atoms, cutoffs)

    neigh = [[] for _ in range(len(atoms))]
    for i, j in zip(i_idx, j_idx):
        neigh[i].append(symbols[j])

    defects = []
    for i, (s, nb) in enumerate(zip(symbols, neigh)):
        c = Counter(nb)
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
