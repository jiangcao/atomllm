"""Export a structure (default: last frame) for VESTA as POSCAR (.vasp) and CIF.
Non-periodic growth output keeps its vacuum box (10 Å around the atoms) as
the cell; atoms are grouped by element.
usage: export_vesta.py IN.extxyz OUT_PREFIX [INDEX=-1]"""
import sys
import numpy as np
from ase.io import read, write

a = read(sys.argv[1], index=int(sys.argv[3]) if len(sys.argv) > 3 else -1)
out = sys.argv[2]
if not a.cell.rank == 3:
    a.center(vacuum=10.0)
a.pbc = True
a = a[np.argsort(a.numbers, kind="stable")]      # group by element for POSCAR
write(out + ".vasp", a, format="vasp", direct=True)
write(out + ".cif", a)
print(f"{len(a)} atoms {a.get_chemical_formula()}, cell {np.round(a.cell.lengths(), 2)} Å -> {out}.vasp, {out}.cif")
