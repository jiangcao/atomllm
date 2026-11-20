"""Render a grown surface structure (window_generation surface output):
substrate Si grey-gold, grown Si gold, O red, Si-O and Si-Si bonds, defects
ringed (blue = interior/judged, light blue dashed = boundary skin).
usage: render_growth.py STRUCT.extxyz OUT.png"""
import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase.io import read
from structure_defects import find_defects

a = read(sys.argv[1]); out = sys.argv[2]
sub = a.get_tags() == 1
P = a.positions.copy(); sym = np.array(a.get_chemical_symbols())
ztop = P[sub, 2].max()
P -= np.array([P[:, 0].min(), P[:, 1].min(), ztop])        # substrate top at z = 0
lo, hi = P[:, :2].min(0), P[:, :2].max(0)
MB = 3.0
zmin = P[:, 2].min()
inner = (np.all((P[:, :2] > lo + MB) & (P[:, :2] < hi - MB), axis=1)
         & (P[:, 2] > zmin + 2 * MB) & (P[:, 2] < 12.0 - MB))
defects = dict(find_defects(a))
D = np.linalg.norm(P[:, None] - P[None], axis=-1)
is_si = sym == "Si"
bond = np.triu(((D < 2.10) & (is_si[:, None] != is_si[None]))
               | ((D < 2.45) & is_si[:, None] & is_si[None]), 1)
short = {"Si_dangling": "Si<4", "Si_overcoord": "Si>4", "O_nonbridging": "NBO",
         "O_overcoord": "O>2", "O_O": "O-O"}

fig = plt.figure(figsize=(15, 7.2))
views = [(12, -60, "perspective"), (0, -90, "side view (x-z)")]
n_in_def = sum(1 for i in defects if inner[i])
for k, (el, az, name) in enumerate(views):
    ax = fig.add_subplot(1, 2, k + 1, projection="3d")
    for i, j in zip(*np.where(bond)):
        both_sub = sub[i] and sub[j]
        ax.plot(*P[[i, j]].T, color="#b5b5b0" if both_sub else "#555555",
                lw=0.6 if both_sub else 1.0, alpha=0.6 if both_sub else 0.9)
    m = sub & is_si
    ax.scatter(*P[m].T, s=45, c="#c9c3a8", edgecolors="#8a8570", linewidths=0.3, depthshade=False)
    m = sub & ~is_si
    ax.scatter(*P[m].T, s=30, c="#e8a0a0", edgecolors="#a05050", linewidths=0.3, depthshade=False)
    m = ~sub & is_si
    ax.scatter(*P[m].T, s=60, c="#e0b040", edgecolors="k", linewidths=0.4, depthshade=False)
    m = ~sub & ~is_si
    ax.scatter(*P[m].T, s=34, c="#d03030", edgecolors="k", linewidths=0.4, depthshade=False)
    for i, lab in defects.items():
        if inner[i]:
            ax.scatter(*P[i], s=280, facecolors="none", edgecolors="#1060ff", linewidths=2.2, depthshade=False)
            ax.text(*(P[i] + 0.35), short.get(lab, lab), color="#1060ff", fontsize=8)
        else:
            ax.scatter(*P[i], s=150, facecolors="none", edgecolors="#9cc0ff", linewidths=1.0,
                       linestyles="--", depthshade=False)
    # judged (interior) region
    x0, y0 = lo + MB; x1, y1 = hi - MB; z0, z1 = zmin + 2 * MB, 12.0 - MB
    for (xa, ya) in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]:
        ax.plot([xa, xa], [ya, ya], [z0, z1], color="#1060ff", lw=0.7, ls=":")
    for zz in (z0, z1):
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], [zz] * 5, color="#1060ff", lw=0.7, ls=":")
    ax.set_box_aspect(np.ptp(P, axis=0)); ax.view_init(elev=el, azim=az)
    ax.set_xlabel("x (Å)"); ax.set_ylabel("y (Å)"); ax.set_zlabel("z (Å)")
    if name.startswith("side"):
        ax.set_yticks([])
    ax.set_title(name, fontsize=10)
fig.suptitle(f"{sys.argv[1].split('/')[-1]}: {(~sub).sum()} grown atoms on {sub.sum()} substrate atoms "
             f"(substrate pale)\nDotted box = judged interior ({inner.sum()} atoms, {n_in_def} defect); "
             f"dashed light rings = {len(defects) - n_in_def} defects in the unjudged boundary skin",
             fontsize=10.5)
fig.subplots_adjust(left=0.0, right=0.98, top=0.9, bottom=0.03, wspace=0.05)
fig.savefig(out, dpi=120)
print("saved", out)
