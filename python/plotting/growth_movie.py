"""Render a surface-growth trajectory (window_generation --traj) as PNG frames.
Per snapshot: substrate pale, grown Si gold / O red, atoms added by this
window outlined, current fill box green (accepted) or red (backtrack).
Ends with a full rotation of the final structure.
usage: growth_movie.py TRAJ.extxyz OUT_DIR [N_GIVEN_OXIDE [TITLE]]
N_GIVEN_OXIDE: oxide atoms of a continued patch (drawn lighter, present from the start)"""
import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase.io import read

traj = read(sys.argv[1], index=":")
out = sys.argv[2]
os.makedirs(out, exist_ok=True)
allp = np.vstack([f.positions for f in traj])
lo, hi = allp.min(0) - 0.5, allp.max(0) + 0.5
n_sub = int(traj[0].get_tags().sum())
n_given = n_sub + (int(sys.argv[3]) if len(sys.argv) > 3 else 0)
start_title = sys.argv[4] if len(sys.argv) > 4 else "Relaxed Si(001) substrate from interface MD (quick_2)"
from structure_defects import BOND_CUTOFFS
SISI = BOND_CUTOFFS[("Si", "Si")]


def bonds(P, sym):
    D = np.linalg.norm(P[:, None] - P[None], axis=-1)
    si = sym == "Si"
    return np.triu(((D < 2.10) & (si[:, None] != si[None])) | ((D < SISI) & si[:, None] & si[None]), 1)


fig = plt.figure(figsize=(8, 8.4))
ax = fig.add_subplot(111, projection="3d")
fig.subplots_adjust(left=0, right=1, bottom=0, top=0.9)
k_img = 0


def draw(f, prev_n, azim, title, box=None, box_color="g"):
    global k_img
    ax.cla()
    P = f.positions; sym = np.array(f.get_chemical_symbols())
    sub = np.arange(len(f)) < n_sub
    si = sym == "Si"
    for i, j in zip(*np.where(bonds(P, sym))):
        s2 = sub[i] and sub[j]
        ax.plot(*P[[i, j]].T, color="#bdbdb8" if s2 else "#555555", lw=0.6 if s2 else 1.0,
                alpha=0.5 if s2 else 0.9)
    ax.scatter(*P[sub & si].T, s=40, c="#cfc9b0", edgecolors="#8a8570", linewidths=0.3, depthshade=False)
    ax.scatter(*P[sub & ~si].T, s=26, c="#eab0b0", edgecolors="#a06060", linewidths=0.3, depthshade=False)
    new = np.arange(len(f)) >= prev_n
    giv = (np.arange(len(f)) >= n_sub) & (np.arange(len(f)) < n_given)
    for m, c, sz in ((giv & si, "#efd590", 50), (giv & ~si, "#f09a9a", 30)):
        ax.scatter(*P[m].T, s=sz, c=c, edgecolors="#777777", linewidths=0.3, depthshade=False)
    for m, c, sz in ((~sub & ~giv & si & ~new, "#e0b040", 60), (~sub & ~giv & ~si & ~new, "#d03030", 34)):
        ax.scatter(*P[m].T, s=sz, c=c, edgecolors="k", linewidths=0.4, depthshade=False)
    for m, c, sz in ((~sub & si & new, "#e0b040", 90), (~sub & ~si & new, "#d03030", 55)):
        ax.scatter(*P[m].T, s=sz, c=c, edgecolors="k", linewidths=1.8, depthshade=False)
    if box is not None:
        b0, b1 = box
        xs, ys, zs = [b0[0], b1[0]], [b0[1], b1[1]], [b0[2], b1[2]]
        for x in xs:
            for y in ys:
                ax.plot([x, x], [y, y], zs, color=box_color, lw=1.4)
        for z in zs:
            ax.plot([xs[0], xs[1], xs[1], xs[0], xs[0]], [ys[0], ys[0], ys[1], ys[1], ys[0]], [z] * 5,
                    color=box_color, lw=1.4)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo); ax.view_init(elev=14, azim=azim)
    ax.set_axis_off()
    fig.suptitle(title, fontsize=11)
    fig.savefig(f"{out}/f{k_img:04d}.png", dpi=100)
    k_img += 1


prev_n = n_given
start = traj[0].copy()
del start[n_given:]
for _ in range(6):
    draw(start, n_given, -60, start_title)
for k, f in enumerate(traj):
    ev = f.info["event"]
    vec = lambda v: np.array([float(x) for x in v.split()]) if isinstance(v, str) else np.asarray(v, float)
    b0, b1 = vec(f.info["box_lo"]), vec(f.info["box_hi"])
    n_grown = len(f) - n_given
    if ev == "accept":
        title = (f"window {f.info['window']}/{f.info['n_windows']} accepted  "
                 f"({n_grown} oxide atoms, {f.info['tries']} tries, {f.info['backtracks']} backtracks)")
        color = "#20a040"
    else:
        title = (f"BACKTRACK to window {f.info['window']}  "
                 f"({n_grown} oxide atoms kept, {f.info['tries']} tries)")
        color = "#e03030"
    for r in range(3):                       # hold each step ~0.5 s at 6 fps
        draw(f, prev_n if ev == "accept" else len(f), -60 + 1.5 * k + 0.5 * r, title, (b0, b1), color)
    prev_n = len(f)
final = traj[-1]
for r in range(72):                          # final full rotation
    draw(final, len(final), -60 + 1.5 * len(traj) + 5 * r,
         f"final: {len(final) - n_given} new atoms" + (f" + {n_given - n_sub} from the 1x1 patch" if n_given > n_sub else "")
         + f" on {n_sub} substrate atoms")
print(f"{k_img} frames in {out}")
