"""Render data/surface/mq_movie_traj.extxyz (from mq_movie_run.py) as a movie:
left = periodic cell (Si gold, O red, Si-O bonds, defective atoms ringed blue),
right = temperature schedule / instantaneous T and number of defective atoms.
writes PNG frames to data/surface/mq_movie_frames/ (assemble with ffmpeg)
"""
import sys, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from ase.io import read
from ase.neighborlist import neighbor_list

out = sys.argv[1] if len(sys.argv) > 1 else "data/surface/mq_movie.gif"
fps = int(sys.argv[2]) if len(sys.argv) > 2 else 15
traj = read("data/surface/mq_movie_traj.extxyz", index=":")
t = np.array([a.info["time_ps"] for a in traj])
Tt = np.array([a.info["T_target"] for a in traj])
Ti = np.array([a.info["T_inst"] for a in traj])
nd = np.array([a.info["n_defects"] for a in traj])
L = np.diag(traj[0].cell.array)
N = len(traj[0])
col = np.where(np.array(traj[0].get_chemical_symbols()) == "Si", "#e0b040", "#d03030")
size = np.where(col == "#e0b040", 80, 45)
phase_name = {"start": "random start", "melt": "melt 5000 K", "anneal": "anneal 2000 K",
              "quench": "quench 2000→300 K", "final": "hold 300 K", "relaxed": "relaxed (FIRE)"}

fig = plt.figure(figsize=(12, 5.6))
ax = fig.add_subplot(1, 2, 1, projection="3d")
axT = fig.add_subplot(1, 2, 2)
fig.subplots_adjust(left=0.0, right=0.92, bottom=0.11, top=0.88, wspace=0.12)

axT.plot(t, Ti, color="#f0a0a0", lw=0.8, label="T instantaneous")
axT.plot(t, Tt, color="#c02020", lw=2, label="T thermostat")
axT.set_xlabel("time (ps)"); axT.set_ylabel("temperature (K)", color="#c02020")
axT.set_xlim(0, t[-1] + 0.3); axT.set_ylim(0, max(Ti.max(), 5500))
axD = axT.twinx()
axD.plot(t, nd / N * 100, color="#1060ff", lw=1.5, label="defective atoms")
axD.set_ylabel("defective atoms (%)", color="#1060ff"); axD.set_ylim(0, 100)
cur = axT.axvline(0, color="k", lw=1)
h1, l1 = axT.get_legend_handles_labels(); h2, l2 = axD.get_legend_handles_labels()
axT.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=9)


def draw(k):
    a = traj[k]
    ax.cla()
    P = a.positions
    i, j, D = neighbor_list("ijD", a, {("Si", "O"): 2.10})
    for ii, jj, dv in zip(i, j, D):
        if ii < jj:
            ax.plot(*np.array([P[ii], P[ii] + dv]).T, color="#555555", lw=0.9, alpha=0.8)
    ax.scatter(*P.T, s=size, c=col, edgecolors="k", linewidths=0.3, depthshade=False)
    bad = a.arrays["defect"].astype(bool)
    ax.scatter(*P[bad].T, s=220, facecolors="none", edgecolors="#1060ff", linewidths=1.8,
               depthshade=False)
    for x0 in (0, L[0]):                      # cell edges
        for y0 in (0, L[1]):
            ax.plot([x0, x0], [y0, y0], [0, L[2]], "k:", lw=0.5)
        for z0 in (0, L[2]):
            ax.plot([x0, x0], [0, L[1]], [z0, z0], "k:", lw=0.5)
    for y0 in (0, L[1]):
        for z0 in (0, L[2]):
            ax.plot([0, L[0]], [y0, y0], [z0, z0], "k:", lw=0.5)
    ax.set_xlim(-0.5, L[0] + 0.5); ax.set_ylim(-0.5, L[1] + 0.5); ax.set_zlim(-0.5, L[2] + 0.5)
    ax.set_box_aspect(tuple(L)); ax.set_axis_off()
    ax.view_init(elev=20, azim=-60 + 0.4 * k)
    ax.set_title(f"SiO2, {N} atoms, MACE-MP-0 — {phase_name[a.info['phase']]}\n"
                 f"t = {a.info['time_ps']:.1f} ps, T = {a.info['T_target']:.0f} K, "
                 f"defective atoms {a.info['n_defects']} (blue rings)", fontsize=10)
    cur.set_xdata([a.info["time_ps"]] * 2)
    return []


import os
fdir = "data/surface/mq_movie_frames"
os.makedirs(fdir, exist_ok=True)
for k in range(len(traj)):
    draw(k)
    fig.savefig(f"{fdir}/f{k:04d}.png", dpi=90)
print("frames in", fdir, "- assemble with ffmpeg")
