"""Plot the 50-crop fair completion test (numbers from data/logs/fair_complete_v4.log)."""
import os
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

shapes = ["4×4×4 Å\n[8,8,8]", "4×4×8 Å\n[8,8,0]", "8×8×4 Å\n[0,0,8]", "8×8×8 Å\n[0,0,0] whole"]
# (defective, judged, clean crops of 50) per shape
data = {
    "v1 fractional":            [(15, 437, 38), (44, 452, 23), (96, 445, 7), (182, 445, 1)],
    "v2 internal coords":       [(22, 443, 33), (63, 453, 13), (110, 450, 5), (151, 505, 2)],
    "v2 + force_bonds":         [(22, 449, 32), (60, 462, 18), (116, 491, 5), (125, 489, 5)],
}
colors = ["#2a78d6", "#eb6834", "#1baf7a"]
ink, muted, grid = "#2b2b29", "#6b6a64", "#e4e3de"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": muted, "axes.labelcolor": ink,
                     "xtick.color": muted, "ytick.color": muted})
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), facecolor="#fcfcfb")
x = np.arange(len(shapes)); w = 0.26
for ax, k, ylab, fmt in ((axes[0], 0, "defective judged atoms (%)", "{:.0f}%"),
                         (axes[1], 2, "defect-free crops (of 50)", "{:d}")):
    ax.set_facecolor("#fcfcfb")
    for i, (name, rows) in enumerate(data.items()):
        v = [100 * d / j for d, j, _ in rows] if k == 0 else [c for _, _, c in rows]
        b = ax.bar(x + (i - 1) * w, v, w - 0.03, color=colors[i], label=name, zorder=3)
        for xi, vi in zip(x + (i - 1) * w, v):
            ax.text(xi, vi + (0.6 if k == 0 else 0.6), fmt.format(round(vi) if k == 0 else vi),
                    ha="center", va="bottom", fontsize=8.5, color=ink)
    ax.set_xticks(x, shapes, fontsize=9)
    ax.set_ylabel(ylab)
    ax.grid(axis="y", color=grid, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.set_xlabel("generated (fill) region; window levels", color=muted)
axes[0].set_ylim(0, 48); axes[1].set_ylim(0, 50)
axes[0].set_title("Defective atoms among judged atoms (lower is better)", color=ink, loc="left")
axes[1].set_title("Crops with no defect at all (higher is better)", color=ink, loc="left")
axes[0].legend(frameon=False, loc="upper left")
fig.suptitle("Fair completion test, SiO2 8 Å windows, 50 held-out crops per shape "
             "(true fill = 0 defective of 446 judged atoms in every shape)", color=ink, fontsize=11, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig("data/surface/fair_complete_v4.png", dpi=130, facecolor=fig.get_facecolor())
print("saved")
