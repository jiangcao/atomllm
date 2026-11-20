"""Generate one window crop with v1 and v2 (same crop as the true fill) and
render the structures with the defects marked. Uses the fair-test setup and
judging rules of fair_complete.py (env = context + past atoms on the low side,
only core atoms whose whole neighbourhood is in the structure are judged).
usage: render_defects.py V1_CKPT V2_CKPT [lx ly lz] [seed]
writes data/surface/render_<levels>_s<seed>.png and .extxyz per method
"""
import os, sys, json, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase import Atoms
from ase.io import read, write
import window_generation as wg
V1Guide, v1_generate = wg.WindowGuide, wg.guided_generate   # before v2 patches wg
import window_generation_v2 as wg2
from window_tokenizer import WindowTokenizer, growth_order
from window_tokenizer_v2 import WindowZTokenizer
from structure_defects import find_defects
from train import load_checkpoint

MAXB = wg.MAX_BOND
v1ck, v2ck = sys.argv[1], sys.argv[2]
levels = [int(x) for x in sys.argv[3:6]] if len(sys.argv) > 5 else [0, 0, 0]
seed = int(sys.argv[6]) if len(sys.argv) > 6 else 0
tok1 = WindowTokenizer.load("data/sio2_win8/tokenizer.json")
tok2 = WindowZTokenizer.load("data/sio2_win8_z/tokenizer.json")
m1, _ = load_checkpoint(v1ck, "cuda"); m1.eval()
m2, _ = load_checkpoint(v2ck, "cuda"); m2.eval()
vr = set(json.load(open("data/sio2_win8/sequences.json"))["val_runs"])
frames = [a for a in read("data/sio2/dataset.extxyz", index=":") if a.info.get("label") in vr]
bad_atoms = [{i for i, _ in find_defects(a)} for a in frames]
W = tok1.W
torch.manual_seed(seed)
rng = np.random.default_rng(seed)


def judge(sym, pos, excl, Lp, n_ctx, n_base):
    """per-atom: judged mask, defect type string ('' = fine)."""
    if len(excl):
        d = pos[:, None] - excl[None]
        d -= np.round(d / Lp) * Lp
        core = np.linalg.norm(d, axis=-1).min(1) > MAXB
    else:
        core = np.ones(len(pos), bool)
    core &= np.all((pos >= 0) & (pos < W), axis=1)
    dd = pos[:, None] - pos[None]
    dm = dd - np.round(dd / Lp) * Lp
    Dm, Dp = np.linalg.norm(dm, axis=-1), np.linalg.norm(dd, axis=-1)
    np.fill_diagonal(Dm, np.inf)
    core &= ~((Dm < MAXB) & (Dp > Dm + 1e-6)).any(1)
    core[n_ctx:n_base] = False
    sym = np.asarray(sym)
    D = Dp.copy(); np.fill_diagonal(D, np.inf)
    si, o = sym == "Si", sym == "O"
    n_o = ((D < 2.10) & o[None]).sum(1)
    n_si = ((D < 2.10) & si[None]).sum(1)
    oo = ((D < 1.70) & o[None]).any(1)
    kind = np.full(len(sym), "", dtype=object)
    kind[si & (n_o < 4)] = "Si under"
    kind[si & (n_o > 4)] = "Si over"
    kind[o & (n_si < 2)] = "O non-bridging"
    kind[o & (n_si > 2)] = "O over"
    kind[o & oo] = "O-O"
    return core, kind


# pick a clean crop with something to fill
while True:
    fi = int(rng.integers(len(frames))); a = frames[fi]
    inside, local, to_cell, Lp = wg.crop_with_transform(a, W, rng)
    if bad_atoms[fi] & set(inside.tolist()):
        continue
    known = tok1.is_known(local[inside], levels)
    ctx_i, fill_i = inside[known], inside[~known]
    if len(fill_i):
        break
t = tok1.thresholds(levels) * (np.asarray(levels) > 0)
if len(ctx_i):
    ctx_i = ctx_i[growth_order(local[ctx_i], np.ones(len(ctx_i), bool), np.full(3, W / 2))]
sym = np.array(a.get_chemical_symbols())
lw = np.where(local > (Lp + W) / 2, local - Lp, local)
out_i = np.setdiff1d(np.arange(len(a)), inside)
past = out_i[np.all(lw[out_i] < W, axis=1) & np.all(lw[out_i] > -2 * MAXB, axis=1)
             & np.any(lw[out_i] < np.where(np.asarray(levels) > 0, t, 0), axis=1)]
excl = lw[np.setdiff1d(out_i, past)]
counts = {e: int((sym[fill_i] == e).sum()) for e in tok1.config["elements"]}
base_s = list(sym[ctx_i]) + list(sym[past])
base_p = np.vstack([lw[ctx_i], lw[past]])
n_ctx, n_base = len(ctx_i), len(ctx_i) + len(past)

results = {"true fill": (list(sym[fill_i]), lw[fill_i])}
for name, tok, G, gen, m in (("v1 (fractional)", tok1, V1Guide, v1_generate, m1),
                             ("v2 (internal coords)", tok2, wg2.ZWindowGuide, wg2.z_generate, m2)):
    for attempt in range(5):
        wg2.FORCE_BONDS = False
        g = G(tok, levels, counts, sym[ctx_i], lw[ctx_i], sym[past], lw[past], t, np.full(3, W))
        out, info = gen(m, g, temperature=0.8, device="cuda")
        if out is not None:
            s_new, p_new = g.new_atoms()
            results[name] = (list(s_new), np.asarray(p_new).reshape(-1, 3))
            break
    else:
        print(f"{name}: generation failed 5x")

tag = "".join(str(x) for x in levels) + f"_s{seed}"
colors = {"Si": "#e0b040", "O": "#d03030"}
fig = plt.figure(figsize=(6 * len(results), 6.6))
for k, (name, (s_new, p_new)) in enumerate(results.items()):
    S = np.array(base_s + list(s_new)); P = np.vstack([base_p, p_new])
    core, kind = judge(S, P, excl, Lp, n_ctx, n_base)
    is_new = np.arange(len(S)) >= n_base
    is_ctx = np.arange(len(S)) < n_ctx
    defect = core & (kind != "")
    nj, nd = int(core.sum()), int(defect.sum())
    # save for OVITO etc.: tags 0 ctx, 1 past, 2 generated; defect flag/kind
    at = Atoms(S, positions=P)
    at.set_tags(np.where(is_new, 2, np.where(is_ctx, 0, 1)))
    at.new_array("judged", core.astype(int))
    at.new_array("defect", defect.astype(int))
    at.info["defect_kinds"] = ";".join(f"{i}:{kind[i]}" for i in np.where(defect)[0])
    fn = name.split()[0].replace("true", "true_fill")
    write(f"data/surface/render_{tag}_{fn}.extxyz", at)

    ax = fig.add_subplot(1, len(results), k + 1, projection="3d")
    D = np.linalg.norm(P[:, None] - P[None], axis=-1)
    for i, j in zip(*np.where(np.triu((D < 2.10) & (S[:, None] != S[None]), 1))):
        faint = not (is_new[i] or is_new[j])
        ax.plot(*P[[i, j]].T, color="#bbbbbb" if faint else "#555555",
                lw=0.6 if faint else 1.2, alpha=0.4 if faint else 0.9)
    for e in ("Si", "O"):
        sel = S == e
        sz = 70 if e == "Si" else 40
        ax.scatter(*P[sel & ~is_new].T, s=sz, c="#c8c8c8", alpha=0.35, depthshade=False)
        ax.scatter(*P[sel & is_new].T, s=sz, c=colors[e], edgecolors="k", linewidths=0.4,
                   depthshade=False)
    ax.scatter(*P[defect].T, s=260, facecolors="none", edgecolors="#1060ff", linewidths=2.2,
               depthshade=False)
    for i in np.where(defect)[0]:
        ax.text(*P[i] + 0.25, {"Si under": "Si-3", "Si over": "Si-5", "O non-bridging": "NBO", "O over": "O-3", "O-O": "O-O"}[kind[i]],
                fontsize=7, color="#1060ff")
    # window box
    for a0 in (0, W):
        for b0 in (0, W):
            for ax_i in range(3):
                pts = np.zeros((2, 3)); pts[:, ax_i] = (0, W)
                o_ = [c for c in range(3) if c != ax_i]
                pts[:, o_[0]] = a0; pts[:, o_[1]] = b0
                ax.plot(*pts.T, color="k", lw=0.4, ls=":")
    kinds = {}
    for i in np.where(defect)[0]:
        kinds[kind[i]] = kinds.get(kind[i], 0) + 1
    ax.set_title(f"{name}\n{len(s_new)} generated atoms; defective {nd}/{nj} judged"
                 + ("\n" + ", ".join(f"{v} {k}" for k, v in kinds.items()) if kinds else "\nno defects"),
                 fontsize=10)
    ax.set_xlim(-2, W + 0.5); ax.set_ylim(-2, W + 0.5); ax.set_zlim(-2, W + 0.5)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=22, azim=-60)
    ax.set_xlabel("x (Å)"); ax.set_ylabel("y (Å)"); ax.set_zlabel("z (Å)")
    print(f"{name}: generated {len(s_new)}, defective {nd}/{nj} judged, {kinds}")
fig.suptitle(f"SiO2 window W={W:.0f} Å, levels {levels} (fill region generated; grey = given "
             f"context/past atoms; blue rings = defects among judged atoms)", fontsize=11, y=0.985)
fig.subplots_adjust(left=0.01, right=0.97, bottom=0.03, top=0.84, wspace=0.08)
out = f"data/surface/render_{tag}.png"
fig.savefig(out, dpi=130)
print("saved", out)
