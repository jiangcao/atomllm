"""
Benchmark generated amorphous structures against a reference ensemble.

Metric suite follows the amorphous-generation literature so our numbers sit
on the same axes as those papers (see papers/README.md):

  Yang & Schwalbe-Koda 2025 (a-SiO2 diffusion), arXiv 2507.05024
      partial g(r), bond-angle distributions, ring statistics,
      elastic moduli (K, G, E, nu) with the ground-truth potential,
      QUESTS information entropy H and overlap (novelty / mode collapse)
  Kwon et al. 2024 (a-C diffusion), arXiv 2312.05472
      coordination fractions, per-atom potential energy distribution
  GaDiT 2026 (a-Si / a-SiO2), arXiv 2611.18420
      energy before vs after fixed-cell relaxation (energy drop per atom)
  plus ours: coordination-defect (trap) fraction from structure_defects.

Every distribution is compared with the reference by a scalar distance, and
the reference is split in two halves to give the noise floor ("ref_A vs
ref_B"): a generator is as good as the data when its distance to ref_B is no
larger than ref_A's.

Usage (on attelas2, see memory notes):
    python benchmark_amorphous.py --ref ../data/dataset.extxyz \\
        --gen llm=generated.extxyz --composition SiO2 \\
        --mace --elastic --device cuda --out ../benchmark/SiO2
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read
from ase.neighborlist import neighbor_list
from scipy.stats import wasserstein_distance

from structure_defects import BOND_CUTOFFS, defect_summary

ANGLE_TRIPLES = {  # (ligand, centre, ligand)
    "SiO2":   [("O", "Si", "O"), ("Si", "O", "Si")],
    "HfO2":   [("O", "Hf", "O"), ("Hf", "O", "Hf")],
    "HfSiO4": [("O", "Si", "O"), ("Si", "O", "Si"), ("O", "Hf", "O"),
               ("Si", "O", "Hf")],
}
RING_BOND = {"SiO2": 2.10}  # Si-O cutoff; ring size = graph ring length / 2
R_MAX, DR = 6.0, 0.02
QUESTS_K, QUESTS_CUT, QUESTS_H = 32, 5.0, 0.015


# ------------------------------------------------------------------ io
def formula_name(atoms: Atoms) -> str:
    c = Counter(atoms.get_chemical_symbols())
    g = np.gcd.reduce(list(c.values()))
    order = [e for e in ("Hf", "Si", "O") if e in c]
    return "".join(f"{e}{c[e] // g if c[e] // g > 1 else ''}" for e in order)


def load(path, composition, step_label=None, defect_free_only=False):
    frames = [a for a in read(path, index=":") if formula_name(a) == composition]
    if step_label:
        frames = [a for a in frames if a.info.get("step_label", "").startswith(step_label)]
    if defect_free_only:
        frames = [a for a in frames if not defect_summary(a)]
    return frames


# ------------------------------------------------------------- metrics
def _cut(a, b):
    return BOND_CUTOFFS.get((a, b), BOND_CUTOFFS.get((b, a)))


def partial_rdf(frames, pairs):
    edges = np.arange(0.0, R_MAX + DR, DR)
    r = 0.5 * (edges[1:] + edges[:-1])
    shell = 4 * np.pi * r**2 * DR
    out = {f"{a}-{b}": np.zeros_like(r) for a, b in pairs}
    for at in frames:
        sym = np.array(at.get_chemical_symbols())
        i, j, d = neighbor_list("ijd", at, R_MAX)
        V = at.get_volume()
        for a, b in pairs:
            na, nb = (sym == a).sum(), (sym == b).sum()
            m = (sym[i] == a) & (sym[j] == b)
            h, _ = np.histogram(d[m], edges)
            out[f"{a}-{b}"] += h * V / (na * nb) / shell
    return r, {k: v / len(frames) for k, v in out.items()}


def bond_angles(frames, triples):
    out = {f"{l1}-{c}-{l2}": [] for l1, c, l2 in triples}
    for at in frames:
        sym = np.array(at.get_chemical_symbols())
        rc = max(v for v in BOND_CUTOFFS.values())
        i, j, D = neighbor_list("ijD", at, rc)
        d = np.linalg.norm(D, axis=1)
        for l1, c, l2 in triples:
            key = f"{l1}-{c}-{l2}"
            for ci in np.where(sym == c)[0]:
                m = i == ci
                vec, js, dd = D[m], j[m], d[m]
                for p in range(len(js)):
                    for q in range(p + 1, len(js)):
                        sp, sq = sym[js[p]], sym[js[q]]
                        if {sp, sq} != {l1, l2} and not (sp == sq == l1 == l2):
                            continue
                        if dd[p] > _cut(c, sp) or dd[q] > _cut(c, sq):
                            continue
                        cos = vec[p] @ vec[q] / (dd[p] * dd[q])
                        out[key].append(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    return {k: np.array(v) for k, v in out.items()}


def coordination(frames, centre, ligands):
    cn = []
    for at in frames:
        sym = np.array(at.get_chemical_symbols())
        cut = {(centre, l): _cut(centre, l) for l in ligands}
        i, j = neighbor_list("ij", at, cut)
        counts = np.bincount(i[(sym[i] == centre) & np.isin(sym[j], ligands)],
                             minlength=len(at))
        cn.extend(counts[sym == centre])
    return np.array(cn)


def ring_sizes(frames, composition):
    if composition not in RING_BOND:
        return None
    from matscipy.rings import ring_statistics
    hist = np.zeros(20)
    for at in frames:
        h = ring_statistics(at, RING_BOND[composition], maxlength=24)
        for n, c in enumerate(h):
            if n % 2 == 0 and n // 2 < len(hist):
                hist[n // 2] += c   # Si-O-...-Si ring of n bonds = n/2 Si
    return hist / max(len(frames), 1)  # rings per structure, index = #Si


def min_distance(frames):
    out = []
    for at in frames:
        _, _, d = neighbor_list("ijd", at, 3.0)
        out.append(d.min() if len(d) else 3.0)
    return np.array(out)


def defects(frames):
    tot, clean = Counter(), 0
    for at in frames:
        s = defect_summary(at)
        tot += s
        clean += not s
    n_atoms = sum(len(a) for a in frames)
    return {"defect_free_fraction": clean / len(frames),
            "defects_per_1000_atoms": 1000 * sum(tot.values()) / n_atoms,
            "by_type": dict(tot)}


def density(frames):
    return np.array([a.get_masses().sum() / a.get_volume() * 1.66053906660
                     for a in frames])


# --------------------------------------------------------- MACE metrics
def mace_metrics(frames, device, elastic=False, fmax=0.02):
    from ase.optimize import FIRE
    from mace.calculators import mace_mp
    calc = mace_mp(model="medium", device=device, default_dtype="float64")
    e0, e1, disp, mods = [], [], [], []
    for at in frames:
        at = at.copy()
        at.calc = calc
        e0.append(at.get_potential_energy() / len(at))
        p0 = at.get_positions()
        FIRE(at, logfile=None).run(fmax=fmax, steps=1000)
        e1.append(at.get_potential_energy() / len(at))
        dp = at.get_positions() - p0
        disp.append(np.sqrt((dp**2).sum(1).mean()))
        if elastic:
            mods.append(elastic_moduli(at, calc, fmax=fmax))
    out = {"E_unrelaxed": np.array(e0), "E_relaxed": np.array(e1),
           "relax_rmsd": np.array(disp)}
    if elastic:
        out["moduli"] = {k: np.array([m[k] for m in mods]) for k in mods[0]}
    return out


def elastic_moduli(at, calc, delta=0.005, fmax=0.02):
    """Relaxed-ion stiffness by central differences of stress (GPa); Hill
    averages. Yang & Schwalbe-Koda use the energy curvature; for small
    strains the two agree."""
    from ase.optimize import FIRE
    from ase.units import GPa
    voigt = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    cell0 = at.get_cell().array.copy()
    C = np.zeros((6, 6))
    for k, (a, b) in enumerate(voigt):
        sig = []
        for s in (+1, -1):
            eps = np.zeros((3, 3))
            if a == b:
                eps[a, a] = s * delta
            else:
                eps[a, b] = eps[b, a] = s * delta / 2   # engineering shear
            x = at.copy()
            x.calc = calc
            x.set_cell(cell0 @ (np.eye(3) + eps), scale_atoms=True)
            FIRE(x, logfile=None).run(fmax=fmax, steps=500)
            sig.append(x.get_stress(voigt=True) / GPa)
        C[:, k] = (sig[0] - sig[1]) / (2 * delta)
    C = 0.5 * (C + C.T)
    S = np.linalg.inv(C)
    KV = (C[0, 0] + C[1, 1] + C[2, 2] + 2 * (C[0, 1] + C[0, 2] + C[1, 2])) / 9
    GV = (C[0, 0] + C[1, 1] + C[2, 2] - C[0, 1] - C[0, 2] - C[1, 2]
          + 3 * (C[3, 3] + C[4, 4] + C[5, 5])) / 15
    KR = 1 / (S[0, 0] + S[1, 1] + S[2, 2] + 2 * (S[0, 1] + S[0, 2] + S[1, 2]))
    GR = 15 / (4 * (S[0, 0] + S[1, 1] + S[2, 2]) - 4 * (S[0, 1] + S[0, 2] + S[1, 2])
               + 3 * (S[3, 3] + S[4, 4] + S[5, 5]))
    K, G = (KV + KR) / 2, (GV + GR) / 2
    return {"K": K, "G": G, "E": 9 * K * G / (3 * K + G),
            "nu": (3 * K - 2 * G) / (2 * (3 * K + G))}


# ------------------------------------------------------------- QUESTS
def quests_metrics(sets, ref_key, seed=0):
    """Entropy H (nats) of each set at equal environment count, and overlap
    of each set with the reference (fraction of envs with dH <= 0)."""
    from quests.descriptor import get_descriptors
    from quests.entropy import delta_entropy, perfect_entropy
    rng = np.random.default_rng(seed)
    X = {k: get_descriptors(v, k=QUESTS_K, cutoff=QUESTS_CUT) for k, v in sets.items()}
    n = min(len(x) for x in X.values())
    out = {}
    for k, x in X.items():
        sub = x[rng.choice(len(x), n, replace=False)]
        out[k] = {"H_nats": float(perfect_entropy(sub, h=QUESTS_H)), "n_envs": n}
        if k != ref_key:
            dH = delta_entropy(x, X[ref_key], h=QUESTS_H)
            out[k]["overlap_with_ref"] = float((dH <= 0).mean())
            out[k]["dH_mean"] = float(dH.mean())
    return out


# ---------------------------------------------------------- comparison
def compute_all(frames, composition, mace, elastic, device):
    elems = sorted(set(frames[0].get_chemical_symbols()))
    pairs = [(a, b) for i, a in enumerate(elems) for b in elems[i:]]
    r, g = partial_rdf(frames, pairs)
    cations = [e for e in elems if e != "O"]
    m = {"n_structures": len(frames), "n_atoms_mean": float(np.mean([len(a) for a in frames])),
         "r": r, "rdf": g, "angles": bond_angles(frames, ANGLE_TRIPLES[composition]),
         "cn": {f"{c}-O": coordination(frames, c, ["O"]) for c in cations},
         "rings": ring_sizes(frames, composition),
         "min_dist": min_distance(frames), "density": density(frames),
         "defects": defects(frames)}
    m["cn"]["O-cation"] = coordination(frames, "O", cations)
    if mace:
        m.update(mace_metrics(frames, device, elastic))
    return m


def distances(m, ref):
    d = {}
    for k in ref["rdf"]:   # integrated |g - g_ref| dr  (Å)
        d[f"rdf_L1_{k}"] = float(np.abs(m["rdf"][k] - ref["rdf"][k]).sum() * DR)
    for k in ref["angles"]:
        if len(m["angles"][k]) and len(ref["angles"][k]):
            d[f"W1_angle_{k}_deg"] = float(wasserstein_distance(m["angles"][k], ref["angles"][k]))
    for k in ref["cn"]:   # total variation between CN histograms
        hi = int(max(m["cn"][k].max(initial=0), ref["cn"][k].max(initial=0))) + 1
        p = np.bincount(m["cn"][k], minlength=hi) / max(len(m["cn"][k]), 1)
        q = np.bincount(ref["cn"][k], minlength=hi) / max(len(ref["cn"][k]), 1)
        d[f"TV_cn_{k}"] = float(0.5 * np.abs(p - q).sum())
    if ref["rings"] is not None:
        p, q = m["rings"] / max(m["rings"].sum(), 1e-12), ref["rings"] / max(ref["rings"].sum(), 1e-12)
        d["TV_ring_size"] = float(0.5 * np.abs(p - q).sum())
    for k in ("E_unrelaxed", "E_relaxed"):
        if k in m and k in ref:
            d[f"W1_{k}_eV"] = float(wasserstein_distance(m[k], ref[k]))
    if "moduli" in m and "moduli" in ref:
        for k in m["moduli"]:
            d[f"dmean_{k}"] = float(m["moduli"][k].mean() - ref["moduli"][k].mean())
    return d


def summary(m):
    s = {"n_structures": m["n_structures"], "n_atoms_mean": m["n_atoms_mean"],
         "density_g_cm3": float(m["density"].mean()),
         "min_dist_A": float(m["min_dist"].min()), **m["defects"],
         "cn_fractions": {k: {int(c): float(n / len(v)) for c, n in zip(*np.unique(v, return_counts=True))}
                          for k, v in m["cn"].items()},
         "angle_mean_std": {k: [float(v.mean()), float(v.std())] for k, v in m["angles"].items() if len(v)}}
    if m["rings"] is not None:
        s["rings_per_structure_by_nSi"] = {i: float(c) for i, c in enumerate(m["rings"]) if c > 0}
    for k in ("E_unrelaxed", "E_relaxed", "relax_rmsd"):
        if k in m:
            s[f"{k}_mean_std"] = [float(m[k].mean()), float(m[k].std())]
    if "E_unrelaxed" in m:
        s["E_drop_on_relax_eV_mean"] = float((m["E_unrelaxed"] - m["E_relaxed"]).mean())
    if "moduli" in m:
        s["moduli_GPa_mean_std"] = {k: [float(v.mean()), float(v.std())] for k, v in m["moduli"].items()}
    return s


def plot(results, composition, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ref = results["ref_B"]
    keys_rdf, keys_ang = list(ref["rdf"]), list(ref["angles"])
    ncol = max(len(keys_rdf), len(keys_ang), 3)
    fig, ax = plt.subplots(3, ncol, figsize=(4 * ncol, 10), squeeze=False)
    for name, m in results.items():
        ls = "--" if name.startswith("ref") else "-"
        for c, k in enumerate(keys_rdf):
            ax[0, c].plot(m["r"], m["rdf"][k], ls, label=name)
            ax[0, c].set(title=f"g(r) {k}", xlabel="r (Å)")
        for c, k in enumerate(keys_ang):
            if len(m["angles"][k]):
                ax[1, c].hist(m["angles"][k], bins=90, range=(0, 180), density=True,
                              histtype="step", ls=ls, label=name)
            ax[1, c].set(title=f"angle {k}", xlabel="deg")
        if m["rings"] is not None:
            ax[2, 0].plot(np.arange(len(m["rings"])), m["rings"] / max(m["rings"].sum(), 1e-12),
                          ls, marker="o", label=name)
            ax[2, 0].set(title="ring size (Si per ring)", xlim=(1, 12))
        if "E_relaxed" in m:
            ax[2, 1].hist(m["E_relaxed"], bins=30, density=True, histtype="step", ls=ls, label=name)
            ax[2, 1].set(title="MACE E/atom relaxed", xlabel="eV")
            ax[2, 2].hist(m["E_unrelaxed"] - m["E_relaxed"], bins=30, density=True,
                          histtype="step", ls=ls, label=name)
            ax[2, 2].set(title="E drop on relaxation", xlabel="eV/atom")
    ax[0, 0].legend()
    fig.suptitle(composition)
    fig.tight_layout()
    fig.savefig(out / f"benchmark_{composition}.png", dpi=120)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref", required=True, help="reference extxyz (melt-quench)")
    p.add_argument("--ref_step_label", default="final",
                   help="keep reference frames whose step_label starts with this ('' = all)")
    p.add_argument("--ref_defect_free", action="store_true",
                   help="compare against defect-free reference frames only")
    p.add_argument("--gen", nargs="+", default=[], metavar="NAME=PATH",
                   help="generated sets, e.g. llm=gen.extxyz llm_T1=gen_T1.extxyz")
    p.add_argument("--composition", default="SiO2")
    p.add_argument("--mace", action="store_true", help="MACE energies + relaxation")
    p.add_argument("--elastic", action="store_true", help="elastic moduli (needs --mace; slow)")
    p.add_argument("--max_structures", type=int, default=None,
                   help="subsample each set to at most this many structures")
    p.add_argument("--no_quests", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="benchmark")
    a = p.parse_args()

    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    ref = load(a.ref, a.composition, a.ref_step_label or None, a.ref_defect_free)
    if len(ref) < 4:
        raise SystemExit(f"only {len(ref)} reference frames for {a.composition}")
    perm = rng.permutation(len(ref))
    sets = {"ref_A": [ref[i] for i in perm[: len(ref) // 2]],
            "ref_B": [ref[i] for i in perm[len(ref) // 2:]]}
    for spec in a.gen:
        name, path = spec.split("=", 1)
        sets[name] = load(path, a.composition)
        if not sets[name]:
            raise SystemExit(f"{path}: no {a.composition} structures")
    if a.max_structures:
        sets = {k: [v[i] for i in rng.permutation(len(v))[: a.max_structures]]
                for k, v in sets.items()}

    results = {}
    for name, frames in sets.items():
        print(f"[{name}] {len(frames)} structures")
        results[name] = compute_all(frames, a.composition, a.mace, a.elastic and a.mace, a.device)

    report = {"composition": a.composition, "reference": a.ref,
              "note": "distances are to ref_B; ref_A row is the noise floor",
              "sets": {k: summary(m) for k, m in results.items()},
              "distance_to_ref_B": {k: distances(m, results["ref_B"])
                                    for k, m in results.items() if k != "ref_B"}}
    if not a.no_quests:
        report["quests"] = quests_metrics(sets, "ref_B", a.seed)

    (out / f"benchmark_{a.composition}.json").write_text(json.dumps(report, indent=1))
    plot(results, a.composition, out)

    names = [k for k in results if k != "ref_B"]
    keys = sorted({k for n in names for k in report["distance_to_ref_B"][n]})
    print(f"\nDistance to ref_B ({a.composition}); ref_A = noise floor")
    print(f"{'metric':32s}" + "".join(f"{n:>14s}" for n in names))
    for k in keys:
        print(f"{k:32s}" + "".join(f"{report['distance_to_ref_B'][n].get(k, float('nan')):14.4f}" for n in names))
    print(f"{'defect_free_fraction':32s}" + "".join(f"{report['sets'][n]['defect_free_fraction']:14.3f}" for n in names))
    if "quests" in report:
        print(f"{'QUESTS H (nats)':32s}" + "".join(f"{report['quests'][n]['H_nats']:14.3f}" for n in names))
        print(f"{'QUESTS overlap w/ ref_B':32s}" + "".join(f"{report['quests'][n]['overlap_with_ref']:14.3f}" for n in names))
    print(f"\nwrote {out}/benchmark_{a.composition}.json and .png")


if __name__ == "__main__":
    main()
