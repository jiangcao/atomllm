"""Physicality report for amorphous SiO2 and Si/SiO2 structures, compared with
MD reference frames (validation runs).

Level 1 (local)   coordination, defects per 100 atoms (structure_defects rules),
                  Si-O length, O-Si-O and Si-O-Si angles, edge-sharing Si pairs
Level 2 (medium)  shortest-path rings in the Si-O-Si network (per Si and pair of
                  its bridges, periodic images tracked), density, g(r) (periodic only)
Interface         Si oxidation states Si0..Si4+ (number of O neighbours of 4-fold Si
                  that also have Si neighbours), per nm^2 of interface
Level 3 (--mace)  MACE-MP-0 per-atom energies of evaluated atoms vs reference
                  (strain indicator: mean excess and fraction above the reference 99th
                  percentile)

Evaluated atoms ("region"):
  periodic frame          all atoms
  frame with array core   those atoms (e.g. anneal_grown.py output)
  grown (tags, no pbc)    interior: >= 5 Å inside the substrate's lateral extent,
                          >= 6 Å above the substrate bottom, >= 3 Å below the top;
                          density from the oxide-only box above the interface.

usage: physicality.py FILE[@INDEX] [FILE ...] [--frames last|all] [--mace] [--out report.json]
       Reference statistics are cached in data/physicality_ref.npz (--rebuild_ref).
"""
import sys, os, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ase.io import read
from ase.neighborlist import neighbor_list
from structure_defects import find_defects, coordination

R_SIO, R_MAX = 2.10, 6.0
LAT_MARGIN = 5.0          # grown structures: lateral margin of the evaluated region (--margin)
MAX_RING = 12
MASS = {"Si": 28.0855, "O": 15.999, "Hf": 178.49}
REF_SETS = {"bulk": "data/sio2_all/dataset.extxyz", "iface": "data/interface/dataset.extxyz"}
VAL_RUNS = "data/sio2_iface_win8_z/sequences.json"


# ---------------------------------------------------------------- regions
def region_of(a):
    """-> (mask of evaluated atoms, density (g/cm3) or None, n_interfaces, area_nm2)."""
    n = len(a)
    if all(a.pbc):
        mass = sum(MASS[s] for s in a.get_chemical_symbols())
        dens = mass / a.get_volume() * 1.66054
        sub = a.arrays.get("substrate")
        n_if = 2 if sub is not None and np.asarray(sub).any() else 0
        c = np.asarray(a.get_cell())
        return np.ones(n, bool), dens, n_if, np.linalg.norm(np.cross(c[0], c[1])) / 100
    if "core" in a.arrays:
        m = a.arrays["core"].astype(bool)
    else:
        m = np.ones(n, bool)
    tags = a.get_tags()
    if not tags.any():
        return m, None, 0, None
    sub, p = tags == 1, a.positions
    lo, hi = p[sub, :2].min(0), p[sub, :2].max(0)
    zb, zt, ztop = p[sub, 2].min(), p[sub, 2].max(), p[~sub, 2].max()
    if "core" not in a.arrays:
        m = (np.all(p[:, :2] > lo + LAT_MARGIN, axis=1) & np.all(p[:, :2] < hi - LAT_MARGIN, axis=1)
             & (p[:, 2] > zb + 6) & (p[:, 2] < ztop - 3))
    # oxide-only density box: inner lateral square, 3 Å above the substrate top to 3 Å below the top
    blo, bhi = np.r_[lo + LAT_MARGIN, zt + 3], np.r_[hi - LAT_MARGIN, ztop - 3]
    dens = None
    if np.all(bhi > blo):
        inb = np.all((p >= blo) & (p < bhi), axis=1)
        dens = sum(MASS[s] for s in np.array(a.get_chemical_symbols())[inb]) / np.prod(bhi - blo) * 1.66054
    return m, dens, 1, np.prod(hi - lo - 2 * LAT_MARGIN) / 100


# ---------------------------------------------------------------- level 1
def local_stats(a, reg):
    sym = np.array(a.get_chemical_symbols())
    i, j, d, D = neighbor_list("ijdD", a, {("Si", "O"): R_SIO})
    si_o = (sym[i] == "Si") & (sym[j] == "O")
    out = {"SiO_len": d[si_o & reg[i]].tolist()}
    osio, siosi = [], []
    for c, other, store in (("Si", "O", osio), ("O", "Si", siosi)):
        sel = (sym[i] == c) & (sym[j] == other) & reg[i]
        ii, DD = i[sel], D[sel]
        order = np.argsort(ii, kind="stable")
        ii, DD = ii[order], DD[order]
        starts = np.r_[0, np.where(np.diff(ii))[0] + 1, len(ii)]
        for s0, s1 in zip(starts[:-1], starts[1:]):
            V = DD[s0:s1]
            if c == "O" and len(V) != 2:
                continue
            U = V / np.linalg.norm(V, axis=1)[:, None]
            cos = np.clip(U @ U.T, -1, 1)
            k = np.triu_indices(len(V), 1)
            store.extend(np.degrees(np.arccos(cos[k])).tolist())
    out["OSiO"], out["SiOSi"] = osio, siosi
    # coordination / defects
    _, n_eff, counts = coordination(a)
    cn_si = [(c.get("O", 0), c.get("Si", 0)) for k, c in enumerate(counts) if sym[k] == "Si" and reg[k]]
    cn_o = [c.get("Si", 0) for k, c in enumerate(counts) if sym[k] == "O" and reg[k]]
    out["Si_cn"] = [o + s for o, s in cn_si]
    out["O_cn"] = cn_o
    out["defects"] = sum(1 for k, _ in find_defects(a) if reg[k])
    out["n_eval"] = int(reg.sum())
    # oxidation states of 4-fold Si that have Si neighbours (+ fully oxidised Si4+)
    ox = [o for o, s in cn_si if o + s == 4 and (s > 0 or o == 4)]
    out["Si_ox"] = ox
    out["si4_mask"] = np.array([sym[k] == "Si" and reg[k] and c.get("O", 0) == 4 and sum(c.values()) == 4
                                for k, c in enumerate(counts)], bool)
    out["Si_ox_sub"] = [o for o, s in cn_si if o + s == 4 and 0 < o < 4]
    # edge sharing: Si pairs sharing >= 2 O (same image)
    o_si = {}
    for a_, b_, S in zip(*neighbor_list("ijS", a, {("Si", "O"): R_SIO})):
        if sym[a_] == "O" and sym[b_] == "Si":
            o_si.setdefault(a_, []).append((b_, tuple(S)))
    pairs = {}
    for o, lst in o_si.items():
        for x in range(len(lst)):
            for y in range(x + 1, len(lst)):
                (s1, S1), (s2, S2) = lst[x], lst[y]
                key = (s1, s2, tuple(np.subtract(S2, S1))) if s1 < s2 else (s2, s1, tuple(np.subtract(S1, S2)))
                pairs[key] = pairs.get(key, 0) + 1
    out["edge_sharing"] = sum(1 for (s1, s2, _), c in pairs.items() if c >= 2 and (reg[s1] or reg[s2]))
    out["n_Si"] = int(((sym == "Si") & reg).sum())
    return out


# ---------------------------------------------------------------- level 2
def si_network(a):
    """Si -> list of (Si, image shift) linked by a 2-fold bridging O."""
    sym = np.array(a.get_chemical_symbols())
    i, j, S = neighbor_list("ijS", a, {("Si", "O"): R_SIO})
    o_nb = {}
    for x, y, s in zip(i, j, S):
        if sym[x] == "O" and sym[y] == "Si":
            o_nb.setdefault(x, []).append((y, s))
    adj = {}
    for o, lst in o_nb.items():
        if len(lst) != 2:
            continue
        (s1, S1), (s2, S2) = lst                       # Si positions relative to O's image 0
        adj.setdefault(s1, []).append((s2, tuple(S2 - S1)))
        adj.setdefault(s2, []).append((s1, tuple(S1 - S2)))
    return adj


def rings_through(adj, v, max_ring=MAX_RING):
    """shortest ring size through v for every pair of its bridges (None if > max)."""
    edges = adj.get(v, [])
    m = len(edges)
    if m < 2:
        return []
    depth = max_ring // 2
    dist, br = {(v, (0, 0, 0)): 0}, {(v, (0, 0, 0)): 0}
    best = {}
    frontier = []
    for e, (w, s) in enumerate(edges):
        st = (w, s)
        if w == v:
            continue
        if st in dist:                                   # 2-ring (edge sharing)
            for f in range(m):
                if br[st] >> f & 1:
                    best[(f, e)] = min(best.get((f, e), 99), 2)
            br[st] |= 1 << e
        else:
            dist[st], br[st] = 1, 1 << e
            frontier.append(st)
    for dcur in range(1, depth + 1):
        nxt = []
        for st in frontier:
            u, su = st
            for w, sw in adj.get(u, []):
                if w == v:
                    continue
                t = (w, tuple(np.add(su, sw)))
                if t not in dist:
                    if dcur < depth:
                        dist[t], br[t] = dcur + 1, br[st]
                        nxt.append(t)
                    continue
                if dist[t] == dcur + 1:                  # even ring meeting at t
                    for f in range(m):
                        for g in range(m):
                            if f < g and ((br[st] >> f & 1 and br[t] >> g & 1) or (br[st] >> g & 1 and br[t] >> f & 1)):
                                best[(f, g)] = min(best.get((f, g), 99), 2 * (dcur + 1))
                    br[t] |= br[st]
                elif dist[t] == dcur:                    # odd ring meeting across edge (st, t)
                    for f in range(m):
                        for g in range(m):
                            if f < g and ((br[st] >> f & 1 and br[t] >> g & 1) or (br[st] >> g & 1 and br[t] >> f & 1)):
                                best[(f, g)] = min(best.get((f, g), 99), 2 * dcur + 1)
        frontier = nxt
    return [best.get((f, g)) for f in range(m) for g in range(f + 1, m)]


def ring_stats(a, reg):
    adj = si_network(a)
    sym = np.array(a.get_chemical_symbols())
    sizes = []
    for v in np.where((sym == "Si") & reg)[0]:
        sizes += [s for s in rings_through(adj, v) if s is not None]
    return sizes


def gr(a, reg, pair, dr=0.05):
    """partial g(r) (periodic frames only), centres in region."""
    sym = np.array(a.get_chemical_symbols())
    i, j, d = neighbor_list("ijd", a, R_MAX)
    sel = (sym[i] == pair[0]) & (sym[j] == pair[1]) & reg[i]
    h, e = np.histogram(d[sel], bins=np.arange(0, R_MAX + dr, dr))
    n_c = ((sym == pair[0]) & reg).sum()
    rho = (sym == pair[1]).sum() / a.get_volume()
    r = 0.5 * (e[1:] + e[:-1])
    return h / max(n_c, 1) / (4 * np.pi * r ** 2 * dr * rho)


# ---------------------------------------------------------------- level 3
_calc = None


def node_energies(a):
    global _calc
    if _calc is None:
        from mace.calculators import mace_mp
        _calc = mace_mp(model="medium", device="cuda", default_dtype="float64")
    b = a.copy(); b.calc = _calc
    b.get_potential_energy()
    return np.asarray(_calc.results["energies"])


# ---------------------------------------------------------------- aggregate
def analyse(frames, mace=False):
    acc = {k: [] for k in ("SiO_len", "OSiO", "SiOSi", "Si_cn", "O_cn", "Si_ox", "Si_ox_sub", "rings",
                           "E_Si", "E_O")}
    tot = dict(defects=0, n_eval=0, edge_sharing=0, n_Si=0, area=0.0)
    dens, grs = [], {p: [] for p in (("Si", "O"), ("O", "O"), ("Si", "Si"))}
    for a in frames:
        reg, de, n_if, area = region_of(a)
        loc = local_stats(a, reg)
        for k in ("SiO_len", "OSiO", "SiOSi", "Si_cn", "O_cn", "Si_ox", "Si_ox_sub"):
            acc[k] += loc[k]
        for k in ("defects", "n_eval", "edge_sharing", "n_Si"):
            tot[k] += loc[k]
        if n_if and area:
            tot["area"] += n_if * area
        acc["rings"] += ring_stats(a, reg)
        if de is not None:
            dens.append(de)
        if all(a.pbc):
            for p in grs:
                grs[p].append(gr(a, reg, p))
        if mace:
            e = node_energies(a)
            sym = np.array(a.get_chemical_symbols())
            acc["E_Si"] += e[loc["si4_mask"]].tolist()          # SiO4 only (not substrate-like Si)
            acc["E_O"] += e[(sym == "O") & reg].tolist()
    res = {k: np.asarray(v, float) for k, v in acc.items()}
    res.update({k: np.asarray(v, float) for k, v in tot.items()})
    res["density"] = np.asarray(dens, float)
    for p, v in grs.items():
        res["gr_" + "".join(p)] = np.mean(v, 0) if v else np.zeros(0)
    res["n_frames"] = np.asarray(len(frames))
    return res


def load_ref(rebuild=False, n_per=60, mace=False, path="data/physicality_ref.npz"):
    if os.path.exists(path) and not rebuild:
        z = np.load(path, allow_pickle=True)
        ref = z["ref"].item()
        if not mace or len(ref["bulk"]["E_Si"]):
            return ref
    vr = set(json.load(open(VAL_RUNS))["val_runs"])
    ref = {}
    rng = np.random.default_rng(0)
    for name, f in REF_SETS.items():
        fr = [a for a in read(f, index=":") if a.info.get("label") in vr]
        fr = [fr[k] for k in sorted(rng.choice(len(fr), size=min(n_per, len(fr)), replace=False))]
        print(f"reference {name}: {len(fr)} validation frames from {f}", flush=True)
        ref[name] = analyse(fr, mace)
    np.savez(path, ref=np.array(ref, dtype=object))
    return ref


# ---------------------------------------------------------------- report
def summary(r):
    def ms(x):
        return (float(np.mean(x)), float(np.std(x))) if len(x) else (np.nan, np.nan)
    s = {}
    s["defects / 100 atoms"] = 100 * float(r["defects"]) / max(float(r["n_eval"]), 1)
    s["Si 4-fold %"] = 100 * float(np.mean(r["Si_cn"] == 4)) if len(r["Si_cn"]) else np.nan
    s["O 2-fold %"] = 100 * float(np.mean(r["O_cn"] == 2)) if len(r["O_cn"]) else np.nan
    for k, lab in (("SiO_len", "Si-O (Å)"), ("OSiO", "O-Si-O (deg)"), ("SiOSi", "Si-O-Si (deg)")):
        m, sd = ms(r[k]); s[lab + " mean"] = m; s[lab + " std"] = sd
    s["Si-O-Si < 120 deg %"] = 100 * float(np.mean(r["SiOSi"] < 120)) if len(r["SiOSi"]) else np.nan
    s["edge-sharing pairs / 100 Si"] = 100 * float(r["edge_sharing"]) / max(float(r["n_Si"]), 1)
    rs = r["rings"]
    for n in range(2, 11):
        s[f"ring {n} %"] = 100 * float(np.mean(rs == n)) if len(rs) else np.nan
    s["density (g/cm3)"] = ms(r["density"])[0]
    if float(r["area"]) > 0:
        for n in (1, 2, 3):
            s[f"Si{n}+ / nm2 interface"] = float(np.sum(r["Si_ox_sub"] == n)) / float(r["area"])
    for e in ("E_Si", "E_O"):
        if len(r[e]):
            s[f"MACE {'SiO4 Si' if e == 'E_Si' else 'O'} energy (eV)"] = ms(r[e])[0]
    return s


def compare(sample, ref):
    from scipy.stats import wasserstein_distance as w1
    out = {}
    for k in ("SiO_len", "OSiO", "SiOSi"):
        if len(sample[k]) and len(ref[k]):
            out[f"W1 {k}"] = float(w1(sample[k], ref[k]))
    for e in ("E_Si", "E_O"):
        if len(sample[e]) and len(ref[e]):
            p99 = np.percentile(ref[e], 99)
            out[f"{e[2:]} mean excess energy (meV)"] = 1000 * float(np.mean(sample[e]) - np.mean(ref[e]))
            out[f"{e[2:]} atoms above ref p99 %"] = 100 * float(np.mean(sample[e] > p99))
    return out


def print_table(cols):
    names = list(cols)
    keys = []
    for c in cols.values():
        keys += [k for k in c if k not in keys]
    w = max(len(k) for k in keys) + 2
    print("".ljust(w) + "".join(n[:14].rjust(15) for n in names))
    for k in keys:
        row = k.ljust(w)
        for n in names:
            v = cols[n].get(k)
            row += ("" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3g}").rjust(15)
        print(row)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--frames", default="last", choices=["last", "all"])
    ap.add_argument("--mace", action="store_true")
    ap.add_argument("--rebuild_ref", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--pool", default=None, help="pool all files into one column with this name")
    ap.add_argument("--margin", type=float, default=5.0, help="grown: lateral margin (Å)")
    a = ap.parse_args()
    LAT_MARGIN = a.margin
    ref = load_ref(a.rebuild_ref, mace=a.mace)
    cols = {f"ref {k}": summary(v) for k, v in ref.items()}
    if a.pool:                                       # all files' frames as one sample
        fr = [x for f in a.files for x in read(f, index="-1:" if a.frames == "last" else ":")]
        r = analyse(fr, a.mace)
        s = summary(r)
        s.update({f"vs bulk: {k}": v for k, v in compare(r, ref["bulk"]).items()})
        s["evaluated atoms"] = float(r["n_eval"])
        cols[a.pool] = s
        a.files = []
    for f in a.files:
        f, _, idx = f.partition("@")                  # FILE@INDEX picks one frame
        fr = read(f, index=f"{idx}:{int(idx) + 1 or None}" if idx else ("-1:" if a.frames == "last" else ":"))
        r = analyse(fr, a.mace)
        name = os.path.basename(f).replace(".extxyz", "") + (f"@{idx}" if idx else "")
        s = summary(r)
        s.update({f"vs bulk: {k}": v for k, v in compare(r, ref["bulk"]).items()})
        s["evaluated atoms"] = float(r["n_eval"])
        cols[name] = s
    print_table(cols)
    if a.out:
        json.dump(cols, open(a.out, "w"), indent=1)
