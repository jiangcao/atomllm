"""Window-completion prompts cut from MD frames (as data/surface/fair_complete.py):
a random crop of a clean region, ctx = known part of the window, env = atoms
already placed on the LOW side of the window (what a raster-scan growth sees),
fill = the rest of the window, to be generated. Judged atoms = window atoms more
than MAX_BOND from every excluded (not ctx / env / fill) atom, so all their
bonds are inside the evaluated structure."""
import json, numpy as np
from ase.io import read
import window_generation as wg
import window_generation_v2 as wg2
from window_tokenizer import growth_order
from structure_defects import find_defects

MAXB = wg.MAX_BOND
SHAPES = ([0, 0, 8], [8, 8, 0], [8, 8, 8], [0, 0, 0])   # fair_complete window shapes


def load_frames(paths, val_runs_file, split):
    """split 'train' -> frames of runs not in val_runs; 'val' -> only those."""
    vr = set(json.load(open(val_runs_file))["val_runs"])
    out = []
    for p in paths:
        for a in read(p, index=":"):
            if (a.info.get("label") in vr) == (split == "val"):
                out.append(a)
    bad = [{i for i, _ in find_defects(a)} for a in out]
    return out, bad


class Prompt:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def guide(self, tok):
        return wg2.ZWindowGuide(tok, self.levels, self.counts, self.sym[self.ctx_i], self.lw[self.ctx_i],
                                self.sym[self.past], self.lw[self.past], self.t, np.full(3, tok.W))


def sample_prompt(frames, bad, tok, rng, levels=None):
    """-> Prompt, or None when the crop is unusable (defect inside, empty fill)."""
    W = tok.W
    levels = list(levels if levels is not None else SHAPES[int(rng.integers(len(SHAPES)))])
    fi = int(rng.integers(len(frames)))
    a = frames[fi]
    inside, local, _, Lp = wg.crop_with_transform(a, W, rng)
    if bad[fi] & set(inside.tolist()):
        return None
    known = tok.is_known(local[inside], levels)
    ctx_i, fill_i = inside[known], inside[~known]
    if not len(fill_i):
        return None
    if len(ctx_i):
        ctx_i = ctx_i[growth_order(local[ctx_i], np.ones(len(ctx_i), bool), np.full(3, W / 2))]
    t = tok.thresholds(levels) * (np.asarray(levels) > 0)
    sym = np.array(a.get_chemical_symbols())
    lw = np.where(local > (Lp + W) / 2, local - Lp, local)       # unwrap below the window
    out_i = np.setdiff1d(np.arange(len(a)), inside)
    past = out_i[np.all(lw[out_i] < W, axis=1) & np.all(lw[out_i] > -2 * MAXB, axis=1)
                 & np.any(lw[out_i] < np.where(np.asarray(levels) > 0, t, 0), axis=1)]
    counts = {e: int((sym[fill_i] == e).sum()) for e in tok.config["elements"]}
    return Prompt(W=W, frame=fi, label=a.info.get("label"), levels=levels, t=t, sym=sym, lw=lw, Lp=Lp,
                  ctx_i=ctx_i, fill_i=fill_i, past=past, counts=counts,
                  excl=lw[np.setdiff1d(out_i, past)])
