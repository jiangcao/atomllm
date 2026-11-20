"""Records a real surface-growth run (window_generation_v2.py surface --record F)
as RL data. Every sampled candidate of every try is one sample:
  ids        full token sequence (prompt + generated)
  act        [(position, token, allowed token ids, log-prob at sampling)]
  window/try where it was sampled
  raw_bad    {defect label: count} of the model's raw output (dead ends included)
  repaired   whether it was relaxed/snapped before the final judgement
  chosen     accepted as the window's structure
  future     (chosen only) blame when later undone by a backtrack it caused, number
             of interior defects left in its atoms at the end, defects among its
             already-final atoms when it was undone, undone / survived
Blame on a backtrack: the window that placed the culprit atom (1.0); a substrate
culprit -> the earliest undone window (0.5); no clear culprit -> the last
accepted window (0.5). Windows undone only as collateral get no blame.
"""
import pickle
from collections import Counter
import numpy as np


class Recorder:
    def __init__(self):
        self.samples, self.stack, self.popped = [], [], []
        self.last_chosen = None
        self.meta = {}

    def on_try(self, n, t, guides, outs, cands, chosen):
        by_g = {c["g"]: c for c in cands}
        self.last_chosen = None
        for k, (g, (out, info)) in enumerate(zip(guides, outs)):
            if out is None or not info.get("act"):
                continue
            c = by_g.get(k)
            if c is None:
                continue
            s = dict(window=n, try_=t, ids=np.asarray(info["ids"], np.int16), act=info["act"],
                     raw_bad=dict(Counter(c["raw_bad"].values())),
                     final_bad=dict(Counter(c["bad"].values())),
                     repaired=bool(c["relaxed"]), chosen=c is chosen, future=None)
            self.samples.append(s)
            if c is chosen:
                self.last_chosen = len(self.samples) - 1

    def on_accept(self, k0, k1):
        sid = self.last_chosen
        if sid is not None:
            self.samples[sid]["future"] = dict(blame=0.0, final_defects=0, undone_defects=0,
                                               undone=False, survived=False)
        self.stack.append((sid, k0, k1))

    def before_backtrack(self, final_defects):
        self.bt_defects = final_defects

    def on_pop(self):
        self.popped.append(self.stack.pop())

    def on_backtrack(self, culprit, n_base):
        pop = self.popped                       # last accepted first
        if pop:
            if culprit is not None and culprit >= n_base:
                hit = [e for e in pop if e[1] <= culprit < e[2]]
                target, w = (hit[0], 1.0) if hit else (pop[-1], 0.5)
            elif culprit is not None:
                target, w = pop[-1], 0.5
            else:
                target, w = pop[0], 0.5
            for e in pop:
                if e[0] is not None:
                    f = self.samples[e[0]]["future"]
                    f["undone"] = True
                    f["undone_defects"] = int(sum(e[1] <= i < e[2] for i in getattr(self, "bt_defects", [])))
            if target[0] is not None:
                self.samples[target[0]]["future"]["blame"] += w
        self.popped = []

    def on_end(self, interior_defects, n_done, n_windows, n_back):
        for sid, k0, k1 in self.stack:
            if sid is None:
                continue
            f = self.samples[sid]["future"]
            f["survived"] = True
            f["final_defects"] = int(sum(k0 <= i < k1 for i in interior_defects))
        self.meta = dict(n_done=n_done, n_windows=n_windows, n_back=n_back,
                         interior_defects=len(interior_defects))

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"samples": self.samples, "meta": self.meta}, f)
        print(f"recorded {len(self.samples)} samples -> {path}", flush=True)
