"""Defects of a completed window prompt (fair-completion rule).
Judged ("core") atoms: inside the window, more than MAX_BOND from every
excluded atom (periodic distance in the crop frame), no neighbour only through
the periodic wrap, never env (past) atoms. Defects: structure_defects rules
(Si-O < 2.10, strict / stretched Si-Si, edge sharing, O needs 2 Si) on the
non-periodic evaluated structure ctx + env + generated."""
import numpy as np
from ase import Atoms
from structure_defects import find_defects
from rl.prompts import MAXB


def judge(pr, s_new, p_new):
    """-> (n_judged, defects) with defects = [(index in eval structure, label)]
    restricted to judged atoms. Eval structure = ctx + env + new."""
    sym = np.array(list(pr.sym[pr.ctx_i]) + list(pr.sym[pr.past]) + list(s_new))
    pos = np.vstack([pr.lw[pr.ctx_i], pr.lw[pr.past], np.asarray(p_new, float).reshape(-1, 3)])
    L = pr.Lp
    if len(pr.excl):
        d = pos[:, None] - pr.excl[None]
        d -= np.round(d / L) * L
        core = np.linalg.norm(d, axis=-1).min(1) > MAXB
    else:
        core = np.ones(len(pos), bool)
    core &= np.all((pos >= 0) & (pos < pr.W), axis=1)          # window atoms only
    dd = pos[:, None] - pos[None]
    dm = dd - np.round(dd / L) * L
    Dm, Dp = np.linalg.norm(dm, axis=-1), np.linalg.norm(dd, axis=-1)
    np.fill_diagonal(Dm, np.inf)
    core &= ~((Dm < MAXB) & (Dp > Dm + 1e-6)).any(1)
    n_ctx = len(pr.ctx_i)
    core[n_ctx:n_ctx + len(pr.past)] = False
    big = np.ptp(pos, axis=0).max() + 50.0
    a = Atoms(sym, positions=pos, cell=[big] * 3, pbc=False)
    defs = [(i, lab) for i, lab in find_defects(a) if core[i]]
    return int(core.sum()), defs
