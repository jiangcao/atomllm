"""
Defect-guided sampling for AtomicGPT.

Generation is conditioned on the Q_clean token, and on top of that every
sampling step is constrained:

  * grammar      : header, then (TYPE, X, Y, Z) groups, then [EOS]
  * stoichiometry: TYPE tokens are limited to the remaining atom counts for
                   the requested composition; [EOS] only when all are placed
  * defects      : at the Z token the atom's final position is known for every
                   candidate bin, so bins that would create a defect that can
                   never be undone are penalised (hard-masked by default):
                     - clashes (pair closer than MIN_DIST)
                     - overcoordination of the new atom or of a neighbour
                       (Si > 4, O > 2 cations [> 4 with Hf], Hf > 8)
                     - O-O (peroxy) bonds
                   If every bin is masked, the atom is resampled.

Undercoordination (e.g. a Si still missing an O) cannot be judged per token
because a later atom may complete it, so it is checked on the finished
structure with structure_defects.find_defects and the sample is rejected.
"""

import numpy as np
import torch
import torch.nn.functional as F
from ase import Atoms
from ase.data import atomic_masses, atomic_numbers

from atomic_tokenizer import HEADER_LEN, AtomicTokenizer
from structure_defects import BOND_CUTOFFS

DENSITY = {"SiO2": 2.2, "HfO2": 9.7, "HfSiO4": 6.5, "Si": 2.33}  # g/cm^3

# Closest allowed approach for any pair (Å)
MIN_DIST = {
    ("Si", "O"): 1.40, ("Si", "Si"): 2.10, ("O", "O"): 2.00,
    ("Hf", "O"): 1.80, ("Hf", "Si"): 2.60, ("Hf", "Hf"): 2.90,
}

MAX_SI_CN = 4      # Si + O neighbours of a Si
MAX_HF_CN = 8      # O neighbours of a Hf


def _pair(table, a, b, default=0.0):
    return table.get((a, b), table.get((b, a), default))


def target_counts(composition: dict, n_atoms: int) -> dict:
    fu = max(1, round(n_atoms / sum(composition.values())))
    return {e: r * fu for e, r in composition.items()}


def cubic_cell_length(counts: dict, density: float) -> float:
    mass = sum(n * atomic_masses[atomic_numbers[e]] for e, n in counts.items())
    return ((mass / 6.02214076e23) / (density / 1e24)) ** (1 / 3)


class DefectGuide:
    """Tracks the partially built structure and masks logits."""

    def __init__(self, tokenizer: AtomicTokenizer, composition: str,
                 n_atoms: int, cell_length: float = None,
                 penalty: float = float("inf")):
        self.tok = tokenizer
        self.vocab = tokenizer.vocab
        self.comp = composition
        ratio = tokenizer.config["compositions"][composition]
        self.counts = target_counts(ratio, n_atoms)
        if cell_length is None:
            cell_length = cubic_cell_length(self.counts, DENSITY[composition])
        # Snap to the tokenizer's cell grid so prompt and decode agree
        res, lo = tokenizer.config["cell_resolution"], tokenizer.config["cell_min"]
        self.L = lo + round((cell_length - lo) / res) * res
        self.cell = np.diag([self.L] * 3)
        self.penalty = penalty
        self.n_bins = tokenizer.config["n_bins"]

        self.type_ids = {e: self.vocab.encode_token(f"TYPE_{e}")
                         for e in tokenizer.config["elements"]}
        self.id2elem = {v: k for k, v in self.type_ids.items()}
        self.axis_ids = {a: torch.tensor(
            [self.vocab.encode_token(f"{a}_{b}") for b in range(self.n_bins)])
            for a in "XYZ"}
        self.eos = tokenizer.eos_id
        self.reset()

    # ---------------- state ----------------

    def reset(self):
        self.sym: list[str] = []
        self.frac = np.zeros((0, 3))
        self.n_siO = np.zeros(0, int)   # Si: # Si/O neighbours
        self.n_cat = np.zeros(0, int)   # O : # cation neighbours
        self.has_hf = np.zeros(0, bool)  # O : bonded to a Hf
        self.n_o = np.zeros(0, int)     # Hf: # O neighbours
        self.remaining = dict(self.counts)

    def prompt_ids(self) -> list[int]:
        v = self.vocab
        ids = [self.tok.bos_id, v.encode_token(f"COMP_{self.comp}"),
               v.encode_token("Q_clean")] + [self.tok.cell_token(self.L)] * 3
        assert len(ids) == HEADER_LEN
        return ids

    def _dists(self, cand_frac: np.ndarray) -> np.ndarray:
        """(m, 3) candidate fractional coords -> (m, n_placed) MIC distances."""
        d = cand_frac[:, None, :] - self.frac[None, :, :]
        d -= np.round(d)
        return np.linalg.norm(d @ self.cell, axis=-1)

    def _bad_positions(self, elem: str, cand_frac: np.ndarray) -> np.ndarray:
        """Boolean mask (m,) of candidate positions that create a defect."""
        m = len(cand_frac)
        if not self.sym:
            return np.zeros(m, bool)
        D = self._dists(cand_frac)
        sym = np.array(self.sym)
        mind = np.array([_pair(MIN_DIST, elem, s) for s in sym])
        cut = np.array([_pair(BOND_CUTOFFS, elem, s) for s in sym])
        bad = (D < mind).any(axis=1)
        B = D < cut                                     # (m, n) bonds formed

        is_si, is_o, is_hf = sym == "Si", sym == "O", sym == "Hf"
        if elem == "Si":
            bad |= (B & (is_si | is_o)).sum(1) > MAX_SI_CN
            bad |= (B & is_si & (self.n_siO + 1 > MAX_SI_CN)).any(1)
            o_max = np.where(self.has_hf, 4, 2)
            bad |= (B & is_o & (self.n_cat + 1 > o_max)).any(1)
        elif elem == "O":
            bad |= (B & is_o).any(1)                     # peroxy
            n_cat = (B & (is_si | is_hf)).sum(1)
            bad |= n_cat > np.where((B & is_hf).any(1), 4, 2)
            bad |= (B & is_si & (self.n_siO + 1 > MAX_SI_CN)).any(1)
            bad |= (B & is_hf & (self.n_o + 1 > MAX_HF_CN)).any(1)
        elif elem == "Hf":
            bad |= (B & is_o).sum(1) > MAX_HF_CN
            bad |= (B & is_o & (self.n_cat + 1 > 4)).any(1)
        return bad

    def add_atom(self, elem: str, f: np.ndarray):
        n = len(self.sym)
        b = (self._dists(f[None])[0] < np.array(
            [_pair(BOND_CUTOFFS, elem, s) for s in self.sym])) if n else np.zeros(0, bool)
        sym = np.array(self.sym)
        is_si, is_o, is_hf = sym == "Si", sym == "O", sym == "Hf"
        if elem in ("Si", "O"):
            self.n_siO = self.n_siO + (b & is_si)
        if elem in ("Si", "Hf"):
            self.n_cat = self.n_cat + (b & is_o)
        if elem == "Hf":
            self.has_hf = self.has_hf | (b & is_o)
        if elem == "O":
            self.n_o = self.n_o + (b & is_hf)

        self.sym.append(elem)
        self.frac = np.vstack([self.frac, f])
        self.n_siO = np.append(self.n_siO, (b & (is_si | is_o)).sum() if elem == "Si" else 0)
        self.n_cat = np.append(self.n_cat, (b & (is_si | is_hf)).sum() if elem == "O" else 0)
        self.has_hf = np.append(self.has_hf, bool((b & is_hf).any()) if elem == "O" else False)
        self.n_o = np.append(self.n_o, (b & is_o).sum() if elem == "Hf" else 0)
        self.remaining[elem] -= 1

    # ---------------- logits masking ----------------

    def mask(self, logits: torch.Tensor, group: list[int]) -> torch.Tensor:
        """
        logits: (vocab,) for the next token; group: tokens of the current
        atom so far (0..3 tokens). Returns masked logits.
        """
        out = torch.full_like(logits, float("-inf"))
        slot = len(group)
        if slot == 0:
            allowed = [self.type_ids[e] for e, n in self.remaining.items() if n > 0]
            allowed = allowed or [self.eos]
            out[allowed] = logits[allowed]
            return out
        axis = "XYZ"[slot - 1]
        ids = self.axis_ids[axis].to(logits.device)
        vals = logits[ids].clone()
        if slot == 3:
            elem = self.id2elem[group[0]]
            fx, fy = [(self._bin(t) + 0.5) / self.n_bins for t in group[1:3]]
            fz = (np.arange(self.n_bins) + 0.5) / self.n_bins
            cand = np.column_stack([np.full(self.n_bins, fx),
                                    np.full(self.n_bins, fy), fz])
            bad = torch.from_numpy(self._bad_positions(elem, cand)).to(logits.device)
            vals = vals - bad.float() * self.penalty if np.isfinite(self.penalty) \
                else vals.masked_fill(bad, float("-inf"))
        out[ids] = vals
        return out

    def _bin(self, token_id: int) -> int:
        return int(self.vocab.decode_id(token_id).split("_")[1])

    def to_atoms(self) -> Atoms:
        return Atoms(self.sym, scaled_positions=self.frac, cell=self.cell, pbc=True)


@torch.no_grad()
def guided_generate(model, guide: DefectGuide, temperature: float = 0.8,
                    top_k: int = 50, max_atom_retries: int = 20,
                    device: str = "cpu"):
    """
    Sample one structure. Returns (Atoms or None, stats dict).
    None means an atom could not be placed without a defect.
    """
    guide.reset()
    ids = guide.prompt_ids()
    group: list[int] = []
    retries, total_retries = 0, 0
    max_len = model.config.max_seq_len

    while True:
        if len(ids) >= max_len:
            return None, {"reason": "max_seq_len", "retries": total_retries}
        x = torch.tensor([ids], device=device)
        logits = model(x)[0][0, -1] / temperature
        logits = guide.mask(logits, group)

        if torch.isinf(logits).all():            # no defect-free Z bin
            retries += 1
            total_retries += 1
            if retries > max_atom_retries:
                return None, {"reason": f"could not place atom {len(guide.sym)+1}",
                              "retries": total_retries}
            del ids[-len(group):]
            group = []
            continue

        if top_k:
            kth = torch.topk(logits, min(top_k, int(torch.isfinite(logits).sum())))[0][-1]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        nxt = int(torch.multinomial(F.softmax(logits, -1), 1))
        ids.append(nxt)

        if nxt == guide.eos:
            return guide.to_atoms(), {"retries": total_retries}
        group.append(nxt)
        if len(group) == 4:
            elem = guide.id2elem[group[0]]
            f = np.array([(guide._bin(t) + 0.5) / guide.n_bins for t in group[1:]])
            guide.add_atom(elem, f)
            group, retries = [], 0
