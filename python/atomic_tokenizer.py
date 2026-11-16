"""
Atomic Sequence Tokenizer for Amorphous Structure LLM
======================================================
Converts periodic atomic structures (ASE Atoms / extxyz) into token sequences
for a GPT-style LLM, and decodes them back *exactly* (up to bin resolution).

Sequence layout:
    [BOS] COMP_<name> Q_<q> L_<a> L_<b> L_<c>  (TYPE_<el> X_<i> Y_<j> Z_<k>)*  [EOS]

    COMP_<name> : composition label (SiO2, HfO2, HfSiO4, Si, mixed)
    Q_<q>       : Q_clean (no coordination defects) or Q_defect. Training uses
                  both; generation is prompted with Q_clean (conditioning).
    L_<n>       : orthorhombic cell length a, b, c in units of cell_resolution
    TYPE_<el>   : element
    X/Y/Z_<i>   : fractional coordinate bin along a / b / c

Fractional coordinates keep periodicity: bonds that cross the cell boundary
survive the roundtrip, so a defect-free structure decodes defect-free.

Atoms are ordered by cluster growing with minimum-image distances (start near
the cell centre, then repeatedly add the unplaced atom nearest to the cluster),
so the model builds structures locally, bond by bond.

Usage:
    tokenizer = AtomicTokenizer().build_vocab()
    ids   = tokenizer.encode(atoms)
    atoms = tokenizer.decode(ids)
"""

import json
from typing import Optional

import numpy as np
from ase import Atoms
from ase.io import read
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

TOKENIZER_CONFIG = {
    "format": "fractional_v2",

    # Fractional coordinate bins per axis.
    # Resolution = L / n_bins: 0.02 Å for a 10 Å cell, 0.06 Å for 30 Å.
    "n_bins": 512,

    # Orthorhombic cell lengths (Å), discretised
    "cell_min": 5.0,
    "cell_max": 40.0,
    "cell_resolution": 0.05,

    "elements": ["Si", "O", "Hf"],

    # Composition labels: name -> stoichiometric ratio
    "compositions": {
        "SiO2":   {"Si": 1, "O": 2},
        "HfO2":   {"Hf": 1, "O": 2},
        "HfSiO4": {"Hf": 1, "Si": 1, "O": 4},
        "Si":     {"Si": 1},
    },

    # Structures with more atoms are skipped (never truncated: truncation
    # would cut bonds and create artificial dangling bonds).
    "max_atoms": 128,
}

HEADER_LEN = 6   # [BOS], COMP, Q, L_a, L_b, L_c
QUALITY_TOKENS = ("Q_clean", "Q_defect")
AXES = ("X", "Y", "Z")


# ============================================================
# VOCABULARY
# ============================================================

class Vocabulary:
    """Maps token strings <-> integer IDs. [PAD]=0 [BOS]=1 [EOS]=2 [UNK]=3."""

    SPECIAL_TOKENS = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]

    def __init__(self):
        self.token2id: dict[str, int] = {}
        self.id2token: dict[int, str] = {}
        for tok in self.SPECIAL_TOKENS:
            self._add(tok)

    def _add(self, token: str) -> int:
        if token not in self.token2id:
            idx = len(self.token2id)
            self.token2id[token] = idx
            self.id2token[idx] = token
        return self.token2id[token]

    def build(self, config: dict):
        for name in list(config["compositions"]) + ["mixed"]:
            self._add(f"COMP_{name}")
        for q in QUALITY_TOKENS:
            self._add(q)
        for n in range(_n_cell_bins(config)):
            self._add(f"L_{n}")
        for elem in config["elements"]:
            self._add(f"TYPE_{elem}")
        for axis in AXES:
            for b in range(config["n_bins"]):
                self._add(f"{axis}_{b}")
        print(f"Vocabulary size: {len(self.token2id)} tokens")

    def __len__(self):
        return len(self.token2id)

    def encode_token(self, token: str) -> int:
        return self.token2id.get(token, self.token2id["[UNK]"])

    def decode_id(self, idx: int) -> str:
        return self.id2token.get(idx, "[UNK]")

    def ids_with_prefix(self, prefix: str) -> list[int]:
        return [i for t, i in self.token2id.items() if t.startswith(prefix)]

    def to_dict(self) -> dict:
        return {"token2id": self.token2id,
                "id2token": {str(k): v for k, v in self.id2token.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "Vocabulary":
        vocab = cls.__new__(cls)
        vocab.token2id = d["token2id"]
        vocab.id2token = {int(k): v for k, v in d["id2token"].items()}
        return vocab


def _n_cell_bins(config: dict) -> int:
    return int(round((config["cell_max"] - config["cell_min"])
                     / config["cell_resolution"])) + 1


# ============================================================
# HELPERS
# ============================================================

def composition_label(atoms: Atoms, config: dict) -> str:
    """Match the structure's stoichiometry to a known composition label."""
    syms = atoms.get_chemical_symbols()
    counts = {e: syms.count(e) for e in set(syms)}
    for name, ratio in config["compositions"].items():
        if set(ratio) != set(counts):
            continue
        k = counts[next(iter(ratio))] / ratio[next(iter(ratio))]
        if all(abs(counts[e] - k * r) < 1e-6 for e, r in ratio.items()):
            return name
    return "mixed"


def cell_lengths_orthorhombic(atoms: Atoms) -> np.ndarray:
    cell = np.asarray(atoms.get_cell())
    off_diag = cell - np.diag(np.diag(cell))
    if np.abs(off_diag).max() > 1e-6:
        raise ValueError("only orthorhombic cells are supported")
    return np.diag(cell).copy()


def cluster_growing_order(atoms: Atoms) -> list[int]:
    """
    Cluster-growing order with minimum-image distances: start with the atom
    nearest the cell centre, then repeatedly add the unplaced atom that is
    nearest to any placed atom.
    """
    n = len(atoms)
    if n == 0:
        return []
    frac = atoms.get_scaled_positions(wrap=True)
    cell = np.asarray(atoms.get_cell())

    def mic_dist(i_frac, others_frac):
        d = others_frac - i_frac
        d -= np.round(d)
        return np.linalg.norm(d @ cell, axis=1)

    start = int(np.argmin(mic_dist(np.full(3, 0.5), frac)))
    order = [start]
    placed = np.zeros(n, dtype=bool)
    placed[start] = True
    min_d = mic_dist(frac[start], frac)
    min_d[start] = np.inf

    for _ in range(n - 1):
        nxt = int(np.argmin(np.where(placed, np.inf, min_d)))
        order.append(nxt)
        placed[nxt] = True
        min_d = np.minimum(min_d, mic_dist(frac[nxt], frac))
    return order


# ============================================================
# ENCODING / DECODING
# ============================================================

def encode_atoms(atoms: Atoms, vocab: Vocabulary, config: dict,
                 add_special: bool = True) -> list[int]:
    """Encode an orthorhombic periodic structure into token IDs."""
    from structure_defects import find_defects
    if len(atoms) > config["max_atoms"]:
        raise ValueError(f"{len(atoms)} atoms > max_atoms={config['max_atoms']}")

    lengths = cell_lengths_orthorhombic(atoms)
    n_cell = _n_cell_bins(config)
    n_bins = config["n_bins"]

    ids = []
    if add_special:
        ids.append(vocab.encode_token("[BOS]"))
    ids.append(vocab.encode_token(f"COMP_{composition_label(atoms, config)}"))
    ids.append(vocab.encode_token("Q_defect" if find_defects(atoms) else "Q_clean"))
    for L in lengths:
        k = int(round((L - config["cell_min"]) / config["cell_resolution"]))
        if not 0 <= k < n_cell:
            raise ValueError(f"cell length {L:.2f} Å outside "
                             f"[{config['cell_min']}, {config['cell_max']}]")
        ids.append(vocab.encode_token(f"L_{k}"))

    frac = atoms.get_scaled_positions(wrap=True)
    symbols = atoms.get_chemical_symbols()
    for i in cluster_growing_order(atoms):
        ids.append(vocab.encode_token(f"TYPE_{symbols[i]}"))
        for axis, f in zip(AXES, frac[i]):
            b = int(np.floor(f * n_bins)) % n_bins
            ids.append(vocab.encode_token(f"{axis}_{b}"))

    if add_special:
        ids.append(vocab.encode_token("[EOS]"))
    return ids


def decode_tokens(token_ids: list[int], vocab: Vocabulary,
                  config: dict) -> Optional[Atoms]:
    """
    Decode token IDs back into a periodic Atoms object.
    Malformed atom groups are dropped; returns None if no cell or no atoms.
    """
    toks = [vocab.decode_id(i) for i in token_ids]
    toks = [t for t in toks if t not in ("[BOS]", "[PAD]", "[UNK]")]
    if "[EOS]" in toks:
        toks = toks[:toks.index("[EOS]")]

    comp = None
    if toks and toks[0].startswith("COMP_"):
        comp = toks.pop(0)[5:]
    if toks and toks[0] in QUALITY_TOKENS:
        toks.pop(0)

    if len(toks) < 3 or not all(t.startswith("L_") for t in toks[:3]):
        return None
    lengths = [config["cell_min"] + int(t[2:]) * config["cell_resolution"]
               for t in toks[:3]]
    toks = toks[3:]

    n_bins = config["n_bins"]
    symbols, frac = [], []
    for g in range(0, len(toks) - 3, 4):
        t, x, y, z = toks[g:g + 4]
        if not (t.startswith("TYPE_") and x.startswith("X_")
                and y.startswith("Y_") and z.startswith("Z_")):
            continue
        symbols.append(t[5:])
        frac.append([(int(c.split("_")[1]) + 0.5) / n_bins for c in (x, y, z)])

    if not symbols:
        return None
    atoms = Atoms(symbols=symbols, scaled_positions=frac,
                  cell=lengths, pbc=True)
    if comp:
        atoms.info["composition"] = comp
    return atoms


# ============================================================
# MAIN TOKENIZER CLASS
# ============================================================

class AtomicTokenizer:
    def __init__(self, config: dict = None):
        self.config = config or TOKENIZER_CONFIG
        self.vocab = Vocabulary()

    def build_vocab(self):
        self.vocab.build(self.config)
        return self

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self.vocab.encode_token("[PAD]")

    @property
    def bos_id(self) -> int:
        return self.vocab.encode_token("[BOS]")

    @property
    def eos_id(self) -> int:
        return self.vocab.encode_token("[EOS]")

    def cell_token(self, length: float) -> int:
        k = int(round((length - self.config["cell_min"])
                      / self.config["cell_resolution"]))
        k = min(max(k, 0), _n_cell_bins(self.config) - 1)
        return self.vocab.encode_token(f"L_{k}")

    def encode(self, atoms: Atoms, add_special: bool = True) -> list[int]:
        return encode_atoms(atoms, self.vocab, self.config, add_special)

    def decode(self, token_ids: list[int]) -> Optional[Atoms]:
        return decode_tokens(token_ids, self.vocab, self.config)

    def encode_dataset(self, extxyz_file: str, save_to: str = None,
                       max_structures: int = None,
                       clean_only: bool = False) -> list[list[int]]:
        """
        Encode a dataset. Every structure gets a Q_clean / Q_defect token from
        the coordination-defect check, so defective frames are still useful
        training data. clean_only=True drops defective frames instead.
        """
        from structure_defects import find_defects

        print(f"\nEncoding dataset: {extxyz_file}")
        structures = read(extxyz_file, index=":")
        if max_structures:
            structures = structures[:max_structures]
        print(f"  Structures loaded: {len(structures)}")

        sequences, lengths = [], []
        n_clean, q_clean = 0, self.vocab.encode_token("Q_clean")
        skipped = {"elements": 0, "defects": 0, "error": 0}
        for atoms in tqdm(structures, desc="Encoding"):
            if not set(atoms.get_chemical_symbols()) <= set(self.config["elements"]):
                skipped["elements"] += 1
                continue
            if clean_only and find_defects(atoms):
                skipped["defects"] += 1
                continue
            try:
                ids = self.encode(atoms)
            except ValueError:
                skipped["error"] += 1
                continue
            sequences.append(ids)
            lengths.append(len(ids))
            n_clean += ids[2] == q_clean

        print(f"\nEncoding complete:")
        print(f"  Encoded  : {len(sequences)} "
              f"({n_clean} Q_clean, {len(sequences) - n_clean} Q_defect)")
        print(f"  Skipped  : {skipped}")
        if lengths:
            print(f"  Seq len  : min={min(lengths)}, max={max(lengths)}, "
                  f"mean={np.mean(lengths):.1f}")
        if save_to:
            self._save_sequences(sequences, save_to)
        return sequences

    def _save_sequences(self, sequences: list[list[int]], path: str):
        with open(path, "w") as f:
            json.dump({"vocab_size": self.vocab_size,
                       "n_sequences": len(sequences),
                       "sequences": sequences}, f)
        print(f"  Sequences saved to: {path}")

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({"config": self.config, "vocab": self.vocab.to_dict()},
                      f, indent=2)
        print(f"Tokenizer saved to: {path}")

    @classmethod
    def load(cls, path: str) -> "AtomicTokenizer":
        with open(path) as f:
            data = json.load(f)
        if data["config"].get("format") != TOKENIZER_CONFIG["format"]:
            raise ValueError(f"{path} was made by an older tokenizer format; "
                             "re-run atomic_tokenizer.py")
        tok = cls(config=data["config"])
        tok.vocab = Vocabulary.from_dict(data["vocab"])
        print(f"Tokenizer loaded from: {path} (vocab {tok.vocab_size})")
        return tok

    def roundtrip_test(self, atoms: Atoms, verbose: bool = True) -> float:
        """Encode+decode; return max minimum-image position error (Å)."""
        from structure_defects import defect_summary

        rec = self.decode(self.encode(atoms))
        order = cluster_growing_order(atoms)
        d = rec.get_scaled_positions() - atoms.get_scaled_positions(wrap=True)[order]
        d -= np.round(d)
        err = np.linalg.norm(d @ np.asarray(rec.get_cell()), axis=1)
        if verbose:
            print(f"Roundtrip: {len(atoms)} atoms, max err {err.max():.3f} Å, "
                  f"defects before {dict(defect_summary(atoms))} "
                  f"after {dict(defect_summary(rec))}")
        return float(err.max())


# ============================================================
# DATASET CLASS
# ============================================================

class AtomicSequenceDataset:
    def __init__(self, sequences, max_length: int = 512, pad_id: int = 0):
        self.sequences = sequences
        self.max_length = max_length
        self.pad_id = pad_id

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx][:self.max_length + 1]
        padded = seq + [self.pad_id] * (self.max_length + 1 - len(seq))
        return padded[:-1], padded[1:]

    @classmethod
    def from_file(cls, sequences_file: str, **kwargs):
        with open(sequences_file) as f:
            return cls(json.load(f)["sequences"], **kwargs)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Tokenize structures for LLM training")
    p.add_argument("--dataset", default="data/dataset.extxyz")
    p.add_argument("--out_tokenizer", default="data/tokenizer.json")
    p.add_argument("--out_sequences", default="data/sequences.json")
    p.add_argument("--max_structures", type=int, default=None)
    p.add_argument("--clean_only", action="store_true",
                   help="Drop defective structures instead of labelling them Q_defect")
    p.add_argument("--test_roundtrip", action="store_true",
                   help="Roundtrip-test the first 5 structures, then exit")
    args = p.parse_args()

    tokenizer = AtomicTokenizer().build_vocab()

    if args.test_roundtrip:
        for a in read(args.dataset, index=":5"):
            tokenizer.roundtrip_test(a)
        raise SystemExit

    sequences = tokenizer.encode_dataset(
        args.dataset, save_to=args.out_sequences,
        max_structures=args.max_structures,
        clean_only=args.clean_only)
    tokenizer.save(args.out_tokenizer)
    print(f"\nNext: python train_llm.py --sequences {args.out_sequences} "
          f"--tokenizer {args.out_tokenizer}")
