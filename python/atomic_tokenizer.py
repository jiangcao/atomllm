"""
Atomic Sequence Tokenizer for Amorphous Structure LLM
======================================================
Converts amorphous structures (ASE Atoms objects / extxyz files)
into token sequences suitable for training a GPT-style LLM.

Strategy:
    - First atom: absolute position (anchor)
    - All subsequent atoms: relative to nearest already-placed neighbor
    - Cluster-growing order: each new atom is nearest to existing cluster
    - Positions encoded as discretized bins (configurable resolution)

Token vocabulary:
    Special    : [PAD], [BOS], [EOS], [UNK]
    Atom types : TYPE_Si, TYPE_O, TYPE_Hf
    Coordinates: X_0..X_N, Y_0..Y_N, Z_0..Z_N  (relative, discretized)
    Distance   : DIST_0..DIST_N                  (to nearest neighbor)

Each atom is encoded as 4 tokens:
    [TYPE, X, Y, Z]  (relative displacement from nearest neighbor)

Requirements:
    pip install ase numpy tqdm

Usage:
    from atomic_tokenizer import AtomicTokenizer
    tokenizer = AtomicTokenizer()
    tokenizer.build_vocab()

    sequences = tokenizer.encode_dataset("data/dataset.extxyz")
    tokenizer.save("tokenizer.json")

    # Later:
    tokenizer2 = AtomicTokenizer.load("tokenizer.json")
    tokens = tokenizer2.encode_atoms(atoms)
    atoms  = tokenizer2.decode_tokens(tokens)
"""

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ase import Atoms
from ase.io import read, write
from typing import Optional


# ============================================================
# CONFIGURATION
# ============================================================

TOKENIZER_CONFIG = {
    # Coordinate discretization
    # Relative displacements are in range [-max_disp, +max_disp] Angstroms
    "max_displacement": 6.0,    # Angstroms — max relative displacement per axis
    "n_bins": 64,               # bins per axis (resolution = 2*max_disp / n_bins)
                                # 64 bins -> 0.19 Å resolution (good enough)

    # Supported elements
    "elements": ["Si", "O", "Hf"],

    # Sequence
    "max_atoms": 128,           # maximum atoms per structure
}

# Derived resolution
_RES = 2 * TOKENIZER_CONFIG["max_displacement"] / TOKENIZER_CONFIG["n_bins"]
print(f"Coordinate resolution: {_RES:.3f} Å per bin")


# ============================================================
# VOCABULARY
# ============================================================

class Vocabulary:
    """
    Maps token strings <-> integer IDs.

    Special tokens:
        [PAD] = 0   padding
        [BOS] = 1   beginning of sequence
        [EOS] = 2   end of sequence
        [UNK] = 3   unknown token
    """

    SPECIAL_TOKENS = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]

    def __init__(self):
        self.token2id: dict[str, int] = {}
        self.id2token: dict[int, str] = {}
        self._add_special_tokens()

    def _add_special_tokens(self):
        for tok in self.SPECIAL_TOKENS:
            self._add(tok)

    def _add(self, token: str) -> int:
        if token not in self.token2id:
            idx = len(self.token2id)
            self.token2id[token] = idx
            self.id2token[idx] = token
        return self.token2id[token]

    def build(self, elements: list[str], n_bins: int):
        """Build full vocabulary from elements and bin count."""
        # Atom type tokens
        for elem in elements:
            self._add(f"TYPE_{elem}")

        # Coordinate tokens: X, Y, Z each get n_bins bins
        for axis in ["X", "Y", "Z"]:
            for b in range(n_bins):
                self._add(f"{axis}_{b}")

        print(f"Vocabulary size: {len(self.token2id)} tokens")
        print(f"  Special  : {len(self.SPECIAL_TOKENS)}")
        print(f"  Types    : {len(elements)}")
        print(f"  X bins   : {n_bins}")
        print(f"  Y bins   : {n_bins}")
        print(f"  Z bins   : {n_bins}")

    def __len__(self):
        return len(self.token2id)

    def encode_token(self, token: str) -> int:
        return self.token2id.get(token, self.token2id["[UNK]"])

    def decode_id(self, idx: int) -> str:
        return self.id2token.get(idx, "[UNK]")

    def to_dict(self) -> dict:
        return {"token2id": self.token2id, "id2token": {
            str(k): v for k, v in self.id2token.items()
        }}

    @classmethod
    def from_dict(cls, d: dict) -> "Vocabulary":
        vocab = cls.__new__(cls)
        vocab.token2id = d["token2id"]
        vocab.id2token = {int(k): v for k, v in d["id2token"].items()}
        return vocab


# ============================================================
# CLUSTER-GROWING SEQUENCE ORDERING
# ============================================================

def cluster_growing_order(atoms: Atoms) -> list[int]:
    """
    Reorder atoms in cluster-growing sequence:
        1. Start with atom closest to centroid
        2. At each step, add atom nearest to existing cluster

    This matches the autoregressive generation strategy exactly.

    Returns:
        List of atom indices in cluster-growing order
    """
    positions = atoms.get_positions()
    n = len(positions)

    if n == 0:
        return []
    if n == 1:
        return [0]

    # Start with atom nearest to centroid
    centroid = positions.mean(axis=0)
    dists_to_centroid = np.linalg.norm(positions - centroid, axis=1)
    start_idx = int(np.argmin(dists_to_centroid))

    ordered = [start_idx]
    remaining = set(range(n)) - {start_idx}

    while remaining:
        # Positions of already-placed atoms
        placed_pos = positions[ordered]  # shape (k, 3)

        # For each remaining atom, find distance to nearest placed atom
        best_idx = None
        best_dist = np.inf

        remaining_list = list(remaining)
        remaining_pos = positions[remaining_list]  # shape (m, 3)

        # Vectorized: distance from each remaining to each placed
        # Shape: (m, k)
        diffs = remaining_pos[:, None, :] - placed_pos[None, :, :]  # (m, k, 3)
        dists = np.linalg.norm(diffs, axis=2)                        # (m, k)
        min_dists = dists.min(axis=1)                                 # (m,)

        nearest = int(np.argmin(min_dists))
        best_idx = remaining_list[nearest]

        ordered.append(best_idx)
        remaining.remove(best_idx)

    return ordered


# ============================================================
# ENCODING (atoms -> tokens)
# ============================================================

def discretize_displacement(disp: float, max_disp: float, n_bins: int) -> int:
    """
    Map a continuous displacement [-max_disp, +max_disp] to bin index [0, n_bins-1].
    Clips out-of-range values to boundary bins.
    """
    # Shift to [0, 2*max_disp], then scale to [0, n_bins]
    normalized = (disp + max_disp) / (2 * max_disp)
    bin_idx = int(np.floor(normalized * n_bins))
    return np.clip(bin_idx, 0, n_bins - 1)


def undiscretize_displacement(bin_idx: int, max_disp: float, n_bins: int) -> float:
    """
    Map bin index back to displacement (bin center value).
    """
    normalized = (bin_idx + 0.5) / n_bins
    return normalized * 2 * max_disp - max_disp


def encode_atoms(atoms: Atoms, vocab: Vocabulary,
                 config: dict, add_special: bool = True) -> list[int]:
    """
    Encode an ASE Atoms object into a list of token IDs.

    Encoding per atom (4 tokens):
        TOKEN 1: TYPE_{element}
        TOKEN 2: X_{bin}   <- relative displacement from nearest neighbor
        TOKEN 3: Y_{bin}
        TOKEN 4: Z_{bin}

    For the FIRST atom (anchor), displacement is relative to cell center.

    Args:
        atoms       : ASE Atoms
        vocab       : Vocabulary object
        config      : tokenizer config dict
        add_special : whether to prepend [BOS] and append [EOS]

    Returns:
        List of integer token IDs
    """
    max_disp = config["max_displacement"]
    n_bins   = config["n_bins"]
    max_atoms = config["max_atoms"]

    # Get cluster-growing order
    order = cluster_growing_order(atoms)
    order = order[:max_atoms]  # truncate if too large

    positions = atoms.get_positions()
    symbols   = atoms.get_chemical_symbols()
    cell_center = atoms.get_cell().sum(axis=0) / 2.0

    token_ids = []

    if add_special:
        token_ids.append(vocab.encode_token("[BOS]"))

    for seq_idx, atom_idx in enumerate(order):
        elem = symbols[atom_idx]
        pos  = positions[atom_idx]

        # Reference point: cell center for first atom, nearest placed atom otherwise
        if seq_idx == 0:
            ref = cell_center
        else:
            placed_positions = positions[order[:seq_idx]]
            dists = np.linalg.norm(placed_positions - pos, axis=1)
            ref = placed_positions[np.argmin(dists)]

        displacement = pos - ref  # (dx, dy, dz)

        # Encode type
        type_token = f"TYPE_{elem}"
        token_ids.append(vocab.encode_token(type_token))

        # Encode displacement per axis
        for axis, (axis_name, d) in enumerate(
            zip(["X", "Y", "Z"], displacement)
        ):
            bin_idx = discretize_displacement(d, max_disp, n_bins)
            token_ids.append(vocab.encode_token(f"{axis_name}_{bin_idx}"))

    if add_special:
        token_ids.append(vocab.encode_token("[EOS]"))

    return token_ids


# ============================================================
# DECODING (tokens -> atoms)
# ============================================================

def decode_tokens(token_ids: list[int], vocab: Vocabulary,
                  config: dict) -> Optional[Atoms]:
    """
    Decode a list of token IDs back into an ASE Atoms object.

    Reconstructs positions by accumulating relative displacements
    along the cluster-growing chain.

    Args:
        token_ids : list of integer token IDs
        vocab     : Vocabulary object
        config    : tokenizer config dict

    Returns:
        ASE Atoms object, or None if decoding fails
    """
    max_disp = config["max_displacement"]
    n_bins   = config["n_bins"]

    # Convert IDs to token strings
    tokens = [vocab.decode_id(i) for i in token_ids]

    # Strip special tokens
    tokens = [t for t in tokens if t not in ["[BOS]", "[EOS]", "[PAD]", "[UNK]"]]

    if len(tokens) % 4 != 0:
        # Truncate to last complete atom
        tokens = tokens[:len(tokens) - (len(tokens) % 4)]

    if len(tokens) == 0:
        return None

    symbols   = []
    positions = []

    # Default box size (will be set after decoding)
    box_length = 15.0

    cell_center = np.array([box_length / 2] * 3)

    for i in range(0, len(tokens), 4):
        chunk = tokens[i:i+4]
        if len(chunk) < 4:
            break

        type_tok, x_tok, y_tok, z_tok = chunk

        # Parse element
        if not type_tok.startswith("TYPE_"):
            continue
        elem = type_tok[5:]
        symbols.append(elem)

        # Parse displacement bins
        try:
            x_bin = int(x_tok.split("_")[1])
            y_bin = int(y_tok.split("_")[1])
            z_bin = int(z_tok.split("_")[1])
        except (IndexError, ValueError):
            symbols.pop()
            continue

        dx = undiscretize_displacement(x_bin, max_disp, n_bins)
        dy = undiscretize_displacement(y_bin, max_disp, n_bins)
        dz = undiscretize_displacement(z_bin, max_disp, n_bins)
        displacement = np.array([dx, dy, dz])

        # Reference: cell center for first atom, nearest placed for rest
        if len(positions) == 0:
            ref = cell_center
        else:
            placed = np.array(positions)
            ref = placed[np.argmin(np.linalg.norm(
                placed - (cell_center + displacement), axis=1
            ))]

        pos = ref + displacement
        positions.append(pos.tolist())

    if len(symbols) == 0:
        return None

    # Determine appropriate box size
    pos_array = np.array(positions)
    span = pos_array.max(axis=0) - pos_array.min(axis=0)
    box_length = max(span.max() + 6.0, 10.0)  # at least 10 Å, with 3Å padding

    # Center structure in box
    pos_array -= pos_array.min(axis=0)
    pos_array += 3.0  # 3 Å padding from edges

    atoms = Atoms(
        symbols=symbols,
        positions=pos_array,
        cell=[box_length, box_length, box_length],
        pbc=True
    )

    return atoms


# ============================================================
# MAIN TOKENIZER CLASS
# ============================================================

class AtomicTokenizer:
    """
    Full tokenizer: encodes/decodes amorphous structures to/from token sequences.

    Usage:
        tokenizer = AtomicTokenizer()
        tokenizer.build_vocab()

        # Encode a dataset
        sequences = tokenizer.encode_dataset("data/dataset.extxyz")

        # Save / load
        tokenizer.save("tokenizer.json")
        tokenizer = AtomicTokenizer.load("tokenizer.json")

        # Single structure
        tokens = tokenizer.encode(atoms)
        atoms  = tokenizer.decode(tokens)
    """

    def __init__(self, config: dict = None):
        self.config = config or TOKENIZER_CONFIG
        self.vocab  = Vocabulary()

    def build_vocab(self):
        """Build vocabulary from config."""
        self.vocab.build(
            elements=self.config["elements"],
            n_bins=self.config["n_bins"]
        )
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

    def encode(self, atoms: Atoms, add_special: bool = True) -> list[int]:
        """Encode a single ASE Atoms object to token IDs."""
        return encode_atoms(atoms, self.vocab, self.config, add_special)

    def decode(self, token_ids: list[int]) -> Optional[Atoms]:
        """Decode token IDs back to ASE Atoms object."""
        return decode_tokens(token_ids, self.vocab, self.config)

    def encode_dataset(self, extxyz_file: str,
                       save_to: str = None,
                       max_structures: int = None) -> list[list[int]]:
        """
        Encode a full dataset from an extxyz file.

        Args:
            extxyz_file   : path to dataset.extxyz
            save_to       : optional .json path to save encoded sequences
            max_structures: limit number of structures (for testing)

        Returns:
            List of token ID sequences
        """
        print(f"\nEncoding dataset: {extxyz_file}")
        structures = read(extxyz_file, index=":")

        if max_structures:
            structures = structures[:max_structures]

        print(f"  Structures loaded: {len(structures)}")

        sequences = []
        skipped   = 0
        lengths   = []

        for atoms in tqdm(structures, desc="Encoding"):
            try:
                # Skip unsupported elements
                syms = set(atoms.get_chemical_symbols())
                if not syms.issubset(set(self.config["elements"])):
                    skipped += 1
                    continue

                token_ids = self.encode(atoms)
                sequences.append(token_ids)
                lengths.append(len(token_ids))

            except Exception as e:
                skipped += 1
                continue

        print(f"\nEncoding complete:")
        print(f"  Encoded  : {len(sequences)}")
        print(f"  Skipped  : {skipped}")
        if lengths:
            print(f"  Seq len  : min={min(lengths)}, "
                  f"max={max(lengths)}, "
                  f"mean={np.mean(lengths):.1f}")
            print(f"  Vocab size: {self.vocab_size}")

        if save_to:
            self._save_sequences(sequences, save_to)

        return sequences

    def _save_sequences(self, sequences: list[list[int]], path: str):
        """Save encoded sequences to JSON."""
        out = {
            "vocab_size": self.vocab_size,
            "n_sequences": len(sequences),
            "sequences": sequences,
        }
        with open(path, "w") as f:
            json.dump(out, f)
        print(f"  Sequences saved to: {path}")

    def save(self, path: str):
        """Save tokenizer (vocab + config) to JSON."""
        data = {
            "config": self.config,
            "vocab": self.vocab.to_dict(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Tokenizer saved to: {path}")

    @classmethod
    def load(cls, path: str) -> "AtomicTokenizer":
        """Load tokenizer from JSON."""
        with open(path) as f:
            data = json.load(f)
        tokenizer = cls(config=data["config"])
        tokenizer.vocab = Vocabulary.from_dict(data["vocab"])
        print(f"Tokenizer loaded from: {path}")
        print(f"  Vocab size: {tokenizer.vocab_size}")
        return tokenizer

    def roundtrip_test(self, atoms: Atoms, verbose: bool = True) -> float:
        """
        Encode then decode a structure and measure position error.
        Useful for checking tokenization quality.

        Returns:
            Mean absolute error in Angstroms
        """
        tokens = self.encode(atoms)
        reconstructed = self.decode(tokens)

        if reconstructed is None:
            print("Roundtrip failed: could not decode")
            return float("inf")

        # Match atoms by order (cluster-growing order is deterministic)
        order_orig = cluster_growing_order(atoms)
        pos_orig = atoms.get_positions()[order_orig[:len(reconstructed)]]
        pos_rec  = reconstructed.get_positions()

        n = min(len(pos_orig), len(pos_rec))
        mae = np.mean(np.linalg.norm(pos_orig[:n] - pos_rec[:n], axis=1))

        if verbose:
            print(f"\nRoundtrip test:")
            print(f"  N atoms original    : {len(atoms)}")
            print(f"  N atoms reconstructed: {len(reconstructed)}")
            print(f"  Token sequence length: {len(tokens)}")
            print(f"  Position MAE        : {mae:.4f} Å")
            print(f"  (Expected ~{_RES/2:.3f} Å = half bin width)")

        return mae


# ============================================================
# DATASET CLASS FOR LLM TRAINING
# ============================================================

class AtomicSequenceDataset:
    """
    PyTorch-compatible dataset of tokenized atomic sequences.
    Used directly for LLM training in the next step.

    Usage:
        dataset = AtomicSequenceDataset(sequences, max_length=512)
        loader  = DataLoader(dataset, batch_size=32, shuffle=True)
    """

    def __init__(self, sequences: list[list[int]],
                 max_length: int = 512,
                 pad_id: int = 0):
        self.sequences  = sequences
        self.max_length = max_length
        self.pad_id     = pad_id

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Returns (input_ids, target_ids) for next-token prediction.
        Input  = sequence[:-1]
        Target = sequence[1:]   (shifted by 1)
        """
        seq = self.sequences[idx]

        # Truncate
        seq = seq[:self.max_length + 1]

        # Pad to max_length + 1
        padded = seq + [self.pad_id] * (self.max_length + 1 - len(seq))

        input_ids  = padded[:-1]
        target_ids = padded[1:]

        # Return as lists (convert to tensors in collate_fn or training loop)
        return input_ids, target_ids

    @classmethod
    def from_file(cls, sequences_file: str, **kwargs) -> "AtomicSequenceDataset":
        """Load from saved sequences JSON."""
        with open(sequences_file) as f:
            data = json.load(f)
        return cls(data["sequences"], **kwargs)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Tokenize amorphous structure dataset for LLM training"
    )
    parser.add_argument("--dataset",   default="data/dataset.extxyz",
                        help="Input extxyz dataset")
    parser.add_argument("--out_tokenizer", default="data/tokenizer.json",
                        help="Output tokenizer file")
    parser.add_argument("--out_sequences", default="data/sequences.json",
                        help="Output encoded sequences file")
    parser.add_argument("--max_structures", type=int, default=None,
                        help="Limit structures (for testing)")
    parser.add_argument("--test_roundtrip", action="store_true",
                        help="Run roundtrip test on first structure")
    args = parser.parse_args()

    # Build tokenizer
    tokenizer = AtomicTokenizer()
    tokenizer.build_vocab()

    # Roundtrip test on a single structure
    if args.test_roundtrip:
        print("\nRunning roundtrip test on synthetic structure...")
        from ase.build import bulk
        test_atoms = read(args.dataset, index="0")
        tokenizer.roundtrip_test(test_atoms)

    # Encode full dataset
    sequences = tokenizer.encode_dataset(
        extxyz_file=args.dataset,
        save_to=args.out_sequences,
        max_structures=args.max_structures
    )

    # Save tokenizer
    tokenizer.save(args.out_tokenizer)

    print("\nReady for LLM training!")
    print(f"  Vocab size : {tokenizer.vocab_size}")
    print(f"  Sequences  : {len(sequences)}")
    print(f"  Next step  : python train_llm.py --sequences {args.out_sequences} "
          f"--tokenizer {args.out_tokenizer}")
