# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project trains a GPT-style autoregressive LLM to generate amorphous atomic structures (HfO2, SiO2, HfSiO4). The pipeline has three stages:

1. **Data generation** (`batch_melt_quench.py`, `batched_md.py`) — Generates amorphous structures via melt-quench molecular dynamics using the MACE-MP-0 universal potential; many runs are batched into one MACE call per MD step (`--batch_size`, default 32)
2. **Tokenization** (`atomic_tokenizer.py`) — Converts ASE Atoms objects into discrete token sequences using cluster-growing order and discretized relative displacements
3. **Training & generation** (`train.py`) — Trains a small GPT (AtomicGPT) on tokenized sequences and generates new structures

## Common Commands

```bash
# Step 1: Generate training data (slow — runs MD simulations)
python batch_melt_quench.py                        # full dataset
python batch_melt_quench.py --n_runs 5 --device cuda  # quick test on GPU
python batch_melt_quench.py --validate             # validate existing dataset

# Step 2: Tokenize structures
python atomic_tokenizer.py --dataset data/dataset.extxyz \
    --out_tokenizer data/tokenizer.json \
    --out_sequences data/sequences.json
python atomic_tokenizer.py --test_roundtrip --dataset data/dataset.extxyz  # roundtrip quality check

# Step 3: Train the model
python train.py --sequences data/sequences.json --tokenizer data/tokenizer.json
python train.py --smoke_test                  # quick sanity check (tiny model, synthetic data)
python train.py --n_layers 12 --d_model 512 --n_heads 8 --device cuda  # larger model

# Generate structures from a trained checkpoint
python train.py --generate --checkpoint checkpoints/checkpoint_best_*.pt \
    --tokenizer data/tokenizer.json --composition SiO2
```

## Architecture Details

### Tokenization scheme (atomic_tokenizer.py)
Each atom encodes as exactly **4 tokens**: `[TYPE, X, Y, Z]`. Positions are relative displacements from the nearest already-placed neighbor (first atom is relative to cell center). Atoms are ordered by **cluster-growing**: start at centroid, greedily add the nearest unplaced atom. This ordering is deterministic and matches the autoregressive generation strategy.

Vocabulary: 4 special tokens (`[PAD]=0, [BOS]=1, [EOS]=2, [UNK]=3`) + element type tokens + 3×N_bins coordinate bin tokens. Default config: 64 bins per axis, ±6.0 Å max displacement, giving ~0.19 Å resolution.

### Model (train.py)
AtomicGPT is a standard pre-norm GPT with causal self-attention, fused QKV projections, GELU activations, and weight-tied input/output embeddings. Default: 6 layers, 256 dim, 8 heads (~2M params). Uses AdamW with cosine-warmup LR schedule.

### Data pipeline
`batch_melt_quench.py` outputs to `data/` — individual `.extxyz` files per run in `data/structures/{composition}/`, combined dataset at `data/dataset.extxyz`, and `data/metadata.json`. The melt-quench protocol: minimize → melt (5000K) → anneal (2000K) → linear quench → equilibrate (300K), collecting snapshots throughout.

## Dependencies

- **Data generation**: `ase`, `mace-torch`, `numpy`, `tqdm`
- **Tokenization**: `ase`, `numpy`, `tqdm`
- **Training**: `torch`, `numpy`, `tqdm`, `matplotlib` (optional, for plotting)

## Key Configuration

Tokenizer config is in `TOKENIZER_CONFIG` dict at top of `atomic_tokenizer.py`. Model hyperparameters are in `GPTConfig` dataclass in `train.py`. Melt-quench parameters are in `CONFIG` dict in `batch_melt_quench.py`. Supported elements: Si, O, Hf.
