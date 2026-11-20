"""
Atomic Structure LLM - GPT Training Script
===========================================
Trains a GPT-style autoregressive transformer on tokenized atomic sequences
produced by atomic_tokenizer.py. The model learns to predict the next atom
(type + position) given all previously placed atoms in the cluster-growing order.

Architecture:
    - Small GPT (6-12 layers) — atomic sequences are simpler than language
    - Causal self-attention (each token attends only to past tokens)
    - Learned positional embeddings
    - Tied input/output embeddings (weight tying)

Training objective:
    Next-token prediction (cross-entropy loss)
    Input:  [BOS, TYPE_Si, X_37, Y_32, Z_33, TYPE_O, ...]
    Target: [TYPE_Si, X_37, Y_32, Z_33, TYPE_O, ..., EOS]

Requirements:
    pip install torch numpy tqdm matplotlib

Usage:
    # Full training
    python train.py \
        --sequences data/sequences.json \
        --tokenizer data/tokenizer.json \
        --out_dir    checkpoints/

    # Quick smoke test (tiny model, few steps)
    python train.py --smoke_test
"""

import os
import json
import math
import time
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm


# ============================================================
# MODEL CONFIGURATION
# ============================================================

@dataclass
class GPTConfig:
    # Vocabulary
    vocab_size:   int = 2251      # overwritten from the sequences file
    pad_token_id: int = 0

    # Architecture
    max_seq_len:  int = 520       # 6 header + 128 atoms × 4 + EOS
    n_layers:     int = 6         # 6 layers for small model, 12 for larger
    n_heads:      int = 8
    d_model:      int = 256       # embedding dimension
    d_ff:         int = 1024      # feedforward hidden dim (4 × d_model)
    dropout:      float = 0.1

    # Training
    batch_size:   int = 32
    lr:           float = 3e-4
    weight_decay: float = 0.1
    max_epochs:   int = 50
    warmup_steps: int = 200
    grad_clip:    float = 1.0

    # Logging
    log_every:    int = 50        # steps
    eval_every:   int = 500       # steps
    save_every:   int = 1000      # steps


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.
    Each token can only attend to itself and previous tokens — 
    essential for autoregressive generation.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0

        self.n_heads = config.n_heads
        self.d_head  = config.d_model // config.n_heads
        self.d_model = config.d_model

        # Fused QKV projection
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj  = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout   = nn.Dropout(config.dropout)

        # Causal mask (upper triangular = future tokens masked out)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
            .view(1, 1, config.max_seq_len, config.max_seq_len)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Compute Q, K, V
        qkv = self.qkv_proj(x)                          # (B, T, 3C)
        q, k, v = qkv.split(self.d_model, dim=2)        # each (B, T, C)

        # Reshape to (B, n_heads, T, d_head)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product attention with causal mask
        scale = 1.0 / math.sqrt(self.d_head)
        attn  = (q @ k.transpose(-2, -1)) * scale       # (B, H, T, T)
        attn  = attn.masked_fill(
            self.causal_mask[:, :, :T, :T] == 0, float("-inf")
        )
        attn  = F.softmax(attn, dim=-1)
        attn  = self.dropout(attn)

        out = attn @ v                                   # (B, H, T, d_head)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

    def forward_cached(self, x, cache, rows, pos, n_keys):
        """Incremental attention with a key-value cache (inference only).
        x (b, N, C) new tokens of cache rows `rows` (b,); pos (b, N) their
        positions (padding -> the dump slot, last index of the cache);
        cache (K, V) each (B_all, H, S, d_head); keys 0..n_keys-1 are read."""
        b, N, C = x.shape
        q, k, v = self.qkv_proj(x).split(self.d_model, dim=2)
        K, V = cache
        r = rows[:, None]
        K[r, :, pos] = k.view(b, N, self.n_heads, self.d_head)
        V[r, :, pos] = v.view(b, N, self.n_heads, self.d_head)
        q = q.view(b, N, self.n_heads, self.d_head).transpose(1, 2)
        kk, vv = K[rows, :, :n_keys], V[rows, :, :n_keys]
        qpos = pos.clamp(max=n_keys - 1)
        allow = torch.arange(n_keys, device=x.device)[None, None, :] <= qpos[:, :, None]
        out = F.scaled_dot_product_attention(q, kk, vv, attn_mask=allow[:, None])
        return self.out_proj(out.transpose(1, 2).reshape(b, N, C))


class TransformerBlock(nn.Module):
    """Single transformer block: attention + feedforward with residuals."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1  = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2  = nn.LayerNorm(config.d_model)
        self.ff   = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))   # pre-norm + residual
        x = x + self.ff(self.ln2(x))
        return x

    def forward_cached(self, x, cache, rows, pos, n_keys):
        x = x + self.attn.forward_cached(self.ln1(x), cache, rows, pos, n_keys)
        return x + self.ff(self.ln2(x))


class AtomicGPT(nn.Module):
    """
    GPT-style autoregressive model for atomic sequence generation.

    Input : token IDs (B, T)
    Output: logits over vocabulary (B, T, vocab_size)

    At generation time, sample next token from logits[:, -1, :]
    to get the next atom type or coordinate bin.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model,
                                      padding_idx=config.pad_token_id)
        self.pos_emb   = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop      = nn.Dropout(config.dropout)

        self.blocks    = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        self.ln_final  = nn.LayerNorm(config.d_model)

        # Output head — weight tied to token embedding (standard GPT trick)
        self.lm_head   = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # weight tying

        # Initialize weights
        self.apply(self._init_weights)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"AtomicGPT: {n_params/1e6:.2f}M parameters")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        """final (layer-normed) hidden states (B, T, d_model)"""
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, \
            f"Sequence length {T} exceeds max {self.config.max_seq_len}"
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)  # (1, T)
        x = self.drop(self.token_emb(input_ids) + self.pos_emb(positions))
        for block in self.blocks:
            x = block(x)
        return self.ln_final(x)

    def forward(self, input_ids: torch.Tensor,
                targets: Optional[torch.Tensor] = None):
        """
        Args:
            input_ids : (B, T) token IDs
            targets   : (B, T) target IDs for loss computation (optional)

        Returns:
            logits : (B, T, vocab_size)
            loss   : scalar cross-entropy loss (if targets provided)
        """
        x      = self.hidden(input_ids)
        logits = self.lm_head(x)   # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Flatten for cross-entropy; ignore padding
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=self.config.pad_token_id,
            )

        return logits, loss

    # ---- key-value cache (inference) ----
    def alloc_cache(self, n_rows: int, device=None):
        """Per layer (K, V), each (n_rows, H, max_seq_len + 1, d_head); the
        extra last slot is a dump for padding tokens."""
        c, p = self.config, next(self.parameters())
        shape = (n_rows, c.n_heads, c.max_seq_len + 1, c.d_model // c.n_heads)
        return [(torch.zeros(shape, dtype=p.dtype, device=device or p.device),
                 torch.zeros(shape, dtype=p.dtype, device=device or p.device))
                for _ in self.blocks]

    @torch.no_grad()
    def forward_cached(self, new_ids, rows, start, n_new, cache):
        """Feed the new tokens of some cache rows.
        new_ids (b, N) right-padded new tokens, rows (b,) cache rows,
        start (b,) number of tokens already cached in each row (all earlier
        positions must hold that row's prefix), n_new (b,) real new tokens.
        Returns the logits at each row's last new token, (b, vocab)."""
        b, N = new_ids.shape
        S = self.config.max_seq_len
        j = torch.arange(N, device=new_ids.device)[None]
        pos = start[:, None] + j
        real = j < n_new[:, None]
        assert int(pos[real].max()) < S, "sequence exceeds max_seq_len"
        pos = torch.where(real, pos, torch.full_like(pos, S))   # dump slot
        x = self.token_emb(new_ids) + self.pos_emb(pos.clamp(max=S - 1))
        n_keys = int((start + n_new).max())
        for blk, kv in zip(self.blocks, cache):
            x = blk.forward_cached(x, kv, rows, pos, n_keys)
        x = self.ln_final(x[torch.arange(b, device=x.device), n_new - 1])
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor,
                 max_new_tokens: int = 400,
                 temperature: float = 1.0,
                 top_k: int = 50,
                 eos_token_id: int = 2) -> torch.Tensor:
        """
        Autoregressive generation — grows the atomic structure token by token.

        Args:
            prompt_ids    : (1, T) seed token IDs (e.g. just [BOS] or a composition prompt)
            max_new_tokens: maximum tokens to generate (~atoms × 4)
            temperature   : sampling temperature (lower = more conservative)
            top_k         : top-k sampling (keeps k most likely tokens)
            eos_token_id  : stop generation at this token

        Returns:
            Full token sequence (1, T + new_tokens)
        """
        self.eval()
        ids = prompt_ids.clone()

        for _ in range(max_new_tokens):
            # Truncate context if too long
            ids_cond = ids[:, -self.config.max_seq_len:]

            logits, _ = self(ids_cond)
            logits = logits[:, -1, :] / temperature  # (1, vocab_size)

            # Top-k filtering
            if top_k is not None:
                top_vals, _ = torch.topk(logits, top_k)
                logits[logits < top_vals[:, -1:]] = float("-inf")

            probs     = F.softmax(logits, dim=-1)
            next_id   = torch.multinomial(probs, num_samples=1)  # (1, 1)
            ids       = torch.cat([ids, next_id], dim=1)

            if next_id.item() == eos_token_id:
                break

        return ids


# ============================================================
# DATASET
# ============================================================

class AtomicSequenceDataset(torch.utils.data.Dataset):
    """Token sequence dataset for next-token prediction."""

    def __init__(self, sequences: list, max_length: int = 512, pad_id: int = 0):
        self.sequences  = sequences
        self.max_length = max_length
        self.pad_id     = pad_id

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx][:self.max_length + 1]
        padded = seq + [self.pad_id] * (self.max_length + 1 - len(seq))
        x = torch.tensor(padded[:-1], dtype=torch.long)
        y = torch.tensor(padded[1:],  dtype=torch.long)
        return x, y

    @classmethod
    def from_file(cls, path: str, **kwargs):
        with open(path) as f:
            data = json.load(f)
        return cls(data["sequences"], **kwargs)


class ULSequenceDataset(AtomicSequenceDataset):
    """Sequences + unlikelihood targets: returns (x, y, m) with m[t] = 1 when
    the target y[t] (= token at position t + 1) is a defect decision (see
    window_tokenizer_v2.encode_window(kinds=...))."""

    def __init__(self, sequences, ul_positions, **kwargs):
        super().__init__(sequences, **kwargs)
        self.ul_positions = ul_positions

    def __getitem__(self, idx):
        x, y = super().__getitem__(idx)
        m = torch.zeros(self.max_length, dtype=torch.float32)
        for p in self.ul_positions[idx]:
            if 1 <= p <= self.max_length:
                m[p - 1] = 1.0
        return x, y, m


def unlikelihood(model, x, y, m, q_pos: int, q_clean_id: int):
    """Unlikelihood of the marked defect decisions under the CLEAN label:
    the rows with targets are re-scored with the label slot set to Q_clean and
    -log(1 - p(target)) is averaged over the marked tokens. Returns
    (loss, mean p) or (None, None) when the batch has no targets."""
    rows = m.sum(1) > 0
    if not rows.any():
        return None, None
    xt = x[rows].clone()
    xt[:, q_pos] = q_clean_id
    logits, _ = model(xt)
    lp = F.log_softmax(logits.float(), dim=-1).gather(-1, y[rows].unsqueeze(-1)).squeeze(-1)
    p = lp.exp()
    mr = m[rows]
    ul = -torch.log1p(-p.clamp(max=1 - 1e-6))
    n = mr.sum()
    return (ul * mr).sum() / n, ((p * mr).sum() / n).detach()


@torch.no_grad()
def evaluate_ul(model, loader, device, q_pos, q_clean_id, max_batches: int = 50):
    """Validation: mean p of the defect decisions under the clean label."""
    model.eval()
    tot, n = 0.0, 0
    for b, (x, y, m) in enumerate(loader):
        if b >= max_batches:
            break
        x, y, m = x.to(device), y.to(device), m.to(device)
        _, p = unlikelihood(model, x, y, m, q_pos, q_clean_id)
        if p is not None:
            tot += p.item(); n += 1
    model.train()
    return tot / max(n, 1)


class SymmetryAugmenter:
    """
    Random symmetry copy of a tokenized periodic structure, done directly on
    the fractional coordinate bins (exact, no re-binning error):
      * translation : b -> (b + s) mod n_bins     (any s, per axis)
      * reflection  : b -> n_bins - 1 - b         (maps bin centres exactly)
      * permutation : swap axes among equal cell lengths
    Atoms are then re-ordered by cluster growing from the cell centre, as the
    tokenizer does, so augmented sequences look like real ones.
    Needs the vocab layout X_0..X_{n-1}, Y_0.., Z_0.. (contiguous ids).
    """

    def __init__(self, tokenizer_path: str, header_len: int = 6):
        with open(tokenizer_path) as f:
            tk = json.load(f)
        t2i = tk["vocab"]["token2id"] if "vocab" in tk else tk["token2id"]
        cfg = tk.get("config", {})
        self.n_bins = cfg.get("n_bins", 512)
        self.x0 = {a: t2i[f"{a}_0"] for a in "XYZ"}
        for a in "XYZ":
            assert t2i[f"{a}_{self.n_bins - 1}"] == self.x0[a] + self.n_bins - 1
        self.header_len = header_len
        self.eos = t2i["[EOS]"]
        self.l0 = t2i["L_0"]
        self.cell_min = cfg.get("cell_min", 5.0)
        self.cell_res = cfg.get("cell_resolution", 0.05)

    def __call__(self, seq: list, rng: np.random.Generator) -> list:
        h = self.header_len
        end = seq.index(self.eos) if self.eos in seq else len(seq)
        body = np.array(seq[h:end])
        if len(body) % 4 or len(body) == 0:
            return seq
        g = body.reshape(-1, 4)
        types = g[:, 0]
        bins = np.stack([g[:, 1 + k] - self.x0[a] for k, a in enumerate("XYZ")], 1)
        L_tok = np.array(seq[3:6])
        lengths = self.cell_min + (L_tok - self.l0) * self.cell_res

        perm = np.arange(3)
        for axes in ([0, 1, 2],) if len(set(L_tok)) == 1 else \
                ([[0, 1]] if L_tok[0] == L_tok[1] else []) + \
                ([[1, 2]] if L_tok[1] == L_tok[2] else []) + \
                ([[0, 2]] if L_tok[0] == L_tok[2] else []):
            axes = np.array(axes)
            perm[axes] = perm[rng.permutation(axes)]
            break
        bins = bins[:, perm]
        flip = rng.random(3) < 0.5
        bins = np.where(flip, self.n_bins - 1 - bins, bins)
        bins = (bins + rng.integers(0, self.n_bins, 3)) % self.n_bins

        frac = (bins + 0.5) / self.n_bins
        order = _cluster_order(frac, lengths)
        out = np.empty((len(order), 4), dtype=np.int64)
        out[:, 0] = types[order]
        for k, a in enumerate("XYZ"):
            out[:, 1 + k] = bins[order, k] + self.x0[a]
        return list(seq[:h]) + out.ravel().tolist() + list(seq[end:])


def _cluster_order(frac: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Cluster-growing order with minimum-image distances (as the tokenizer)."""
    n = len(frac)

    def dist(f):
        d = frac - f
        d -= np.round(d)
        return np.linalg.norm(d * lengths, axis=1)

    start = int(np.argmin(dist(np.full(3, 0.5))))
    order = np.empty(n, dtype=np.int64)
    order[0] = start
    placed = np.zeros(n, bool)
    placed[start] = True
    min_d = dist(frac[start])
    for k in range(1, n):
        nxt = int(np.argmin(np.where(placed, np.inf, min_d)))
        order[k] = nxt
        placed[nxt] = True
        min_d = np.minimum(min_d, dist(frac[nxt]))
    return order


class AugmentedSequenceDataset(AtomicSequenceDataset):
    """AtomicSequenceDataset that returns a fresh random symmetry copy each time."""

    def __init__(self, sequences, augmenter: SymmetryAugmenter, **kwargs):
        super().__init__(sequences, **kwargs)
        self.augmenter = augmenter

    def __getitem__(self, idx):
        rng = np.random.default_rng()
        seq = self.augmenter(self.sequences[idx], rng)
        seq = seq[:self.max_length + 1]
        padded = seq + [self.pad_id] * (self.max_length + 1 - len(seq))
        return (torch.tensor(padded[:-1], dtype=torch.long),
                torch.tensor(padded[1:], dtype=torch.long))


# ============================================================
# TRAINING UTILITIES
# ============================================================

class CosineWarmupScheduler:
    """Linear warmup then cosine decay learning rate schedule."""

    def __init__(self, optimizer, warmup_steps: int, max_steps: int, min_lr: float = 1e-5):
        self.optimizer    = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps    = max_steps
        self.min_lr       = min_lr
        self.base_lrs     = [g["lr"] for g in optimizer.param_groups]
        self.step_count   = 0

    def step(self):
        self.step_count += 1
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = self._compute_lr(base_lr)

    def _compute_lr(self, base_lr: float) -> float:
        s = self.step_count
        if s < self.warmup_steps:
            return base_lr * s / self.warmup_steps
        progress = (s - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        progress = min(progress, 1.0)
        return self.min_lr + 0.5 * (base_lr - self.min_lr) * (
            1 + math.cos(math.pi * progress)
        )


@torch.no_grad()
def evaluate(model: AtomicGPT, loader: DataLoader,
             device: str, max_batches: int = 50) -> float:
    """Compute mean validation loss."""
    model.eval()
    total_loss = 0.0
    n_batches  = 0

    for batch in loader:
        if n_batches >= max_batches:
            break
        x, y = batch[0].to(device), batch[1].to(device)
        _, loss = model(x, y)
        total_loss += loss.item()
        n_batches  += 1

    model.train()
    return total_loss / max(n_batches, 1)


def save_checkpoint(model: AtomicGPT, optimizer: torch.optim.Optimizer,
                    step: int, val_loss: float, out_dir: str, tag: str = ""):
    """Save model checkpoint."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fname = f"checkpoint_{tag}_step{step:06d}_loss{val_loss:.4f}.pt"
    path  = os.path.join(out_dir, fname)
    torch.save({
        "step":        step,
        "val_loss":    val_loss,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "config":      asdict(model.config),
    }, path)
    print(f"  Checkpoint saved: {path}")
    return path


def load_checkpoint(path: str, device: str = "cpu"):
    """Load model from checkpoint."""
    ckpt   = torch.load(path, map_location=device)
    config = GPTConfig(**ckpt["config"])
    model  = AtomicGPT(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded checkpoint: {path} (step={ckpt['step']}, val_loss={ckpt['val_loss']:.4f})")
    return model, ckpt


# ============================================================
# TRAINING LOOP
# ============================================================

def train(config: GPTConfig, sequences_file: str,
          out_dir: str = "checkpoints", device: str = "auto",
          augment_tokenizer: Optional[str] = None, num_workers: int = 0,
          init_checkpoint: Optional[str] = None, ul_weight: float = 0.0):
    """augment_tokenizer: tokenizer.json path -> random symmetry augmentation
    of the training split (validation stays un-augmented).
    ul_weight > 0: add ul_weight x unlikelihood of the defect decisions of
    defective crops re-labelled Q_clean (sequences file needs ul_positions)."""

    # Device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    # Dataset
    print(f"\nLoading dataset: {sequences_file}")
    full_dataset = AtomicSequenceDataset.from_file(
        sequences_file, max_length=config.max_seq_len
    )
    with open(sequences_file) as f:
        meta = json.load(f)
    val_seqs = meta.get("val_sequences")
    val_ul_loader = None
    if ul_weight > 0:
        assert "ul_positions" in meta and not augment_tokenizer, \
            "--ul_weight needs ul_positions in the sequences file (window_tokenizer_v2.py), no --augment"
        q_pos, q_clean_id = meta["q_pos"], meta["q_clean_id"]
        full_dataset = ULSequenceDataset(meta["sequences"], meta["ul_positions"],
                                         max_length=config.max_seq_len)
        n_ul = sum(len(u) > 0 for u in meta["ul_positions"])
        print(f"  Unlikelihood: weight {ul_weight}, {n_ul} train sequences with "
              f"{sum(map(len, meta['ul_positions']))} targets")
        if meta.get("val_ul_positions") and len(meta["val_ul_positions"]) == len(val_seqs):
            val_ul_loader = DataLoader(
                ULSequenceDataset(val_seqs, meta["val_ul_positions"], max_length=config.max_seq_len),
                batch_size=config.batch_size, shuffle=False, num_workers=0)
    del meta
    if val_seqs:
        # Pre-split (e.g. window crops, held out by MD run)
        train_set = full_dataset
        val_set = AtomicSequenceDataset(val_seqs, max_length=config.max_seq_len)
        n_train, n_val = len(train_set), len(val_set)
    else:
        n_total = len(full_dataset)
        n_val   = max(int(0.05 * n_total), 100)
        n_train = n_total - n_val
        train_set, val_set = random_split(
            full_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )
    print(f"  Train: {n_train}  |  Val: {n_val}")
    if augment_tokenizer:
        aug_ds = AugmentedSequenceDataset(full_dataset.sequences,
                                          SymmetryAugmenter(augment_tokenizer),
                                          max_length=config.max_seq_len)
        train_set = torch.utils.data.Subset(aug_ds, train_set.indices)
        print("  Augmentation: random translation / reflection / axis permutation")

    train_loader = DataLoader(train_set, batch_size=config.batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=True, persistent_workers=num_workers > 0)
    val_loader   = DataLoader(val_set,   batch_size=config.batch_size,
                              shuffle=False, num_workers=0)

    # Model
    model = AtomicGPT(config).to(device)
    if init_checkpoint:
        # warm start (fine-tuning on new data with the same tokenizer/architecture)
        ckpt = torch.load(init_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"  Initialised from {init_checkpoint} (val_loss={ckpt['val_loss']:.4f})")

    # Optimizer (AdamW with weight decay on non-bias/norm params)
    decay_params     = [p for n, p in model.named_parameters()
                        if p.dim() >= 2]
    no_decay_params  = [p for n, p in model.named_parameters()
                        if p.dim() < 2]
    optimizer = torch.optim.AdamW([
        {"params": decay_params,    "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=config.lr, betas=(0.9, 0.95))

    total_steps = config.max_epochs * len(train_loader)
    scheduler   = CosineWarmupScheduler(optimizer, config.warmup_steps, total_steps)

    # Training
    print(f"\nStarting training: {config.max_epochs} epochs, {total_steps} steps")
    print(f"{'='*60}")

    best_val_loss = float("inf")
    step          = 0
    train_losses  = []
    val_losses    = []

    for epoch in range(config.max_epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch in train_loader:
            x, y = batch[0].to(device), batch[1].to(device)

            # Forward
            _, loss = model(x, y)
            ce = loss
            if ul_weight > 0:
                ul, p_ul = unlikelihood(model, x, y, batch[2].to(device), q_pos, q_clean_id)
                if ul is not None:
                    loss = loss + ul_weight * ul

            # Backward
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            step       += 1

            # Logging
            if step % config.log_every == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                extra = (f" | ce {ce.item():.4f} | ul {ul.item():.4f} p {p_ul.item():.3f}"
                         if ul_weight > 0 and ul is not None else "")
                print(f"  step {step:5d} | loss {loss.item():.4f} | lr {lr_now:.2e}{extra}")
                train_losses.append((step, loss.item()))

            # Validation
            if step % config.eval_every == 0:
                val_loss = evaluate(model, val_loader, device)
                val_losses.append((step, val_loss))
                print(f"\n>>> Validation loss: {val_loss:.4f} (best: {best_val_loss:.4f})")
                if val_ul_loader is not None:
                    print(f">>> Val p(defect decision | Q_clean): "
                          f"{evaluate_ul(model, val_ul_loader, device, q_pos, q_clean_id):.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(model, optimizer, step, val_loss, out_dir, tag="best")

            # Periodic checkpoint
            if step % config.save_every == 0:
                save_checkpoint(model, optimizer, step,
                                epoch_loss / (step - epoch * len(train_loader)), out_dir)

        dt = time.time() - t0
        avg_loss = epoch_loss / len(train_loader)
        print(f"\nEpoch {epoch+1}/{config.max_epochs} | "
              f"avg_loss={avg_loss:.4f} | time={dt:.1f}s")

    # Final checkpoint
    final_val = evaluate(model, val_loader, device)
    save_checkpoint(model, optimizer, step, final_val, out_dir, tag="final")

    # Save training curves
    curves = {"train": train_losses, "val": val_losses}
    with open(os.path.join(out_dir, "training_curves.json"), "w") as f:
        json.dump(curves, f)

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Best val loss : {best_val_loss:.4f}")
    print(f"  Checkpoints   : {out_dir}/")
    print(f"{'='*60}")

    return model


# ============================================================
# GENERATION DEMO
# ============================================================

def generate_structure(model_path: str, tokenizer_path: str,
                       composition: str = "SiO2",
                       n_atoms: int = 72,
                       n_samples: int = 10,
                       max_tries: int = 100,
                       temperature: float = 0.8,
                       penalty: float = float("inf"),
                       relax: bool = False,
                       device: str = "cpu",
                       save_to: str = "generated.extxyz"):
    """
    Generate defect-free structures with the trained model.

    Prompts with Q_clean and uses DefectGuide (guided_generation.py) to
    mask clashes, overcoordination and O-O bonds at every Z token. Finished
    structures are checked for any remaining defect (e.g. undercoordinated
    Si = missing O) and rejected; optionally relaxed with MACE first.

    Args:
        composition : "SiO2", "HfO2", "HfSiO4"
        n_atoms     : target atom count (rounded to whole formula units)
        n_samples   : number of defect-free structures wanted
        max_tries   : give up after this many sampled structures
        penalty     : logit penalty for defect-creating Z bins (inf = mask)
        relax       : relax with MACE-MP-0 (fixed cell) before the final check
    """
    from ase.io import write
    from atomic_tokenizer import AtomicTokenizer
    from guided_generation import DefectGuide, guided_generate
    from structure_defects import defect_summary

    tokenizer = AtomicTokenizer.load(tokenizer_path)
    model, _ = load_checkpoint(model_path, device)
    model.eval()

    guide = DefectGuide(tokenizer, composition, n_atoms, penalty=penalty)
    print(f"\nGenerating {n_samples} x {composition} {guide.counts}, "
          f"cell {guide.L:.2f} Å, T={temperature}")

    calc = None
    if relax:
        from ase.optimize import FIRE
        from mace.calculators import mace_mp
        calc = mace_mp(model="medium", device=device, default_dtype="float64")

    kept, stats = [], {"aborted": 0, "defective": 0}
    for attempt in range(1, max_tries + 1):
        atoms, info = guided_generate(model, guide, temperature=temperature,
                                      device=device)
        if atoms is None:
            stats["aborted"] += 1
            continue
        if calc is not None:
            atoms.calc = calc
            FIRE(atoms, logfile=None).run(fmax=0.02, steps=500)
            atoms.info["energy_per_atom"] = atoms.get_potential_energy() / len(atoms)
            atoms.calc = None
        defects = defect_summary(atoms)
        if defects:
            stats["defective"] += 1
            print(f"  try {attempt}: rejected, defects {dict(defects)}")
            continue
        atoms.info.update({"generated_by": "AtomicGPT", "composition": composition,
                           "temperature": temperature, "relaxed": relax})
        kept.append(atoms)
        print(f"  try {attempt}: defect-free ({len(kept)}/{n_samples})")
        if len(kept) >= n_samples:
            break

    print(f"\nKept {len(kept)} defect-free structures from {attempt} tries "
          f"(aborted {stats['aborted']}, rejected {stats['defective']})")
    if kept:
        write(save_to, kept)
        print(f"  Saved to: {save_to}")
    return kept


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train GPT on atomic sequences for amorphous structure generation"
    )
    parser.add_argument("--sequences",  default="data/sequences.json")
    parser.add_argument("--tokenizer",  default="data/tokenizer.json")
    parser.add_argument("--out_dir",    default="checkpoints")
    parser.add_argument("--device",     default="auto")
    parser.add_argument("--n_layers",   type=int,   default=6)
    parser.add_argument("--d_model",    type=int,   default=256)
    parser.add_argument("--n_heads",    type=int,   default=8)
    parser.add_argument("--max_epochs", type=int,   default=50)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--max_seq_len", type=int,  default=520,
                        help="Context length; sequences are padded to it")
    parser.add_argument("--dropout",    type=float, default=0.1)
    parser.add_argument("--augment",    action="store_true",
                        help="Random translation/reflection/axis-permutation copies "
                             "of each training structure every epoch")
    parser.add_argument("--num_workers", type=int,  default=4,
                        help="DataLoader workers (used with --augment)")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Quick test with tiny model + synthetic data")
    parser.add_argument("--generate",   action="store_true",
                        help="Generate structure from checkpoint")
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint path for generation")
    parser.add_argument("--composition", default="SiO2",
                        help="Composition for generation: SiO2, HfO2, HfSiO4")
    parser.add_argument("--ul_weight", type=float, default=0.0,
                        help="unlikelihood weight for defect decisions of defective crops "
                             "re-labelled Q_clean (needs ul_positions, window_tokenizer_v2.py)")
    parser.add_argument("--init_checkpoint", default=None,
                        help="Warm start from this checkpoint (same vocab/architecture)")
    parser.add_argument("--n_atoms",    type=int,   default=72)
    parser.add_argument("--n_samples",  type=int,   default=10)
    parser.add_argument("--max_tries",  type=int,   default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--penalty",    type=float, default=float("inf"),
                        help="Logit penalty for defect-creating tokens (inf = forbid)")
    parser.add_argument("--relax",      action="store_true",
                        help="Relax generated structures with MACE before the defect check")
    args = parser.parse_args()

    if args.generate:
        # Generation mode
        assert args.checkpoint, "Provide --checkpoint for generation"
        device = args.device if args.device != "auto" else (
            "cuda" if torch.cuda.is_available() else "cpu")
        generate_structure(
            model_path=args.checkpoint,
            tokenizer_path=args.tokenizer,
            composition=args.composition,
            n_atoms=args.n_atoms,
            n_samples=args.n_samples,
            max_tries=args.max_tries,
            temperature=args.temperature,
            penalty=args.penalty,
            relax=args.relax,
            device=device,
            save_to=f"generated_{args.composition}.extxyz"
        )

    elif args.smoke_test:
        # Quick smoke test with random data
        print("Running smoke test...")
        config = GPTConfig(
            vocab_size=199, n_layers=2, d_model=64,
            n_heads=4, d_ff=256, max_epochs=2,
            batch_size=4, log_every=5, eval_every=20,
            save_every=50
        )
        # Synthetic random sequences
        sequences = [
            [1] + list(np.random.randint(4, 199, size=np.random.randint(20, 60))) + [2]
            for _ in range(200)
        ]
        with open("/tmp/smoke_sequences.json", "w") as f:
            json.dump({"sequences": sequences}, f)
        train(config, "/tmp/smoke_sequences.json", out_dir="/tmp/smoke_checkpoints")

    else:
        # Full training
        with open(args.sequences) as f:
            vocab_size = json.load(f)["vocab_size"]
        config = GPTConfig(
            vocab_size=vocab_size,
            n_layers=args.n_layers,
            d_model=args.d_model,
            n_heads=args.n_heads,
            d_ff=4 * args.d_model,
            max_seq_len=args.max_seq_len,
            dropout=args.dropout,
            lr=args.lr,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
        )
        train(config, args.sequences, out_dir=args.out_dir, device=args.device,
              augment_tokenizer=args.tokenizer if args.augment else None,
              num_workers=args.num_workers if args.augment else 0,
              init_checkpoint=args.init_checkpoint, ul_weight=args.ul_weight)
