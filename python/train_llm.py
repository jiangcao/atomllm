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
    python train_llm.py \
        --sequences data/sequences.json \
        --tokenizer data/tokenizer.json \
        --out_dir    checkpoints/

    # Quick smoke test (tiny model, few steps)
    python train_llm.py --smoke_test
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
    vocab_size:   int = 199       # matches AtomicTokenizer vocab (4 special + 3 types + 3*64 bins)
    pad_token_id: int = 0

    # Architecture
    max_seq_len:  int = 512       # ~128 atoms × 4 tokens each
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
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, \
            f"Sequence length {T} exceeds max {self.config.max_seq_len}"

        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)  # (1, T)

        x = self.drop(self.token_emb(input_ids) + self.pos_emb(positions))

        for block in self.blocks:
            x = block(x)

        x      = self.ln_final(x)
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

    for x, y in loader:
        if n_batches >= max_batches:
            break
        x, y = x.to(device), y.to(device)
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
          out_dir: str = "checkpoints", device: str = "auto"):

    # Device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    # Dataset
    print(f"\nLoading dataset: {sequences_file}")
    full_dataset = AtomicSequenceDataset.from_file(
        sequences_file, max_length=config.max_seq_len
    )
    n_total = len(full_dataset)
    n_val   = max(int(0.05 * n_total), 100)
    n_train = n_total - n_val

    train_set, val_set = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"  Train: {n_train}  |  Val: {n_val}")

    train_loader = DataLoader(train_set, batch_size=config.batch_size,
                              shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=config.batch_size,
                              shuffle=False, num_workers=0)

    # Model
    model = AtomicGPT(config).to(device)

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

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # Forward
            _, loss = model(x, y)

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
                print(f"  step {step:5d} | loss {loss.item():.4f} | lr {lr_now:.2e}")
                train_losses.append((step, loss.item()))

            # Validation
            if step % config.eval_every == 0:
                val_loss = evaluate(model, val_loader, device)
                val_losses.append((step, val_loss))
                print(f"\n>>> Validation loss: {val_loss:.4f} (best: {best_val_loss:.4f})")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(model, optimizer, step, val_loss, out_dir, tag="best")

            # Periodic checkpoint
            if step % config.save_every == 0:
                save_checkpoint(model, optimizer, step, epoch_loss / step, out_dir)

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
                       n_atoms: int = 60,
                       temperature: float = 0.8,
                       device: str = "cpu",
                       save_to: str = "generated.extxyz"):
    """
    Generate a new amorphous structure using the trained LLM.

    Args:
        model_path     : path to checkpoint .pt file
        tokenizer_path : path to tokenizer.json
        composition    : "SiO2", "HfO2", or "HfSiO4"
        n_atoms        : target number of atoms
        temperature    : sampling temperature (0.5 = conservative, 1.0 = diverse)
        save_to        : output extxyz file
    """
    # Load tokenizer
    import sys
    sys.path.append(".")
    from atomic_tokenizer import AtomicTokenizer

    tokenizer = AtomicTokenizer.load(tokenizer_path)

    # Load model
    model, _ = load_checkpoint(model_path, device)
    model.eval()

    # Build composition prompt
    # Map composition to atom type sequence for priming
    composition_prompts = {
        "SiO2":   ["TYPE_Si", "TYPE_O", "TYPE_O"],
        "HfO2":   ["TYPE_Hf", "TYPE_O", "TYPE_O"],
        "HfSiO4": ["TYPE_Hf", "TYPE_Si", "TYPE_O", "TYPE_O", "TYPE_O", "TYPE_O"],
    }

    prompt_types  = composition_prompts.get(composition, ["TYPE_Si"])
    prompt_tokens = (
        [tokenizer.bos_id] +
        [tokenizer.vocab.encode_token(t) for t in prompt_types]
    )

    prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long).to(device)

    print(f"\nGenerating {composition} structure (~{n_atoms} atoms)...")
    print(f"  Temperature : {temperature}")
    print(f"  Prompt      : {prompt_types}")

    # Generate
    max_tokens = n_atoms * 4 + 10  # 4 tokens per atom + buffer
    with torch.no_grad():
        output_ids = model.generate(
            prompt_tensor,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=50,
            eos_token_id=tokenizer.eos_id
        )

    # Decode to atoms
    token_list = output_ids[0].tolist()
    atoms = tokenizer.decode(token_list)

    if atoms is None:
        print("Generation failed: could not decode structure")
        return None

    print(f"  Generated   : {len(atoms)} atoms")
    print(f"  Elements    : {set(atoms.get_chemical_symbols())}")

    # Save
    from ase.io import write
    atoms.info["generated_by"] = "AtomicGPT"
    atoms.info["composition"]  = composition
    atoms.info["temperature"]  = temperature
    write(save_to, atoms)
    print(f"  Saved to    : {save_to}")

    return atoms


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
    parser.add_argument("--smoke_test", action="store_true",
                        help="Quick test with tiny model + synthetic data")
    parser.add_argument("--generate",   action="store_true",
                        help="Generate structure from checkpoint")
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint path for generation")
    parser.add_argument("--composition", default="SiO2",
                        help="Composition for generation: SiO2, HfO2, HfSiO4")
    args = parser.parse_args()

    if args.generate:
        # Generation mode
        assert args.checkpoint, "Provide --checkpoint for generation"
        generate_structure(
            model_path=args.checkpoint,
            tokenizer_path=args.tokenizer,
            composition=args.composition,
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
        config = GPTConfig(
            vocab_size=199,
            n_layers=args.n_layers,
            d_model=args.d_model,
            n_heads=args.n_heads,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
        )
        train(config, args.sequences, out_dir=args.out_dir, device=args.device)
