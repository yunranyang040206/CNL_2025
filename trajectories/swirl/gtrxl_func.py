"""
gtrxl_func.py — helper functions only (no training loop)

- Causal GTrXL encoder that produces deterministic history embeddings h_t = f(x_0:t)
- No actions as encoder input (by design for SWIRL identifiability)
- Utilities:
  - build chunk datasets for pretraining objectives
  - compute/export embeddings for SWIRL/JAX
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def causal_attn_mask(L: int, M: int, device: torch.device) -> torch.Tensor:
    """
    Additive attention mask for nn.MultiheadAttention (batch_first=True).

    Query length = L (current segment)
    Key length   = M + L (memory + current segment)

    Allows each position i to attend to:
      - all memory positions
      - current positions up to itself (causal)
    """
    mask = torch.zeros((L, M + L), device=device)
    for i in range(L):
        if i + 1 < L:
            mask[i, M + i + 1 : M + L] = float("-inf")
    return mask


def load_xs(xs_path: str) -> np.ndarray:
    """
    Load trajectories from:
      - .npy: expects (N, T) int
      - .npz: expects key 'xs' as (N, T) int
    """
    p = Path(xs_path)
    if p.suffix == ".npy":
        xs = np.load(p).astype(np.int64)
        if xs.ndim != 2:
            raise ValueError(f"{xs_path} must be (N,T). Got shape {xs.shape}")
        return xs
    if p.suffix == ".npz":
        data = np.load(p)
        if "xs" not in data:
            raise ValueError(f"{xs_path} missing key 'xs'")
        xs = data["xs"].astype(np.int64)
        if xs.ndim != 2:
            raise ValueError(f"{xs_path}['xs'] must be (N,T). Got shape {xs.shape}")
        return xs
    raise ValueError(f"Unsupported xs_path suffix: {p.suffix} (use .npy or .npz)")


def save_npz_xs(out_npz: str, xs: np.ndarray) -> None:
    out = Path(out_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, xs=xs.astype(np.int64))


# -----------------------------
# Positional Encoding
# -----------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (absolute)."""
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        # x: (B, L, D)
        L = x.size(1)
        return x + self.pe[start_pos:start_pos + L, :].unsqueeze(0).to(x.device)


# -----------------------------
# GRU-type gating
# -----------------------------

class GRUGatingUnit(nn.Module):
    """
    GRU-type gating used in GTrXL:
      r = σ(W_r y + U_r x)
      z = σ(W_z y + U_z x - b_g)
      ĥ = tanh(W_g y + U_g (r ⊙ x))
      out = (1-z) ⊙ x + z ⊙ ĥ
    """
    def __init__(self, d_model: int, bias_init: float = 2.0):
        super().__init__()
        self.W_r = nn.Linear(d_model, d_model, bias=True)
        self.U_r = nn.Linear(d_model, d_model, bias=False)

        self.W_z = nn.Linear(d_model, d_model, bias=True)
        self.U_z = nn.Linear(d_model, d_model, bias=False)

        self.W_g = nn.Linear(d_model, d_model, bias=True)
        self.U_g = nn.Linear(d_model, d_model, bias=False)

        self.b_g = nn.Parameter(torch.full((d_model,), float(bias_init)))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r = torch.sigmoid(self.W_r(y) + self.U_r(x))
        z = torch.sigmoid(self.W_z(y) + self.U_z(x) - self.b_g)
        h_hat = torch.tanh(self.W_g(y) + self.U_g(r * x))
        return (1.0 - z) * x + z * h_hat


# -----------------------------
# GTrXL block
# -----------------------------

class GTrXLBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        gate_bias_init: float = 2.0,
    ):
        super().__init__()
        self.ln_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.dropout_attn = nn.Dropout(dropout)
        self.gate_attn = GRUGatingUnit(d_model, bias_init=gate_bias_init)

        self.ln_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout_ffn = nn.Dropout(dropout)
        self.gate_ffn = GRUGatingUnit(d_model, bias_init=gate_bias_init)

    def forward(self, x: torch.Tensor, mem: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, L, D), mem: (B, M, D)
        B, L, D = x.shape
        device = x.device

        if mem is None:
            mem_kv = x
            M = 0
        else:
            mem_kv = torch.cat([mem.detach(), x], dim=1)
            M = mem.size(1)

        # Attention (pre-norm + causal mask)
        q = self.ln_attn(x)
        k = mem_kv
        v = mem_kv

        attn_mask = causal_attn_mask(L=L, M=M, device=device)
        y, _ = self.attn(q, k, v, attn_mask=attn_mask)
        y = self.dropout_attn(y)
        y = F.relu(y)
        x = self.gate_attn(x, y)

        # FFN (pre-norm + gating)
        y2 = self.ffn(self.ln_ffn(x))
        y2 = self.dropout_ffn(y2)
        y2 = F.relu(y2)
        out = self.gate_ffn(x, y2)

        return out, out


# -----------------------------
# Encoder
# -----------------------------

class GTrXLEncoder(nn.Module):
    """
    Causal encoder producing embeddings h_t that depend only on x_0:t.

    Input:
      x: (B, L) int64 state IDs
    Output:
      h: (B, L, D)
    """
    def __init__(
        self,
        n_states: int,
        d_model: int = 64, # dimension of the history embeddings h_t
        n_layers: int = 2, # number of GTrXL layers (short term structure -> movement pattern + longer term intention-> thirst cycle)
        n_heads: int = 4, # attention heads (64/4=16 dim per head)
        d_ff: int = 128, # Width of the feed-forward network inside each transformer block; Controls how non-linear the transformation of history is.
        # d_ff = 2*d_model is common
        dropout: float = 0.1,
        max_len: int = 4096,
        mem_len: int = 0,
        gate_bias_init: float = 2.0,
    ):
        super().__init__()
        self.n_states = n_states
        self.d_model = d_model
        self.n_layers = n_layers
        self.mem_len = mem_len

        self.state_emb = nn.Embedding(n_states, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            GTrXLBlock(d_model, n_heads, d_ff, dropout=dropout, gate_bias_init=gate_bias_init)
            for _ in range(n_layers)
        ])
        self.out_ln = nn.LayerNorm(d_model)

    def init_memory(self, batch_size: int, device: torch.device) -> List[torch.Tensor]:
        if self.mem_len <= 0:
            return []
        return [torch.zeros((batch_size, 0, self.d_model), device=device) for _ in range(self.n_layers)]

    def update_memory(self, mem: torch.Tensor, new_mem: torch.Tensor) -> torch.Tensor:
        if self.mem_len <= 0:
            return torch.zeros_like(mem[:, :0, :])
        cat = torch.cat([mem, new_mem], dim=1)
        if cat.size(1) <= self.mem_len:
            return cat
        return cat[:, -self.mem_len :, :]

    def forward(
        self,
        x: torch.Tensor,  # (B, L)
        mems: Optional[List[torch.Tensor]] = None,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        B, L = x.shape
        device = x.device
        if mems is None:
            mems = self.init_memory(B, device)

        out = self.state_emb(x)
        out = self.pos_enc(out, start_pos=start_pos)
        out = self.dropout(out)

        new_mems: List[torch.Tensor] = []
        for i, block in enumerate(self.blocks):
            mem_i = mems[i] if (self.mem_len > 0 and len(mems) > 0) else None
            out, mem_candidate = block(out, mem=mem_i)
            if self.mem_len > 0:
                updated = self.update_memory(
                    mem_i if mem_i is not None else torch.zeros((B, 0, self.d_model), device=device),
                    mem_candidate
                )
                new_mems.append(updated)
            else:
                new_mems.append(torch.zeros((B, 0, self.d_model), device=device))

        out = self.out_ln(out)
        return out, new_mems


class PretrainHeads(nn.Module):
    """Heads for Objective A (next-state CE) and Objective B (occupancy BCE)."""
    def __init__(self, d_model: int, n_states: int):
        super().__init__()
        self.next_state = nn.Linear(d_model, n_states)
        self.occupancy = nn.Linear(d_model, n_states)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.next_state(h), self.occupancy(h)


# -----------------------------
# Generic chunk dataset (for training script)
# -----------------------------

class TrajChunkDataset(Dataset):
    """
    Generic chunk dataset for state-only sequences.

    - xs: (N, T) int64
    - returns x (L), x_next (L), occ (L, n_states)

    Modes:
      - fixed_starts: deterministic chunk starts (good for sanity tests)
      - random_starts: random starts each __getitem__ (good for training)
    """
    def __init__(
        self,
        xs: np.ndarray,
        traj_ids: List[int],
        n_states: int,
        chunk_len: int,
        horizon_H: int,
        samples_per_traj: int,
        seed: int = 0,
        fixed_starts: Optional[List[int]] = None,
    ):
        super().__init__()
        self.xs = xs.astype(np.int64)
        self.traj_ids = traj_ids
        self.n_states = int(n_states)
        self.L = int(chunk_len)
        self.H = int(horizon_H)
        self.samples_per_traj = int(samples_per_traj)
        self.rng = np.random.default_rng(seed)

        N, T = self.xs.shape
        need = self.L + 1 + self.H
        if T < need:
            raise ValueError(f"T={T} too short for chunk_len={self.L}, H={self.H} (need >= {need})")
        self.max_start = T - need

        self.fixed_starts = fixed_starts[:] if fixed_starts is not None else None
        if self.fixed_starts is not None:
            # Build deterministic sample list: (traj, start)
            self.samples = []
            for tid in traj_ids:
                for st in self.fixed_starts:
                    if st < 0 or st > self.max_start:
                        raise ValueError(f"fixed start {st} invalid (max_start={self.max_start})")
                    self.samples.append((tid, st))
        else:
            self.samples = None  # random starts

    def __len__(self) -> int:
        if self.samples is not None:
            return len(self.samples)
        return len(self.traj_ids) * self.samples_per_traj

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.samples is not None:
            tid, st = self.samples[idx]
        else:
            tid = self.traj_ids[idx % len(self.traj_ids)]
            st = int(self.rng.integers(0, self.max_start + 1))

        x_full = self.xs[tid, st : st + self.L + 1 + self.H]  # length L+1+H
        x_in = x_full[: self.L]
        x_next = x_full[1 : self.L + 1]

        occ = np.zeros((self.L, self.n_states), dtype=np.float32)
        for t in range(self.L):
            fut = x_full[t + 1 : t + 1 + self.H]
            occ[t, fut] = 1.0

        return {
            "x": torch.from_numpy(x_in),
            "x_next": torch.from_numpy(x_next),
            "occ": torch.from_numpy(occ),
        }


# -----------------------------
# Export embeddings (for SWIRL/JAX)
# -----------------------------

@torch.no_grad()
def export_embeddings(
    ckpt_path: str,
    xs_path: str,
    out_path: str,
    n_states: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    d_ff: int,
    mem_len: int = 0,
    gate_bias_init: float = 2.0,
    batch_size: int = 64,
    device: Optional[str] = None,
) -> None:
    """
    Compute h for all trajectories and save.

    Inputs:
      ckpt_path: torch checkpoint containing encoder weights (key 'encoder')
      xs_path:   .npy (N,T) or .npz with 'xs'
      out_path:  .npy or .npz (if .npz, saves key 'h')
    """
    xs = load_xs(xs_path)
    N, T = xs.shape

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    ckpt = torch.load(ckpt_path, map_location=dev)
    enc_state = ckpt["encoder"] if "encoder" in ckpt else ckpt

    encoder = GTrXLEncoder(
        n_states=n_states,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=0.0,
        mem_len=mem_len,
        gate_bias_init=gate_bias_init,
    ).to(dev)
    encoder.load_state_dict(enc_state)
    encoder.eval()

    h_all = np.zeros((N, T, d_model), dtype=np.float32)

    for i0 in range(0, N, batch_size):
        i1 = min(N, i0 + batch_size)
        x = torch.from_numpy(xs[i0:i1]).to(dev)
        h, _ = encoder(x, mems=None, start_pos=0)
        h_all[i0:i1] = h.detach().cpu().numpy().astype(np.float32)

    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    if outp.suffix == ".npy":
        np.save(outp, h_all)
    elif outp.suffix == ".npz":
        np.savez_compressed(outp, h=h_all)
    else:
        raise ValueError("out_path must end with .npy or .npz")

    print(f"[export] xs: {xs.shape} -> h: {h_all.shape} saved to {out_path}")
