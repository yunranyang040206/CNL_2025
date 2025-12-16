"""
gtrxl.py — Gated Transformer-XL (GTrXL) feature encoder + pretraining (Objectives A & B)

- Inputs are discrete state IDs x_t (0..126) and action IDs a_t (0..3) per trajectory.
- The model is a *causal* GTrXL-style encoder producing deterministic history embeddings h_t.
- Pretraining uses:
  (A) next-state prediction (cross-entropy)
  (B) future-occupancy prediction over horizon H (multi-label BCE)

IMPORTANT: This is a *deterministic feature extractor* intended to be frozen and exported to SWIRL/JAX.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


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
    Build an additive attention mask for nn.MultiheadAttention (batch_first=True).

    Query length = L (current segment)
    Key length   = M + L (memory + current segment)

    We allow each position i in [0..L-1] to attend to:
      - all memory positions [0..M-1]
      - current positions [M..M+i] (i.e., up to itself in the current segment)

    Disallow attending to future positions in current segment.
    """
    mask = torch.zeros((L, M + L), device=device)
    if L > 0:
        # For each query row i, disallow keys corresponding to current positions > i
        # Current segment keys are indices [M .. M+L-1]
        # Disallowed keys: [M+i+1 .. M+L-1]
        for i in range(L):
            if i + 1 < L:
                mask[i, M + i + 1 : M + L] = float("-inf")
    return mask


# -----------------------------
# Positional Encoding
# -----------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.

    NOTE: Transformer-XL uses relative positional encodings.
    Absolute time is meaningful in your behavioral data, so we use absolute encodings here.
    """
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """
        x: (B, L, D)
        """
        L = x.size(1)
        return x + self.pe[start_pos:start_pos + L, :].unsqueeze(0).to(x.device)


# -----------------------------
# GRU-type gating (used in GTrXL paper)
# -----------------------------

class GRUGatingUnit(nn.Module):
    """
    GRU-type gating used as an *untied activation function in depth*.

    Corresponds to Parisotto et al. "Stabilizing Transformers for RL", Section 3.2 (Gating Layers),
    specifically the GRU-type gating equations:
      r = σ(W_r y + U_r x)
      z = σ(W_z y + U_z x - b_g)
      ĥ = tanh(W_g y + U_g (r ⊙ x))
      g(x,y) = (1 - z) ⊙ x + z ⊙ ĥ

    Here:
      x = residual stream input (skip path)
      y = submodule output (after ReLU as recommended for identity-map reordering)
    """
    def __init__(self, d_model: int, bias_init: float = 2.0):
        super().__init__()
        self.W_r = nn.Linear(d_model, d_model, bias=True)
        self.U_r = nn.Linear(d_model, d_model, bias=False)

        self.W_z = nn.Linear(d_model, d_model, bias=True)
        self.U_z = nn.Linear(d_model, d_model, bias=False)

        self.W_g = nn.Linear(d_model, d_model, bias=True)
        self.U_g = nn.Linear(d_model, d_model, bias=False)

        # b_g in the paper is a bias that encourages identity mapping at init.
        # Implement as a learnable parameter added to the z pre-activation (subtracted).
        self.b_g = nn.Parameter(torch.full((d_model,), float(bias_init)))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        x, y: (B, L, D)
        """
        r = torch.sigmoid(self.W_r(y) + self.U_r(x))
        z = torch.sigmoid(self.W_z(y) + self.U_z(x) - self.b_g)
        h_hat = torch.tanh(self.W_g(y) + self.U_g(r * x))
        return (1.0 - z) * x + z * h_hat


# -----------------------------
# GTrXL Block
# -----------------------------

class GTrXLBlock(nn.Module):
    """
    One GTrXL layer block:

    - Identity Map Reordering: LayerNorm is applied to the input stream of each submodule
      (self-attention and FFN) to preserve an identity path.

    - ReLU before residual/gating connection: recommended due to two linear layers path
      created by reordering (Parisotto et al., Section 3.1).

    - Replace residual additions with gating layers (Parisotto et al., Section 3.2).

    Also supports Transformer-XL style segment-level recurrence via an optional memory.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        gate_bias_init: float = 2.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # Pre-norm (Identity Map Reordering) for attention submodule
        self.ln_attn = nn.LayerNorm(d_model)

        # Multi-head self-attention
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)

        self.dropout_attn = nn.Dropout(dropout)
        self.gate_attn = GRUGatingUnit(d_model, bias_init=gate_bias_init)

        # Pre-norm for FFN submodule
        self.ln_ffn = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout_ffn = nn.Dropout(dropout)
        self.gate_ffn = GRUGatingUnit(d_model, bias_init=gate_bias_init)

    def forward(
        self,
        x: torch.Tensor,
        mem: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x:   (B, L, D) current segment embeddings
        mem: (B, M, D) previous segment memory (detached/stop-grad)

        Returns:
          out: (B, L, D)
          new_mem: (B, M', D) memory to carry forward (usually concat then cut)
        """
        B, L, D = x.shape
        device = x.device

        if mem is None:
            mem_kv = x
            M = 0
        else:
            # "SG(M)" in the paper: stop gradient through memory.
            mem_kv = torch.cat([mem.detach(), x], dim=1)
            M = mem.size(1)

        # ---- Recurrent Multi-Head Attention (RMHA) with Identity Map Reordering ----
        # Corresponds to paper Eq. block for TrXL-I / GTrXL:
        #   Y^(l) = RMHA( LN([SG(M^(l-1)), E^(l-1)]) )
        #   E^(l) = g_MHA( E^(l-1), ReLU(Y^(l)) )
        x_norm = self.ln_attn(x)

        # Attention mask for causal attention with memory
        attn_mask = causal_attn_mask(L=L, M=M, device=device)  # (L, M+L)

        # Query is current segment, keys/values include memory + current
        # But we normalize only current x; for KV we use mem_kv directly (common simplification).
        # If you want strict: also LN the concatenated stream before attn.
        q = x_norm
        k = mem_kv
        v = mem_kv

        y, _ = self.attn(q, k, v, attn_mask=attn_mask)
        y = self.dropout_attn(y)

        # ReLU before gating/residual as suggested in Identity Map Reordering discussion
        y = F.relu(y)

        x = self.gate_attn(x, y)

        # ---- Feedforward with Identity Map Reordering + gating ----
        # Corresponds to:
        #   E^(l) = f( LN(Y^(l)) )
        #   E^(l) = g_MLP( Y^(l), ReLU(E^(l)) )
        x_norm2 = self.ln_ffn(x)
        y2 = self.ffn(x_norm2)
        y2 = self.dropout_ffn(y2)
        y2 = F.relu(y2)
        out = self.gate_ffn(x, y2)

        # Memory update: return the *post-block* representation as new memory candidates.
        return out, out


# -----------------------------
# GTrXL Encoder (stack + embeddings)
# -----------------------------

class GTrXLEncoder(nn.Module):
    """
    Causal GTrXL encoder producing deterministic history embeddings h_t.

    Inputs are discrete states and actions. We embed and add positional encoding.

    Supports segment recurrence by passing a list of per-layer memories.
    """
    def __init__(
        self,
        n_states: int = 127,
        n_actions: int = 4,
        d_model: int = 64,
        n_layers: int = 3,
        n_heads: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        max_len: int = 2048,
        mem_len: int = 0,
        gate_bias_init: float = 2.0,
    ):
        super().__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.d_model = d_model
        self.n_layers = n_layers
        self.mem_len = mem_len

        self.state_emb = nn.Embedding(n_states, d_model)
        self.act_emb = nn.Embedding(n_actions, d_model)

        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            GTrXLBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                gate_bias_init=gate_bias_init,
            )
            for _ in range(n_layers)
        ])

        self.out_ln = nn.LayerNorm(d_model)

    def init_memory(self, batch_size: int, device: torch.device) -> List[torch.Tensor]:
        """
        Initialize empty memory for each layer.
        """
        if self.mem_len <= 0:
            return []
        return [torch.zeros((batch_size, 0, self.d_model), device=device) for _ in range(self.n_layers)]

    def update_memory(self, mem: torch.Tensor, new_mem: torch.Tensor) -> torch.Tensor:
        """
        Keep the last mem_len tokens.
        """
        if self.mem_len <= 0:
            return torch.zeros_like(mem[:, :0, :])
        cat = torch.cat([mem, new_mem], dim=1)
        if cat.size(1) <= self.mem_len:
            return cat
        return cat[:, -self.mem_len :, :]

    def forward(
        self,
        x: torch.Tensor,  # (B, L) int64 states
        a: torch.Tensor,  # (B, L) int64 actions
        mems: Optional[List[torch.Tensor]] = None,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Returns:
          h: (B, L, D) deterministic embeddings for each timestep
          new_mems: list of per-layer memory tensors
        """
        B, L = x.shape
        device = x.device

        if mems is None:
            mems = self.init_memory(B, device)

        tok = self.state_emb(x) + self.act_emb(a)
        tok = self.pos_enc(tok, start_pos=start_pos)
        tok = self.dropout(tok)

        new_mems: List[torch.Tensor] = []
        out = tok

        for i, block in enumerate(self.blocks):
            mem_i = mems[i] if (self.mem_len > 0 and len(mems) > 0) else None
            out, mem_candidate = block(out, mem=mem_i)
            if self.mem_len > 0:
                updated = self.update_memory(mem_i if mem_i is not None else torch.zeros((B, 0, self.d_model), device=device), mem_candidate)
                new_mems.append(updated)
            else:
                new_mems.append(torch.zeros((B, 0, self.d_model), device=device))

        out = self.out_ln(out)
        return out, new_mems


# -----------------------------
# Pretraining Heads (Objective A & B)
# -----------------------------

class PretrainHeads(nn.Module):
    """
    Two heads on top of h_t:
      - next-state logits (Objective A)
      - future-occupancy logits (Objective B)
    """
    def __init__(self, d_model: int, n_states: int):
        super().__init__()
        self.next_state = nn.Linear(d_model, n_states)
        self.occupancy = nn.Linear(d_model, n_states)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # h: (B, L, D)
        ns_logits = self.next_state(h)         # (B, L, n_states)
        occ_logits = self.occupancy(h)         # (B, L, n_states)
        return ns_logits, occ_logits


# -----------------------------
# Training
# -----------------------------

@dataclass
class TrainConfig:
    npz_path: str
    out_dir: str
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # model
    d_model: int = 64
    n_layers: int = 3
    n_heads: int = 4
    d_ff: int = 256
    dropout: float = 0.1
    mem_len: int = 0
    gate_bias_init: float = 2.0

    # data
    chunk_len: int = 128
    horizon_H: int = 25
    batch_size: int = 64
    num_workers: int = 2

    # optimization
    lr: float = 3e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    epochs: int = 10
    lambda_occ: float = 0.5

class LabyrinthChunkDataset(Dataset):
    """
    Samples fixed-length chunks from trajectories (xs, acs).

    Expects an .npz containing:
      xs: (N, T) int
      acs: (N, T) int
    """
    def __init__(
        self,
        npz_path: str,
        chunk_len: int = 128,
        horizon_H: int = 25,
        n_states: int = 127,
        split: str = "train",
        test_every: int = 5,
    ):
        super().__init__()
        data = np.load(npz_path)
        xs = data["xs"].astype(np.int64)   # (N, T)
        acs = data["acs"].astype(np.int64) # (N, T)
        assert xs.shape == acs.shape

        # mimic your run_labyrinth.py split: every 5th to test
        idx = np.arange(xs.shape[0])
        test_idx = idx[(idx % test_every) == 0]
        train_idx = idx[(idx % test_every) != 0]
        use_idx = train_idx if split == "train" else test_idx

        self.xs = xs[use_idx]
        self.acs = acs[use_idx]
        self.N, self.T = self.xs.shape
        self.chunk_len = chunk_len
        self.H = horizon_H
        self.n_states = n_states

        # Precompute valid start positions: need x_{t+L} for next-state targets, and up to t+L+H for occupancy
        self.max_start = self.T - (chunk_len + 1 + horizon_H)
        if self.max_start < 0:
            raise ValueError(f"Traj length T={self.T} too short for chunk_len={chunk_len}, H={horizon_H}")

        # We'll sample uniformly at __getitem__ time; dataset length can be large
        self.virtual_len = self.N * max(1, self.max_start)

    def __len__(self) -> int:
        return self.virtual_len

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Map idx to a trajectory and a random start (deterministic mapping for reproducibility)
        traj_i = idx % self.N
        # a simple hash-like start selection
        start = (idx // self.N) % (self.max_start + 1)

        x = self.xs[traj_i, start : start + self.chunk_len + 1 + self.H]  # include +1 for next-state, +H for occupancy
        a = self.acs[traj_i, start : start + self.chunk_len + 1 + self.H]

        # Inputs to encoder: length L (we use first L positions)
        x_in = x[: self.chunk_len]
        a_in = a[: self.chunk_len]

        # Objective A target: next state for each position in chunk (predict x_{t+1})
        x_next = x[1 : self.chunk_len + 1]  # length L

        # Objective B target: future occupancy multi-hot in next H steps for each position
        # For each t in [0..L-1], look at x[t+1 : t+1+H] and mark visited states.
        occ = np.zeros((self.chunk_len, self.n_states), dtype=np.float32)
        for t in range(self.chunk_len):
            fut = x[t + 1 : t + 1 + self.H]
            occ[t, fut] = 1.0

        return {
            "x": torch.from_numpy(x_in),
            "a": torch.from_numpy(a_in),
            "x_next": torch.from_numpy(x_next),
            "occ": torch.from_numpy(occ),
        }


def train(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg.device)

    ds_train = LabyrinthChunkDataset(
        npz_path=cfg.npz_path,
        chunk_len=cfg.chunk_len,
        horizon_H=cfg.horizon_H,
        split="train",
    )
    ds_test = LabyrinthChunkDataset(
        npz_path=cfg.npz_path,
        chunk_len=cfg.chunk_len,
        horizon_H=cfg.horizon_H,
        split="test",
    )

    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    dl_test = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    encoder = GTrXLEncoder(
        n_states=127,
        n_actions=4,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        mem_len=cfg.mem_len,
        gate_bias_init=cfg.gate_bias_init,
    ).to(device)

    heads = PretrainHeads(d_model=cfg.d_model, n_states=127).to(device)

    opt = torch.optim.AdamW(list(encoder.parameters()) + list(heads.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay)

    def run_epoch(dl: DataLoader, train_mode: bool) -> Dict[str, float]:
        encoder.train(train_mode)
        heads.train(train_mode)

        total_loss = 0.0
        total_ns = 0.0
        total_occ = 0.0
        total_acc = 0.0
        n_batches = 0

        for batch in dl:
            x = batch["x"].to(device)
            a = batch["a"].to(device)
            x_next = batch["x_next"].to(device)
            occ = batch["occ"].to(device)

            h, _ = encoder(x, a, mems=None, start_pos=0)
            ns_logits, occ_logits = heads(h)

            # Objective A: next-state CE
            # ns_logits: (B, L, 127), x_next: (B, L)
            ns_loss = F.cross_entropy(ns_logits.reshape(-1, 127), x_next.reshape(-1))

            # next-state accuracy
            with torch.no_grad():
                preds = ns_logits.argmax(dim=-1)
                acc = (preds == x_next).float().mean().item()

            # Objective B: occupancy BCE
            occ_loss = F.binary_cross_entropy_with_logits(occ_logits, occ)

            loss = ns_loss + cfg.lambda_occ * occ_loss

            if train_mode:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(heads.parameters()), cfg.grad_clip)
                opt.step()

            total_loss += float(loss.item())
            total_ns += float(ns_loss.item())
            total_occ += float(occ_loss.item())
            total_acc += float(acc)
            n_batches += 1

        return {
            "loss": total_loss / max(1, n_batches),
            "ns_loss": total_ns / max(1, n_batches),
            "occ_loss": total_occ / max(1, n_batches),
            "ns_acc": total_acc / max(1, n_batches),
        }

    best_val = float("inf")
    for ep in range(cfg.epochs):
        tr = run_epoch(dl_train, train_mode=True)
        va = run_epoch(dl_test, train_mode=False)

        print(f"[epoch {ep+1:03d}] train loss={tr['loss']:.4f} (ns={tr['ns_loss']:.4f}, occ={tr['occ_loss']:.4f}, acc={tr['ns_acc']:.3f}) "
              f"| val loss={va['loss']:.4f} (ns={va['ns_loss']:.4f}, occ={va['occ_loss']:.4f}, acc={va['ns_acc']:.3f})")

        ckpt = {
            "encoder": encoder.state_dict(),
            "heads": heads.state_dict(),
            "cfg": cfg.__dict__,
            "epoch": ep + 1,
            "val_loss": va["loss"],
        }
        torch.save(ckpt, out_dir / "last.pt")
        if va["loss"] < best_val:
            best_val = va["loss"]
            torch.save(ckpt, out_dir / "best.pt")

    print(f"Done. Best val loss: {best_val:.4f}. Saved to: {out_dir}")


# -----------------------------
# Export embeddings for SWIRL
# -----------------------------

@torch.no_grad()
def export_embeddings(
    ckpt_path: str,
    npz_path: str,
    out_npz: str,
    batch_size: int = 64,
) -> None:
    """
    Load pretrained encoder, compute h_{n,t} for all trajectories, save as .npz for JAX/SWIRL.

    Output:
      h: (N, T, D) float32
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["cfg"]

    encoder = GTrXLEncoder(
        n_states=127,
        n_actions=4,
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        d_ff=cfg["d_ff"],
        dropout=0.0,
        mem_len=cfg.get("mem_len", 0),
        gate_bias_init=cfg.get("gate_bias_init", 2.0),
    ).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()

    data = np.load(npz_path)
    xs = data["xs"].astype(np.int64)
    acs = data["acs"].astype(np.int64)
    N, T = xs.shape

    h_all = np.zeros((N, T, cfg["d_model"]), dtype=np.float32)

    for i0 in range(0, N, batch_size):
        i1 = min(N, i0 + batch_size)
        x = torch.from_numpy(xs[i0:i1]).to(device)
        a = torch.from_numpy(acs[i0:i1]).to(device)

        # process full sequences in one go (no recurrence needed, but can be large)
        h, _ = encoder(x, a, mems=None, start_pos=0)
        h_all[i0:i1] = h.detach().cpu().numpy().astype(np.float32)

    np.savez_compressed(out_npz, h=h_all)
    print(f"Saved embeddings: {out_npz}  (h shape={h_all.shape})")
