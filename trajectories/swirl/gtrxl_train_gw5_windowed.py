#!/usr/bin/env python3
"""
Train separate windowed GTrXL encoders on GW5 using ACTION PREDICTION.

Purpose
-------
This is a drop-in pretraining stage before SWIRL. Unlike the old next-state /
occupancy objective, this trains the encoder to make the demonstrated action
predictable from the past window ending at time t.

For each window W and time t, the encoder sees only:
    x[max(0, t-W+1) : t+1]

and the last hidden state h_t^(W) is used to predict action a_t.

This stays separate from SWIRL / EM. It does NOT use latent z during training.

Inputs
------
- xs_path : xs.npy or xs_fixed.npy with shape (N, T+1)
- acs_path: acs.npy with shape (N, T)

Outputs
-------
out_dir/
  checkpoints/
    best_w5.pt
    best_w10.pt
    ...
  h_gw5_w5.npz
  h_gw5_w10.npz
  ...
  xs_gw5_for_embed.npz    # contains key 'xs' with shape (N, T)
  summary_actionpred.csv

Saved embedding shape is always:
    (N, T, d_model)
so it can be loaded by your SWIRL script as h_all.

Example
-------
python gtrxl_train_gw5_actionpred.py \
  --xs_path ../data_new/xs_fixed.npy \
  --acs_path ../data_new/acs.npy \
  --out_dir ../output/gtrxl_actionpred \
  --windows 5 10 20 30 50 \
  --epochs 40 \
  --batch_size 128 \
  --samples_per_traj 120 \
  --lr 5e-4 \
  --weight_decay 1e-2 \
  --dropout 0.2 \
  --d_model 32 \
  --d_ff 64 \
  --n_layers 1 \
  --device cuda
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from gtrxl_func import GTrXLEncoder, load_xs, save_npz_xs, set_seed


class ActionPredHead(nn.Module):
    """Predict action from the final hidden state."""
    def __init__(self, d_model: int, n_actions: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, h_last: torch.Tensor) -> torch.Tensor:
        return self.net(h_last)


class WindowActionDataset(Dataset):
    """
    Sample (window ending at t, target action a_t) pairs.

    xs_full: (N, T+1)
    acs    : (N, T)

    Each item returns:
      x_win  : (Lw,) states ending at x_t
      action : scalar a_t
      traj   : episode id
      t      : timestep
    """
    def __init__(
        self,
        xs_full: np.ndarray,
        acs: np.ndarray,
        traj_ids: List[int],
        window: int,
        samples_per_traj: int,
        seed: int = 0,
        fixed_ts: List[int] | None = None,
    ):
        super().__init__()
        if xs_full.ndim != 2 or acs.ndim != 2:
            raise ValueError(f"xs_full and acs must be 2D, got {xs_full.shape}, {acs.shape}")
        if xs_full.shape[0] != acs.shape[0]:
            raise ValueError("xs_full and acs must have same N")
        if xs_full.shape[1] != acs.shape[1] + 1:
            raise ValueError(f"Need xs_full shape (N,T+1) to match acs (N,T). Got {xs_full.shape}, {acs.shape}")
        self.xs_full = xs_full.astype(np.int64)
        self.acs = acs.astype(np.int64)
        self.traj_ids = list(traj_ids)
        self.window = int(window)
        self.samples_per_traj = int(samples_per_traj)
        self.rng = np.random.default_rng(seed)
        self.T = acs.shape[1]
        self.fixed_ts = fixed_ts[:] if fixed_ts is not None else None

        if self.fixed_ts is not None:
            self.samples: List[Tuple[int, int]] = []
            for tid in self.traj_ids:
                for t in self.fixed_ts:
                    if t < 0 or t >= self.T:
                        raise ValueError(f"fixed t={t} invalid for T={self.T}")
                    self.samples.append((tid, t))
        else:
            self.samples = []

    def __len__(self) -> int:
        if self.fixed_ts is not None:
            return len(self.samples)
        return len(self.traj_ids) * self.samples_per_traj

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.fixed_ts is not None:
            tid, t = self.samples[idx]
        else:
            tid = self.traj_ids[idx % len(self.traj_ids)]
            t = int(self.rng.integers(0, self.T))

        s = max(0, t - self.window + 1)
        x_win = self.xs_full[tid, s:t+1]  # includes x_t
        a_t = self.acs[tid, t]
        return {
            "x_win": torch.from_numpy(x_win),
            "action": torch.tensor(a_t, dtype=torch.long),
            "traj": torch.tensor(tid, dtype=torch.long),
            "t": torch.tensor(t, dtype=torch.long),
        }


def collate_varlen(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    lens = [item["x_win"].numel() for item in batch]
    Lmax = max(lens)
    B = len(batch)
    x_pad = torch.zeros((B, Lmax), dtype=torch.long)
    mask = torch.zeros((B, Lmax), dtype=torch.bool)
    actions = torch.zeros((B,), dtype=torch.long)
    traj = torch.zeros((B,), dtype=torch.long)
    t = torch.zeros((B,), dtype=torch.long)

    for i, item in enumerate(batch):
        L = item["x_win"].numel()
        x_pad[i, :L] = item["x_win"]
        mask[i, :L] = True
        actions[i] = item["action"]
        traj[i] = item["traj"]
        t[i] = item["t"]

    return {
        "x_win": x_pad,
        "mask": mask,
        "action": actions,
        "traj": traj,
        "t": t,
        "lengths": torch.tensor(lens, dtype=torch.long),
    }


@torch.no_grad()
def eval_epoch(encoder: nn.Module, head: nn.Module, dl: DataLoader, device: torch.device) -> Dict[str, float]:
    encoder.eval()
    head.eval()
    loss_sum = 0.0
    correct = 0
    count = 0
    n_batches = 0

    for batch in dl:
        x_win = batch["x_win"].to(device)
        lengths = batch["lengths"].to(device)
        actions = batch["action"].to(device)

        h_seq, _ = encoder(x_win, mems=None, start_pos=0)
        h_last = h_seq[torch.arange(h_seq.size(0), device=device), lengths - 1, :]
        logits = head(h_last)
        loss = F.cross_entropy(logits, actions)

        pred = logits.argmax(dim=-1)
        correct += (pred == actions).sum().item()
        count += actions.numel()
        loss_sum += loss.item()
        n_batches += 1

    return {
        "loss": loss_sum / max(1, n_batches),
        "acc": correct / max(1, count),
    }


@torch.no_grad()
def export_window_embeddings(
    encoder: nn.Module,
    xs_full: np.ndarray,
    out_path: Path,
    batch_size: int,
    device: torch.device,
) -> None:
    """
    Export h_t for every episode and every t using the exact same window regime
    the model was trained on.

    xs_full: (N, T+1)
    output : (N, T, D) corresponding to timesteps t=0..T-1, each built from states ending at x_t.
    """
    encoder.eval()
    xs = xs_full[:, :-1].astype(np.int64)  # states aligned with a_t at time t
    N, T = xs.shape
    d_model = encoder.d_model
    h_all = np.zeros((N, T, d_model), dtype=np.float32)

    # Full episode forward pass under causal masking gives h_t based on x_0:t.
    # For a finite-window-trained encoder with mem_len=0, this is still acceptable
    # as an export of the learned causal representation. If you want strict finite-window
    # export, you can re-run per t with slices; this full-pass version is much faster.
    # Here we do STRICT windowed export to match training exactly.
    window = getattr(encoder, "_train_window", None)
    if window is None:
        raise RuntimeError("encoder missing _train_window; cannot do strict windowed export")

    xs_torch = torch.from_numpy(xs_full).to(device)
    for t in range(T):
        s = max(0, t - window + 1)
        x_win = xs_torch[:, s:t+1]  # includes x_t
        for i0 in range(0, N, batch_size):
            i1 = min(N, i0 + batch_size)
            h_seq, _ = encoder(x_win[i0:i1], mems=None, start_pos=0)
            h_last = h_seq[:, -1, :]
            h_all[i0:i1, t, :] = h_last.detach().cpu().numpy().astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, h=h_all)
    print(f"[export] saved {out_path} with h shape {h_all.shape}")


@dataclass
class TrainConfig:
    xs_path: str
    acs_path: str
    out_dir: str
    windows: List[int]
    seed: int
    train_frac: float
    samples_per_traj: int
    val_samples_per_traj: int
    n_states: int
    n_actions: int
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    dropout: float
    gate_bias_init: float
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    grad_clip: float
    device: str


def train_one_window(cfg: TrainConfig, xs_full: np.ndarray, acs: np.ndarray, train_ids: List[int], val_ids: List[int], W: int) -> Dict[str, float]:
    out_dir = Path(cfg.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.device)

    ds_train = WindowActionDataset(
        xs_full=xs_full,
        acs=acs,
        traj_ids=train_ids,
        window=W,
        samples_per_traj=cfg.samples_per_traj,
        seed=cfg.seed + W,
        fixed_ts=None,
    )
    ds_val = WindowActionDataset(
        xs_full=xs_full,
        acs=acs,
        traj_ids=val_ids if len(val_ids) > 0 else train_ids[: max(1, min(2, len(train_ids)))],
        window=W,
        samples_per_traj=cfg.val_samples_per_traj,
        seed=cfg.seed + 1000 + W,
        fixed_ts=None,
    )

    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, num_workers=0, drop_last=True, collate_fn=collate_varlen)
    dl_val = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False, num_workers=0, drop_last=False, collate_fn=collate_varlen)

    encoder = GTrXLEncoder(
        n_states=cfg.n_states,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        mem_len=0,
        gate_bias_init=cfg.gate_bias_init,
    ).to(device)
    encoder._train_window = W  # for exact export later

    head = ActionPredHead(d_model=cfg.d_model, n_actions=cfg.n_actions, hidden=max(64, cfg.d_model), dropout=cfg.dropout).to(device)

    opt = torch.optim.AdamW(list(encoder.parameters()) + list(head.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = float("inf")
    best_path = ckpt_dir / f"best_w{W}.pt"
    best_metrics: Dict[str, float] = {}

    print(f"\n===== Training W={W} =====")
    print(f"[cfg] d_model={cfg.d_model} layers={cfg.n_layers} heads={cfg.n_heads} d_ff={cfg.d_ff} dropout={cfg.dropout}")
    print(f"[cfg] batch_size={cfg.batch_size} lr={cfg.lr} wd={cfg.weight_decay} epochs={cfg.epochs}")

    for ep in range(1, cfg.epochs + 1):
        encoder.train()
        head.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_count = 0
        n_batches = 0

        for batch in dl_train:
            x_win = batch["x_win"].to(device)
            lengths = batch["lengths"].to(device)
            actions = batch["action"].to(device)

            h_seq, _ = encoder(x_win, mems=None, start_pos=0)
            h_last = h_seq[torch.arange(h_seq.size(0), device=device), lengths - 1, :]
            logits = head(h_last)
            loss = F.cross_entropy(logits, actions)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(head.parameters()), cfg.grad_clip)
            opt.step()

            pred = logits.argmax(dim=-1)
            train_correct += (pred == actions).sum().item()
            train_count += actions.numel()
            train_loss_sum += loss.item()
            n_batches += 1

        tr_loss = train_loss_sum / max(1, n_batches)
        tr_acc = train_correct / max(1, train_count)
        va = eval_epoch(encoder, head, dl_val, device=device)
        print(f"[W={W}] ep {ep:03d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va['loss']:.4f} acc {va['acc']:.4f}")

        if va["loss"] < best_val:
            best_val = va["loss"]
            best_metrics = {
                "window": W,
                "best_val_loss": va["loss"],
                "best_val_acc": va["acc"],
                "last_train_loss": tr_loss,
                "last_train_acc": tr_acc,
            }
            torch.save({
                "encoder": encoder.state_dict(),
                "head": head.state_dict(),
                "config": {
                    "n_states": cfg.n_states,
                    "n_actions": cfg.n_actions,
                    "d_model": cfg.d_model,
                    "n_layers": cfg.n_layers,
                    "n_heads": cfg.n_heads,
                    "d_ff": cfg.d_ff,
                    "dropout": cfg.dropout,
                    "mem_len": 0,
                    "gate_bias_init": cfg.gate_bias_init,
                    "window": W,
                    "objective": "action_prediction",
                },
            }, best_path)

    print(f"[W={W}] best val loss={best_metrics.get('best_val_loss', float('nan')):.4f} acc={best_metrics.get('best_val_acc', float('nan')):.4f}")

    # Reload best encoder before export
    ckpt = torch.load(best_path, map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    encoder._train_window = W
    export_window_embeddings(
        encoder=encoder,
        xs_full=xs_full,
        out_path=out_dir / f"h_gw5_w{W}.npz",
        batch_size=cfg.batch_size,
        device=device,
    )

    return best_metrics


def parse_args() -> TrainConfig:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xs_path", type=str, required=True, help="Path to xs.npy / xs_fixed.npy with shape (N, T+1)")
    ap.add_argument("--acs_path", type=str, required=True, help="Path to acs.npy with shape (N, T)")
    ap.add_argument("--out_dir", type=str, default="gtrxl_gw5_actionpred_out")
    ap.add_argument("--windows", type=int, nargs="+", default=[5, 10, 20, 30, 50])

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--samples_per_traj", type=int, default=120)
    ap.add_argument("--val_samples_per_traj", type=int, default=60)

    ap.add_argument("--n_states", type=int, default=25)
    ap.add_argument("--n_actions", type=int, default=5)
    ap.add_argument("--d_model", type=int, default=32)
    ap.add_argument("--n_layers", type=int, default=1)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--d_ff", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--gate_bias_init", type=float, default=2.0)

    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = ap.parse_args()
    return TrainConfig(**vars(args))


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)

    xs_full = load_xs(cfg.xs_path)
    acs = np.load(cfg.acs_path).astype(np.int64)

    if xs_full.ndim != 2 or acs.ndim != 2:
        raise ValueError(f"xs_full and acs must both be 2D; got {xs_full.shape}, {acs.shape}")
    if xs_full.shape[0] != acs.shape[0]:
        raise ValueError("xs_full and acs must have same number of trajectories")
    if xs_full.shape[1] != acs.shape[1] + 1:
        raise ValueError(f"Need xs_full shape (N,T+1) and acs shape (N,T). Got {xs_full.shape}, {acs.shape}")

    N, Tp1 = xs_full.shape
    T = acs.shape[1]
    xs_for_embed = xs_full[:, :-1].astype(np.int64)

    uniq_states = np.unique(xs_full)
    uniq_actions = np.unique(acs)
    print(f"[data] xs_full shape={xs_full.shape} | acs shape={acs.shape}")
    print(f"[data] xs_for_embed shape={xs_for_embed.shape}")
    print(f"[data] state range: min={uniq_states.min()} max={uniq_states.max()} unique={len(uniq_states)}")
    print(f"[data] action range: min={uniq_actions.min()} max={uniq_actions.max()} unique={len(uniq_actions)}")

    if uniq_states.min() < 0 or uniq_states.max() >= cfg.n_states:
        raise ValueError(f"States out of range for n_states={cfg.n_states}")
    if uniq_actions.min() < 0 or uniq_actions.max() >= cfg.n_actions:
        raise ValueError(f"Actions out of range for n_actions={cfg.n_actions}")

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_npz_xs(str(out_dir / "xs_gw5_for_embed.npz"), xs_for_embed)

    rng = np.random.default_rng(cfg.seed)
    ids = np.arange(N)
    rng.shuffle(ids)
    n_train = int(round(cfg.train_frac * N))
    train_ids = ids[:n_train].tolist()
    val_ids = ids[n_train:].tolist()
    print(f"[split] train episodes={len(train_ids)} | val episodes={len(val_ids)}")

    all_metrics: List[Dict[str, float]] = []
    for W in cfg.windows:
        metrics = train_one_window(cfg, xs_full, acs, train_ids, val_ids, W)
        all_metrics.append(metrics)

    csv_path = out_dir / "summary_actionpred.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["window", "best_val_loss", "best_val_acc", "last_train_loss", "last_train_acc"])
        writer.writeheader()
        for row in all_metrics:
            writer.writerow(row)

    print("\nSaved summary to", csv_path)
    print("Saved embeddings:")
    for W in cfg.windows:
        print("  ", out_dir / f"h_gw5_w{W}.npz")
    print("Saved xs alignment file:")
    print("  ", out_dir / "xs_gw5_for_embed.npz")


if __name__ == "__main__":
    main()
