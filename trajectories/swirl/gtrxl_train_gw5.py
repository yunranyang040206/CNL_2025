# gtrxl_train_gw5.py
"""
Train GTrXL encoder on GW5 (state-only) and export causal history embeddings h_t = f(x_0:t).

Inputs (GW5 output folder):
  - xs.npy : (N, T+1) int states 0..24

Outputs:
  - checkpoints/best.pt : torch checkpoint with encoder+heads+config
  - h_gw5.npz : {h: (N, T, d_model)} float32  (T = original T, i.e., xs.shape[1]-1)

Design constraints:
  - NO actions as input to encoder (important for SWIRL identifiability)
  - Causal embedding: h_t uses only x_0..x_t
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from gtrxl_func import (
    set_seed,
    load_xs,
    TrajChunkDataset,
    GTrXLEncoder,
    PretrainHeads,
    export_embeddings,
)

# -----------------------------
# Evaluation helper
# -----------------------------
@torch.no_grad()
def eval_epoch(encoder, heads, dl, device, lambda_occ: float):
    encoder.eval()
    heads.eval()

    loss_sum = 0.0
    ns_sum = 0.0
    occ_sum = 0.0
    correct = 0
    count = 0
    n_batches = 0

    for batch in dl:
        x = batch["x"].to(device)           # (B, L)
        x_next = batch["x_next"].to(device) # (B, L)
        occ = batch["occ"].to(device)       # (B, L, S)

        h, _ = encoder(x)
        ns_logits, occ_logits = heads(h)

        ns_loss = F.cross_entropy(
            ns_logits.reshape(-1, ns_logits.size(-1)),
            x_next.reshape(-1),
        )
        occ_loss = F.binary_cross_entropy_with_logits(occ_logits, occ)

        loss = ns_loss + lambda_occ * occ_loss

        pred = ns_logits.argmax(dim=-1)
        correct += (pred == x_next).sum().item()
        count += x_next.numel()

        loss_sum += loss.item()
        ns_sum += ns_loss.item()
        occ_sum += occ_loss.item()
        n_batches += 1

    if n_batches == 0:
        return {"loss": float("nan"), "ns": float("nan"), "occ": float("nan"), "acc": float("nan")}

    return {
        "loss": loss_sum / n_batches,
        "ns": ns_sum / n_batches,
        "occ": occ_sum / n_batches,
        "acc": correct / max(1, count),
    }


def main():
    ap = argparse.ArgumentParser()

    # Paths
    ap.add_argument("--xs_path", type=str, default="output/xs.npy",
                    help="GW5 xs.npy (N, T+1) or an .npz with key 'xs'")
    ap.add_argument("--out_dir", type=str, default="gtrxl_gw5_out",
                    help="Where to save checkpoints + embeddings")

    # Data
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--chunk_len", type=int, default=128)
    ap.add_argument("--H", type=int, default=25)
    ap.add_argument("--samples_per_traj", type=int, default=300)

    # Model
    ap.add_argument("--n_states", type=int, default=25)
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--d_ff", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--mem_len", type=int, default=0)
    ap.add_argument("--gate_bias_init", type=float, default=2.0)

    # Training
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--lambda_occ", type=float, default=0.5)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = ap.parse_args()
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    # -----------------------------
    # Load GW5 xs and convert to (N, T) for training
    # GW5 xs.npy is (N, T+1) states; we train on x_0..x_{T-1} to predict x_1..x_T
    # Our dataset class expects xs shape (N, T) for chunks, so we drop the last state.
    # -----------------------------
    xs_full = load_xs(args.xs_path)  # (N, T+1) typically
    if xs_full.ndim != 2:
        raise ValueError(f"xs must be 2D (N,T). Got {xs_full.shape}")

    N, Tp1 = xs_full.shape
    if Tp1 < 2:
        raise ValueError(f"xs seems too short: {xs_full.shape}")

    # training xs: (N, T) where T = Tp1-1
    xs = xs_full[:, :-1].astype(np.int64)
    N, T = xs.shape

    # Basic sanity prints
    uniq = np.unique(xs)
    print(f"[data] xs_path={args.xs_path}")
    print(f"[data] xs_full shape: {xs_full.shape}  (GW5 expects N episodes, T+1 states)")
    print(f"[data] xs used  shape: {xs.shape}       (dropping last state for chunking)")
    print(f"[data] state range in file: min={uniq.min()} max={uniq.max()} (unique={len(uniq)})")
    if uniq.min() < 0 or uniq.max() >= args.n_states:
        print(f"[warn] states exceed n_states={args.n_states}. You may need to set --n_states to {uniq.max()+1}")

    # Split by episodes (trajectories)
    rng = np.random.default_rng(args.seed)
    ids = np.arange(N)
    rng.shuffle(ids)
    n_train = int(round(args.train_frac * N))
    train_ids = ids[:n_train].tolist()
    val_ids = ids[n_train:].tolist()

    print(f"[split] train episodes: {len(train_ids)} | val episodes: {len(val_ids)}")
    print(f"[cfg] chunk_len={args.chunk_len}, H={args.H}, samples_per_traj={args.samples_per_traj}")
    print(f"[cfg] effective train samples per epoch ~ {len(train_ids)*args.samples_per_traj}")
    print(f"[cfg] effective val   samples per epoch ~ {max(1,len(val_ids))*max(100, args.samples_per_traj//2)}")

    # Build datasets
    ds_train = TrajChunkDataset(
        xs=xs,
        traj_ids=train_ids,
        n_states=args.n_states,
        chunk_len=args.chunk_len,
        horizon_H=args.H,
        samples_per_traj=args.samples_per_traj,
        seed=args.seed,
        fixed_starts=None,  # random starts
    )
    ds_val = TrajChunkDataset(
        xs=xs,
        traj_ids=val_ids if len(val_ids) > 0 else train_ids[: max(1, min(2, len(train_ids)))],
        n_states=args.n_states,
        chunk_len=args.chunk_len,
        horizon_H=args.H,
        samples_per_traj=max(100, args.samples_per_traj // 2),
        seed=args.seed + 1,
        fixed_starts=None,
    )

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

    # -----------------------------
    # Model
    # -----------------------------
    encoder = GTrXLEncoder(
        n_states=args.n_states,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        mem_len=args.mem_len,
        gate_bias_init=args.gate_bias_init,
    ).to(device)

    heads = PretrainHeads(d_model=args.d_model, n_states=args.n_states).to(device)

    opt = torch.optim.AdamW(
        list(encoder.parameters()) + list(heads.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print(f"[model] n_states={args.n_states}, d_model={args.d_model}, layers={args.n_layers}, heads={args.n_heads}, d_ff={args.d_ff}, dropout={args.dropout}")
    print(f"[opt] lr={args.lr}, weight_decay={args.weight_decay}, lambda_occ={args.lambda_occ}, grad_clip={args.grad_clip}")
    print(f"[dev] device={device}\n")

    # -----------------------------
    # Train loop
    # -----------------------------
    best_val = float("inf")
    best_path = ckpt_dir / "best.pt"

    for ep in range(1, args.epochs + 1):
        encoder.train()
        heads.train()

        for batch in dl_train:
            x = batch["x"].to(device)
            x_next = batch["x_next"].to(device)
            occ = batch["occ"].to(device)

            h, _ = encoder(x)
            ns_logits, occ_logits = heads(h)

            ns_loss = F.cross_entropy(
                ns_logits.reshape(-1, ns_logits.size(-1)),
                x_next.reshape(-1),
            )
            occ_loss = F.binary_cross_entropy_with_logits(occ_logits, occ)

            #loss = ns_loss + args.lambda_occ * occ_loss

            occ_w = args.lambda_occ * min(1.0, (ep + 1) / 10.0)
            loss = ns_loss + occ_w * occ_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(heads.parameters()), args.grad_clip)
            opt.step()

        train_m = eval_epoch(encoder, heads, dl_train, device, args.lambda_occ)
        val_m = eval_epoch(encoder, heads, dl_val, device, args.lambda_occ)

        print(
            f"[epoch {ep:03d}] "
            f"train loss={train_m['loss']:.4f} (ns={train_m['ns']:.4f}, occ={train_m['occ']:.4f}, acc={train_m['acc']:.3f}) | "
            f"val loss={val_m['loss']:.4f} (ns={val_m['ns']:.4f}, occ={val_m['occ']:.4f}, acc={val_m['acc']:.3f})"
        )

        # Save best checkpoint by val loss
        if val_m["loss"] < best_val:
            best_val = val_m["loss"]
            torch.save(
                {
                    "encoder": encoder.state_dict(),
                    "heads": heads.state_dict(),
                    "config": vars(args),
                    "val_loss": best_val,
                },
                best_path,
            )
            print(f"  [save] new best -> {best_path} (val_loss={best_val:.4f})")

    print(f"\n[done] best val_loss={best_val:.4f} @ {best_path}")

    # -----------------------------
    # Export embeddings for SWIRL / JAX
    # Important: we export for xs with shape (N, T) (same as used for training)
    # But SWIRL likely expects embeddings aligned to action times t=0..T-1.
    # We'll export h with shape (N, T, d_model).
    # -----------------------------
    # Save a temp xs.npz that contains the (N,T) xs we trained on (no last state)
    xs_npz = out_dir / "xs_gw5_for_embed.npz"
    np.savez_compressed(xs_npz, xs=xs)

    out_embed = out_dir / "h_gw5.npz"
    export_embeddings(
        ckpt_path=str(best_path),
        xs_path=str(xs_npz),
        out_path=str(out_embed),
        n_states=args.n_states,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        mem_len=args.mem_len,
        gate_bias_init=args.gate_bias_init,
        batch_size=64,
        device=str(device),
    )

    print(f"[ok] Embeddings exported to {out_embed} (key='h')")


if __name__ == "__main__":
    main()
