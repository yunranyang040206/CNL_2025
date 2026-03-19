#!/usr/bin/env python3
"""
Plot GW5 ground-truth and learned reward landscapes in a way that matches data_generator.py.

- Uses SoftVI convention:
    Q[s,a] = R[s,a] + discount * sum_{s'} P[s,a,s'] * V[s']
    V[s]   = tau * logsumexp(Q[s,:] / tau)
- Avoids the incorrect einsum contraction that can flatten the value landscape.
- Can compare different state summaries from (K,S,A):
    * immediate_mean   : mean_a R(s,a)
    * immediate_max    : max_a R(s,a)
    * soft_value       : V(s) under generator SoftVI  [recommended default]
    * policy_expected  : sum_a pi(a|s) R(s,a)
    * chosen_action    : R(s, argmax_a pi(a|s))
    * action_gap       : top1_a R - top2_a R

Example usage:
python plot_gridworld.py \
  --result_npz /home/yunran-yang/EngSci/Computational_Neuro_Lab/CNL_2025/trajectories/output/swirl_result/gw5/0_gw5_amortize.npz \
  --rg /home/yunran-yang/EngSci/Computational_Neuro_Lab/CNL_2025/trajectories/data_new/RG_sa.npy \
  --trans_prob /home/yunran-yang/EngSci/Computational_Neuro_Lab/CNL_2025/trajectories/data_new/trans_prob.npy \
  --perm 1,0 \
  --summary soft_value \
  --tau 1.0 \
  --discount 0.95 \
  --vi_iters 50 \
  --water_state 24 \
  --out /home/yunran-yang/EngSci/Computational_Neuro_Lab/CNL_2025/trajectories/output/reward_heatmaps.png

"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp


EPS = 1e-12


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--result_npz", required=True, help="Path to .npz containing learned R_avg (K,S,A).")
    p.add_argument("--rg", default=None, help="Optional GT RG_sa.npy or state map with shape (K,S,A) or (K,S).")
    p.add_argument("--trans_prob", required=True, help="Path to transition matrix trans_prob.npy with shape (S,A,S).")
    p.add_argument(
        "--summary",
        choices=["immediate_mean", "immediate_max", "soft_value", "policy_expected", "chosen_action", "action_gap"],
        default="soft_value",
        help="How to convert (K,S,A) into a state map (K,S).",
    )
    p.add_argument("--discount", type=float, default=0.95)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--vi_iters", type=int, default=50)
    p.add_argument("--grid", type=int, default=5)
    p.add_argument("--perm", default=None, help='Optional learned-mode permutation, e.g. "1,0"')
    p.add_argument("--water_state", type=int, default=None)
    p.add_argument("--start_state", type=int, default=None)
    p.add_argument("--same_scale", action="store_true", help="Use one common color scale across all panels.")
    p.add_argument("--title", default=None)
    p.add_argument("--out", default="reward_maps_generator_exact.png")
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def apply_perm(arr: np.ndarray, perm_str: str | None) -> np.ndarray:
    if perm_str is None:
        return arr
    perm = [int(x) for x in perm_str.split(",")]
    if len(perm) != arr.shape[0]:
        raise ValueError(f"Permutation length {len(perm)} does not match K={arr.shape[0]}")
    return arr[perm]


def load_reward_from_npz(npz_path: str) -> np.ndarray:
    data = np.load(npz_path, allow_pickle=True)
    for key in ["R_avg", "R_learn", "Rs_learn", "reward_avg"]:
        if key in data:
            arr = np.asarray(data[key], dtype=float)
            if arr.ndim != 3:
                raise ValueError(f"{key} must have shape (K,S,A), got {arr.shape}")
            return arr
    raise KeyError(
        f"No learned reward found in {npz_path}. Expected one of: R_avg, R_learn, Rs_learn, reward_avg. "
        f"Keys found: {list(data.keys())}"
    )


def load_optional_reward(path: str | None, K: int, S: int) -> np.ndarray | None:
    if path is None:
        return None
    arr = np.asarray(np.load(path, allow_pickle=True), dtype=float)
    if arr.ndim == 2:
        if arr.shape != (K, S):
            raise ValueError(f"Reward file has shape {arr.shape}, expected {(K, S)}")
        return arr
    if arr.ndim == 3:
        if arr.shape[0] != K or arr.shape[1] != S:
            raise ValueError(f"Reward file has shape {arr.shape}, expected (K={K}, S={S}, A)")
        return arr
    raise ValueError(f"Reward file must have shape (K,S) or (K,S,A), got {arr.shape}")


def soft_vi_value_and_policy(trans_prob: np.ndarray, reward_sa: np.ndarray, discount: float, tau: float, vi_iters: int):
    """Exact SoftVI used by data_generator.py.

    trans_prob: (S,A,S)
    reward_sa : (S,A)
    returns   : V (S,), pi (S,A), Q (S,A)
    """
    tp = np.asarray(trans_prob, dtype=float)
    r = np.asarray(reward_sa, dtype=float)

    if tp.ndim != 3:
        raise ValueError(f"trans_prob must be (S,A,S), got {tp.shape}")
    if r.ndim != 2:
        raise ValueError(f"reward_sa must be (S,A), got {r.shape}")

    S, A, S2 = tp.shape
    if S != S2 or r.shape != (S, A):
        raise ValueError(f"Incompatible shapes: trans_prob={tp.shape}, reward_sa={r.shape}")

    V = np.zeros(S, dtype=float)
    for _ in range(vi_iters):
        EV = np.einsum("sak,k->sa", tp, V)  # exact: sum over next-state s'
        Q = r + discount * EV
        V = tau * logsumexp(Q / tau, axis=1)

    EV = np.einsum("sak,k->sa", tp, V)
    Q = r + discount * EV
    logpi = Q / tau - logsumexp(Q / tau, axis=1, keepdims=True)
    pi = np.exp(logpi)
    return V, pi, Q


def summarize_reward(R: np.ndarray, summary: str, trans_prob: np.ndarray, discount: float, tau: float, vi_iters: int) -> np.ndarray:
    if R.ndim == 2:
        return R
    if R.ndim != 3:
        raise ValueError(f"Reward must have shape (K,S) or (K,S,A), got {R.shape}")

    if summary == "immediate_mean":
        return np.mean(R, axis=-1)
    if summary == "immediate_max":
        return np.max(R, axis=-1)
    if summary == "action_gap":
        top2 = np.partition(R, kth=-2, axis=-1)[..., -2:]
        return top2[..., 1] - top2[..., 0]

    K, S, _ = R.shape
    out = np.zeros((K, S), dtype=float)
    for k in range(K):
        V, pi, _ = soft_vi_value_and_policy(trans_prob, R[k], discount=discount, tau=tau, vi_iters=vi_iters)
        if summary == "soft_value":
            out[k] = V
        elif summary == "policy_expected":
            out[k] = np.sum(pi * R[k], axis=-1)
        elif summary == "chosen_action":
            a_star = np.argmax(pi, axis=-1)
            out[k] = R[k, np.arange(S), a_star]
        else:
            raise ValueError(f"Unsupported summary '{summary}'")
    return out


def state_to_grid(Rs: np.ndarray, grid: int) -> np.ndarray:
    K, S = Rs.shape
    if S != grid * grid:
        raise ValueError(f"State count S={S} does not match grid={grid}")
    return Rs.reshape(K, grid, grid)


def mark_state(ax, s: int, grid: int, label: str, marker: str):
    r, c = divmod(int(s), grid)
    ax.scatter([c], [r], s=110, marker=marker, edgecolors="black", linewidths=1.2)
    ax.text(c + 0.10, r + 0.12, label, fontsize=9, weight="bold")


def draw_one(ax, arr2d: np.ndarray, title: str, grid: int, start_state: int | None, water_state: int | None, vmin=None, vmax=None):
    im = ax.imshow(arr2d, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(range(grid))
    ax.set_yticks(range(grid))
    if start_state is not None:
        mark_state(ax, start_state, grid, "Start", "o")
    if water_state is not None:
        mark_state(ax, water_state, grid, "Goal", "*")
    return im


def main():
    args = parse_args()

    trans_prob = np.asarray(np.load(args.trans_prob), dtype=float)
    learned = load_reward_from_npz(args.result_npz)
    learned = apply_perm(learned, args.perm)
    K, S, _ = learned.shape

    if trans_prob.shape[0] != S or trans_prob.shape[2] != S:
        raise ValueError(f"trans_prob shape {trans_prob.shape} incompatible with S={S}")

    gt = load_optional_reward(args.rg, K=K, S=S)
    learned_s = summarize_reward(learned, args.summary, trans_prob, args.discount, args.tau, args.vi_iters)
    gt_s = None if gt is None else summarize_reward(gt, args.summary, trans_prob, args.discount, args.tau, args.vi_iters)

    learned_g = state_to_grid(learned_s, args.grid)
    gt_g = None if gt_s is None else state_to_grid(gt_s, args.grid)

    if gt_g is None:
        fig, axes = plt.subplots(K, 1, figsize=(5.6, 4.6 * K), squeeze=False)
        axes = axes[:, 0]
        all_vals = learned_g[np.isfinite(learned_g)]
        vmin = vmax = None
        if args.same_scale and all_vals.size:
            vmin, vmax = float(all_vals.min()), float(all_vals.max())
        for k in range(K):
            im = draw_one(
                axes[k], learned_g[k], f"Learned mode {k} ({args.summary})", args.grid,
                args.start_state, args.water_state, vmin=vmin, vmax=vmax,
            )
            plt.colorbar(im, ax=axes[k], fraction=0.046, pad=0.04)
    else:
        fig, axes = plt.subplots(K, 2, figsize=(10.5, 4.6 * K), squeeze=False)
        vmin = vmax = None
        if args.same_scale:
            all_vals = np.concatenate([
                gt_g[np.isfinite(gt_g)].ravel(),
                learned_g[np.isfinite(learned_g)].ravel(),
            ])
            if all_vals.size:
                vmin, vmax = float(all_vals.min()), float(all_vals.max())
        for k in range(K):
            im0 = draw_one(
                axes[k, 0], gt_g[k], f"GT mode {k} ({args.summary})", args.grid,
                args.start_state, args.water_state, vmin=vmin, vmax=vmax,
            )
            plt.colorbar(im0, ax=axes[k, 0], fraction=0.046, pad=0.04)
            im1 = draw_one(
                axes[k, 1], learned_g[k], f"Learned mode {k} ({args.summary})", args.grid,
                args.start_state, args.water_state, vmin=vmin, vmax=vmax,
            )
            plt.colorbar(im1, ax=axes[k, 1], fraction=0.046, pad=0.04)

    if args.title:
        fig.suptitle(args.title, fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
    else:
        fig.tight_layout()

    out = Path(args.out)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Saved figure to: {out}")

    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
