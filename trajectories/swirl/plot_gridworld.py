"""
Plot GW5 rewards using the SAME visual style as your original plot_gridworld.py,
but plotting the *correct* learned reward (from saved R_state params) and the GT RG.npy.

What you will see:

A) 25x25 panels (history-dependent / state-pair reward)
   - Each row index is "current state s" (0..24)
   - Each column index is "previous state s_prev" (0..24)
   - Pixel value is a scalar reward for that (s, s_prev) pair

B) 5x5 panels (collapsed-to-state view)
   - For each current state s, we average over previous states:
        R_5x5[k, s] = mean_{s_prev} R_25x25[k, s, s_prev]
   - Then reshape s (0..24) into 5x5 for a gridworld-like visualization.

How state-action reward collapses to these maps:
  - The reward net outputs R(k, expanded_state, a) where expanded_state encodes (s, s_prev).
  - We reduce over actions using either mean or max:
        R_pair(k, expanded_state) = mean_a R(k, expanded_state, a)   (default)
    then reshape expanded_state -> (s, s_prev) -> 25x25.

Why trans_probs.npy is needed (unlike your old script):
  - Your old script plotted new_Rs, which already had 25 states.
  - Here we reconstruct *expanded* dynamics (625 expanded states) consistent with GW5,
    and we also use trans_probs to mask impossible (s_prev -> s) pairs when comparing to RG.

CLI example (your paths):
python plot_gridworld_style_fixed.py \
  --result_npz /home/yunran-yang/.../12345Long_NM_gw5_net1.npz \
  --trans_probs /home/yunran-yang/.../trans_probs.npy \
  --rg /home/yunran-yang/.../RG.npy \
  --config default.yaml

"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import yaml
import matplotlib.patches as patches

import jax
import jax.numpy as jnp
from flax import linen as nn

from gw5_analysis import get_reward_nm


# -------------------------------------------------------------
# Load config and extract HOME/WATER (same as your original)
# -------------------------------------------------------------
def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def infer_home_water(cfg):
    rewards_cfg = cfg.get("rewards", {})

    state_based = rewards_cfg.get("state_based", {})
    if isinstance(state_based, dict) and len(state_based) > 0:
        home_state = int(next(iter(state_based.keys())))
    else:
        home_state = 0

    water_state = None
    for entry in rewards_cfg.get("sequence_based", []):
        if isinstance(entry, dict) and entry.get("to_state") is not None:
            water_state = int(entry["to_state"])
            break

    if water_state is None:
        grid_n = cfg["grid"]["size"]
        water_state = grid_n * grid_n - 1

    return home_state, water_state


# -------------------------------------------------------------
# Reward net definition (copied from run_gw5.py)
# -------------------------------------------------------------
class MLP(nn.Module):
    subnet_size: int
    hidden_size: int
    output_size: int
    n_hidden: int
    C: int
    expand: bool = False

    def setup(self):
        self.subnet = nn.Dense(self.subnet_size)
        self.dense1 = nn.Dense(self.hidden_size)
        self.dense2 = nn.Dense(self.n_hidden)
        if self.expand:
            self.reshape_func = lambda x: jnp.tile(jnp.expand_dims(x, axis=-1), (1,) * (x.ndim) + (self.C,)) / self.C
        else:
            self.reshape_func = lambda x: x.reshape(*x.shape[:-1], self.C, self.C)

    def __call__(self, x):
        x = self.reshape_func(x)
        x = jax.vmap(self.subnet, in_axes=-1)(x)
        x = x.reshape(*x.shape[1:-1], x.shape[0] * x.shape[-1])
        x = self.dense1(x)
        x = nn.leaky_relu(x)
        x = self.dense2(x)
        x = nn.leaky_relu(x)
        x = jnp.expand_dims(x, axis=-1)
        x = jnp.tile(x, (1, self.output_size))  # (K, A)
        return x


# -------------------------------------------------------------
# Expanded transitions: expanded_state = s*C + s_prev
# -------------------------------------------------------------
def build_expanded_trans_probs(trans_probs: np.ndarray) -> np.ndarray:
    C, A, _ = trans_probs.shape
    new_tp = np.zeros((C * C, A, C * C), dtype=trans_probs.dtype)

    for s_prev in range(C):
        for s in range(C):
            idx = s * C + s_prev
            for a in range(A):
                nonzero = np.nonzero(trans_probs[s, a])[0]
                for s_prime in nonzero:
                    new_idx = s_prime * C + s
                    new_tp[idx, a, new_idx] = trans_probs[s, a, s_prime]
    return new_tp


# -------------------------------------------------------------
# Compute learned 25x25 reward from saved R_state params
# -------------------------------------------------------------
def compute_learned_reward_25x25(trans_probs, R_params, reduce_actions="mean"):
    """
    Returns:
      learned_pair: (K, 25, 25) where [k, s, s_prev]
      learned_raw:  (K, 625, A) raw reward net outputs (before action reduction)
    """
    C, A, _ = trans_probs.shape
    K = None

    # Rebuild reward net with training hyperparams
    model = MLP(subnet_size=4, hidden_size=16, output_size=A, n_hidden=2, C=C, expand=False)
    # n_hidden=2 is typical; if your run has different K, we will infer K from params output below.

    apply_fn = model.apply

    new_trans_probs = build_expanded_trans_probs(trans_probs)  # (625, A, 625)
    learned_raw = get_reward_nm(new_trans_probs, R_params, apply_fn)  # (K, 625, A) based on net output
    learned_raw = np.array(learned_raw)

    # Infer K safely
    K = learned_raw.shape[0]

    # Reduce over actions -> scalar per expanded_state
    if reduce_actions == "mean":
        learned_scalar = learned_raw.mean(axis=2)  # (K, 625)
    elif reduce_actions == "max":
        learned_scalar = learned_raw.max(axis=2)
    else:
        raise ValueError("reduce_actions must be 'mean' or 'max'")

    # Map expanded_state idx -> (s, s_prev)
    # idx = s*C + s_prev, so reshape (K, C, C) with axis1=s, axis2=s_prev
    learned_pair = learned_scalar.reshape(K, C, C)
    return learned_pair, learned_raw


def mask_impossible_pairs(trans_probs, M_25x25):
    """
    Mask impossible (s_prev -> s) pairs (set to NaN) using trans_probs.
    This matches the spirit of your RG masking in evaluation.
    """
    # invalid_transitions[s_prev, s] := (no action can take s_prev -> s)
    # In your earlier code, invalid_transitions built from trans_probs == 0 across actions.
    invalid = np.all(trans_probs == 0, axis=1)  # (C, C) indexed [s, s_prime]
    # We need invalid_pairs[s, s_prev] for M[s, s_prev].
    # If M index is (current s, previous s_prev), the impossible condition is:
    #   previous s_prev cannot transition to current s under any action => invalid[s_prev, s] == True
    invalid_pairs = invalid.T  # now [s_prime, s] -> [current s, prev s_prev]
    out = M_25x25.copy()
    out[:, invalid_pairs] = np.nan
    return out


# -------------------------------------------------------------
# Plotting utilities (matching your old style as closely as possible)
# -------------------------------------------------------------
def add_grid_and_circles(ax, grid_n, circle=True):
    for x in range(grid_n + 1):
        ax.axvline(x - 0.5, color="#eeeeee", lw=1)
        ax.axhline(x - 0.5, color="#eeeeee", lw=1)
    if circle:
        for r in range(grid_n):
            for c in range(grid_n):
                ax.add_patch(patches.Circle((c, r), 0.1, color="#dddddd"))

def mark_home_water(ax, home_r, home_c, water_r, water_c):
    ax.scatter(home_c, home_r, marker="s", s=140, edgecolors="yellow", facecolors="none", linewidths=2)
    ax.text(home_c, home_r, "H", color="yellow", ha="center", va="center", fontsize=10)
    ax.scatter(water_c, water_r, marker="o", s=140, edgecolors="cyan", facecolors="none", linewidths=2)
    ax.text(water_c, water_r, "W", color="cyan", ha="center", va="center", fontsize=10)


def plot_panel(ax, M, cmap, title, vmin, vmax, grid_n=None, add_grid=False, add_circles=False):
    im = ax.imshow(M, cmap=cmap, origin="upper", vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontweight="bold", pad=10)
    if grid_n is not None and add_grid:
        ax.set_xlim(-0.5, grid_n - 0.5)
        ax.set_ylim(grid_n - 0.5, -0.5)
        ax.set_aspect("equal")
        add_grid_and_circles(ax, grid_n, circle=add_circles)
    return im


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Plot GT and learned rewards with your original visual style.")
    parser.add_argument("--result_npz", type=str, required=True, help="Training output .npz containing new_R_state.")
    parser.add_argument("--trans_probs", type=str, required=True, help="Path to trans_probs.npy")
    parser.add_argument("--rg", type=str, required=True, help="Path to RG.npy")
    parser.add_argument("--config", type=str, required=False, default=None, help="Config YAML filename (or full path).")
    parser.add_argument("--reduce_actions", type=str, default="mean", choices=["mean", "max"])
    parser.add_argument("--out_png", type=str, default=None)
    args = parser.parse_args()

    # Load trans_probs + RG
    trans_probs = np.load(args.trans_probs)
    RG = np.load(args.rg)  # expected (K,25,25)
    C, A, _ = trans_probs.shape

    # Load R_state params
    result = np.load(args.result_npz, allow_pickle=True)
    if "new_R_state" not in result:
        raise KeyError("new_R_state not found in result npz. Available keys: " + str(list(result.keys())))
    R_params = result["new_R_state"].item() if (isinstance(result["new_R_state"], np.ndarray) and result["new_R_state"].dtype == object) else result["new_R_state"]

    # Compute learned reward in the correct space (K,25,25)
    learned_25x25, learned_raw = compute_learned_reward_25x25(trans_probs, R_params, reduce_actions=args.reduce_actions)

    # Mask impossible pairs in BOTH GT and learned (for a fair comparison)
    RG_masked = mask_impossible_pairs(trans_probs, RG)
    learned_masked = mask_impossible_pairs(trans_probs, learned_25x25)

    # Collapse to 5x5 by averaging over previous state axis
    # M[k, s, s_prev] -> mean over s_prev -> M_state[k, s] -> reshape 5x5
    RG_5x5 = np.nanmean(RG_masked, axis=2).reshape(RG.shape[0], 5, 5)
    learned_5x5 = np.nanmean(learned_masked, axis=2).reshape(learned_25x25.shape[0], 5, 5)

    # HOME/WATER from config (optional)
    home_r = home_c = water_r = water_c = None
    if args.config is not None:
        config_path = args.config
        if not os.path.exists(config_path):
            # if user passed filename like "default.yaml", try ../configs/
            config_path = os.path.join("..", "configs", args.config)
        if os.path.exists(config_path):
            cfg = load_yaml(config_path)
            grid_n = cfg["grid"]["size"]
            home_state, water_state = infer_home_water(cfg)
            home_r, home_c = divmod(home_state, grid_n)
            water_r, water_c = divmod(water_state, grid_n)

    K = RG.shape[0]
    mode_cmaps = ["Blues", "Reds", "Greens", "Purples"]

    # Figure layout: 2 rows (modes) x 4 columns:
    #   GT 25x25 | Learned 25x25 | GT 5x5 | Learned 5x5
    fig, axes = plt.subplots(K, 4, figsize=(5.5 * 4, 6 * K), squeeze=False)

    # Shared color limits per "type"
    # (A) 25x25
    finite_25 = np.concatenate([RG_masked[np.isfinite(RG_masked)], learned_masked[np.isfinite(learned_masked)]])
    if finite_25.size == 0:
        vmin25, vmax25 = 0.0, 1.0
    else:
        vmin25, vmax25 = float(np.min(finite_25)), float(np.max(finite_25))

    # (B) 5x5
    finite_5 = np.concatenate([RG_5x5[np.isfinite(RG_5x5)], learned_5x5[np.isfinite(learned_5x5)]])
    if finite_5.size == 0:
        vmin5, vmax5 = 0.0, 1.0
    else:
        vmin5, vmax5 = float(np.min(finite_5)), float(np.max(finite_5))

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    for k in range(K):
        cmap = mode_cmaps[k % len(mode_cmaps)]

        # --- GT 25x25
        ax = axes[k, 0]
        im = plot_panel(ax, RG_masked[k], cmap=cmap, title=f"GT RG (mode {k})\n25×25", vmin=vmin25, vmax=vmax25)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.10)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(f"Reward (GT, mode {k})")

        # --- Learned 25x25
        ax = axes[k, 1]
        im = plot_panel(ax, learned_masked[k], cmap=cmap, title=f"Learned (mode {k})\n25×25", vmin=vmin25, vmax=vmax25)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.10)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(f"Reward (learned, mode {k})")

        # --- GT 5x5 (your original grid style)
        ax = axes[k, 2]
        im = plot_panel(ax, RG_5x5[k], cmap=cmap, title=f"GT reduced (mode {k})\n5×5", vmin=vmin5, vmax=vmax5,
                        grid_n=5, add_grid=True, add_circles=True)
        if home_r is not None:
            mark_home_water(ax, home_r, home_c, water_r, water_c)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.10)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(f"Reward (GT reduced, mode {k})")

        # --- Learned 5x5
        ax = axes[k, 3]
        im = plot_panel(ax, learned_5x5[k], cmap=cmap, title=f"Learned reduced (mode {k})\n5×5", vmin=vmin5, vmax=vmax5,
                        grid_n=5, add_grid=True, add_circles=True)
        if home_r is not None:
            mark_home_water(ax, home_r, home_c, water_r, water_c)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.10)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(f"Reward (learned reduced, mode {k})")

    plt.subplots_adjust(wspace=0.4, hspace=0.35)
    plt.tight_layout(rect=[0, 0, 0.98, 0.96])

    if args.out_png is None:
        out_png = os.path.splitext(args.result_npz)[0] + "_GT_vs_learned.png"
    else:
        out_png = args.out_png

    fig.savefig(out_png, dpi=300)
    print(f"\nSaved PNG to:\n  {out_png}\n")


if __name__ == "__main__":
    main()