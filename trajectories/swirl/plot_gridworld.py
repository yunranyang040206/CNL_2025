"""
Visualize the reward function learned by the S-2 SWIRL model on the 5×5 gridworld.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import yaml
import matplotlib.patches as patches  # NEW: for small grey circles


# -------------------------------------------------------------
# Load config and extract HOME/WATER
# -------------------------------------------------------------

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def infer_home_water(cfg):
    """Extract HOME and WATER from YAML config."""
    rewards_cfg = cfg.get("rewards", {})

    # HOME
    state_based = rewards_cfg.get("state_based", {})
    if isinstance(state_based, dict) and len(state_based) > 0:
        home_state = int(next(iter(state_based.keys())))  # first key
    else:
        home_state = 0  # fallback top-left

    # WATER
    water_state = None
    for entry in rewards_cfg.get("sequence_based", []):
        if (
            isinstance(entry, dict)
            and entry.get("to_state") is not None
        ):
            water_state = int(entry["to_state"])

    if water_state is None:
        grid_n = cfg["grid"]["size"]
        water_state = grid_n * grid_n - 1  # fallback bottom-right

    return home_state, water_state


# -------------------------------------------------------------
# Extract S-2 reward from new_Rs
# -------------------------------------------------------------

def extract_state_reward(new_Rs, n_states):
    """
    Convert new_Rs into (K, n_states) reward.

    Expected format (which your files contain):
        new_Rs shape = (n_states, 1, K)

    Output:
        R_state[k, s]
    """
    arr = np.asarray(new_Rs)

    if arr.shape[0] == n_states and arr.shape[1] == 1:
        # shape (n_states, 1, K)
        return arr[:, 0, :].T  # → shape (K, n_states)

    elif arr.shape[0] != n_states:
        raise ValueError(
            f"Unexpected new_Rs shape {arr.shape}. "
            f"Expected (n_states,1,K) with n_states={n_states}."
        )

    else:
        raise ValueError(f"Cannot interpret new_Rs with shape {arr.shape}")


# -------------------------------------------------------------
# Main plotting function
# -------------------------------------------------------------

def plot_s2_rewards(dataset_name, seed, config_filename):
    """
    dataset_name: e.g. "default_20251124_135442"
    seed:        e.g. "12345"
    config_filename: e.g. "default.yaml"
    """

    # -------------------------------
    # Locate required files
    # -------------------------------

    # Config file:
    config_path = os.path.join("..", "configs", config_filename)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    # Learned S-2 SWIRL result:
    s2_result_path = os.path.join(
        "..", "output", "swirl_result", dataset_name, f"{seed}Long_NM_gw5_net2.npz"
    )
    if not os.path.exists(s2_result_path):
        raise FileNotFoundError(f"Learned S-2 reward file not found: {s2_result_path}")

    print(f"Loading learned S-2 reward from:\n  {s2_result_path}")

    # -------------------------------
    # Load config and determine grid
    # -------------------------------

    cfg = load_yaml(config_path)
    grid_n = cfg["grid"]["size"]
    n_states = grid_n * grid_n

    home_state, water_state = infer_home_water(cfg)
    home_r, home_c = divmod(home_state, grid_n)
    water_r, water_c = divmod(water_state, grid_n)

    # -------------------------------
    # Load learned reward
    # -------------------------------

    result = np.load(s2_result_path, allow_pickle=True)
    if "new_Rs" not in result:
        raise KeyError(f"`new_Rs` key not found in file {s2_result_path}")

    new_Rs = result["new_Rs"]
    R_state = extract_state_reward(new_Rs, n_states)  # → (K, n_states)
    K = R_state.shape[0]

    R_grid = R_state.reshape(K, grid_n, grid_n)

    # -------------------------------
    # Plot
    # -------------------------------

    fig, axes = plt.subplots(1, K, figsize=(5.5 * K, 6), squeeze=False)

    axes = axes[0]

    vmin = R_grid.min()
    vmax = R_grid.max()

    for k in range(K):
        ax = axes[k]

        mode_cmaps = ["Blues", "Reds", "Greens", "Purples"]
        cmap = mode_cmaps[k % len(mode_cmaps)]

        im = ax.imshow(
            R_grid[k],
            cmap=cmap,    
            origin="upper",
            vmin=vmin,
            vmax=vmax,
        )

        # Set coordinate limits and aspect
        ax.set_xlim(-0.5, grid_n - 0.5)
        ax.set_ylim(grid_n - 0.5, -0.5)
        ax.set_aspect("equal")

        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])


        ax.set_title(f"{dataset_name}\nS-2 reward – mode {k}", fontweight="bold", pad=10)
        for x in range(grid_n + 1):
            ax.axvline(x - 0.5, color="#eeeeee", lw=1)
            ax.axhline(x - 0.5, color="#eeeeee", lw=1)
        for r in range(grid_n):
            for c in range(grid_n):
                circle = patches.Circle((c, r), 0.1, color="#dddddd")
                ax.add_patch(circle)

        # HOME 
        ax.scatter(home_c, home_r, marker="s", s=140,
                   edgecolors="yellow", facecolors="none", linewidths=2)
        ax.text(home_c, home_r, "H", color="yellow",
                ha="center", va="center", fontsize=10)

        # WATER 
        ax.scatter(water_c, water_r, marker="o", s=140,
                   edgecolors="cyan", facecolors="none", linewidths=2)
        ax.text(water_c, water_r, "W", color="cyan",
                ha="center", va="center", fontsize=10)
        
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.10)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(f"Reward (mode {k})")

    plt.subplots_adjust(wspace=0.4, hspace=0.2)
    plt.tight_layout(rect=[0, 0, 0.98, 0.96])
    
    # Save PNG in swirl_result/<dataset_name>/
    out_png = os.path.join(
        "..", "output", "swirl_result", dataset_name,
        f"{dataset_name}_S2_rewardmap_{seed}.png"
    )
    fig.savefig(out_png, dpi=300)
    print(f"\nSaved PNG to:\n  {out_png}\n")


# -------------------------------------------------------------
# CLI
# -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize reward maps learned by the S-2 SWIRL model (5×5 gridworld)."
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="Name of dataset folder inside ../output/, e.g. default_20251124_135442",
    )
    parser.add_argument(
        "--seed",
        type=str,
        default="12345",
        help="Seed prefix used in swirl_result/<dataset>/<seed>_NM_gw5_net2.npz",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config YAML filename inside ../configs/, e.g. default.yaml",
    )

    args = parser.parse_args()
    plot_s2_rewards(args.dataset, args.seed, args.config)


if __name__ == "__main__":
    main()
