import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.patches as patches
from matplotlib.lines import Line2D

# ===========================
# CONFIGURATION
# ===========================
CONFIG = {
    "data_dir": "./output",
    "output_dir": "./plots",
    "grid_size": 5,
    "jitter": 0.08,
}

sns.set_theme(style="white", context="talk", font_scale=1.0)


def load_data():
    return {
        "xs": np.load(f"{CONFIG['data_dir']}/xs.npy", allow_pickle=True)[0],
        "zs": np.load(f"{CONFIG['data_dir']}/zs.npy", allow_pickle=True)[0],
        "RG": np.load(f"{CONFIG['data_dir']}/RG.npy"),
    }


def setup_grid(ax, title):
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(4.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontweight="bold", pad=10)
    for x in range(6):
        ax.axvline(x - 0.5, color="#eeeeee", lw=1)
        ax.axhline(x - 0.5, color="#eeeeee", lw=1)
    for r in range(5):
        for c in range(5):
            circle = patches.Circle((c, r), 0.1, color="#dddddd")
            ax.add_patch(circle)


# ===========================
# PLOT 1: REWARD MAPS
# ===========================
def plot_reward_schematic(data):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    setup_grid(ax1, "Mode 1: Water Seeking\n(Sparse Reward = 10)")
    ax1.arrow(0, 4, 3, 0, color="#e63946", width=0.05, head_width=0, linestyle="--")
    ax1.text(
        1.5,
        3.8,
        "Required History\n(Must traverse bottom)",
        color="#e63946",
        fontsize=10,
        ha="center",
    )
    ax1.arrow(3.2, 4, 0.6, 0, color="#e63946", width=0.08, head_width=0.3)
    rect = patches.Rectangle(
        (3.6, 3.6), 0.8, 0.8, linewidth=2, edgecolor="#e63946", facecolor="#ffcccc"
    )
    ax1.add_patch(rect)
    ax1.text(4, 4, "+10", color="#e63946", fontweight="bold", ha="center", va="center")

    setup_grid(ax2, "Mode 2: Exploration\n(Zero Reward)")
    ax2.text(
        2,
        2,
        "Uniform Reward (0.0)\nNo History Requirement",
        ha="center",
        va="center",
        color="#457b9d",
        style="italic",
    )

    rules = "SWITCHING RULES:\n1. Start Thirsty (Mode 1).\n2. If +10 Reward -> Satiated (Mode 2).\n3. Satiated 100 steps -> Thirsty (Mode 1)."
    fig.text(
        0.5,
        0.05,
        rules,
        ha="center",
        fontsize=12,
        bbox=dict(boxstyle="round", facecolor="#f0f0f0", edgecolor="none", pad=1),
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(f"{CONFIG['output_dir']}/1_reward_maps_and_rules.png")
    plt.close()


# ===========================
# PLOT 2: FULL TRAJECTORY (Explicit Labels)
# ===========================
def plot_full_trajectory(data):
    fig, ax = plt.subplots(figsize=(10, 10))
    setup_grid(ax, "Full 500-Step Trajectory\n(Highlighting Start of Thirst Cycles)")

    traj = data["xs"]
    modes = data["zs"]
    coords = np.array([divmod(s, 5) for s in traj])
    ys, xs = coords[:, 0], coords[:, 1]

    xs = xs + np.random.normal(0, 0.08, size=xs.shape)
    ys = ys + np.random.normal(0, 0.08, size=ys.shape)

    # Loop safely
    num_steps = min(len(traj) - 1, len(modes))

    for i in range(num_steps):
        color = "#e63946" if modes[i] == 1 else "#457b9d"
        alpha = 0.8 if modes[i] == 1 else 0.3
        width = 2.0 if modes[i] == 1 else 1.0
        zorder = 10 if modes[i] == 1 else 1
        ax.plot(
            [xs[i], xs[i + 1]],
            [ys[i], ys[i + 1]],
            color=color,
            alpha=alpha,
            lw=width,
            zorder=zorder,
        )

        # 1. Very First Start (Always label t=0)
        if i == 0:
            if modes[0] == 1:
                ax.text(
                    xs[0],
                    ys[0],
                    "t=0\nStart Thirsty",
                    fontsize=9,
                    fontweight="bold",
                    bbox=dict(
                        facecolor="#ffcccc", alpha=1.0, pad=0.5, edgecolor="#e63946"
                    ),
                    zorder=40,
                )
            else:
                ax.plot(xs[0], ys[0], "go", markersize=12, label="Start", zorder=20)

        # 2. Switches
        if i > 0 and modes[i] != modes[i - 1]:

            # Explore (0) -> Water (1) [THIRST ONSET]
            if modes[i - 1] == 0 and modes[i] == 1:
                label = f"t={i}\nThirsty"
                ax.text(
                    xs[i],
                    ys[i],
                    label,
                    fontsize=9,
                    fontweight="bold",
                    bbox=dict(
                        facecolor="white", alpha=0.9, pad=0.5, edgecolor="#e63946"
                    ),
                    zorder=40,
                    ha="right",
                    va="bottom",
                )

            # Water (1) -> Explore (0) [DRINKING DONE]
            if modes[i - 1] == 1 and modes[i] == 0:
                ax.plot(
                    xs[i],
                    ys[i],
                    marker="D",
                    color="gold",
                    markersize=14,
                    markeredgecolor="black",
                    mew=2,
                    zorder=30,
                )

    legend_elements = [
        Line2D([0], [0], color="#e63946", lw=3, label="Water (Seek)"),
        Line2D([0], [0], color="#457b9d", lw=3, label="Explore (Random)"),
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor="gold",
            markeredgecolor="black",
            markersize=12,
            label="Finished Drinking",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="white",
            markeredgecolor="#e63946",
            markersize=10,
            label="Thirst Onset",
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.savefig(f"{CONFIG['output_dir']}/2_full_trajectory.png")
    plt.close()


# ===========================
# PLOT 3: ZOOMED SEGMENTS (Precise Endpoint)
# ===========================
def plot_zoomed_segments(data):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    traj = data["xs"]
    modes = data["zs"]

    # 1. Identify the Switching Index
    # 'end_water' is the index in ZS where the mode becomes 0.
    # This corresponds to the state index where the Explore phase BEGINS.
    end_water = np.argmax(modes == 0)

    # This state is SHARED. It is the end of Water and Start of Explore.
    # So we use 'end_water' as the anchor for both.
    join_point_idx = end_water

    # Find end of Explore
    end_explore = join_point_idx + np.argmax(modes[join_point_idx:] == 1)

    def plot_seg(ax, start_idx, end_idx, title, color):
        setup_grid(ax, title)

        # Slice including the endpoint
        segment = traj[start_idx : end_idx + 1]

        coords = np.array([divmod(s, 5) for s in segment])
        ys, xs = coords[:, 0], coords[:, 1]
        xs = xs + np.random.normal(0, 0.06, size=xs.shape)
        ys = ys + np.random.normal(0, 0.06, size=ys.shape)

        ax.plot(xs, ys, color=color, lw=2.5, alpha=0.8, marker=".", markersize=5)

        # Text Labels
        ax.text(
            xs[0],
            ys[0],
            f"Start\nt={start_idx}",
            ha="right",
            fontsize=9,
            fontweight="bold",
        )
        ax.text(
            xs[-1],
            ys[-1],
            f"End\nt={end_idx}",
            ha="left",
            fontsize=9,
            fontweight="bold",
        )

        ax.plot(xs[0], ys[0], "go", markersize=8)

        if color == "#e63946":  # Water end
            ax.plot(
                xs[-1],
                ys[-1],
                marker="D",
                color="gold",
                markersize=12,
                markeredgecolor="black",
            )
        else:  # Explore end
            ax.plot(xs[-1], ys[-1], "k*", markersize=12)

    # Plot Water: 0 -> join_point
    plot_seg(ax1, 0, join_point_idx, "First 'Water' Cycle\n(Efficient Path)", "#e63946")

    # Plot Explore: join_point -> end_explore
    plot_seg(
        ax2,
        join_point_idx,
        end_explore,
        "First 'Explore' Cycle\n(Random Walk - 100 steps)",
        "#457b9d",
    )

    plt.savefig(f"{CONFIG['output_dir']}/3_zoomed_cycles.png")
    plt.close()


if __name__ == "__main__":
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    data = load_data()
    plot_reward_schematic(data)
    plot_full_trajectory(data)
    plot_zoomed_segments(data)
    print(f"Final plots saved to {CONFIG['output_dir']}/")
