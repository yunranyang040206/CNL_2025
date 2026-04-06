"""
Latent Social Drive Profiles for the CalMS21 SWIRL experiment.

This is a pure posthoc analysis script.
It does NOT modify training, and it does NOT require retraining.

Main use:
    python analyze_latent_social_drive_profiles.py --K 2 --seeds 30 --model net2 --compressed

What it computes:
1. Reward-drive profile per hidden mode:
       D_z^(b) = mean_{s != b} r_z(s, b)
2. Policy-drive profile per hidden mode:
       P_z^(b) = mean_{s != b} pi_z(b | s)
3. Empirical next-behavior frequency by hidden mode:
       Pr(a_t = b | z_t = z_hat)

By default, self-transitions s->s are excluded in the drive averages, because
the compressed CalMS21 pipeline removes consecutive self-transitions and the
existing repo heatmaps also mask the diagonal.
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flax import linen as nn

jax.config.update("jax_enable_x64", True)

C = 4
BEHAVIOR_NAMES = ["Attack", "Investigation", "Mount", "Other"]


# ---------------------------------------------------------------------
# 1) CLI
# ---------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Latent Social Drive Profiles for CalMS21 SWIRL")
    p.add_argument("--K", type=int, default=2, help="number of hidden modes")
    p.add_argument("--seeds", type=int, nargs="+", default=[30], help="one or more seeds")
    p.add_argument("--model", choices=["net1", "net2"], default="net2",
                   help="net2 = S-2, net1 = S-1")
    p.add_argument("--compressed", action="store_true", default=True,
                   help="use compressed dataset/results (default)")
    p.add_argument("--uncompressed", action="store_true",
                   help="override compressed and use standard dataset/results")
    p.add_argument("--reward-extractor", choices=["expanded", "first_order"], default="expanded",
                   help="expanded matches the current repo's evaluation convention")
    p.add_argument("--include-self", action="store_true",
                   help="include self-transitions s->s in drive averages")
    p.add_argument("--data-folder", default="../data/")
    p.add_argument("--results-folder", default="../results/")
    return p.parse_args()


# ---------------------------------------------------------------------
# 2) Rebuild the exact CalMS21 state/action arrays from saved seqs
#    This mirrors the existing run/analyze scripts so the analysis stays
#    aligned with the repo.
# ---------------------------------------------------------------------
def load_sequences(data_folder, compressed=True):
    seq_name = "compressed_seqs.npy" if compressed else "seqs.npy"
    tp_name = "compressed_trans_probs.npy" if compressed else "trans_probs.npy"

    seqs = np.load(os.path.join(data_folder, seq_name))
    trans_probs = np.load(os.path.join(data_folder, tp_name))

    T_traj = seqs.shape[1] - 1
    trajs = []
    for i in range(seqs.shape[0]):
        traj = []
        for j in range(seqs.shape[1] - 1):
            s = int(seqs[i, j])
            a = int(seqs[i, j + 1])
            if s < C and a < C:
                traj.append([s, a, 1, a])
        if len(traj) >= T_traj:
            trajs.append(traj[:T_traj])

    trajs = np.array(trajs)
    xs = trajs[:, :, 0]   # current behavior label
    acs = trajs[:, :, 1]  # next behavior label

    return xs, acs, trans_probs


# ---------------------------------------------------------------------
# 3) Rebuild the reward-network apply_fn
#    We define the same MLP architecture inline so this script stays
#    independent of create_train_state / optax imports.
# ---------------------------------------------------------------------
class MLP(nn.Module):
    subnet_size: int
    hidden_size: int
    output_size: int
    n_hidden: int
    expand: bool

    def setup(self):
        self.dense1 = nn.Dense(self.hidden_size)
        self.dense2 = nn.Dense(self.n_hidden * self.output_size)

    def __call__(self, x):
        x = self.dense1(x)
        x = nn.leaky_relu(x)
        x = self.dense2(x)
        return x.reshape((self.n_hidden, self.output_size))


def make_apply_fn(K):
    model = MLP(
        subnet_size=4,
        hidden_size=16,
        output_size=C,
        n_hidden=K,
        expand=False,
    )
    return model.apply


# ---------------------------------------------------------------------
# 4) Extract reward and policy
#    Default is "expanded" because the existing repo's analysis/evaluation
#    path uses vinet_expand / get_reward_m for the saved Caltech models.
# ---------------------------------------------------------------------
def extract_reward_policy(trans_probs, params, apply_fn, reward_extractor="expanded"):
    from caltech_analysis import get_reward_m, get_reward_nm
    from swirl_func import vinet_expand, vinet

    if reward_extractor == "expanded":
        reward = np.array(get_reward_m(trans_probs, params, apply_fn))  # (K, C, C)
        policy, _, _ = vinet_expand(trans_probs, params, apply_fn)
    else:
        reward = np.array(get_reward_nm(trans_probs, params, apply_fn))  # (K, C, C)
        policy, _, _ = vinet(trans_probs, params, apply_fn)

    return reward, np.array(policy)


# ---------------------------------------------------------------------
# 5) Compute latent social drive profiles
#    Main metric:
#      reward_drive[z, b] = mean_s r_z(s, b)
#      policy_drive[z, b] = mean_s pi_z(b | s)
#
#    For compressed data, excluding self-transitions is more faithful to
#    the repo because s->s transitions are removed and diagonals are not
#    interpreted in the existing reward heatmaps.
# ---------------------------------------------------------------------
def compute_drive_matrices(reward, policy, include_self=False):
    K, C1, C2 = reward.shape
    assert C1 == C2 == C

    valid = np.ones((C, C), dtype=bool)
    if not include_self:
        np.fill_diagonal(valid, False)

    reward_drive = np.zeros((K, C), dtype=float)
    policy_drive = np.zeros((K, C), dtype=float)

    for z in range(K):
        for b in range(C):
            state_mask = valid[:, b]
            reward_drive[z, b] = reward[z, state_mask, b].mean()
            policy_drive[z, b] = policy[z, state_mask, b].mean()

    return reward_drive, policy_drive


# ---------------------------------------------------------------------
# 6) Empirical next-behavior frequencies by hidden mode
#    Because gamma_t is not saved in the current result files, we use
#    Viterbi hard assignments z_hat_t from the existing npz outputs.
# ---------------------------------------------------------------------
def empirical_behavior_given_mode(acs, zs, K):
    empirical = np.full((K, C), np.nan)
    occupancy = np.zeros(K, dtype=float)

    for z in range(K):
        mask = (zs == z)
        occupancy[z] = mask.mean()
        denom = mask.sum()
        if denom > 0:
            for b in range(C):
                empirical[z, b] = np.mean(acs[mask] == b)

    return empirical, occupancy


# ---------------------------------------------------------------------
# 7) Save helpers
# ---------------------------------------------------------------------
def write_per_run_csv(path, rows):
    fieldnames = [
        "seed", "mode", "behavior",
        "reward_drive", "policy_drive",
        "empirical_freq", "occupancy"
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_heatmap(mat, title, outpath, vmin=None, vmax=None, cmap="viridis", fmt="{:.3f}"):
    fig, ax = plt.subplots(figsize=(6, max(2.8, 0.8 * mat.shape[0])), dpi=200)
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks(range(C))
    ax.set_xticklabels(BEHAVIOR_NAMES, rotation=30, ha="right")
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels([f"h{i+1}" for i in range(mat.shape[0])])

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isfinite(val):
                ax.text(j, i, fmt.format(val), ha="center", va="center",
                        color="white", fontsize=8)

    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.9)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def rank_summary(mat):
    # Returns descending mode ranking for each target behavior
    return {
        BEHAVIOR_NAMES[b]: [int(x) + 1 for x in np.argsort(-mat[:, b])]
        for b in range(mat.shape[1])
    }


# ---------------------------------------------------------------------
# 8) Main
# ---------------------------------------------------------------------
def main():
    args = parse_args()
    compressed = not args.uncompressed

    xs, acs, trans_probs = load_sequences(args.data_folder, compressed=compressed)
    apply_fn = make_apply_fn(args.K)

    suffix = "compressed" if compressed else "standard"
    prefix = f"drive_profiles_{suffix}_K{args.K}_{args.model}_seeds{'-'.join(map(str, args.seeds))}"

    results_dir = Path(args.results_folder)
    results_dir.mkdir(parents=True, exist_ok=True)

    per_run_rows = []
    reward_runs = []
    policy_runs = []
    empirical_runs = []
    occupancy_runs = []

    for seed in args.seeds:
        result_name = f"{args.K}_{seed}_NM_caltech_{'compressed_' if compressed else ''}{args.model}.npz"
        result_path = results_dir / result_name
        if not result_path.exists():
            raise FileNotFoundError(f"Missing result file: {result_path}")

        data = np.load(result_path, allow_pickle=True)
        params = data["new_R_state"].item()
        zs = np.array(data["viterbi_zs"])

        reward, policy = extract_reward_policy(
            trans_probs, params, apply_fn,
            reward_extractor=args.reward_extractor
        )
        reward_drive, policy_drive = compute_drive_matrices(
            reward, policy, include_self=args.include_self
        )
        empirical, occupancy = empirical_behavior_given_mode(acs, zs, args.K)

        reward_runs.append(reward_drive)
        policy_runs.append(policy_drive)
        empirical_runs.append(empirical)
        occupancy_runs.append(occupancy)

        for z in range(args.K):
            for b, bname in enumerate(BEHAVIOR_NAMES):
                per_run_rows.append({
                    "seed": seed,
                    "mode": z + 1,
                    "behavior": bname,
                    "reward_drive": float(reward_drive[z, b]),
                    "policy_drive": float(policy_drive[z, b]),
                    "empirical_freq": float(empirical[z, b]),
                    "occupancy": float(occupancy[z]),
                })

    reward_runs = np.array(reward_runs)
    policy_runs = np.array(policy_runs)
    empirical_runs = np.array(empirical_runs)
    occupancy_runs = np.array(occupancy_runs)

    reward_mean = reward_runs.mean(axis=0)
    reward_std = reward_runs.std(axis=0)

    policy_mean = policy_runs.mean(axis=0)
    policy_std = policy_runs.std(axis=0)

    empirical_mean = np.nanmean(empirical_runs, axis=0)
    empirical_std = np.nanstd(empirical_runs, axis=0)

    occupancy_mean = occupancy_runs.mean(axis=0)
    occupancy_std = occupancy_runs.std(axis=0)

    csv_path = results_dir / f"{prefix}_per_run.csv"
    write_per_run_csv(csv_path, per_run_rows)

    reward_pdf = results_dir / f"{prefix}_reward_mean.pdf"
    policy_pdf = results_dir / f"{prefix}_policy_mean.pdf"
    empirical_pdf = results_dir / f"{prefix}_empirical_mean.pdf"
    summary_txt = results_dir / f"{prefix}_summary.txt"

    save_heatmap(
        reward_mean,
        f"Reward-drive profiles ({suffix}, {args.model})",
        reward_pdf
    )
    save_heatmap(
        policy_mean,
        f"Policy-drive profiles ({suffix}, {args.model})",
        policy_pdf,
        vmin=0.0, vmax=1.0
    )
    save_heatmap(
        empirical_mean,
        f"Empirical next-behavior frequency by Viterbi mode ({suffix}, {args.model})",
        empirical_pdf,
        vmin=0.0, vmax=1.0
    )

    with open(summary_txt, "w") as f:
        f.write("Latent Social Drive Profiles\n")
        f.write(f"dataset={suffix}\n")
        f.write(f"K={args.K}\n")
        f.write(f"model={args.model}\n")
        f.write(f"seeds={args.seeds}\n")
        f.write(f"reward_extractor={args.reward_extractor}\n")
        f.write(f"include_self={args.include_self}\n\n")

        f.write("Mode occupancy (mean ± std across seeds)\n")
        for z in range(args.K):
            f.write(f"  h{z+1}: {occupancy_mean[z]:.4f} ± {occupancy_std[z]:.4f}\n")
        f.write("\n")

        f.write("Reward-drive matrix (mean ± std)\n")
        for z in range(args.K):
            parts = [
                f"{BEHAVIOR_NAMES[b]}={reward_mean[z, b]:.4f}±{reward_std[z, b]:.4f}"
                for b in range(C)
            ]
            f.write(f"  h{z+1}: " + ", ".join(parts) + "\n")
        f.write("\n")

        f.write("Policy-drive matrix (mean ± std)\n")
        for z in range(args.K):
            parts = [
                f"{BEHAVIOR_NAMES[b]}={policy_mean[z, b]:.4f}±{policy_std[z, b]:.4f}"
                for b in range(C)
            ]
            f.write(f"  h{z+1}: " + ", ".join(parts) + "\n")
        f.write("\n")

        f.write("Empirical next-behavior frequency by Viterbi mode (mean ± std)\n")
        for z in range(args.K):
            parts = [
                f"{BEHAVIOR_NAMES[b]}={empirical_mean[z, b]:.4f}±{empirical_std[z, b]:.4f}"
                for b in range(C)
            ]
            f.write(f"  h{z+1}: " + ", ".join(parts) + "\n")
        f.write("\n")

        f.write("Reward-drive rankings by behavior (best mode first)\n")
        for bname, ranks in rank_summary(reward_mean).items():
            f.write(f"  {bname}: {ranks}\n")
        f.write("\n")

        f.write("Policy-drive rankings by behavior (best mode first)\n")
        for bname, ranks in rank_summary(policy_mean).items():
            f.write(f"  {bname}: {ranks}\n")

    print(f"Saved {csv_path}")
    print(f"Saved {summary_txt}")
    print(f"Saved {reward_pdf}")
    print(f"Saved {policy_pdf}")
    print(f"Saved {empirical_pdf}")


if __name__ == "__main__":
    main()