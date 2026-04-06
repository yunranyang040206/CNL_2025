"""
Persistence / Commitment Metric for CalMS21 SWIRL.

Purpose:
Test whether the inferred hidden mode captures how stable a behavior is once it begins.
This is a posthoc analysis only. It does not retrain SWIRL.

IMPORTANT:
This metric should be run primarily on the STANDARD / UNCOMPRESSED pipeline,
because the compressed pipeline removes consecutive self-transitions and therefore
destroys bout-duration information.
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import jax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flax import linen as nn

jax.config.update("jax_enable_x64", True)

C = 4
BEHAVIOR_NAMES = ["Attack", "Investigation", "Mount", "Other"]


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Persistence / Commitment Metric for CalMS21 SWIRL")
    p.add_argument("--K", type=int, default=2)
    p.add_argument("--seeds", type=int, nargs="+", default=[30])
    p.add_argument("--model", choices=["net1", "net2"], default="net2")
    p.add_argument("--compressed", action="store_true", help="not recommended; kept only for completeness")
    p.add_argument("--uncompressed", action="store_true", default=True,
                   help="use standard uncompressed dataset/results (recommended)")
    p.add_argument("--reward-extractor", choices=["expanded", "first_order"], default="expanded")
    p.add_argument("--bout-mode", choices=["onset", "majority"], default="onset")
    p.add_argument("--data-folder", default="../data/")
    p.add_argument("--results-folder", default="../results/")
    return p.parse_args()


# ---------------------------------------------------------------------
# Load raw label sequences
# ---------------------------------------------------------------------
def load_raw_sequences(data_folder, compressed=False):
    seq_name = "compressed_seqs.npy" if compressed else "seqs.npy"
    tp_name = "compressed_trans_probs.npy" if compressed else "trans_probs.npy"

    seqs = np.load(os.path.join(data_folder, seq_name))
    trans_probs = np.load(os.path.join(data_folder, tp_name))
    return seqs, trans_probs


# ---------------------------------------------------------------------
# Same reward network architecture
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
# Reward/policy extraction
# ---------------------------------------------------------------------
def extract_reward_policy(trans_probs, params, apply_fn, reward_extractor="expanded"):
    from caltech_analysis import get_reward_m, get_reward_nm
    from swirl_func import vinet_expand, vinet

    if reward_extractor == "expanded":
        reward = np.array(get_reward_m(trans_probs, params, apply_fn))
        policy, _, _ = vinet_expand(trans_probs, params, apply_fn)
    else:
        reward = np.array(get_reward_nm(trans_probs, params, apply_fn))
        policy, _, _ = vinet(trans_probs, params, apply_fn)

    return reward, np.array(policy)


# ---------------------------------------------------------------------
# Bout extraction from raw per-frame labels
# seqs has shape (N, Tframes). We use the frame labels directly.
# viterbi_zs corresponds to transitions, so we align a bout with the
# transition indices that span its frames.
# ---------------------------------------------------------------------
def extract_bouts_from_seqs(seqs, zs, bout_mode="onset"):
    """
    seqs: (N, Tframes)
    zs:   (N, Tframes-1) Viterbi hidden states aligned to transitions
    Returns a list of bouts:
      {
        "traj": n,
        "behavior": b,
        "start": s,
        "end": e,          # inclusive frame index
        "length": L,
        "mode": z
      }
    """
    N, Tframes = seqs.shape
    bouts = []

    for n in range(N):
        labels = seqs[n]
        ztraj = zs[n]
        start = 0

        for t in range(1, Tframes):
            if labels[t] != labels[start]:
                end = t - 1
                b = int(labels[start])
                L = end - start + 1

                # transitions covered by this bout:
                # from start to end-1, if length >= 2
                if L == 1:
                    # use the nearest available transition index
                    z_candidates = [ztraj[min(start, len(ztraj) - 1)]]
                else:
                    z_candidates = ztraj[start:end]

                if bout_mode == "onset":
                    z = int(z_candidates[0])
                else:
                    vals, counts = np.unique(z_candidates, return_counts=True)
                    z = int(vals[np.argmax(counts)])

                bouts.append({
                    "traj": n,
                    "behavior": b,
                    "start": start,
                    "end": end,
                    "length": L,
                    "mode": z,
                })
                start = t

        # final bout
        end = Tframes - 1
        b = int(labels[start])
        L = end - start + 1
        if L == 1:
            z_candidates = [ztraj[min(start, len(ztraj) - 1)]]
        else:
            z_candidates = ztraj[start:end]

        if bout_mode == "onset":
            z = int(z_candidates[0])
        else:
            vals, counts = np.unique(z_candidates, return_counts=True)
            z = int(vals[np.argmax(counts)])

        bouts.append({
            "traj": n,
            "behavior": b,
            "start": start,
            "end": end,
            "length": L,
            "mode": z,
        })

    return bouts


# ---------------------------------------------------------------------
# Summaries by behavior and mode
# ---------------------------------------------------------------------
def summarize_bouts(bouts, K):
    mean_len = np.full((K, C), np.nan)
    median_len = np.full((K, C), np.nan)
    count = np.zeros((K, C), dtype=int)

    for z in range(K):
        for b in range(C):
            lengths = [bt["length"] for bt in bouts if bt["mode"] == z and bt["behavior"] == b]
            count[z, b] = len(lengths)
            if len(lengths) > 0:
                mean_len[z, b] = np.mean(lengths)
                median_len[z, b] = np.median(lengths)

    return mean_len, median_len, count


# ---------------------------------------------------------------------
# Correlation helper across valid (z,b) cells
# ---------------------------------------------------------------------
def valid_corr(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan
    return np.corrcoef(x[mask], y[mask])[0, 1]


# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------
def save_heatmap(mat, title, outpath, fmt="{:.2f}", cmap="viridis"):
    fig, ax = plt.subplots(figsize=(6, max(2.8, 0.8 * mat.shape[0])), dpi=200)
    im = ax.imshow(mat, aspect="auto", cmap=cmap)
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


def save_scatter(x, y, labels, title, xlabel, ylabel, outpath):
    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=200)
    mask = np.isfinite(x) & np.isfinite(y)
    ax.scatter(x[mask], y[mask])

    for xi, yi, lab in zip(x[mask], y[mask], np.array(labels)[mask]):
        ax.annotate(lab, (xi, yi), fontsize=8, alpha=0.8)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    args = parse_args()
    compressed = args.compressed and not args.uncompressed

    if compressed:
        print("WARNING: compressed pipeline removes self-transitions; persistence results may be uninformative.")

    seqs, trans_probs = load_raw_sequences(args.data_folder, compressed=compressed)
    apply_fn = make_apply_fn(args.K)

    suffix = "compressed" if compressed else "standard"
    prefix = f"persistence_{suffix}_K{args.K}_{args.model}_seeds{'-'.join(map(str, args.seeds))}_{args.bout_mode}"

    results_dir = Path(args.results_folder)
    results_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    mean_runs = []
    median_runs = []
    reward_persist_runs = []
    policy_persist_runs = []
    count_runs = []

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

        # self-persistence scores
        reward_persist = np.array([[reward[z, b, b] for b in range(C)] for z in range(args.K)])
        policy_persist = np.array([[policy[z, b, b] for b in range(C)] for z in range(args.K)])

        bouts = extract_bouts_from_seqs(seqs, zs, bout_mode=args.bout_mode)
        mean_len, median_len, count = summarize_bouts(bouts, args.K)

        mean_runs.append(mean_len)
        median_runs.append(median_len)
        reward_persist_runs.append(reward_persist)
        policy_persist_runs.append(policy_persist)
        count_runs.append(count)

        for z in range(args.K):
            for b, bname in enumerate(BEHAVIOR_NAMES):
                rows.append({
                    "seed": seed,
                    "mode": z + 1,
                    "behavior": bname,
                    "reward_self_persistence": float(reward_persist[z, b]),
                    "policy_self_persistence": float(policy_persist[z, b]),
                    "mean_bout_length": float(mean_len[z, b]) if np.isfinite(mean_len[z, b]) else np.nan,
                    "median_bout_length": float(median_len[z, b]) if np.isfinite(median_len[z, b]) else np.nan,
                    "bout_count": int(count[z, b]),
                })

    mean_runs = np.array(mean_runs)
    median_runs = np.array(median_runs)
    reward_persist_runs = np.array(reward_persist_runs)
    policy_persist_runs = np.array(policy_persist_runs)
    count_runs = np.array(count_runs)

    mean_len_mean = np.nanmean(mean_runs, axis=0)
    median_len_mean = np.nanmean(median_runs, axis=0)
    reward_persist_mean = np.mean(reward_persist_runs, axis=0)
    policy_persist_mean = np.mean(policy_persist_runs, axis=0)
    count_mean = np.mean(count_runs, axis=0)

    reward_corr_mean = valid_corr(reward_persist_mean.flatten(), mean_len_mean.flatten())
    policy_corr_mean = valid_corr(policy_persist_mean.flatten(), mean_len_mean.flatten())
    reward_corr_median = valid_corr(reward_persist_mean.flatten(), median_len_mean.flatten())
    policy_corr_median = valid_corr(policy_persist_mean.flatten(), median_len_mean.flatten())

    # save csv
    csv_path = results_dir / f"{prefix}_per_run.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed", "mode", "behavior",
                "reward_self_persistence", "policy_self_persistence",
                "mean_bout_length", "median_bout_length", "bout_count"
            ]
        )
        writer.writeheader()
        writer.writerows(rows)

    # save heatmaps
    reward_pdf = results_dir / f"{prefix}_reward_self.pdf"
    policy_pdf = results_dir / f"{prefix}_policy_self.pdf"
    mean_pdf = results_dir / f"{prefix}_mean_bout_length.pdf"
    median_pdf = results_dir / f"{prefix}_median_bout_length.pdf"
    count_pdf = results_dir / f"{prefix}_bout_count.pdf"

    save_heatmap(reward_persist_mean, f"Reward self-persistence ({suffix}, {args.model})", reward_pdf, fmt="{:.3f}")
    save_heatmap(policy_persist_mean, f"Policy self-persistence ({suffix}, {args.model})", policy_pdf, fmt="{:.3f}")
    save_heatmap(mean_len_mean, f"Mean bout length by mode ({suffix}, {args.model})", mean_pdf, fmt="{:.2f}")
    save_heatmap(median_len_mean, f"Median bout length by mode ({suffix}, {args.model})", median_pdf, fmt="{:.2f}")
    save_heatmap(count_mean, f"Bout count by mode ({suffix}, {args.model})", count_pdf, fmt="{:.0f}")

    # save scatter plots
    labels = [f"h{z+1}-{BEHAVIOR_NAMES[b]}" for z in range(args.K) for b in range(C)]
    reward_scatter = results_dir / f"{prefix}_reward_vs_mean_scatter.pdf"
    policy_scatter = results_dir / f"{prefix}_policy_vs_mean_scatter.pdf"

    save_scatter(
        reward_persist_mean.flatten(),
        mean_len_mean.flatten(),
        labels,
        f"Reward self-persistence vs mean bout length ({suffix}, {args.model})",
        "reward self-persistence r_z(b,b)",
        "mean bout length",
        reward_scatter
    )
    save_scatter(
        policy_persist_mean.flatten(),
        mean_len_mean.flatten(),
        labels,
        f"Policy self-persistence vs mean bout length ({suffix}, {args.model})",
        "policy self-persistence pi_z(b|b)",
        "mean bout length",
        policy_scatter
    )

    # save summary
    summary_path = results_dir / f"{prefix}_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Persistence / Commitment Metric\n")
        f.write(f"dataset={suffix}\n")
        f.write(f"K={args.K}\n")
        f.write(f"model={args.model}\n")
        f.write(f"seeds={args.seeds}\n")
        f.write(f"reward_extractor={args.reward_extractor}\n")
        f.write(f"bout_mode={args.bout_mode}\n\n")

        f.write("Correlation with mean bout length\n")
        f.write(f"  reward_self_vs_mean_length: {reward_corr_mean:.4f}\n")
        f.write(f"  policy_self_vs_mean_length: {policy_corr_mean:.4f}\n\n")

        f.write("Correlation with median bout length\n")
        f.write(f"  reward_self_vs_median_length: {reward_corr_median:.4f}\n")
        f.write(f"  policy_self_vs_median_length: {policy_corr_median:.4f}\n\n")

        f.write("Per-mode, per-behavior summary\n")
        for z in range(args.K):
            f.write(f"\nh{z+1}\n")
            for b, bname in enumerate(BEHAVIOR_NAMES):
                f.write(
                    f"  {bname}: reward_self={reward_persist_mean[z,b]:.4f}, "
                    f"policy_self={policy_persist_mean[z,b]:.4f}, "
                    f"mean_len={mean_len_mean[z,b]:.3f}, "
                    f"median_len={median_len_mean[z,b]:.3f}, "
                    f"count={int(round(count_mean[z,b]))}\n"
                )

    print(f"Saved {csv_path}")
    print(f"Saved {summary_path}")
    print(f"Saved {reward_pdf}")
    print(f"Saved {policy_pdf}")
    print(f"Saved {mean_pdf}")
    print(f"Saved {median_pdf}")
    print(f"Saved {count_pdf}")
    print(f"Saved {reward_scatter}")
    print(f"Saved {policy_scatter}")


if __name__ == "__main__":
    main()