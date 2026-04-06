"""
Escalation-Potential Metric for CalMS21 SWIRL.

Purpose:
Test whether the inferred hidden mode contains information about future
escalation into Attack beyond the current visible behavior label.

This is a pure posthoc script. It does not retrain SWIRL and does not
modify any existing training or analysis code.
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

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.model_selection import train_test_split

jax.config.update("jax_enable_x64", True)

C = 4
ATTACK = 0
BEHAVIOR_NAMES = ["Attack", "Investigation", "Mount", "Other"]


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Escalation-Potential Metric for CalMS21 SWIRL")
    p.add_argument("--K", type=int, default=2)
    p.add_argument("--seeds", type=int, nargs="+", default=[30])
    p.add_argument("--model", choices=["net1", "net2"], default="net2")
    p.add_argument("--compressed", action="store_true", default=True)
    p.add_argument("--uncompressed", action="store_true")
    p.add_argument("--reward-extractor", choices=["expanded", "first_order"], default="expanded")
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 5, 10])
    p.add_argument("--data-folder", default="../data/")
    p.add_argument("--results-folder", default="../results/")
    return p.parse_args()


# ---------------------------------------------------------------------
# Data loading: same convention as existing repo
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
    xs = trajs[:, :, 0]   # current label
    acs = trajs[:, :, 1]  # next label

    return xs, acs, trans_probs


# ---------------------------------------------------------------------
# Same network architecture as training
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
# Build escalation dataset
# ---------------------------------------------------------------------
def future_attack_within_k(acs, k):
    """
    acs: (N, T), where acs[:, t] is the immediate next behavior at time t
    Return y_k: (N, T), where y_k[n, t] = 1 if Attack appears in acs[n, t:t+k]
    """
    N, T = acs.shape
    y = np.zeros((N, T), dtype=int)
    for n in range(N):
        for t in range(T):
            end = min(T, t + k)
            y[n, t] = int(np.any(acs[n, t:end] == ATTACK))
    return y


def one_hot_behavior(xs):
    return np.eye(C)[xs]


def one_hot_mode(zs, K):
    return np.eye(K)[zs]


def build_probe_dataset(xs, acs, zs, reward, policy, k):
    """
    Keep only current non-attack states.
    Build:
      baseline features: current behavior only
      reward features: current behavior + reward-to-attack
      policy features: current behavior + policy-to-attack
      full features: current behavior + one-hot mode + reward + policy
    """
    y_k = future_attack_within_k(acs, k)

    reward_to_attack = reward[zs, xs, ATTACK]
    policy_to_attack = policy[zs, xs, ATTACK]

    mask = (xs != ATTACK)

    x_behavior = one_hot_behavior(xs)[mask]
    x_mode = one_hot_mode(zs, reward.shape[0])[mask]
    x_reward = reward_to_attack[mask][:, None]
    x_policy = policy_to_attack[mask][:, None]
    y = y_k[mask]

    X_baseline = x_behavior
    X_reward = np.concatenate([x_behavior, x_reward], axis=1)
    X_policy = np.concatenate([x_behavior, x_policy], axis=1)
    X_full = np.concatenate([x_behavior, x_mode, x_reward, x_policy], axis=1)

    return {
        "baseline": X_baseline,
        "reward": X_reward,
        "policy": X_policy,
        "full": X_full,
        "y": y,
    }


# ---------------------------------------------------------------------
# Fit transparent logistic probes
# ---------------------------------------------------------------------
def evaluate_probe(X, y, random_state=0):
    # stratified split for transparency and stability
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=random_state
    )

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_tr, y_tr)

    prob = clf.predict_proba(X_te)[:, 1]
    pred = (prob >= 0.5).astype(int)

    metrics = {
        "auc": roc_auc_score(y_te, prob),
        "ap": average_precision_score(y_te, prob),
        "f1": f1_score(y_te, pred),
        "positive_rate": float(y.mean()),
    }
    return metrics


# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------
def plot_metric_curves(horizons, curves, title, ylabel, outpath):
    fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
    for name, vals in curves.items():
        ax.plot(horizons, vals, marker="o", label=name)
    ax.set_xlabel("Prediction horizon k")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    args = parse_args()
    compressed = not args.uncompressed

    xs_all, acs_all, trans_probs = load_sequences(args.data_folder, compressed=compressed)
    apply_fn = make_apply_fn(args.K)

    results_dir = Path(args.results_folder)
    results_dir.mkdir(parents=True, exist_ok=True)

    suffix = "compressed" if compressed else "standard"
    prefix = f"escalation_{suffix}_K{args.K}_{args.model}_seeds{'-'.join(map(str, args.seeds))}"

    rows = []

    auc_curves = {"baseline": [], "reward": [], "policy": [], "full": []}
    ap_curves = {"baseline": [], "reward": [], "policy": [], "full": []}
    f1_curves = {"baseline": [], "reward": [], "policy": [], "full": []}

    for k in args.horizons:
        # collect across seeds, then average
        horizon_metrics = {name: [] for name in ["baseline", "reward", "policy", "full"]}

        for seed in args.seeds:
            result_name = f"{args.K}_{seed}_NM_caltech_{'compressed_' if compressed else ''}{args.model}.npz"
            result_path = results_dir / result_name
            if not result_path.exists():
                raise FileNotFoundError(f"Missing result file: {result_path}")

            data = np.load(result_path, allow_pickle=True)
            params = data["new_R_state"].item()
            zs_all = np.array(data["viterbi_zs"])

            reward, policy = extract_reward_policy(
                trans_probs, params, apply_fn,
                reward_extractor=args.reward_extractor
            )

            ds = build_probe_dataset(xs_all, acs_all, zs_all, reward, policy, k)

            for name in ["baseline", "reward", "policy", "full"]:
                metrics = evaluate_probe(ds[name], ds["y"], random_state=seed)
                horizon_metrics[name].append(metrics)

                rows.append({
                    "seed": seed,
                    "horizon": k,
                    "model_type": name,
                    **metrics,
                })

        for name in ["baseline", "reward", "policy", "full"]:
            auc_curves[name].append(np.mean([m["auc"] for m in horizon_metrics[name]]))
            ap_curves[name].append(np.mean([m["ap"] for m in horizon_metrics[name]]))
            f1_curves[name].append(np.mean([m["f1"] for m in horizon_metrics[name]]))

    # save csv
    csv_path = results_dir / f"{prefix}_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["seed", "horizon", "model_type", "auc", "ap", "f1", "positive_rate"]
        )
        writer.writeheader()
        writer.writerows(rows)

    # save plots
    auc_pdf = results_dir / f"{prefix}_auc.pdf"
    ap_pdf = results_dir / f"{prefix}_ap.pdf"
    f1_pdf = results_dir / f"{prefix}_f1.pdf"

    plot_metric_curves(args.horizons, auc_curves,
                       f"Escalation potential: AUC ({suffix}, {args.model})",
                       "AUC", auc_pdf)
    plot_metric_curves(args.horizons, ap_curves,
                       f"Escalation potential: AP ({suffix}, {args.model})",
                       "Average Precision", ap_pdf)
    plot_metric_curves(args.horizons, f1_curves,
                       f"Escalation potential: F1 ({suffix}, {args.model})",
                       "F1", f1_pdf)

    # save text summary
    summary_path = results_dir / f"{prefix}_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Escalation-Potential Metric\n")
        f.write(f"dataset={suffix}\n")
        f.write(f"K={args.K}\n")
        f.write(f"model={args.model}\n")
        f.write(f"seeds={args.seeds}\n")
        f.write(f"horizons={args.horizons}\n")
        f.write(f"reward_extractor={args.reward_extractor}\n\n")

        for metric_name, curves in [("AUC", auc_curves), ("AP", ap_curves), ("F1", f1_curves)]:
            f.write(metric_name + "\n")
            for name, vals in curves.items():
                pretty = ", ".join([f"k={k}:{v:.4f}" for k, v in zip(args.horizons, vals)])
                f.write(f"  {name}: {pretty}\n")
            f.write("\n")

    print(f"Saved {csv_path}")
    print(f"Saved {summary_path}")
    print(f"Saved {auc_pdf}")
    print(f"Saved {ap_pdf}")
    print(f"Saved {f1_pdf}")


if __name__ == "__main__":
    main()