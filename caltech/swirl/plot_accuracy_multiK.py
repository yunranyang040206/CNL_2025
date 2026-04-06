"""
Plot final train/test accuracy across K values for the compressed SWIRL models.

Accuracy is computed properly: at each time step t, the predicted action uses
only past observations a_0,...,a_{t-1} via the forward-algorithm hidden state
distribution. This is comparable to the LL/step metric.

Usage:
  python plot_accuracy_multiK.py [seed]

Output (../results/):
  compressed_multiK_{seed}_net1_accuracy.pdf
  compressed_multiK_{seed}_net2_accuracy.pdf
"""

import sys
import glob
import os
import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap
from functools import partial
from jax.scipy.special import logsumexp as jax_logsumexp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 30

C           = 4
folder      = '../data/'
save_folder = '../results/'

# ── load data ────────────────────────────────────────────────────────────────
seqs        = np.load(folder + 'compressed_seqs.npy')
trans_probs = np.load(folder + 'compressed_trans_probs.npy')

T_traj = seqs.shape[1] - 1
trajs  = []
for i in range(seqs.shape[0]):
    traj = []
    for j in range(seqs.shape[1] - 1):
        s, a = int(seqs[i, j]), int(seqs[i, j + 1])
        if s < C and a < C:
            traj.append([s, a])
    if len(traj) >= T_traj:
        trajs.append(traj[:T_traj])

trajs = np.array(trajs)
xs    = trajs[:, :, 0]   # current behavior, shape (N, T)
acs   = trajs[:, :, 1]   # next behavior,    shape (N, T)

# ── one-hot helpers ───────────────────────────────────────────────────────────
def one_hot_jax(z, K_):
    z   = jnp.atleast_1d(z).astype(int)
    shp = z.shape
    N   = z.size
    zoh = jnp.zeros((N, K_))
    zoh = zoh.at[jnp.arange(N), jnp.ravel(z)].set(1)
    return jnp.reshape(zoh, shp + (K_,))

one_hotx_partial = lambda xs_: one_hot_jax(xs_[:, None], C)
one_hota_partial = lambda acs_: one_hot_jax(acs_[:, None], C)

all_xohs = vmap(one_hotx_partial)(xs)
all_aohs = vmap(one_hota_partial)(acs)

# ── forward algorithm + predictive accuracy ───────────────────────────────────
from swirl_func import comp_ll_jax, comp_transP, vinet_expand
from caltech_models import MLP


def make_apply_fn(K_):
    model = MLP(subnet_size=4, hidden_size=16, output_size=C, n_hidden=K_, expand=False)
    return model.apply


def predictive_accuracy_traj(pi0, Ps, log_likes, pi_emit, xs_traj):
    """
    At each step t, predict a_t using only past observations a_0,...,a_{t-1}.

    pi0:       (K,)     initial hidden state distribution
    Ps:        (T-1, K, K)  per-step transition matrices
    log_likes: (T, K)   log P(actual a_t | z_t, s_t) for each hidden state
    pi_emit:   (K, C, C) emission probabilities pi_emit[k, s, a]
    xs_traj:   (T,)     current behavior indices
    """
    T = log_likes.shape[0]
    preds = np.empty(T, dtype=int)

    # t=0: predict using prior pi0 (no past actions seen yet)
    marginal = np.einsum('k,ka->a', pi0, pi_emit[:, xs_traj[0], :])
    preds[0] = int(np.argmax(marginal))

    # alpha: log unnormalized hidden state posterior after observing a_0
    alpha = np.log(pi0 + 1e-300) + log_likes[0]

    for t in range(1, T):
        # one-step-ahead prediction using only past (through alpha)
        m = alpha.max()
        pred_log_z = np.log(np.exp(alpha - m) @ Ps[t - 1] + 1e-300) + m  # (K,)

        # normalize to probabilities
        pred_z = np.exp(pred_log_z - pred_log_z.max())
        pred_z /= pred_z.sum()

        # marginal action distribution under predicted hidden states
        marginal = np.einsum('k,ka->a', pred_z, pi_emit[:, xs_traj[t], :])
        preds[t] = int(np.argmax(marginal))

        # update alpha with actual a_t
        alpha = pred_log_z + log_likes[t]

    return preds


def compute_accuracy_set(pi0, log_Ps, Rs, R_params, apply_fn, indices):
    """Compute predictive accuracy over a subset of trajectories."""
    pi_emit, _, _ = vinet_expand(trans_probs, R_params, apply_fn)
    pi_emit       = np.array(pi_emit)
    logemit       = jnp.log(jnp.array(pi_emit))

    subset_xohs = all_xohs[indices]
    subset_aohs = all_aohs[indices]
    subset_xs   = xs[indices]
    subset_acs  = acs[indices]

    # compute per-step emission log-likelihoods and transition matrices
    log_likes_all = np.array(vmap(partial(comp_ll_jax, logemit))(subset_xohs, subset_aohs))
    Ps_all        = np.array(vmap(partial(comp_transP, jnp.array(log_Ps), jnp.array(Rs)))(subset_xohs))

    correct = 0
    total   = 0
    for i in range(len(indices)):
        preds = predictive_accuracy_traj(pi0, Ps_all[i], log_likes_all[i], pi_emit, subset_xs[i])
        correct += int(np.sum(preds == subset_acs[i]))
        total   += len(preds)

    return correct / total


# ── scan all K files ──────────────────────────────────────────────────────────
def load_entries(model_tag):
    pattern = save_folder + f'*_{seed}_NM_caltech_compressed_{model_tag}.npz'
    found   = sorted(glob.glob(pattern),
                     key=lambda p: int(os.path.basename(p).split('_')[0]))
    entries = []
    for fpath in found:
        try:
            d        = np.load(fpath, allow_pickle=True)
            k_val    = int(d['K'])
            tr_idx   = d['train_indices']
            te_idx   = d['test_indices']
            R_params = d['new_R_state'].item()
            log_Ps   = np.array(d['new_log_Ps'])
            Rs       = np.array(d['new_Rs'])
            logpi0   = np.array(d['new_logpi0'])
            pi0      = np.array(jnp.exp(jnp.array(logpi0) - jax_logsumexp(jnp.array(logpi0))))

            apply_fn = make_apply_fn(k_val)

            print(f"  K={k_val}  computing train accuracy...", flush=True)
            train_acc = compute_accuracy_set(pi0, log_Ps, Rs, R_params, apply_fn, tr_idx)
            print(f"  K={k_val}  computing test accuracy...", flush=True)
            test_acc  = compute_accuracy_set(pi0, log_Ps, Rs, R_params, apply_fn, te_idx)

            print(f"  K={k_val}  train_acc={train_acc:.4f}  test_acc={test_acc:.4f}")
            entries.append((k_val, train_acc, test_acc))
        except Exception as e:
            print(f"  Skipping {fpath}: {e}")
    return entries


for model_tag, model_label in [('net1', 'S-1'), ('net2', 'S-2')]:
    print(f"\n=== {model_label} ({model_tag}) ===")
    entries = load_entries(model_tag)
    if not entries:
        print("  No results found.")
        continue

    k_vals     = [e[0] for e in entries]
    train_accs = [e[1] for e in entries]
    test_accs  = [e[2] for e in entries]

    x     = np.arange(len(k_vals))
    width = 0.35
    c_tr  = '#1f77b4'
    c_te  = '#ff7f0e'

    fig, ax = plt.subplots(figsize=(max(5, 2 * len(k_vals) + 1), 4), dpi=200)
    bars_tr = ax.bar(x - width / 2, train_accs, width, label='Train', color=c_tr)
    bars_te = ax.bar(x + width / 2, test_accs,  width, label='Test',  color=c_te)

    for bar in bars_tr:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    for bar in bars_te:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

    ax.axhline(y=0.25, color='gray', linestyle='--', linewidth=1.2, label='Random (25%)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'K={k}' for k in k_vals])
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Predictive train/test accuracy — compressed {model_label}, seed={seed}')
    ax.legend()
    ax.set_ylim(0, min(1.0, max(train_accs + test_accs) * 1.15))

    plt.tight_layout()
    path = save_folder + f'compressed_multiK_{seed}_{model_tag}_accuracy.pdf'
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")

print("\nDone.")
