"""
Load-only analysis script for the Caltech SWIRL experiment.

Loads pre-trained net1 / net2 results (no retraining), computes metrics,
prints a comparison table, and saves plots to ../results/.

Usage:
  python analyze_caltech.py [K] [seed]

Output files (../results/):
  {K}_{seed}_training_curves.pdf
  {K}_{seed}_Rsa_net1.pdf
  {K}_{seed}_Rsa_net2.pdf
  {K}_{seed}_segments.svg
"""

import sys
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

# ── CLI args ──────────────────────────────────────────────────────────────────
K    = int(sys.argv[1]) if len(sys.argv) > 1 else 2
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 30

# ── constants ─────────────────────────────────────────────────────────────────
C              = 4
BEHAVIOR_NAMES = ['Attack', 'Investigation', 'Mount', 'Other']
folder         = '../data/'
save_folder    = '../results/'

# ── reconstruct data (same logic as run_caltech.py) ───────────────────────────
seqs        = np.load(folder + 'seqs.npy')
trans_probs = np.load(folder + 'trans_probs.npy')

T_traj = seqs.shape[1] - 1   # = 1999 steps per sequence
trajs = []
for i in range(seqs.shape[0]):
    traj = []
    for j in range(seqs.shape[1] - 1):
        s, a = int(seqs[i, j]), int(seqs[i, j + 1])
        if s < C and a < C:
            traj.append([s, a, 1, a])
    if len(traj) >= T_traj:
        trajs.append(traj[:T_traj])

trajs = np.array(trajs)
xs    = trajs[:, :, 0]
acs   = trajs[:, :, 1]

def one_hot_jax(z, K_):
    z   = jnp.atleast_1d(z).astype(int)
    shp = z.shape
    N   = z.size
    zoh = jnp.zeros((N, K_))
    zoh = zoh.at[jnp.arange(N), jnp.ravel(z)].set(1)
    return jnp.reshape(zoh, shp + (K_,))

def one_hot_jax2(z, z_prev, K_):
    z   = z * K_ + z_prev
    z   = jnp.atleast_1d(z).astype(int)
    K2  = K_ * K_
    shp = z.shape
    N   = z.size
    zoh = jnp.zeros((N, K2))
    zoh = zoh.at[jnp.arange(N), jnp.ravel(z)].set(1)
    return jnp.reshape(zoh, shp + (K2,)), z

n_states  = C
n_actions = C

one_hotx_partial  = lambda xs_:           one_hot_jax(xs_[:, None],  n_states)
one_hotx2_partial = lambda xs_, xs_prev:  one_hot_jax2(xs_[:, None], xs_prev[:, None], n_states)
one_hota_partial  = lambda acs_:          one_hot_jax(acs_[:, None], n_actions)

all_xohs               = vmap(one_hotx_partial)(xs)
all_xohs2, all_xs2     = vmap(one_hotx2_partial)(xs, jnp.roll(xs, 1))
all_aohs               = vmap(one_hota_partial)(acs)

# ── load results ──────────────────────────────────────────────────────────────
net1 = np.load(save_folder + f'{K}_{seed}_NM_caltech_net1.npz', allow_pickle=True)
net2 = np.load(save_folder + f'{K}_{seed}_NM_caltech_net2.npz', allow_pickle=True)

K_saved = int(net1['K'])
if K_saved != K:
    raise ValueError(f"K mismatch: CLI K={K}, file K={K_saved}")

train_indices = net1['train_indices']
test_indices  = net1['test_indices']
N_train       = len(train_indices)
T             = all_xohs.shape[1]

# ── reconstruct apply_fn (no retraining) ─────────────────────────────────────
from caltech_models import MLP

def make_apply_fn(K_):
    model = MLP(subnet_size=4, hidden_size=16, output_size=C, n_hidden=K_, expand=False)
    return model.apply   # pure fn: apply_fn({'params': dict}, x) -> (K_, C)

apply_fn = make_apply_fn(K)

R_params1 = net1['new_R_state'].item()
R_params2 = net2['new_R_state'].item()

# ── BIC helper ────────────────────────────────────────────────────────────────
def count_params(K_, C_=4, input_size_=16, hidden_size_=16):
    pi0_params    = max(K_ - 1, 0)
    trans_params  = K_ * (K_ - 1)
    R_params_cnt  = K_ * C_
    mlp_params    = (input_size_ * hidden_size_ + hidden_size_) + (hidden_size_ * K_ * C_ + K_ * C_)
    return pi0_params + trans_params + R_params_cnt + mlp_params

def bic(total_ll, K_):
    n_params = count_params(K_)
    return -2.0 * float(total_ll) + n_params * np.log(N_train * T)

# ── metrics table ─────────────────────────────────────────────────────────────
print(f"\n{'Metric':<30} {'net1 (S-1)':>14} {'net2 (S-2)':>14}")
print("-" * 60)

for label, data in [('net1 (S-1)', net1), ('net2 (S-2)', net2)]:
    pass  # just to define scope

rows = [
    ('Train LL/step',  float(net1['train_ll']),       float(net2['train_ll'])),
    ('Test  LL/step',  float(net1['test_ll']),         float(net2['test_ll'])),
    ('Total train LL', float(net1['total_train_ll']),  float(net2['total_train_ll'])),
    ('BIC',            bic(net1['total_train_ll'], K), bic(net2['total_train_ll'], K)),
    ('# params',       float(count_params(K)),         float(count_params(K))),
]

for name, v1, v2 in rows:
    print(f"  {name:<28} {v1:>14.4f} {v2:>14.4f}")

# ── state occupancy + behavior frequency ─────────────────────────────────────
zs1 = net1['viterbi_zs']   # (N, T)
zs2 = net2['viterbi_zs']

print(f"\n{'State occupancy (net1)':}")
for k in range(K):
    frac = float(np.mean(zs1 == k))
    print(f"  h{k+1}: {frac:.3f}")

print(f"\n{'State occupancy (net2)':}")
for k in range(K):
    frac = float(np.mean(zs2 == k))
    print(f"  h{k+1}: {frac:.3f}")

print(f"\nBehavior frequency per state (net1)  [Attack / Invest / Mount / Other]")
for k in range(K):
    mask = (zs1 == k)
    freqs = [float(np.mean(acs[mask] == c)) for c in range(C)]
    print(f"  h{k+1}: " + "  ".join(f"{f:.3f}" for f in freqs))

print(f"\nBehavior frequency per state (net2)  [Attack / Invest / Mount / Other]")
for k in range(K):
    mask = (zs2 == k)
    freqs = [float(np.mean(acs[mask] == c)) for c in range(C)]
    print(f"  h{k+1}: " + "  ".join(f"{f:.3f}" for f in freqs))

# ── plots ─────────────────────────────────────────────────────────────────────

# 1. Training curves (train + test LL per epoch)
train_LL1 = list(net1['train_LL_list'])
test_LL1  = list(net1['test_LL_list'])
train_LL2 = list(net2['train_LL_list'])
test_LL2  = list(net2['test_LL_list'])

N_train = len(train_indices)
N_test  = len(test_indices)

# normalize to per-step LL for interpretable y-axis
train_LL1_ps = [ll / (N_train * T) for ll in train_LL1]
test_LL1_ps  = [ll / (N_test  * T) for ll in test_LL1]
train_LL2_ps = [ll / (N_train * T) for ll in train_LL2]
test_LL2_ps  = [ll / (N_test  * T) for ll in test_LL2]

fig, axes = plt.subplots(1, 2, figsize=(13, 4), dpi=200)
c1, c2 = '#1f77b4', '#ff7f0e'
SKIP = 5   # skip first iterations where the large initial drop occurs

all_curves = [train_LL1_ps, test_LL1_ps, train_LL2_ps, test_LL2_ps]

# panel 0: full view
ax = axes[0]
ax.plot(train_LL1_ps, color=c1, linestyle='-',  label='net1 train (S-1)')
ax.plot(test_LL1_ps,  color=c1, linestyle='--', label='net1 test  (S-1)')
ax.plot(train_LL2_ps, color=c2, linestyle='-',  label='net2 train (S-2)')
ax.plot(test_LL2_ps,  color=c2, linestyle='--', label='net2 test  (S-2)')
ax.axvline(x=49.5, color='gray', linestyle=':', linewidth=1, label='phase 1→2')
ax.set_xlabel('EM iteration'); ax.set_ylabel('LL / step')
ax.set_title(f'K={K}, seed={seed} (full)'); ax.legend(fontsize=7)

# panel 1: zoom past initial drop — tight y-axis on converged region
ax = axes[1]
ax.plot(train_LL1_ps, color=c1, linestyle='-',  label='net1 train (S-1)')
ax.plot(test_LL1_ps,  color=c1, linestyle='--', label='net1 test  (S-1)')
ax.plot(train_LL2_ps, color=c2, linestyle='-',  label='net2 train (S-2)')
ax.plot(test_LL2_ps,  color=c2, linestyle='--', label='net2 test  (S-2)')
ax.axvline(x=49.5, color='gray', linestyle=':', linewidth=1, label='phase 1→2')
ax.set_xlim(SKIP, len(train_LL1_ps) - 1)
ys_zoom = [v for c in all_curves for v in c[SKIP:]]
margin = (max(ys_zoom) - min(ys_zoom)) * 0.15 or 1e-5
ax.set_ylim(min(ys_zoom) - margin, max(ys_zoom) + margin)
ax.set_xlabel('EM iteration'); ax.set_ylabel('LL / step')
ax.set_title(f'K={K}, seed={seed} (iter {SKIP}+)'); ax.legend(fontsize=7)

plt.suptitle('Training curves', y=1.01)
plt.tight_layout()
path = save_folder + f'{K}_{seed}_training_curves.pdf'
plt.savefig(path, bbox_inches='tight')
print(f"\nSaved {path}")
plt.close()

# 2. Reward heatmaps
from caltech_analysis import get_reward_m, get_reward_nm

def normalize(data):
    mn, mx = np.nanmin(data), np.nanmax(data)
    return (data - mn) / (mx - mn + 1e-8)

def plot_reward_heatmap(reward_arr, title, fname):
    reward_arr = np.array(reward_arr).reshape((K, C, C))
    ncols = max(K, 1)
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4), dpi=200)
    if K == 1:
        axes = [axes]
    for j in range(K):
        ax = axes[j]
        plot_data = reward_arr[j].copy()
        np.fill_diagonal(plot_data, np.nan)
        im = ax.imshow(normalize(plot_data), cmap='viridis', aspect='auto')
        ax.set_xticks(range(C))
        ax.set_xticklabels(BEHAVIOR_NAMES, rotation=30, ha='right', fontsize=8)
        ax.set_yticks(range(C))
        ax.set_yticklabels(BEHAVIOR_NAMES, fontsize=8)
        ax.set_xlabel('action (next behavior)')
        ax.set_ylabel('state (current behavior)')
        ax.set_title(f'Hidden state {j + 1}')
        ax.grid(False)
        plt.colorbar(im, ax=ax)
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(fname, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Saved {fname}")

# net1 uses get_reward_m (vinet_expand / S-1)
reward1 = get_reward_m(trans_probs, R_params1, apply_fn)
plot_reward_heatmap(reward1,
    f'Reward R(state, action) — net1 (S-1), K={K}',
    save_folder + f'{K}_{seed}_Rsa_net1.pdf')

# net2: both networks have input_size=16, so use get_reward_m for both
reward2 = get_reward_m(trans_probs, R_params2, apply_fn)
plot_reward_heatmap(reward2,
    f'Reward R(state, action) — net2 (S-2), K={K}',
    save_folder + f'{K}_{seed}_Rsa_net2.pdf')

# 3. Segmentation raster (net1 top, net2 bottom)
fig, axes = plt.subplots(2, 1, figsize=(14, 5), dpi=200)
for ax, zs, label in [(axes[0], zs1, f'net1 (S-1)'), (axes[1], zs2, f'net2 (S-2)')]:
    ax.imshow(zs + 1, aspect='auto', cmap='inferno', vmin=0, vmax=K + 1)
    colors = plt.cm.magma(np.linspace(0, 1, K + 1))
    for idx in range(K):
        ax.plot([], [], color=colors[idx + 1], label=f'h{idx + 1}')
    ax.legend(loc='upper right', fontsize=7)
    ax.set_xlabel('time step')
    ax.set_ylabel('trajectory')
    ax.set_title(f'Hidden-state segmentation — {label} (K={K})')
    ax.grid(False)
plt.tight_layout()
path = save_folder + f'{K}_{seed}_segments.svg'
plt.savefig(path, bbox_inches='tight')
plt.close()
print(f"Saved {path}")

# ── multi-K comparison plot ───────────────────────────────────────────────────
# Scan results folder for all K values available for this seed, then plot
# per-step train and test LL curves together on one figure per model (net1/net2).
import glob as _glob
import os as _os

def _load_multi_K(model_tag, seed_):
    pattern = save_folder + f'*_{seed_}_NM_caltech_{model_tag}.npz'
    found = sorted(_glob.glob(pattern),
                   key=lambda p: int(_os.path.basename(p).split('_')[0]))
    entries = []
    for fpath in found:
        try:
            d = np.load(fpath, allow_pickle=True)
            k_val = int(d['K'])
            n_tr  = len(d['train_indices'])
            n_te  = len(d['test_indices'])
            t_len = T  # same trajectory length for all runs
            tll   = [ll / (n_tr * t_len) for ll in d['train_LL_list']]
            tell  = [ll / (n_te * t_len) for ll in d['test_LL_list']]
            entries.append((k_val, tll, tell))
        except Exception:
            pass
    return entries

for model_tag, model_label in [('net1', 'S-1'), ('net2', 'S-2')]:
    entries = _load_multi_K(model_tag, seed)
    if len(entries) < 2:
        continue   # nothing interesting to compare
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(entries)))

    all_mk_curves = [c for _, tll, tell in entries for c in (tll, tell)]
    n_iters_mk = len(entries[0][1])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=200)

    # panel 0: full
    ax = axes[0]
    for (k_val, tll, tell), col in zip(entries, colors):
        ax.plot(tll,  color=col, linestyle='-',  label=f'K={k_val} train')
        ax.plot(tell, color=col, linestyle='--', label=f'K={k_val} test')
    ax.axvline(x=49.5, color='gray', linestyle=':', linewidth=1)
    ax.set_xlabel('EM iteration'); ax.set_ylabel('LL / step')
    ax.set_title(f'Multi-K — {model_label}, seed={seed} (full)')
    ax.legend(fontsize=7, ncol=2)

    # panel 1: skip first SKIP iters, tight ylim
    ax = axes[1]
    for (k_val, tll, tell), col in zip(entries, colors):
        ax.plot(tll,  color=col, linestyle='-',  label=f'K={k_val} train')
        ax.plot(tell, color=col, linestyle='--', label=f'K={k_val} test')
    ax.axvline(x=49.5, color='gray', linestyle=':', linewidth=1)
    ax.set_xlim(SKIP, n_iters_mk - 1)
    ys_mk = [v for c in all_mk_curves for v in c[SKIP:]]
    margin_mk = (max(ys_mk) - min(ys_mk)) * 0.15 or 1e-5
    ax.set_ylim(min(ys_mk) - margin_mk, max(ys_mk) + margin_mk)
    ax.set_xlabel('EM iteration'); ax.set_ylabel('LL / step')
    ax.set_title(f'Multi-K — {model_label}, seed={seed} (iter {SKIP}+)')
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    path = save_folder + f'multiK_{seed}_{model_tag}_comparison.pdf'
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")

print("\nDone.")
