#!/usr/bin/env python3
"""
GW5 training script (paper-style SWIRL transitions) + h_t (GTrXL embeddings) for time-varying, mode-specific rewards.
  - Uses swirl_func.jaxnet_e_step_batch2 (expects logemit_list)
  - Uses swirl_func.trans_m_step_jax_optax
  - Uses swirl_func.emit_m_step_jaxnet_optax2 (expects one_hot_hs)
  - Uses swirl_func.pi0_m_step
  - Uses swirl_func.soft_vi_sa to build logemit_list from the reward net
"""
import os
import sys
import numpy as np

import jax
import jax.numpy as jnp
from jax import vmap
from jax.scipy.special import logsumexp as jax_logsumexp

import optax
from flax import linen as nn
from flax.training import train_state

# ---- import your algorithmic pieces from swirl_func.py
from swirl_func import (
    soft_vi_sa,
    jaxnet_e_step_batch2,
    trans_m_step_jax_optax,
    emit_m_step_jaxnet_optax2,
    pi0_m_step,
)

jax.config.update("jax_enable_x64", True)

# -----------------------------
# Args / paths
# -----------------------------
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0

DATASET = sys.argv[2] if len(sys.argv) > 2 else "gw5"

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

data_folder = os.path.join(BASE_DIR, "output")
save_folder = os.path.join(BASE_DIR, "output", "swirl_result", DATASET)

os.makedirs(save_folder, exist_ok=True)

# Your embedding folder
EMBED_DIR = "/home/yunran-yang/EngSci/Computational_Neuro_Lab/CNL_2025/trajectories/output/gtrxl_small_H5"

# -----------------------------
# Load dataset artifacts
# -----------------------------
trans_probs = np.load(os.path.join(data_folder, "trans_prob.npy"), allow_pickle=True)  # (S,A,S)
xs = np.load(os.path.join(data_folder, "xs.npy"), allow_pickle=True)[:200]
acs = np.load(os.path.join(data_folder, "acs.npy"), allow_pickle=True)[:200]
zs = np.load(os.path.join(data_folder, "zs.npy"), allow_pickle=True)[:200]  # optional but used for acc

S, A, S2 = trans_probs.shape
assert S == S2

xs = np.asarray(xs, dtype=int)
acs = np.asarray(acs, dtype=int)
zs = np.asarray(zs, dtype=int)

# align decision-time states with actions
if xs.shape[1] == acs.shape[1] + 1:
    xs_dec = xs[:, :-1]
else:
    xs_dec = xs[:, :acs.shape[1]]

# -----------------------------
# Load embeddings + verify alignment
# -----------------------------
h_npz = np.load(os.path.join(EMBED_DIR, "h_gw5.npz"), allow_pickle=True)
print("h_gw5.npz keys:", h_npz.files)
h_all = h_npz["h"] if "h" in h_npz.files else h_npz[h_npz.files[0]]  # (N,T_embed,H)

xs_npz = np.load(os.path.join(EMBED_DIR, "xs_gw5_for_embed.npz"), allow_pickle=True)
print("xs_gw5_for_embed.npz keys:", xs_npz.files)
xs_for_embed = xs_npz["xs"] if "xs" in xs_npz.files else xs_npz[xs_npz.files[0]]

n = min(xs_for_embed.shape[0], xs_dec.shape[0])
t_min = min(xs_for_embed.shape[1], xs_dec.shape[1])

same_prefix = np.all(xs_for_embed[:n, :t_min] == xs_dec[:n, :t_min])
same_shift = False
if xs_dec.shape[1] == xs_for_embed.shape[1] + 1:
    same_shift = np.all(xs_for_embed[:n, :] == xs_dec[:n, 1:1+xs_for_embed.shape[1]])
elif xs_for_embed.shape[1] == xs_dec.shape[1] + 1:
    same_shift = np.all(xs_for_embed[:n, 1:1+xs_dec.shape[1]] == xs_dec[:n, :])

print(f"[align] xs_dec: {xs_dec.shape}, xs_for_embed: {xs_for_embed.shape}")
print(f"[align] same_prefix(overlap)={same_prefix}, same_shift(off-by-one)={same_shift}")
if not (same_prefix or same_shift):
    raise ValueError("Embedding xs do NOT match xs_dec (even with off-by-one shift).")

# trim to common T
T = min(xs_dec.shape[1], acs.shape[1], zs.shape[1], h_all.shape[1])
xs_dec = xs_dec[:, :T]
acs = acs[:, :T]
zs = zs[:, :T]
h_all = h_all[:xs_dec.shape[0], :T]  # (N,T,H)

print(f"[trim] N={xs_dec.shape[0]}, T={T}, S={S}, A={A}, H={h_all.shape[-1]}")

# -----------------------------
# Train/test split (every 5th episode test)
# -----------------------------
test_indices = np.arange(0, xs_dec.shape[0], 5).astype(int)
train_indices = np.setdiff1d(np.arange(xs_dec.shape[0]), test_indices).astype(int)

train_xs, test_xs = xs_dec[train_indices], xs_dec[test_indices]
train_acs, test_acs = acs[train_indices], acs[test_indices]
train_zs, test_zs = zs[train_indices], zs[test_indices]
train_hs, test_hs = h_all[train_indices], h_all[test_indices]

# -----------------------------
# One-hot (JAX) helpers (matches your preference)
# -----------------------------
def one_hot_jax(z, K):
    z = jnp.atleast_1d(z).astype(int)
    shp = z.shape
    N = z.size
    zoh = jnp.zeros((N, K))
    zoh = zoh.at[jnp.arange(N), jnp.ravel(z)].set(1)
    zoh = jnp.reshape(zoh, shp + (K,))
    return zoh

def make_xoh(xs_int):
    '''one-hot state vector'''
    # (N,T) -> (N,T,1,S)
    return one_hot_jax(jnp.array(xs_int), S)[:, :, None, :]

def make_aoh(acs_int):
    # (N,T) -> (N,T,1,A)
    return one_hot_jax(jnp.array(acs_int), A)[:, :, None, :]

def make_hoh(hs_arr):
    # (N,T,H) -> (N,T,1,H)
    return jnp.array(hs_arr)[:, :, None, :]

train_xoh = make_xoh(train_xs)
test_xoh  = make_xoh(test_xs)
train_aoh = make_aoh(train_acs)
test_aoh  = make_aoh(test_acs)
train_hoh = make_hoh(train_hs)
test_hoh  = make_hoh(test_hs)

trans_probs_j = jnp.array(trans_probs)

# -----------------------------
# Reward network (paper-style emissions)
# Input: [onehot(s), h_t]  => (S+H)
# Output per state: (K*A)  -> reshape to (S,K,A)
# -----------------------------
K = 2  # GW5 has two modes

class RewardNet(nn.Module):
    hidden_size: int
    K: int
    A: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_size)(x)
        x = nn.leaky_relu(x)
        x = nn.Dense(self.hidden_size)(x)
        x = nn.leaky_relu(x)
        x = nn.Dense(self.K * self.A)(x)
        return x  # (S, K*A) when x is (S, S+H)

def create_reward_state(rng, input_size, hidden_size, K, A, lr):
    model = RewardNet(hidden_size=hidden_size, K=K, A=A)
    params = model.init(rng, jnp.ones((1, input_size)))["params"]
    tx = optax.adam(lr)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

# -----------------------------
# Build logemit_list (N,T,K,S,A) using swirl_func.soft_vi_sa
# -----------------------------
def build_logemit_list(R_state, hoh, trans_probs_j, discount=0.95, vi_iters=50):
    """
    hoh: (N,T,1,H)
    """
    S = trans_probs_j.shape[0]
    eyeS = jnp.eye(S)

    # squeeze to (N,T,H)
    hs = hoh[:, :, 0, :]

    def per_t(h_t):
        # build input (S, S+H)
        h_rep = jnp.repeat(h_t[None, :], S, axis=0)      # (S,H)
        inp = jnp.concatenate([eyeS, h_rep], axis=1)     # (S,S+H)
        out = R_state.apply_fn({"params": R_state.params}, inp)  # (S, K*A)
        out = out.reshape(S, K, A)                       # (S,K,A)
        r_ksa = jnp.transpose(out, (1, 0, 2))            # (K,S,A)

        # for each mode k, compute pi_k(s,a) via soft VI on r_sa
        pi_ksa = vmap(lambda r_sa: soft_vi_sa(trans_probs_j, r_sa, discount=discount, threshold=vi_iters))(r_ksa)
        return jnp.log(pi_ksa + 1e-20)                   # (K,S,A)

    def per_traj(h_TH):
        return vmap(per_t)(h_TH)                         # (T,K,S,A)

    # (N,T,K,S,A)
    return vmap(per_traj)(hs)

# -----------------------------
# EM training
# -----------------------------
def em_train(logpi0, log_Ps, Rs, R_state, iters=50, trans_iters=200, emit_iters=200):
    LL_list = []
    for it in range(iters):
        pi0 = jnp.exp(logpi0 - jax_logsumexp(logpi0))

        logemit_train = build_logemit_list(R_state, train_hoh, trans_probs_j, discount=0.95, vi_iters=50)

        # E-step (from swirl_func)
        gamma, xi, alpha = jaxnet_e_step_batch2(
            pi0,
            log_Ps,
            Rs,
            trans_probs_j,
            train_xoh,   # transitions condition on xoh
            train_xoh,   # emissions also use xoh (keep split but same)
            train_aoh,
            logemit_train
        )

        # dataset LL = sum over traj log p(x,a)
        LL = jnp.sum(jax_logsumexp(alpha[:, -1, :], axis=-1))
        LL_list.append(float(LL))
        print(it)
        print(LL)

        # M-step pi0
        logpi0 = pi0_m_step(gamma)

        # M-step transitions (from swirl_func)
        log_Ps, Rs = trans_m_step_jax_optax(
            log_Ps, Rs,
            (gamma, xi),
            train_xoh,
            num_iters=trans_iters,
            learning_rate=5e-3
        )

        # M-step emissions/reward net (from swirl_func)
        R_state = emit_m_step_jaxnet_optax2(
            R_state,
            trans_probs_j,
            gamma,
            train_xoh,
            train_aoh,
            train_hoh,
            num_iters=emit_iters,
            batch_size = 16,
            discount=0.95,
            vi_threshold=50,
            lr=3e-4
        )

    return logpi0, log_Ps, Rs, R_state, LL_list

# init params
rng = jax.random.PRNGKey(seed)
H = train_hs.shape[-1]
input_size = S + H
R_state = create_reward_state(rng, input_size, hidden_size=64, K=K, A=A, lr=3e-4)

import numpy.random as npr
npr.seed(seed)

logpi0_start = jnp.log(jnp.array([0.5, 0.5]))

Ps = 0.95 * np.eye(K) + 0.05 * npr.rand(K, K)
Ps /= Ps.sum(axis=1, keepdims=True)
log_Ps_start = jnp.log(jnp.array(Ps))

Rs_start = jnp.zeros((S, 1, K))  # this matches comp_log_transP's dot(x, Rs[:,0,:]) usage


new_logpi0, new_log_Ps, new_Rs, new_R_state, LL_list = em_train(
    logpi0_start, log_Ps_start, Rs_start, R_state,
    iters=50, trans_iters=200, emit_iters=200
)

# -----------------------------
# Quantitative evaluation (your requested 5 numbers)
# -----------------------------
def mode_acc(gamma, z_true):
    z_hat = np.argmax(np.array(gamma), axis=-1)

    # permutation-invariant for K=2
    acc = np.mean(z_hat == z_true)
    acc_flip = np.mean((1 - z_hat) == z_true)

    # helpful debug: class balance + confusion-ish counts
    p1 = np.mean(z_true == 1)
    print(f"[DEBUG acc] frac(z_true==1)={p1:.3f}, acc={acc:.3f}, acc_flip={acc_flip:.3f}")

    return float(max(acc, acc_flip))

def mode_metrics(gamma, z_true):
    z_hat = np.argmax(np.array(gamma), axis=-1)

    # permutation invariant for K=2
    acc = np.mean(z_hat == z_true)
    acc_flip = np.mean((1 - z_hat) == z_true)
    if acc_flip > acc:
        z_hat = 1 - z_hat
        acc = acc_flip

    # confusion
    tp = np.sum((z_hat == 1) & (z_true == 1))
    tn = np.sum((z_hat == 0) & (z_true == 0))
    fp = np.sum((z_hat == 1) & (z_true == 0))
    fn = np.sum((z_hat == 0) & (z_true == 1))

    tpr = tp / (tp + fn + 1e-9)  # recall for class 1
    tnr = tn / (tn + fp + 1e-9)  # recall for class 0
    bal_acc = 0.5 * (tpr + tnr)

    prec = tp / (tp + fp + 1e-9)
    f1 = 2 * prec * tpr / (prec + tpr + 1e-9)

    return acc, bal_acc, f1, (tp, fp, fn, tn)

# train acc
pi0 = jnp.exp(new_logpi0 - jax_logsumexp(new_logpi0))
logemit_train = build_logemit_list(new_R_state, train_hoh, trans_probs_j)
train_gamma, _, train_alpha = jaxnet_e_step_batch2(pi0, new_log_Ps, new_Rs, trans_probs_j,
                                                   train_xoh, train_xoh, train_aoh, logemit_train)
acc1 = mode_acc(train_gamma, train_zs)
acc, bal_acc, f1, conf = mode_metrics(train_gamma, train_zs)

# test acc + loglik
logemit_test = build_logemit_list(new_R_state, test_hoh, trans_probs_j)
test_gamma, _, test_alpha = jaxnet_e_step_batch2(pi0, new_log_Ps, new_Rs, trans_probs_j,
                                                 test_xoh, test_xoh, test_aoh, logemit_test)
test_acc1 = mode_acc(test_gamma, test_zs)

per_traj = float(jnp.mean(jax_logsumexp(test_alpha[:, -1, :], axis=-1)))
T = test_alpha.shape[1]
per_step = per_traj / float(T)

# reward correlation (time-averaged predicted reward vs RG)

best_corr1 = float("nan")
rg_path = os.path.join(data_folder, "RG.npy")
if os.path.exists(rg_path):
    RG = np.load(rg_path, allow_pickle=True)  # expect (K,S,A) or compatible
    RG = np.asarray(RG)

    # compute time-averaged predicted rewards:
    # for each (n,t): out is (S,K,A)
    S_ = S
    eyeS = jnp.eye(S_)

    def rewards_from_h(h_t):
        h_rep = jnp.repeat(h_t[None, :], S_, axis=0)
        inp = jnp.concatenate([eyeS, h_rep], axis=1)
        out = new_R_state.apply_fn({"params": new_R_state.params}, inp)  # (S,K*A)
        out = out.reshape(S_, K, A)
        return out  # (S,K,A)

    # (N,T,S,K,A)
    R_pred = vmap(vmap(rewards_from_h))(train_hoh[:, :, 0, :])
    # average over (n,t) -> (S,K,A)
    R_avg = np.array(jnp.mean(R_pred, axis=(0, 1)))
    # reorder to (K,S,A)
    R_avg = np.transpose(R_avg, (1, 0, 2))

    if RG.shape != R_avg.shape:
        # try to coerce if RG is (K,S,S) from older code (state-state reward); then correlation is not meaningful
        # We'll only compute corr when shapes match.
        best_corr1 = float("nan")
    else:
        best_corr1 = float(np.corrcoef(R_avg.flatten(), RG.flatten())[0, 1])
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
print("[DEBUG R_corr] RG shape:", None if RG is None else RG.shape)
print("[DEBUG R_corr] R_avg shape:", R_avg.shape)
print("[DEBUG R_corr] std RG:", None if RG is None else float(np.std(RG)))
print("[DEBUG R_corr] std R_avg:", float(np.std(R_avg)))

print("S1:",
      "acc", acc1,
      "[DEBUG metrics] acc=", acc, "bal_acc=", bal_acc, "f1=", f1, "conf(tp,fp,fn,tn)=", conf,
      "test_acc", test_acc1,
      "R_corr", best_corr1,
      "test_loglik_per_traj", per_traj, "test_loglik_per_step", per_step, "T", int(T))

# Save
out_path = os.path.join(save_folder, f"{seed}_gw5_ht_paperstyle_from_swirl_func.npz")
np.savez(out_path,
         new_logpi0=np.array(new_logpi0),
         new_log_Ps=np.array(new_log_Ps),
         new_Rs=np.array(new_Rs),
         LL_list=np.array(LL_list))
print("Saved:", out_path)
