"""
Run SWIRL on the Caltech Mouse Social Behavior dataset (CalMS21 Task 1).

Behavior classes (C = 4):
  0: attack  |  1: investigation  |  2: mount  |  3: other

Two model variants are trained, mirroring the spontda pipeline:
  S-1  (net1): first-order model  -- vinet_expand, input = C one-hot
  S-2  (net2): second-order model -- vinet,        input = C×C one-hot

Results are saved to ../results/:
  K_seed_NM_caltech_net1.npz
  K_seed_NM_caltech_net2.npz

Usage:
  python run_caltech.py [K] [seed]
"""

import sys
import numpy as np
import numpy.random as npr

import jax
import jax.numpy as jnp
from jax import vmap
from functools import partial
from jax.scipy.special import logsumexp as jax_logsumexp

jax.config.update("jax_enable_x64", True)

# ── CLI args ──────────────────────────────────────────────────────────────────
K    = int(sys.argv[1]) if len(sys.argv) > 1 else 2
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 30

# ── constants ─────────────────────────────────────────────────────────────────
C               = 4   # behavior classes
BEHAVIOR_NAMES  = ['Attack', 'Investigation', 'Mount', 'Other']
folder          = '../data/'
save_folder     = '../results/'

# ── load data ─────────────────────────────────────────────────────────────────
seqs        = np.load(folder + 'seqs.npy')           # (n_videos, MAX_LEN)
trans_probs = np.load(folder + 'trans_probs.npy')    # (C, C, C)

# Build (state, action) trajectories.
# state  = seqs[i, t]
# action = seqs[i, t+1]   (next behavior = chosen action)
T_traj = seqs.shape[1] - 1   # = 1999 steps per sequence (s,a pairs)
trajs = []
for i in range(seqs.shape[0]):
    traj = []
    for j in range(seqs.shape[1] - 1):
        s, a = int(seqs[i, j]), int(seqs[i, j + 1])
        if s < C and a < C:
            traj.append([s, a, 1, a])          # [state, action, reward, obs]
    if len(traj) >= T_traj:
        trajs.append(traj[:T_traj])

trajs = np.array(trajs)
print(f"Trajectories: {trajs.shape[0]}  (length {T_traj} each)")

xs  = trajs[:, :, 0]   # (N, T) current states
acs = trajs[:, :, 1]   # (N, T) actions = next states

# ── train / test split ────────────────────────────────────────────────────────
test_indices  = np.arange(0, xs.shape[0], 5)
train_indices = np.setdiff1d(np.arange(xs.shape[0]), test_indices)

test_xs,  test_acs  = xs[test_indices],  acs[test_indices]
train_xs, train_acs = xs[train_indices], acs[train_indices]
print(f"Train: {len(train_indices)}   Test: {len(test_indices)}")

n_states, n_actions, _ = trans_probs.shape   # (C, C, C)

# ── one-hot helpers ───────────────────────────────────────────────────────────
def one_hot_jax(z, K):
    z   = jnp.atleast_1d(z).astype(int)
    shp = z.shape
    N   = z.size
    zoh = jnp.zeros((N, K))
    zoh = zoh.at[jnp.arange(N), jnp.ravel(z)].set(1)
    return jnp.reshape(zoh, shp + (K,))

def one_hot_jax2(z, z_prev, K):
    """Second-order one-hot: encodes (current, previous) state pair."""
    z   = z * K + z_prev
    z   = jnp.atleast_1d(z).astype(int)
    K2  = K * K
    shp = z.shape
    N   = z.size
    zoh = jnp.zeros((N, K2))
    zoh = zoh.at[jnp.arange(N), jnp.ravel(z)].set(1)
    return jnp.reshape(zoh, shp + (K2,)), z

one_hotx_partial  = lambda xs:            one_hot_jax(xs[:, None],  n_states)
one_hotx2_partial = lambda xs, xs_prev:   one_hot_jax2(xs[:, None], xs_prev[:, None], n_states)
one_hota_partial  = lambda acs:           one_hot_jax(acs[:, None], n_actions)

train_xohs               = vmap(one_hotx_partial)(train_xs)
train_xohs2, train_xs2   = vmap(one_hotx2_partial)(train_xs, jnp.roll(train_xs, 1))
train_aohs               = vmap(one_hota_partial)(train_acs)

all_xohs                 = vmap(one_hotx_partial)(xs)
all_xohs2, all_xs2       = vmap(one_hotx2_partial)(xs, jnp.roll(xs, 1))
all_aohs                 = vmap(one_hota_partial)(acs)

test_xohs                = vmap(one_hotx_partial)(test_xs)
test_xohs2, test_xs2     = vmap(one_hotx2_partial)(test_xs, jnp.roll(test_xs, 1))
test_aohs                = vmap(one_hota_partial)(test_acs)

# ── load ARHMM initialisation ─────────────────────────────────────────────────
arhmm_params  = np.load(folder + f'{K}_{seed}_arhmm_caltech.npz', allow_pickle=True)
logpi0_start  = arhmm_params['logpi0_start']
log_Ps_start  = arhmm_params['log_Ps_start']
Rs_start      = arhmm_params['Rs_start']

# ── import SWIRL functions ────────────────────────────────────────────────────
from swirl_func import (
    pi0_m_step,
    trans_m_step_jax_jaxopt,
    emit_m_step_jaxnet_optax2,
    emit_m_step_jaxnet_optax2_expand,
    jaxnet_e_step_batch2,
    jaxnet_e_step_batch,
)

# ── neural-network reward model ───────────────────────────────────────────────
from caltech_models import MLP, create_train_state

rng            = jax.random.PRNGKey(0)
input_size     = C * C      # 16
hidden_size    = 16
learning_rate  = 5e-3

R_state  = create_train_state(rng, 4, learning_rate, K, input_size, hidden_size, C)
R_state2 = create_train_state(rng, 4, learning_rate, K, input_size, hidden_size, C)

# ── expanded (second-order) transition matrix ─────────────────────────────────
# new_trans_probs[s*C+s_prev, a, s'*C+s] = trans_probs[s, a, s']
n_state, n_action, _ = trans_probs.shape
new_trans_probs = np.zeros((n_state * n_state, n_action, n_state * n_state))
for s_prev in range(n_state):
    for s in range(n_state):
        for a in range(n_action):
            for s_prime in range(n_state):
                if trans_probs[s, a, s_prime] > 0:
                    new_trans_probs[s * n_state + s_prev, a,
                                    s_prime * n_state + s] = trans_probs[s, a, s_prime]

# ── EM training: second-order model (net2) ────────────────────────────────────
def em_train_net2(logpi0, log_Ps, Rs, R_state, n_iter=100,
                   init=True, trans=True, emit=True):
    train_LL_list = []
    test_LL_list  = []
    for i in range(n_iter):
        print(f"  iter {i}", flush=True)
        pi0 = jnp.exp(logpi0 - jax_logsumexp(logpi0))

        all_gamma, all_xi, all_alphas = jaxnet_e_step_batch2(
            pi0, log_Ps, Rs, R_state,
            new_trans_probs, train_xohs, train_xohs2, train_aohs)

        train_ll = jnp.sum(jax_logsumexp(all_alphas[:, -1], axis=-1))

        _, _, test_alphas = jaxnet_e_step_batch2(
            pi0, log_Ps, Rs, R_state,
            new_trans_probs, test_xohs, test_xohs2, test_aohs)
        test_ll = jnp.sum(jax_logsumexp(test_alphas[:, -1], axis=-1))

        print(f"    Train LL = {train_ll:.2f}   Test LL = {test_ll:.2f}", flush=True)

        new_logpi0 = pi0_m_step(all_gamma) if init else logpi0

        if trans:
            new_log_Ps, new_Rs = trans_m_step_jax_jaxopt(
                log_Ps, Rs, (all_gamma, all_xi), jnp.array(train_xohs))
        else:
            new_log_Ps, new_Rs = log_Ps, Rs

        if emit:
            new_R_state = emit_m_step_jaxnet_optax2_expand(
                R_state, jnp.array(trans_probs), all_gamma,
                jnp.array(train_xohs), jnp.array(train_aohs), num_iters=800)
            new_R_state = emit_m_step_jaxnet_optax2(
                new_R_state, jnp.array(new_trans_probs), all_gamma,
                jnp.array(train_xohs2), jnp.array(train_aohs), num_iters=200)
        else:
            new_R_state = R_state

        train_LL_list.append(float(train_ll))
        test_LL_list.append(float(test_ll))
        logpi0, log_Ps, Rs, R_state = new_logpi0, new_log_Ps, new_Rs, new_R_state

    return logpi0, log_Ps, Rs, R_state, train_LL_list, test_LL_list

# ── EM training: first-order model (net1) ─────────────────────────────────────
def em_train_net1(logpi0, log_Ps, Rs, R_state, n_iter=100,
                   init=True, trans=True, emit=True):
    train_LL_list = []
    test_LL_list  = []
    for i in range(n_iter):
        print(f"  iter {i}", flush=True)
        pi0 = jnp.exp(logpi0 - jax_logsumexp(logpi0))

        all_gamma, all_xi, all_alphas = jaxnet_e_step_batch(
            pi0, log_Ps, Rs, R_state,
            trans_probs, train_xohs, train_aohs)

        train_ll = jnp.sum(jax_logsumexp(all_alphas[:, -1], axis=-1))

        _, _, test_alphas = jaxnet_e_step_batch(
            pi0, log_Ps, Rs, R_state,
            trans_probs, test_xohs, test_aohs)
        test_ll = jnp.sum(jax_logsumexp(test_alphas[:, -1], axis=-1))

        print(f"    Train LL = {train_ll:.2f}   Test LL = {test_ll:.2f}", flush=True)

        new_logpi0 = pi0_m_step(all_gamma) if init else logpi0

        if trans:
            new_log_Ps, new_Rs = trans_m_step_jax_jaxopt(
                log_Ps, Rs, (all_gamma, all_xi), jnp.array(train_xohs))
        else:
            new_log_Ps, new_Rs = log_Ps, Rs

        if emit:
            new_R_state = emit_m_step_jaxnet_optax2_expand(
                R_state, jnp.array(trans_probs), all_gamma,
                jnp.array(train_xohs), jnp.array(train_aohs), num_iters=800)
        else:
            new_R_state = R_state

        train_LL_list.append(float(train_ll))
        test_LL_list.append(float(test_ll))
        logpi0, log_Ps, Rs, R_state = new_logpi0, new_log_Ps, new_Rs, new_R_state

    return logpi0, log_Ps, Rs, R_state, train_LL_list, test_LL_list

# ── run ───────────────────────────────────────────────────────────────────────
from jax.lib import xla_bridge
print(f"\nBackend: {xla_bridge.get_backend().platform}\n")

# S-2: second-order model
print("=== S-2 (net2): second-order model ===")
print("--- phase 1: emit only ---")
lp2, lP2, R2, Rs2, LL2a_train, LL2a_test = em_train_net2(
    jnp.array(logpi0_start), jnp.array(log_Ps_start), jnp.array(Rs_start),
    R_state, n_iter=50, init=False, trans=False)

print("--- phase 2: full EM ---")
lp2, lP2, R2, Rs2, LL2b_train, LL2b_test = em_train_net2(
    jnp.array(lp2), jnp.array(lP2), jnp.array(R2),
    Rs2, n_iter=50)
LL2_train = LL2a_train + LL2b_train
LL2_test  = LL2a_test  + LL2b_test

jnp.savez(save_folder + f'{K}_{seed}_NM_caltech_net2.npz',
          new_logpi0=lp2, new_log_Ps=lP2, new_Rs=R2,
          new_R_state=Rs2.params, train_LL_list=LL2_train, test_LL_list=LL2_test)
print(f"Saved {K}_{seed}_NM_caltech_net2.npz\n")

# S-1: first-order model
print("=== S-1 (net1): first-order model ===")
print("--- phase 1: emit only ---")
lp1, lP1, R1, Rs1, LL1a_train, LL1a_test = em_train_net1(
    jnp.array(logpi0_start), jnp.array(log_Ps_start), jnp.array(Rs_start),
    R_state2, n_iter=50, init=False, trans=False)

print("--- phase 2: full EM ---")
lp1, lP1, R1, Rs1, LL1b_train, LL1b_test = em_train_net1(
    jnp.array(lp1), jnp.array(lP1), jnp.array(R1),
    Rs1, n_iter=50)
LL1_train = LL1a_train + LL1b_train
LL1_test  = LL1a_test  + LL1b_test

# ── post-training evaluation ──────────────────────────────────────────────────
from swirl_func import vinet_expand, comp_ll_jax, comp_transP, _viterbi_JAX, forward

def comp_LLloss(pi0, trans_Ps, lls):
    alphas = vmap(partial(forward, jnp.array(pi0)))(trans_Ps, lls)
    return float(jnp.sum(jax_logsumexp(alphas[:, -1], axis=-1)))

def eval_net1(logpi0, log_Ps, Rs, R_params, apply_fn):
    pi0      = jnp.exp(logpi0 - jax_logsumexp(logpi0))
    pi, _, _ = vinet_expand(trans_probs, R_params, apply_fn)
    logemit  = jnp.log(pi)
    lls_all  = vmap(partial(comp_ll_jax, logemit))(all_xohs,  all_aohs)
    lls_test = vmap(partial(comp_ll_jax, logemit))(test_xohs, test_aohs)
    Ps_all   = vmap(partial(comp_transP, jnp.array(log_Ps), jnp.array(Rs)))(all_xohs)
    Ps_test  = vmap(partial(comp_transP, jnp.array(log_Ps), jnp.array(Rs)))(test_xohs)
    zs       = vmap(partial(_viterbi_JAX, pi0))(Ps_all, lls_all)
    lls_train = lls_all[train_indices]; Ps_train = Ps_all[train_indices]
    total_train = comp_LLloss(pi0, Ps_train, lls_train)
    total_test  = comp_LLloss(pi0, Ps_test,  lls_test)
    T = all_xohs.shape[1]
    return (total_train / (len(train_indices) * T),
            total_test  / (len(test_indices)  * T),
            total_train, np.array(zs))

def eval_net2(logpi0, log_Ps, Rs, R_params, apply_fn):
    pi0      = jnp.exp(logpi0 - jax_logsumexp(logpi0))
    pi, _, _ = vinet_expand(trans_probs, R_params, apply_fn)
    logemit  = jnp.log(pi)
    lls_all  = vmap(partial(comp_ll_jax, logemit))(all_xohs,  all_aohs)
    lls_test = vmap(partial(comp_ll_jax, logemit))(test_xohs, test_aohs)
    Ps_all   = vmap(partial(comp_transP, jnp.array(log_Ps), jnp.array(Rs)))(all_xohs)
    Ps_test  = vmap(partial(comp_transP, jnp.array(log_Ps), jnp.array(Rs)))(test_xohs)
    zs       = vmap(partial(_viterbi_JAX, pi0))(Ps_all, lls_all)
    lls_train = lls_all[train_indices]; Ps_train = Ps_all[train_indices]
    total_train = comp_LLloss(pi0, Ps_train, lls_train)
    total_test  = comp_LLloss(pi0, Ps_test,  lls_test)
    T = all_xohs.shape[1]
    return (total_train / (len(train_indices) * T),
            total_test  / (len(test_indices)  * T),
            total_train, np.array(zs))

print("=== Post-training evaluation ===")
train_ll1, test_ll1, total_ll1, zs1 = eval_net1(lp1, lP1, R1, Rs1.params, Rs1.apply_fn)
print(f"[net1] Train LL/step: {train_ll1:.4f}   Test LL/step: {test_ll1:.4f}")
train_ll2, test_ll2, total_ll2, zs2 = eval_net2(lp2, lP2, R2, Rs2.params, Rs2.apply_fn)
print(f"[net2] Train LL/step: {train_ll2:.4f}   Test LL/step: {test_ll2:.4f}")

common = dict(train_indices=train_indices, test_indices=test_indices,
              K=np.array(K), seed=np.array(seed))

jnp.savez(save_folder + f'{K}_{seed}_NM_caltech_net1.npz',
          new_logpi0=lp1, new_log_Ps=lP1, new_Rs=R1, new_R_state=Rs1.params,
          train_LL_list=LL1_train, test_LL_list=LL1_test,
          train_ll=np.array(train_ll1), test_ll=np.array(test_ll1),
          total_train_ll=np.array(total_ll1), viterbi_zs=zs1, **common)
print(f"Saved {K}_{seed}_NM_caltech_net1.npz")

jnp.savez(save_folder + f'{K}_{seed}_NM_caltech_net2.npz',
          new_logpi0=lp2, new_log_Ps=lP2, new_Rs=R2, new_R_state=Rs2.params,
          train_LL_list=LL2_train, test_LL_list=LL2_test,
          train_ll=np.array(train_ll2), test_ll=np.array(test_ll2),
          total_train_ll=np.array(total_ll2), viterbi_zs=zs2, **common)
print(f"Saved {K}_{seed}_NM_caltech_net2.npz")
print("Done.")
