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

from swirl_func import (
    emit_m_step_jaxnet_optax2_student,
    soft_vi_sa,
    soft_vi_fn,
    jaxnet_e_step_batch2,
    trans_m_step_jax_optax,
    emit_m_step_jaxnet_optax2,
    pi0_m_step,
    distill_step,
    amortized_e_step_batch,
    create_inf_state,
    semi_amortized_e_step_batch
)

jax.config.update("jax_enable_x64", True)

# -----------------------------
# Args / paths
# -----------------------------
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0

DATASET = sys.argv[2] if len(sys.argv) > 2 else "gw5"

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

data_folder = os.path.join(BASE_DIR, "data_new")
save_folder = os.path.join(BASE_DIR, "output", "swirl_result", DATASET)

os.makedirs(save_folder, exist_ok=True)

# Your embedding folder
EMBED_DIR = "/home/yunran-yang/EngSci/Computational_Neuro_Lab/CNL_2025/trajectories/output/gtrxl_newset"

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

# align states with actions (GW5: action a_t aligns with NEXT state x_{t+1})
if xs.shape[1] == acs.shape[1] + 1:
    xs_ref_for_embed = xs[:, :-1]          # x_t (length = #actions)
else:
    xs_ref_for_embed = xs[:, :acs.shape[1]]


# -----------------------------
# Load embeddings + verify alignment
# -----------------------------
h_npz = np.load(os.path.join(EMBED_DIR, "h_gw5.npz"), allow_pickle=True)
print("h_gw5.npz keys:", h_npz.files)
h_all = h_npz["h"] if "h" in h_npz.files else h_npz[h_npz.files[0]]  # (N,T_embed,H)
print("[DEBUG] h_all shape:", h_all.shape)


xs_npz = np.load(os.path.join(EMBED_DIR, "xs_gw5_for_embed.npz"), allow_pickle=True)
print("xs_gw5_for_embed.npz keys:", xs_npz.files)
xs_for_embed = xs_npz["xs"] if "xs" in xs_npz.files else xs_npz[xs_npz.files[0]]

n = min(xs_for_embed.shape[0], xs_ref_for_embed.shape[0])
t_min = min(xs_for_embed.shape[1], xs_ref_for_embed.shape[1])

same_prefix = np.all(xs_for_embed[:n, :t_min] == xs_ref_for_embed[:n, :t_min])

# also allow internal 1-step shift even when lengths are equal
same_shift = False
if xs_ref_for_embed.shape[1] == xs_for_embed.shape[1]:
    # xs_for_embed matches xs_ref shifted by 1 (drop one end)
    same_shift = (
        np.all(xs_for_embed[:n, :-1] == xs_ref_for_embed[:n, 1:]) or
        np.all(xs_for_embed[:n, 1:]  == xs_ref_for_embed[:n, :-1])
    )
elif xs_ref_for_embed.shape[1] == xs_for_embed.shape[1] + 1:
    same_shift = np.all(xs_for_embed[:n, :] == xs_ref_for_embed[:n, 1:1+xs_for_embed.shape[1]])
elif xs_for_embed.shape[1] == xs_ref_for_embed.shape[1] + 1:
    same_shift = np.all(xs_for_embed[:n, 1:1+xs_ref_for_embed.shape[1]] == xs_ref_for_embed[:n, :])

print(f"[align] xs_ref_for_embed: {xs_ref_for_embed.shape}, xs_for_embed: {xs_for_embed.shape}")
print(f"[align] same_prefix(overlap)={same_prefix}, same_shift(off-by-one)={same_shift}")
if not (same_prefix or same_shift):
    raise ValueError("Embedding xs do NOT match xs_ref_for_embed (even with off-by-one shift).")

# xs_dec = xs_ref_for_embed[:, 1:]   # x_{t+1}
# acs    = acs[:, :-1]               # a_t aligned with x_{t+1}
# zs     = zs[:, :-1]                # keep labels aligned with a_t
# h_all  = h_all[:, :-1]       # h_t aligned with a_t (and x_{t+1})

# T = min(xs_dec.shape[1], acs.shape[1], zs.shape[1], h_all.shape[1])
# xs_dec = xs_dec[:, :T]
# acs = acs[:, :T]
# zs = zs[:, :T]
# h_all = h_all[:xs_dec.shape[0], :T]  # (N,T,H)

# print(f"[trim] N={xs_dec.shape[0]}, T={T}, S={S}, A={A}, H={h_all.shape[-1]}")
# Use x_t to score a_t (matches data_generator.py)
T = min(xs_ref_for_embed.shape[1], acs.shape[1], zs.shape[1], h_all.shape[1])

xs_dec = xs_ref_for_embed[:, :T]   # x_t
acs    = acs[:, :T]                # a_t
zs     = zs[:, :T]                 # z_t
h_all  = h_all[:, :T]              # h_t

print(f"[trim] N={xs_dec.shape[0]}, T={T}, S={S}, A={A}, H={h_all.shape[-1]}")



H_red = 4 # try 4, 8, 16, 32, 64
# h_all = h_all[:, :, :H_red]
print(f"[dim-reduce] Using first H_red={H_red} dims, new H={h_all.shape[-1]}")

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
def build_logemit_list(R_state, hoh, trans_probs_j, discount=0.95, vi_iters=50, tau=1.0):
    """
    hoh: (N,T,1,H)
    Returns log pi_{t,k}(a|s) with shape (N,T,K,S,A)

    tau: SoftVI temperature. We divide rewards by tau before planning.
    """
    S = trans_probs_j.shape[0]
    eyeS = jnp.eye(S)
    eps = 1e-20

    # squeeze to (N,T,H)
    hs = hoh[:, :, 0, :]

    def per_t(h_t):
        # build input (S, S+H)
        h_rep = jnp.repeat(h_t[None, :], S, axis=0)      # (S,H)
        inp = jnp.concatenate([eyeS, h_rep], axis=1)     # (S,S+H)

        out = R_state.apply_fn({"params": R_state.params}, inp)  # (S, K*A)
        out = out.reshape(S, K, A)                       # (S,K,A)
        r_ksa = jnp.transpose(out, (1, 0, 2))            # (K,S,A)

        # Step D: SoftVI with temperature (divide rewards by tau)
        pi_ksa = vmap(
            lambda r_sa: soft_vi_sa(trans_probs_j, r_sa / float(tau), discount=discount, threshold=vi_iters)
        )(r_ksa)

        return jnp.log(pi_ksa + eps)  # (K,S,A)

    def per_traj(h_TH):
        return vmap(per_t)(h_TH)  # (T,K,S,A)

    return vmap(per_traj)(hs)    # (N,T,K,S,A)

rng = jax.random.PRNGKey(seed)
H = train_hs.shape[-1]
S = train_xoh.shape[-1] 

inf_state = create_inf_state(
    jax.random.PRNGKey(seed + 123),K=K, H=H, S=S, A=A,
    lr=1e-4,d_h=64, d_x=32, d_a=16, d_model=64,n_hidden=2,
    init_stay_bias=3.0
)

# -----------------------------
# EM training
# -----------------------------
def em_train(
    logpi0, log_Ps, Rs, R_state, inf_state,
    train_hoh, train_xoh, train_aoh, trans_probs_j,
    *,
    iters=50,
    teacher_warmup=0,          # NEW: first N iters are PURE teacher-only (no node/edge priors)
    lam_node_max=0.0,           # NEW: max node weight after warmup (if you want it on)
    lam_node_ramp=0.0,           # NEW: ramp length (iters) from 0 -> lam_node_max
    use_node_priors=False,      # NEW: default False for correctness + debuggability
    distill_every=5,
    distill_steps=3,
    trans_iters=200,
    emit_iters=200,
    vi_discount=0.95,
    vi_iters=50,
    tau=1.0,
    do_student_diag=True,
    eta_mstep=0.0,              # NEW: mix student posterior into M-step (default OFF)
    seed=0
):
    """
    Corrected EM training loop with clean teacher/student separation.

    Key correction:
      - Teacher E-step is PURE teacher-only by default (lam_node=lam_edge=0).
        This avoids the circular dependency where an untrained student biases the teacher posterior.
      - Student is trained by distillation to match teacher posteriors.
      - Optional: after warmup, you may gradually turn on node priors (use_node_priors=True),
        which lets h_t influence z_t via the node network in a controlled way.

    Why this is theoretically correct:
      - Standard EM requires the E-step posterior to be computed from the current model parameters
        (teacher). Adding a learned q_phi(z|h) term into the teacher posterior before q_phi is trained
        breaks the intended EM objective and can cause self-reinforcing errors.
      - Distillation should approximate the teacher posterior, not define it (at least early on).
    """

    LL_list = []
    distill_list = []
    student_finite_list = []

    # small helper: schedule lam_node for teacher E-step
    def lam_node_schedule(it: int) -> float:
        if not use_node_priors:
            return 0.0
        # warmup: keep it purely teacher
        if it < teacher_warmup:
            return 0.0
        # ramp: linear from 0 -> lam_node_max
        t = it - teacher_warmup
        if lam_node_ramp <= 0:
            return float(lam_node_max)
        frac = min(1.0, max(0.0, t / float(lam_node_ramp)))
        return float(frac * lam_node_max)

    for it in range(iters):
        # ---- normalize pi0 ----
        pi0 = jnp.exp(logpi0 - jax_logsumexp(logpi0))

        # ---- build emissions from current reward model (this is still teacher model) ----
        # logemit_train = build_logemit_list(
        #     R_state, train_hoh, trans_probs_j,
        #     discount=vi_discount, vi_iters=vi_iters, tau=tau
        # )

        # ---- build emissions ----
        logemit_train = build_logemit_list(
            R_state, train_hoh, trans_probs_j,
            discount=vi_discount, vi_iters=vi_iters, tau=tau
        )

        pi = jnp.exp(logemit_train)  # (N,T,K,S,A)
        ent = -jnp.sum(pi * logemit_train, axis=-1).mean()   # mean entropy over a

        # ---- Teacher E-step (PURE by default) ----
        lam_node_t = lam_node_schedule(it)
        lam_edge_t = 0.0  # keep off unless you explicitly want edge priors too

        gamma_T, xi_T, alpha_T = semi_amortized_e_step_batch(
            inf_state,
            train_hoh, train_xoh, train_aoh,
            logemit_train,
            pi0, log_Ps, Rs, trans_probs_j,
            lam_edge=lam_edge_t,
            lam_node=lam_node_t,
            lam_prior=0.0,
            trust_gate=True,      # ok; with lam_node=0 it has no effect
            trust_temp=1.0,
            trust_floor=0.0,
            trust_cap=1.0,
            eta_post=0.0,         # IMPORTANT: do NOT blend posteriors inside teacher E-step
        )

        # Teacher log-likelihood (track this)
        LL = jnp.sum(jax_logsumexp(alpha_T[:, -1, :], axis=-1))
        LL_list.append(float(LL))

        # ---- Distill student on a schedule ----
        distill_on = (distill_every > 0) and (it % distill_every == 0)
        class_w = jnp.array([1.0, 5.0], dtype=jnp.float32)

        if distill_on:
            last_distillL = float("nan")
            for _ in range(distill_steps):
                inf_state, distillL, aux = distill_step(
                    inf_state, train_hoh, train_xoh, train_aoh,
                    gamma_T, xi_T,
                    clip_norm=1.0, ent_w=0.00, class_w=class_w,
                    mi_ent_w=0.00, mi_temp=0.5
                )
                last_distillL = float(distillL)

            distill_list.append(last_distillL)

            print(f"[distill] it={it} KL(T||S)={float(aux['kl_T_S_mean']):.6f} "
                  f"H_T={float(aux['ent_T_mean']):.4f} H_S={float(aux['ent_S_mean']):.4f} "
                  f"qbar={np.array(aux['qbar'])}")

        else:
            distill_list.append(float("nan"))

        # ---- Student diagnostics (optional) ----
        if do_student_diag:
            gamma_S, xi_S, _ = amortized_e_step_batch(inf_state, train_hoh, train_xoh, train_aoh)
            finite = bool(jnp.isfinite(gamma_S).all() & jnp.isfinite(xi_S).all())
            student_finite_list.append(finite)
        else:
            gamma_S = gamma_T
            xi_S = xi_T
            finite = True
            student_finite_list.append(True)

        # ---- Posterior used for M-step ----
        # Correct default: use teacher posterior for EM.
        # Optional (advanced): mix in student posterior later, but keep OFF while debugging.
        eta = float(eta_mstep)
        if eta > 0.0:
            gamma_used = (1.0 - eta) * gamma_T + eta * gamma_S
            xi_used = (1.0 - eta) * xi_T + eta * xi_S
        else:
            gamma_used = gamma_T
            xi_used = xi_T

        # ---- Logging ----
        print(f"[it {it}] trueLL={float(LL):.3f} lam_node_teacher={lam_node_t:.3f} "
              f"distill_on={distill_on} student_finite={finite} eta_mstep={eta:.2f}")

        # ---- M-step ----
        # pi0 M-step should return logpi0 (log space)
        logpi0 = pi0_m_step(gamma_used)

        # transition + reward params
        log_Ps, Rs = trans_m_step_jax_optax(
            log_Ps, Rs,
            (gamma_used, xi_used),
            train_xoh,
            num_iters=trans_iters,
            learning_rate=5e-4
        )

        R_state = emit_m_step_jaxnet_optax2(
            R_state, trans_probs_j, gamma_used, train_xoh, train_aoh, train_hoh,
            num_iters=emit_iters, batch_size=16,
            discount=vi_discount, vi_threshold=vi_iters,
            lr=3e-4, weight_decay=5e-3, tau=tau, seed=seed + it
        )

    return logpi0, log_Ps, Rs, R_state, inf_state, LL_list, distill_list, student_finite_list


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


new_logpi0, new_log_Ps, new_Rs, new_R_state, inf_state, LL_list, distill_list, _ = em_train(
    logpi0_start,
    log_Ps_start,
    Rs_start,
    R_state,
    inf_state,
    train_hoh=train_hoh,
    train_xoh=train_xoh,
    train_aoh=train_aoh,
    trans_probs_j=trans_probs_j,
    iters=50,
    teacher_warmup=0,        # pure teacher for first 10 iterations
    use_node_priors=True,    # IMPORTANT: start clean
    lam_node_max=0.2,
    lam_node_ramp=0.0,
    distill_every=1,
    trans_iters=200,
    emit_iters=50,
    eta_mstep=0.0             # do NOT mix student into M-step yet
)



# Save


from gw5_eval import best_perm_corr_KSA, mode_acc, mode_metrics, build_oracle_policy_gw5, learned_policy_from_R_avg, evaluate_policy_agreement, build_oracle_policy_from_RG, greedy_to_water_accuracy


# -----------------------------
# Quantitative evaluation (your requested 5 numbers)
# -----------------------------

# train acc
pi0 = jnp.exp(new_logpi0 - jax_logsumexp(new_logpi0))
logemit_train = build_logemit_list(new_R_state, train_hoh, trans_probs_j)
train_gamma, _, train_alpha = jaxnet_e_step_batch2(pi0, new_log_Ps, new_Rs, trans_probs_j,
                                                   train_xoh, train_xoh, train_aoh, logemit_train)

np.savez(
    "gamma_T_latest.npz",
    gamma=np.array(train_gamma),
    train_indices=np.array(train_indices, dtype=np.int64)
)
print("Saved gamma_T_latest.npz (train posterior + train_indices)")

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
rg_path = os.path.join(data_folder, "RG_sa.npy")
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

    # ---- NEW: gamma-weighted averaging over (n,t), separately per mode k ----
    gamma_np = np.array(train_gamma)
    if gamma_np.ndim == 4:      # (N,T,1,K) -> (N,T,K)
        gamma_np = gamma_np[:, :, 0, :]
    W = jnp.array(gamma_np)      # (N,T,K)

    # make R_pred -> (N,T,K,S,A) so it lines up with W
    R_pred = jnp.transpose(R_pred, (0, 1, 3, 2, 4))  # (N,T,K,S,A)

    num = jnp.sum(W[:, :, :, None, None] * R_pred, axis=(0, 1))            # (K,S,A)
    den = jnp.sum(W, axis=(0, 1))[:, None, None] + 1e-12                   # (K,1,1)
    R_avg = np.array(num / den)    

    if RG.shape != R_avg.shape:
        # try to coerce if RG is (K,S,S) from older code (state-state reward); then correlation is not meaningful
        # We'll only compute corr when shapes match.
        best_corr1 = float("nan")
    else:
        mask_SA = (np.sum(np.asarray(trans_probs), axis=-1) > 1e-12)  # (S, A)

        best_corr1, best_perm, per_mode_corrs, calib_params = best_perm_corr_KSA(
            R_pred=R_avg,
            R_gt=RG,
            mask_SA=mask_SA
        )

        print("[DEBUG R_corr] best_corr:", best_corr1)
        print("[DEBUG R_corr] best_perm:", best_perm)
        print("[DEBUG R_corr] per_mode_corrs:", per_mode_corrs)
        print("[DEBUG R_corr] calib (a,b) per mode:", calib_params)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
print("[DEBUG R_corr] RG shape:", None if RG is None else RG.shape)
print("[DEBUG R_corr] R_avg shape:", R_avg.shape)
print("[DEBUG R_corr] std RG:", None if RG is None else float(np.std(RG)))
print("[DEBUG R_corr] std R_avg:", float(np.std(R_avg)))

# ============================
# Time-windowed evaluation
# ============================
# NOTE: we evaluate how the *learned* reward/policy changes across time windows.
# Ground-truth RG is stationary, so windowed R_corr is mainly checking "does the model
# stay consistent" or "does it drift" (useful when your model *could* be time-varying).

# Load transition dynamics + oracle once (reuse for all windows)
from gw5_eval import build_oracle_policy_gw5, learned_policy_from_R_avg, evaluate_policy_agreement, policy_kl
trans_prob = np.load(os.path.join(data_folder, "trans_prob.npy")).astype(float)  # (S,A,S)
RG = np.load(os.path.join(data_folder, "RG_sa.npy"))
pi_oracle = build_oracle_policy_from_RG(trans_prob, RG, discount=0.95, vi_iters=50, tau=1.0)


# Choose windows (edit as you like)
# Example: 5 equal windows across the trajectory length T
win_edges = np.linspace(0, int(T), num=6, dtype=int)  # 0..T split into 5 windows

prev_Ravg_w = None
prev_pi_w = None

for wi in range(len(win_edges) - 1):
    t0, t1 = int(win_edges[wi]), int(win_edges[wi + 1])
    if t1 <= t0 + 1:
        continue

    Ww = W[:, t0:t1, :]                  # (N, Tw, K)
    Rw = R_pred[:, t0:t1, :, :, :]       # (N, Tw, K, S, A)

    num_w = jnp.sum(Ww[:, :, :, None, None] * Rw, axis=(0, 1))            # (K,S,A)
    den_w = jnp.sum(Ww, axis=(0, 1))[:, None, None] + 1e-12               # (K,1,1)
    R_avg_w = np.array(num_w / den_w)                                     # (K,S,A)

    mode_mass = np.array(Ww).mean(axis=(0,1))   # (K,)
    print(f"[WIN {wi}] mode_mass={mode_mass}")
    

    # ----- ΔR drift across windows (relative L2) -----
    if prev_Ravg_w is None:
        dR = float("nan")
        dRk = None
    else:
        diff = R_avg_w - prev_Ravg_w                    
        denom_norm = np.linalg.norm(prev_Ravg_w) + 1e-12
        dR = float(np.linalg.norm(diff) / denom_norm)

        # per-mode RMS drift (new diagnostic) -> (K,)
        dRk = np.sqrt(np.mean(diff**2, axis=(1,2)))
    
    dRk_str = "None" if dRk is None else np.array2string(dRk, precision=4)

    # ----- windowed reward correlation vs stationary RG -----
    if (RG is not None) and (RG.shape == R_avg_w.shape):
        mask_SA = (np.sum(np.asarray(trans_prob), axis=-1) > 1e-12)  # (S,A)
        corr_w, perm_w, per_mode_corrs_w, _ = best_perm_corr_KSA(
            R_pred=R_avg_w, R_gt=RG, mask_SA=mask_SA
        )
    else:
        corr_w, perm_w, per_mode_corrs_w = float("nan"), None, None


    # ----- windowed policy agreement -----
    pi_learn_w = learned_policy_from_R_avg(R_avg_w, soft_vi_fn, trans_prob)
    if perm_w is not None:
        pi_learn_w = pi_learn_w[list(perm_w)]
    met_w = evaluate_policy_agreement(pi_oracle, pi_learn_w)

    # ----- Δπ drift across windows (KL between learned policies) -----
    if prev_pi_w is None:
        dKL_pi = float("nan")
    else:
        # average KL across modes (simple scalar)
        dKL_pi = float(np.nanmean(policy_kl(prev_pi_w, pi_learn_w)))

    print(
        f"[WIN {wi}] t={t0}:{t1}  "
        f"dR={dR:.4f} dRk={dRk_str}"
        f"dR={dR:.4f}  dKL_pi={dKL_pi:.4f}  "
        f"Rcorr={corr_w} perm={perm_w}  "
        f"KL_oracle->learn={met_w['KL_oracle_to_learn']}  "
        f"CE_oracle_vs_learn={met_w['CE_oracle_vs_learn']}  "
        f"H_learn={met_w['H_learn']}"
    )

    prev_Ravg_w = R_avg_w
    prev_pi_w = pi_learn_w


# --- Policy agreement evaluation (oracle vs learned) ---
trans_prob = np.load(os.path.join(data_folder, "trans_prob.npy")).astype(float)  # (S,A,S)

# 1) Oracle policy (ground truth behavior)
pi_oracle = build_oracle_policy_gw5(trans_prob)

# 2) Learned policy from time-averaged reward (stable, comparable to oracle)
# NOTE: we pass a wrapper that matches (R_sa, trans_prob)->pi_sa
def soft_vi_wrapper(R_sa_k, trans_prob_):
    # soft_vi_sa expects (trans_probs, rewards_sa) in this codebase
    return np.array(soft_vi_sa(trans_prob_, R_sa_k))


pi_learn = learned_policy_from_R_avg(R_avg, soft_vi_wrapper, trans_prob)

metrics = evaluate_policy_agreement(pi_oracle, pi_learn)
#acc_water = greedy_to_water_accuracy(pi_learn, trans_prob, water_state=24, mode_water=1)

print("[DEBUG policy] KL oracle->learn per mode:", metrics["KL_oracle_to_learn"])
print("[DEBUG policy] CE oracle vs learn per mode:", metrics["CE_oracle_vs_learn"])
print("[DEBUG policy] Entropy oracle per mode:", metrics["H_oracle"])
print("[DEBUG policy] Entropy learn  per mode:", metrics["H_learn"])
#print("[DEBUG policy] Greedy-to-water accuracy (mode1):", acc_water)


print("S1:",
      "acc", acc1,
      "[DEBUG metrics] acc=", acc, "bal_acc=", bal_acc, "f1=", f1, "conf(tp,fp,fn,tn)=", conf,
      "test_acc", test_acc1,
      "R_corr", best_corr1,
      "test_loglik_per_traj", per_traj, "test_loglik_per_step", per_step, "T", int(T))

out_path = os.path.join(save_folder, f"{seed}_gw5_amortize.npz")
np.savez(
    out_path,
    new_logpi0=np.array(new_logpi0),
    new_log_Ps=np.array(new_log_Ps),
    new_Rs_state=np.array(new_Rs),  # transition params (Rs)
    new_R_state=np.array(new_R_state.params, dtype=object),  # reward net params
    LL_list=np.array(LL_list),
    R_avg = np.array(R_avg)
)
print("Saved:", out_path)
