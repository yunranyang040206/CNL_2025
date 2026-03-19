#!/usr/bin/env python3
"""
Training with reward ambiguity regularization + more diagnostics
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax import vmap
from jax.scipy.special import logsumexp as jax_logsumexp

import optax
from flax import linen as nn
from flax.training import train_state

from CNL_2025.gw5.swirl_func_reg import (
    soft_vi_sa,
    soft_vi_fn,
    jaxnet_e_step_batch2,
    trans_m_step_jax_optax,
    emit_m_step_jaxnet_optax2,
    pi0_m_step,
    distill_step,
    amortized_e_step_batch,
    create_inf_state,
    semi_amortized_e_step_batch,
    shaped_rewards_from_h,

)
from gw5_eval_clean import (
    avg_top2_gap,
    best_perm_corr_KSA,
    build_oracle_policy_from_RG,
    build_true_behavior_oracle_gw5,
    evaluate_policy_agreement,
    learned_policy_from_R_avg,
    mode_acc,
    mode_metrics,
    per_state_action_metrics,
    per_state_policy_metrics,
    realized_lls_from_logemit,
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

# Embedding folder
EMBED_DIR = "/home/yunran-yang/EngSci/Computational_Neuro_Lab/CNL_2025/trajectories/output/gtrxl_newset1"

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
EMBED_FILE = sys.argv[3] if len(sys.argv) > 3 else "h_gw5.npz"
h_npz = np.load(os.path.join(EMBED_DIR, EMBED_FILE), allow_pickle=True)
print(f"{EMBED_FILE} keys:", h_npz.files)
h_all = h_npz["h"] if "h" in h_npz.files else h_npz[h_npz.files[0]]  # (N,T_embed,H)
print(f"[embed] using embedding file: {EMBED_FILE}")
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

T = min(xs_ref_for_embed.shape[1], acs.shape[1], zs.shape[1], h_all.shape[1])

xs_dec = xs_ref_for_embed[:, :T]   # x_t
acs    = acs[:, :T]                # a_t
zs     = zs[:, :T]                 # z_t
h_all  = h_all[:, :T]              # h_t

print(f"[trim] N={xs_dec.shape[0]}, T={T}, S={S}, A={A}, H={h_all.shape[-1]}")
print(f" Using H={h_all.shape[-1]}")

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
# One-hot helpers
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
        x = nn.Dense(self.K * self.A + self.K)(x)
        return x

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

    IMPORTANT:
    Use the SAME AIRL-style shaped reward as in training:
        r_shaped(s,k,a;h_t) = r_base(s,k,a;h_t) + gamma * E[phi(s',k;h_t)] - phi(s,k;h_t)
    """
    eps = 1e-20

    # squeeze to (N,T,H)
    hs = hoh[:, :, 0, :]   # (N,T,H)

    def per_t(h_t):
        # --- use the SAME reward construction as training ---
        # expected shape: (K,S,A)
        r_ksa = shaped_rewards_from_h(
            R_state,
            h_t,
            trans_probs_j,
            discount=discount
        )

        # planning from shaped reward
        pi_ksa = jax.vmap(
            lambda r_sa: soft_vi_sa(
                trans_probs_j,
                r_sa / float(tau),
                discount=discount,
                threshold=vi_iters
            )
        )(r_ksa)   # (K,S,A)

        return jnp.log(pi_ksa + eps)

    def per_traj(h_TH):
        return jax.vmap(per_t)(h_TH)   # (T,K,S,A)

    return jax.vmap(per_traj)(hs)      # (N,T,K,S,A)

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
    teacher_warmup=0,
    lam_node_max=0.0,
    lam_node_ramp=0.0,
    use_node_priors=False,
    distill_every=5,
    distill_steps=3,
    trans_iters=200,
    emit_iters=200,
    vi_discount=0.95,
    vi_iters=50,
    tau=1.0,
    do_student_diag=True,
    eta_mstep=0.0,
    oracle_warm_iters=0,
    seed=0
):
    """
    Corrected EM training loop with clean teacher/student separation.

    Key correction:
      - Teacher E-step is PURE teacher-only by default (lam_node=lam_edge=0).
        This avoids the circular dependency where an untrained student biases the teacher posterior.
      - Student is trained by distillation to match teacher posteriors.

    Why this is theoretically correct:
      - Standard EM requires the E-step posterior to be computed from the current model parameters
        (teacher). Adding a learned q_phi(z|h) term into the teacher posterior before q_phi is trained
        breaks the intended EM objective and can cause self-reinforcing errors.
      - Distillation should approximate the teacher posterior, not define it (at least early on).
    """

    LL_list = []
    distill_list = []
    student_finite_list = []
    acc_hist = []
    emit_gap_hist = []
    node_gap_hist = []
    mode_mass_hist = []
    d_logpi0_hist = []
    d_logPs_hist = []
    pi0_ent_hist = []
    Ps_row_ent_hist = []

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
    
    oracle_gamma = jax.nn.one_hot(jnp.array(train_zs), K)

    def l2_change(a_new, a_old):
        da = jnp.asarray(a_new) - jnp.asarray(a_old)
        return float(jnp.sqrt(jnp.sum(da * da)))

    def prob_entropy(p, eps=1e-12):
        p = jnp.clip(jnp.asarray(p), eps, 1.0)
        p = p / jnp.sum(p, axis=-1, keepdims=True)
        return -jnp.sum(p * jnp.log(p), axis=-1)

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
        lam_edge_t = 0.0  # edge prior kept off by default

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

        # ---- Teacher diagnostics ----
        lls_train = realized_lls_from_logemit(logemit_train, train_xoh, train_aoh)
        emit_gap = avg_top2_gap(lls_train)

        node_logits_list = []
        for n in range(train_hoh.shape[0]):
            h_n = train_hoh[n, :, 0, :]
            x_n = train_xoh[n, :, 0, :]
            a_n = train_aoh[n, :, 0, :]
            node_logits_n, _ = inf_state.apply_fn({"params": inf_state.params}, h_n, x_n, a_n, train=False)
            node_logits_list.append(np.asarray(node_logits_n))
        node_logits_all = np.stack(node_logits_list, axis=0)
        node_gap = avg_top2_gap(node_logits_all)
        acc_now = mode_acc(gamma_T, train_zs)
        mode_mass_now = np.asarray(gamma_T).mean(axis=(0,1))

        acc_hist.append(float(acc_now))
        emit_gap_hist.append(float(emit_gap))
        node_gap_hist.append(float(node_gap))
        mode_mass_hist.append(mode_mass_now)

        print(f"[diag it {it}] acc={acc_now:.4f} emit_gap={emit_gap:.4f} node_gap={node_gap:.4f} mode_mass={mode_mass_now}")

        # ---- Posterior used for M-step ----
        # Correct default: use teacher posterior for EM.
        # Optional (advanced): mix in student posterior later, but keep OFF while debugging.
        eta = float(eta_mstep)
        if it < oracle_warm_iters:
            gamma_used = oracle_gamma
            xi_used = xi_T
            print(f"[warm-start] using ORACLE gamma for M-step at it={it}")
        else:
            if eta > 0.0:
                gamma_used = (1.0 - eta) * gamma_T + eta * gamma_S
                xi_used = (1.0 - eta) * xi_T + eta * xi_S
            else:
                gamma_used = gamma_T
                xi_used = xi_T
        print(f"[it {it}] using_oracle_gamma = {it < oracle_warm_iters}")
        print(f"[it {it}] gamma_used mode mass = {np.asarray(gamma_used).mean(axis=(0,1))}")
        print(f"[it {it}] gamma_T    mode mass = {np.asarray(gamma_T).mean(axis=(0,1))}")

        if it < oracle_warm_iters:
            print(f"[it {it}] oracle-gamma acc = {mode_acc(gamma_used, train_zs):.6f}")
            print(f"[it {it}] teacher-gamma acc = {mode_acc(gamma_T, train_zs):.6f}")


                # ---- Logging ----
        print(f"[it {it}] trueLL={float(LL):.3f} lam_node_teacher={lam_node_t:.3f} "
              f"distill_on={distill_on} student_finite={finite} eta_mstep={eta:.2f}")

        # ---- M-step ----
        logpi0_prev = logpi0
        log_Ps_prev = log_Ps

        # pi0 M-step should return logpi0 (log space)
        logpi0 = pi0_m_step(gamma_used)

        # transition + reward params
        log_Ps, Rs = trans_m_step_jax_optax(
            log_Ps, Rs,
            (gamma_used, xi_used),
            train_xoh,
            num_iters=trans_iters,
            learning_rate=5e-4,
            ridge_R = 1e-4, # L2 regulation on rewards
            dir_Ps   = 0.0, #Dirichlet smoothing for the transition matrix
            uni_Ps   = 0.0,  # Mix transition matrix with a uniform distribution to introduce uncertainty (avoid collapse)
        )

        R_state = emit_m_step_jaxnet_optax2(
            R_state, trans_probs_j, gamma_used, train_xoh, train_aoh, train_hoh,
            num_iters=emit_iters, batch_size=16,
            discount=vi_discount, vi_threshold=vi_iters,
            lr=3e-4, weight_decay=5e-3, tau=tau, seed=seed + it,
            lam_base=1e-3, lam_smooth=1e-3, lam_phi=1e-4,center_phi=True,
        )

        d_logpi0 = l2_change(logpi0, logpi0_prev)
        d_logPs = l2_change(log_Ps, log_Ps_prev)
        pi0_now = jnp.exp(logpi0 - jax_logsumexp(logpi0))
        Ps_now = jnp.exp(log_Ps - jax.nn.logsumexp(log_Ps, axis=-1, keepdims=True))
        pi0_ent = float(prob_entropy(pi0_now))
        Ps_row_ent = np.asarray(prob_entropy(Ps_now), dtype=float)

        d_logpi0_hist.append(d_logpi0)
        d_logPs_hist.append(d_logPs)
        pi0_ent_hist.append(pi0_ent)
        Ps_row_ent_hist.append(Ps_row_ent)

        if (it % 5 == 0) or (it == iters - 1):
            print(f"[ident it {it}] d_logpi0={d_logpi0:.6f} d_logPs={d_logPs:.6f} "
                  f"H(pi0)={pi0_ent:.4f} H(P_rows)={np.round(Ps_row_ent, 4)}")

    diag_hist = {
        "acc": np.array(acc_hist, dtype=float),
        "emit_gap": np.array(emit_gap_hist, dtype=float),
        "node_gap": np.array(node_gap_hist, dtype=float),
        "mode_mass": np.array(mode_mass_hist, dtype=float),
        "d_logpi0": np.array(d_logpi0_hist, dtype=float),
        "d_logPs": np.array(d_logPs_hist, dtype=float),
        "pi0_ent": np.array(pi0_ent_hist, dtype=float),
        "Ps_row_ent": np.array(Ps_row_ent_hist, dtype=float),
    }
    return logpi0, log_Ps, Rs, R_state, inf_state, LL_list, distill_list, student_finite_list, diag_hist


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


new_logpi0, new_log_Ps, new_Rs, new_R_state, inf_state, LL_list, distill_list, _, diag_hist = em_train(
    logpi0_start,
    log_Ps_start,
    Rs_start,
    R_state,
    inf_state,
    train_hoh=train_hoh,
    train_xoh=train_xoh,
    train_aoh=train_aoh,
    trans_probs_j=trans_probs_j,
    iters=35,
    teacher_warmup=0,
    use_node_priors=False,
    lam_node_max=0.0,
    lam_node_ramp=0.0,
    distill_every=0,
    trans_iters=200,
    emit_iters=50,
    eta_mstep=0.0,
    oracle_warm_iters=3
)

its = np.arange(len(diag_hist["acc"]))
plt.figure(figsize=(7, 5))
plt.plot(its, diag_hist["acc"], label="teacher acc")
plt.plot(its, diag_hist["emit_gap"], label="emit gap")
plt.plot(its, diag_hist["node_gap"], label="node gap")
plt.xlabel("EM iteration")
plt.ylabel("value")
plt.title("Teacher posterior diagnostics")
plt.legend()
plt.tight_layout()
diag_plot_path = os.path.join(save_folder, f"{seed}_teacher_diagnostic_plot.png")
plt.savefig(diag_plot_path, dpi=200)
plt.close()
print("Saved:", diag_plot_path)

plt.figure(figsize=(7, 5))
mode_mass = np.asarray(diag_hist["mode_mass"])
for k in range(mode_mass.shape[1]):
    plt.plot(its, mode_mass[:, k], label=f"mode {k} mass")
plt.xlabel("EM iteration")
plt.ylabel("mean posterior mass")
plt.title("Teacher mode mass over EM")
plt.legend()
plt.tight_layout()
mode_plot_path = os.path.join(save_folder, f"{seed}_teacher_mode_mass_plot.png")
plt.savefig(mode_plot_path, dpi=200)
plt.close()
print("Saved:", mode_plot_path)



# -----------------------------
# Quantitative evaluation
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
        r_ksa = shaped_rewards_from_h(
            new_R_state,
            h_t,
            trans_probs_j,
            discount=0.95
        )  # (K,S,A)
        return jnp.transpose(r_ksa, (1, 0, 2))  # -> (S,K,A)


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
# stay consistent or drift over time.

# Load transition dynamics + oracle once (reuse for all windows)
from gw5_eval_clean import learned_policy_from_logemit_avg, policy_kl
trans_prob = np.load(os.path.join(data_folder, "trans_prob.npy")).astype(float)  # (S,A,S)
RG = np.load(os.path.join(data_folder, "RG_sa.npy"))
pi_oracle = build_oracle_policy_from_RG(trans_prob, RG, discount=0.95, vi_iters=50, tau=1.0)


# Choose windows
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

# 1) Oracle policy (true handcrafted behavior)
pi_oracle = build_true_behavior_oracle_gw5(trans_prob)

# 2) Learned policy from time-averaged reward (stable, comparable to oracle)
def soft_vi_wrapper(R_sa_k, trans_prob_):
    return np.array(soft_vi_sa(trans_prob_, np.asarray(R_sa_k) / 1.0))

# 2a) Learned policy from time-averaged reward
def soft_vi_wrapper(R_sa_k, trans_prob_):
    return np.array(soft_vi_sa(trans_prob_, np.asarray(R_sa_k) / 1.0))

pi_learn_Ravg = learned_policy_from_R_avg(R_avg, soft_vi_wrapper, trans_prob)

# 2b) Learned policy from actual timestep emissions, posterior-weighted
pi_learn_pibar = learned_policy_from_logemit_avg(logemit_train, train_gamma)

# Compare both
metrics_Ravg = evaluate_policy_agreement(pi_oracle, pi_learn_Ravg)
state_dbg_Ravg = per_state_policy_metrics(pi_oracle, pi_learn_Ravg)

metrics_pibar = evaluate_policy_agreement(pi_oracle, pi_learn_pibar)
state_dbg_pibar = per_state_policy_metrics(pi_oracle, pi_learn_pibar)

print("[DEBUG policy | R_avg ] KL oracle->learn per mode:", metrics_Ravg["KL_oracle_to_learn"])
print("[DEBUG policy | R_avg ] CE oracle vs learn per mode:", metrics_Ravg["CE_oracle_vs_learn"])
print("[DEBUG policy | R_avg ] Entropy oracle per mode:", metrics_Ravg["H_oracle"])
print("[DEBUG policy | R_avg ] Entropy learn  per mode:", metrics_Ravg["H_learn"])

print("[DEBUG policy | pi_bar] KL oracle->learn per mode:", metrics_pibar["KL_oracle_to_learn"])
print("[DEBUG policy | pi_bar] CE oracle vs learn per mode:", metrics_pibar["CE_oracle_vs_learn"])
print("[DEBUG policy | pi_bar] Entropy oracle per mode:", metrics_pibar["H_oracle"])
print("[DEBUG policy | pi_bar] Entropy learn  per mode:", metrics_pibar["H_learn"])

print("[DEBUG per-state KL | R_avg ] worst states mode 0:",
      np.argsort(-state_dbg_Ravg["kl_state"][0])[:10],
      state_dbg_Ravg["kl_state"][0][np.argsort(-state_dbg_Ravg["kl_state"][0])[:10]])
print("[DEBUG per-state KL | R_avg ] worst states mode 1:",
      np.argsort(-state_dbg_Ravg["kl_state"][1])[:10],
      state_dbg_Ravg["kl_state"][1][np.argsort(-state_dbg_Ravg["kl_state"][1])[:10]])

print("[DEBUG per-state KL | pi_bar] worst states mode 0:",
      np.argsort(-state_dbg_pibar["kl_state"][0])[:10],
      state_dbg_pibar["kl_state"][0][np.argsort(-state_dbg_pibar["kl_state"][0])[:10]])
print("[DEBUG per-state KL | pi_bar] worst states mode 1:",
      np.argsort(-state_dbg_pibar["kl_state"][1])[:10],
      state_dbg_pibar["kl_state"][1][np.argsort(-state_dbg_pibar["kl_state"][1])[:10]])

from gw5_eval_clean import inspect_action_semantics, print_worst_action_states, save_action_metric_heatmaps, summarize_action_metrics
action_dbg = per_state_action_metrics(pi_oracle, pi_learn_pibar)
summarize_action_metrics(action_dbg)
print_worst_action_states(pi_oracle, pi_learn_pibar, action_dbg, top_m=10)
save_action_metric_heatmaps(
    action_dbg,
    grid=5,
    out_path=os.path.join(save_folder, f"{seed}_action_metric_heatmaps.png")
)

action_dbg = per_state_action_metrics(pi_oracle, pi_learn_pibar)
summarize_action_metrics(action_dbg)
print_worst_action_states(pi_oracle, pi_learn_pibar, action_dbg, top_m=10)
# Inspect action semantics on worst states from current diagnostics
states_to_check = {
    0: [18, 7, 11, 17, 8],
    1: [24, 19, 23, 9, 21],
}
inspect_action_semantics(
    pi_oracle,
    pi_learn_pibar,
    trans_prob,
    states_to_check,
    mode_names=["mode 0", "mode 1"]
)

# Save per-state KL / entropy heatmaps
fig, axes = plt.subplots(2, 3, figsize=(10, 6))
for k in range(2):
    axes[k, 0].imshow(state_dbg_pibar["kl_state"][k].reshape(5, 5))
    axes[k, 0].set_title(f"mode {k} KL")
    axes[k, 1].imshow(state_dbg_pibar["H_oracle_state"][k].reshape(5, 5))
    axes[k, 1].set_title(f"mode {k} H oracle")
    axes[k, 2].imshow((state_dbg_pibar["H_learn_state"][k] - state_dbg_pibar["H_oracle_state"][k]).reshape(5, 5))
    axes[k, 2].set_title(f"mode {k} H learn-oracle")
for ax in axes.ravel():
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
plt.tight_layout()
state_plot_path = os.path.join(save_folder, f"{seed}_policy_state_heatmaps.png")
plt.savefig(state_plot_path, dpi=200)
plt.close()
print("Saved:", state_plot_path)

# # -------------------------------------------------
# # Save per-state top-2 action-gap heatmaps
# # gap = p(best action) - p(second-best action)
# # Larger gap = stronger preference for best action
# # -------------------------------------------------
# gap_oracle = per_state_top2_gap(pi_oracle)         # (K,S)
# gap_learn  = per_state_top2_gap(pi_learn_pibar)    # (K,S)
# gap_diff   = gap_learn - gap_oracle                # (K,S)

# fig, axes = plt.subplots(2, 3, figsize=(10, 6))

# for k in range(2):
#     im0 = axes[k, 0].imshow(gap_oracle[k].reshape(5, 5))
#     axes[k, 0].set_title(f"mode {k} gap_oracle")

#     im1 = axes[k, 1].imshow(gap_learn[k].reshape(5, 5))
#     axes[k, 1].set_title(f"mode {k} gap_learn")

#     im2 = axes[k, 2].imshow(gap_diff[k].reshape(5, 5))
#     axes[k, 2].set_title(f"mode {k} gap_learn-oracle")

# for ax in axes.ravel():
#     ax.set_xticks(range(5))
#     ax.set_yticks(range(5))

# plt.tight_layout()
# gap_plot_path = os.path.join(save_folder, f"{seed}_policy_gap_heatmaps.png")
# plt.savefig(gap_plot_path, dpi=200)
# plt.close()
# print("Saved:", gap_plot_path)

# # -------------------------------------------------
# # Q-gap diagnostic for stay action (assume stay_a=4, goal_state=24)
# # -------------------------------------------------
# stay_a = 4
# goal_state = 24

# if ("RG" in locals()) and (RG is not None) and (RG.shape == R_avg.shape):
#     perm_for_q = best_perm if ("best_perm" in locals()) and (best_perm is not None) else tuple(range(R_avg.shape[0]))

#     R_learn_aligned = np.asarray(R_avg)[list(perm_for_q)]   # (K,S,A)
#     R_oracle = np.asarray(RG)                               # (K,S,A)

#     K_q = R_oracle.shape[0]
#     stay_gap_oracle = []
#     stay_gap_learn = []

#     fig, axes = plt.subplots(2, 3, figsize=(10, 6))

#     for k in range(K_q):
#         Vok, Qok, _ = soft_vi_value_q_np(
#             trans_prob,
#             R_oracle[k],
#             discount=0.95,
#             threshold=100,
#             tau=1.0,
#         )
#         Vlk, Qlk, _ = soft_vi_value_q_np(
#             trans_prob,
#             R_learn_aligned[k],
#             discount=0.95,
#             threshold=100,
#             tau=1.0,
#         )

#         best_nonstay_ok, gap_ok = best_nonstay_action_and_gap(Qok, stay_a=stay_a)
#         best_nonstay_lk, gap_lk = best_nonstay_action_and_gap(Qlk, stay_a=stay_a)

#         stay_gap_oracle.append(gap_ok)
#         stay_gap_learn.append(gap_lk)

#         axes[k, 0].imshow(gap_ok.reshape(5, 5))
#         axes[k, 0].set_title(f"mode {k} Q(stay)-Q(best move) oracle")

#         axes[k, 1].imshow(gap_lk.reshape(5, 5))
#         axes[k, 1].set_title(f"mode {k} Q(stay)-Q(best move) learn")

#         axes[k, 2].imshow((gap_lk - gap_ok).reshape(5, 5))
#         axes[k, 2].set_title(f"mode {k} stay-gap learn-oracle")

#         print(
#             f"[DEBUG stay-Q] mode {k}, goal={goal_state}: "
#             f"oracle Qstay={Qok[goal_state, stay_a]:.6f}, "
#             f"oracle best_nonstay_a={int(best_nonstay_ok[goal_state])}, "
#             f"oracle best_nonstay_Q={Qok[goal_state, best_nonstay_ok[goal_state]]:.6f}, "
#             f"oracle stay_gap={gap_ok[goal_state]:.6f}"
#         )
#         print(
#             f"[DEBUG stay-Q] mode {k}, goal={goal_state}: "
#             f"learn  Qstay={Qlk[goal_state, stay_a]:.6f}, "
#             f"learn  best_nonstay_a={int(best_nonstay_lk[goal_state])}, "
#             f"learn  best_nonstay_Q={Qlk[goal_state, best_nonstay_lk[goal_state]]:.6f}, "
#             f"learn  stay_gap={gap_lk[goal_state]:.6f}"
#         )

#     stay_gap_oracle = np.asarray(stay_gap_oracle)   # (K,S)
#     stay_gap_learn  = np.asarray(stay_gap_learn)    # (K,S)

#     for ax in axes.ravel():
#         ax.set_xticks(range(5))
#         ax.set_yticks(range(5))

#     plt.tight_layout()
#     stayq_plot_path = os.path.join(save_folder, f"{seed}_stay_qgap_heatmaps.png")
#     plt.savefig(stayq_plot_path, dpi=200)
#     plt.close()
#     print("Saved:", stayq_plot_path)
# else:
#     print("[DEBUG stay-Q] skipped stay-Q diagnostic because RG and R_avg were unavailable or shape-mismatched")
# Optional: print worst states where learned gap is too small
# for k in range(2):
#     worst = np.argsort(gap_diff[k])[:10]   # most negative = learned much softer than oracle
#     print(f"[DEBUG top2-gap] mode {k} worst states:", worst, gap_diff[k][worst])

ident_path = os.path.join(save_folder, f"{seed}_ident_debug_metrics.npz")
np.savez(
    ident_path,
    d_logpi0=np.array(diag_hist["d_logpi0"], dtype=float),
    d_logPs=np.array(diag_hist["d_logPs"], dtype=float),
    pi0_ent=np.array(diag_hist["pi0_ent"], dtype=float),
    Ps_row_ent=np.array(diag_hist["Ps_row_ent"], dtype=float),
)
print("Saved:", ident_path)

print("S1:",
      "acc", acc1,
      "[DEBUG metrics] acc=", acc, "bal_acc=", bal_acc, "f1=", f1, "conf(tp,fp,fn,tn)=", conf,
      "test_acc", test_acc1,
      "R_corr", best_corr1,
      "test_loglik_per_traj", per_traj, "test_loglik_per_step", per_step, "T", int(T))

