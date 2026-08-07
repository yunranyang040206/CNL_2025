#!/usr/bin/env python3
"""
Freeze-RG posterior test (GW5) — CONSISTENT with current_run_gw5.py

What this tests:
  - We freeze rewards to the ground-truth RG_sa.npy (so policy/emissions are fixed).
  - Then we run the teacher E-step (semi_amortized_e_step_batch) and sweep lam_node.
  - If lam_node helps align gamma(z=1) with z_true (thirst), then your node-net wiring
    can inject h_t into the posterior. If not, the issue is deeper (features, scaling,
    or how node potentials are formed/used).

This script matches your training loader:
  data_new/{trans_prob.npy, xs.npy, acs.npy, zs.npy, RG_sa.npy}
  gw5_embed/{h_gw5.npz, xs_gw5_for_embed.npz}
"""

import os
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp as jax_logsumexp

from new_swirl_func import (
    soft_vi_sa,
    create_inf_state,
    semi_amortized_e_step_batch,
)

# -----------------------------
# Config (match training defaults)
# -----------------------------
SEED = 0
N_USE = 200         # you use [:200] in training
TRAIN_N = 160       # your train split
DISCOUNT = 0.95
VI_ITERS = 50
TAU = 1.0           # set this to whatever you trained with (you tried 2 elsewhere)
K = 2               # GW5 has 2 modes
GRID_SIZE = 5       # not used directly here, but kept for clarity

LAM_EDGE = 0.0
LAM_PRIOR = 0.0
TRUST_GATE = True
TRUST_TEMP = 1.0
TRUST_FLOOR = 0.0
TRUST_CAP = 1.0

LAM_NODE_SWEEP = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EMBED_DIR = os.path.join(BASE_DIR, "output", "gtrxl_newset")


# -----------------------------
# Helpers
# -----------------------------
def one_hot_int(x: np.ndarray, depth: int) -> np.ndarray:
    # x: (...,) int
    out = np.zeros(x.shape + (depth,), dtype=np.float32)
    idx = np.indices(x.shape)
    out[(*idx, x)] = 1.0
    return out


def corrcoef_safe(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.sqrt((a * a).sum()) * np.sqrt((b * b).sum()) + 1e-12)
    return float((a * b).sum() / denom)


def build_logemit_from_RG(trans_prob: jnp.ndarray, RG_sa: np.ndarray,
                          discount: float, vi_iters: int, tau: float) -> jnp.ndarray:
    """
    Builds log-emission tensor logemit with shape (K,S,A), using:
      pi_k(s,a) = SoftVI(trans_prob, RG[k]/tau)
      logemit[k,s,a] = log pi_k(s,a)

    Then caller will broadcast to (N,T,K,S,A) like your training does.
    """
    RG_sa = np.asarray(RG_sa, dtype=np.float32)
    assert RG_sa.ndim == 3, "RG_sa must be (K,S,A)"
    K_, S, A = RG_sa.shape
    assert K_ == K, f"Expected K={K}, got {K_}"

    def solve_pi_for_mode(r_sa):
        pi = soft_vi_sa(trans_prob, r_sa / float(tau), discount=discount, threshold=vi_iters)
        pi = jnp.clip(pi, 1e-20, 1.0)
        pi = pi / jnp.sum(pi, axis=-1, keepdims=True)
        return pi

    pi_ksa = jax.vmap(solve_pi_for_mode)(jnp.array(RG_sa))  # (K,S,A)
    logpi_ksa = jnp.log(pi_ksa)
    return logpi_ksa


# -----------------------------
# Paths (match current_run_gw5.py)
# -----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_folder = os.path.join(BASE_DIR, "data_new")

# -----------------------------
# Load dataset artifacts (MATCH training)
# -----------------------------
trans_probs = np.load(os.path.join(data_folder, "trans_prob.npy"), allow_pickle=True)  # (S,A,S)
xs = np.load(os.path.join(data_folder, "xs.npy"), allow_pickle=True)[:N_USE]
acs = np.load(os.path.join(data_folder, "acs.npy"), allow_pickle=True)[:N_USE]
zs = np.load(os.path.join(data_folder, "zs.npy"), allow_pickle=True)[:N_USE]
RG = np.load(os.path.join(data_folder, "RG_sa.npy"), allow_pickle=True)              # (K,S,A)

xs = np.asarray(xs, dtype=int)
acs = np.asarray(acs, dtype=int)
zs = np.asarray(zs, dtype=int)
trans_probs = np.asarray(trans_probs, dtype=np.float32)
RG = np.asarray(RG, dtype=np.float32)

S, A, S2 = trans_probs.shape
assert S == S2, "trans_prob must be (S,A,S)"
assert RG.shape == (K, S, A), f"RG_sa must be (K,S,A) = {(K,S,A)}, got {RG.shape}"

# align decision-time states with actions (MATCH training)
if xs.shape[1] == acs.shape[1] + 1:
    xs_dec = xs[:, :-1]
else:
    xs_dec = xs[:, :acs.shape[1]]

# -----------------------------
# Load embeddings + verify alignment (MATCH training)
# -----------------------------
h_npz = np.load(os.path.join(EMBED_DIR, "h_gw5.npz"), allow_pickle=True)
h_all = h_npz["h"] if "h" in h_npz.files else h_npz[h_npz.files[0]]  # (N,T,H)

xs_npz = np.load(os.path.join(EMBED_DIR, "xs_gw5_for_embed.npz"), allow_pickle=True)
xs_for_embed = xs_npz["xs"] if "xs" in xs_npz.files else xs_npz[xs_npz.files[0]]

n = min(xs_for_embed.shape[0], xs_dec.shape[0])
t_min = min(xs_for_embed.shape[1], xs_dec.shape[1])

same_prefix = np.all(xs_for_embed[:n, :t_min] == xs_dec[:n, :t_min])
same_shift = False
if xs_dec.shape[1] == xs_for_embed.shape[1] + 1:
    same_shift = np.all(xs_for_embed[:n, :] == xs_dec[:n, 1:1 + xs_for_embed.shape[1]])
elif xs_for_embed.shape[1] == xs_dec.shape[1] + 1:
    same_shift = np.all(xs_for_embed[:n, 1:1 + xs_dec.shape[1]] == xs_dec[:n, :])

print(f"[align] xs_dec: {xs_dec.shape}, xs_for_embed: {xs_for_embed.shape}")
print(f"[align] same_prefix(overlap)={same_prefix}, same_shift(off-by-one)={same_shift}")
if not (same_prefix or same_shift):
    raise ValueError("Embedding xs do NOT match xs_dec (even with off-by-one shift).")

# trim to common T (MATCH training)
T = min(xs_dec.shape[1], acs.shape[1], zs.shape[1], h_all.shape[1])
xs_dec = xs_dec[:, :T]
acs = acs[:, :T]
zs = zs[:, :T]
h_all = h_all[:xs_dec.shape[0], :T]

print(f"[trim] N={xs_dec.shape[0]}, T={T}, S={S}, A={A}, H={h_all.shape[-1]}")
print("[load] trans_prob:", trans_probs.shape, "RG:", RG.shape, "xs_dec:", xs_dec.shape, "acs:", acs.shape, "zs:", zs.shape, "h:", h_all.shape)

# -----------------------------
# Build one-hot tensors like training (N,T,1,dim)
# -----------------------------
xoh = one_hot_int(xs_dec, S)[:, :, None, :]   # (N,T,1,S)
aoh = one_hot_int(acs, A)[:, :, None, :]      # (N,T,1,A)
hoh = np.asarray(h_all, dtype=np.float32)[:, :, None, :]  # (N,T,1,H)

# train split like your script (first TRAIN_N)
train_xoh = jnp.array(xoh[:TRAIN_N])
train_aoh = jnp.array(aoh[:TRAIN_N])
train_hoh = jnp.array(hoh[:TRAIN_N])
train_z = zs[:TRAIN_N]

# -----------------------------
# Create inf_state like training
# -----------------------------
trans_probs_j = jnp.array(trans_probs)

Hdim = train_hoh.shape[-1]
Sdim = train_xoh.shape[-1]

inf_state = create_inf_state(
    jax.random.PRNGKey(SEED + 123),
    K=K, H=Hdim, S=Sdim, A=A,
    lr=1e-4, d_h=64, d_x=32, d_a=16, d_model=64,
    n_hidden=2,
    init_stay_bias=3.0
)

# -----------------------------
# Build frozen log-emissions from RG (then broadcast to N,T)
# -----------------------------
logpi_ksa = build_logemit_from_RG(trans_probs_j, RG, DISCOUNT, VI_ITERS, TAU)  # (K,S,A)

# broadcast to (N,T,K,S,A) to match semi_amortized_e_step_batch usage
logemit_train = jnp.broadcast_to(logpi_ksa[None, None, :, :, :], (TRAIN_N, T, K, S, A))
print("logemit_train:", logemit_train.shape)

# -----------------------------
# Run lam_node sweep
# -----------------------------
print("\n=== Freeze-RG TEST: lam_node sweep ===")

# baseline gamma for Δgamma measurement
gamma0 = None

for lam_node in LAM_NODE_SWEEP:
    pi0_dummy = jnp.ones((K,), dtype=jnp.float32) / K
    log_Ps_dummy = jnp.log(jnp.ones((K, K), dtype=jnp.float32) / K)
    Rs_dummy = jnp.zeros((K, 1, S), dtype=jnp.float32)   # <-- FIXED shape

    gamma_T, xi_T, alpha_T = semi_amortized_e_step_batch(
        inf_state,
        train_hoh, train_xoh, train_aoh,
        logemit_train,
        pi0_dummy,
        log_Ps_dummy,
        Rs_dummy,
        trans_probs_j,
        lam_edge=LAM_EDGE,
        lam_node=float(lam_node),
        lam_prior=LAM_PRIOR,
        trust_gate=TRUST_GATE,
        trust_temp=TRUST_TEMP,
        trust_floor=TRUST_FLOOR,
        trust_cap=TRUST_CAP,
        eta_post=0.0,
    )

    # gamma_T expected shape: (N,T,K)
    gamma_np = np.array(gamma_T)

    # thirst proxy: z_true==1 (mode 1 in your generator)
    thirsty = (train_z == 1).astype(np.float32)          # (N,T)
    gamma1 = gamma_np[:, :, 1]                           # (N,T)

    c = corrcoef_safe(thirsty, gamma1)

    # mode mass across all time/trajectories
    mode_mass = gamma_np.reshape(-1, K).mean(axis=0)

    # Δgamma vs baseline
    if gamma0 is None:
        gamma0 = gamma_np
        mean_dg = 0.0
    else:
        mean_dg = float(np.mean(np.abs(gamma_np - gamma0)))

    print(f"lam_node={lam_node:>5.1f}: corr(z_true==1, gamma1)={c:+.4f}  mean|Δgamma|={mean_dg:.6f}  mode_mass={mode_mass}")

print("\nNotes:")
print("  - If mean|Δgamma| increases with lam_node: node term is actually affecting gamma (wiring works).")
print("  - If corr does NOT improve (or gets worse): either h_t doesn't encode thirst for your node net,")
print("    or the node potential form/scale is wrong, or your z_true isn't aligned with 'mode 1' as assumed.")
