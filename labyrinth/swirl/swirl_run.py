#!/usr/bin/env python3
"""
Memory-optimized history-dependent SWIRL training for labyrinth mice data.

Main changes vs the prior trainer:
- disables JAX GPU preallocation by default
- runs the E-step in outer trajectory chunks, so logemit is never materialized for all N at once
- uses a streaming reward summary (R_avg) instead of building a giant (N,T,K,S,A) tensor
- passes small, explicit reward-M-step batch/time-chunk settings through to swirl_func_optimized
"""

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # CHANGED
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")  # CHANGED

import gc
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp as jax_logsumexp

import optax
from flax import linen as nn
from flax.training import train_state
from flax.serialization import to_bytes

from swirl_func import (
    soft_vi_sa,
    shaped_rewards_from_h,
    create_inf_state,
    semi_amortized_e_step_batch,
    trans_m_step_jax_optax,
    emit_m_step_jaxnet_optax2,
    pi0_m_step,
)

jax.config.update("jax_enable_x64", False)


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


def create_reward_state(rng, input_size, hidden_size, K, A, lr, weight_decay=0.0):
    model = RewardNet(hidden_size=hidden_size, K=K, A=A)
    params = model.init(rng, jnp.ones((1, input_size), dtype=jnp.float32))['params']
    tx = optax.adamw(lr, weight_decay=weight_decay)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def infer_valid_len_from_xs(xs: np.ndarray) -> np.ndarray:
    valid_len = np.full((xs.shape[0],), xs.shape[1], dtype=np.int64)
    for i in range(xs.shape[0]):
        bad = np.where(xs[i] < 0)[0]
        if bad.size > 0:
            valid_len[i] = int(bad[0])
    return valid_len


def safe_one_hot_int(arr: np.ndarray, K: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.int64)
    out = np.zeros(arr.shape + (K,), dtype=np.float32)
    mask = (arr >= 0) & (arr < K)
    if np.any(mask):
        flat_idx = np.where(mask.ravel())[0]
        flat_vals = arr.ravel()[flat_idx]
        out.reshape(-1, K)[flat_idx, flat_vals] = 1.0
    return out


def make_xoh(xs_int: np.ndarray, S: int) -> jnp.ndarray:
    return jnp.array(safe_one_hot_int(xs_int, S)[:, :, None, :], dtype=jnp.float32)


def make_aoh(acs_int: np.ndarray, A: int) -> jnp.ndarray:
    return jnp.array(safe_one_hot_int(acs_int, A)[:, :, None, :], dtype=jnp.float32)


def make_hoh(hs_arr: np.ndarray) -> jnp.ndarray:
    return jnp.array(hs_arr[:, :, None, :], dtype=jnp.float32)


def align_embeddings(xs_raw: np.ndarray, h_all: np.ndarray, xs_embed: np.ndarray | None = None):
    n = min(xs_raw.shape[0], h_all.shape[0])
    t = min(xs_raw.shape[1], h_all.shape[1])
    xs_raw = xs_raw[:n, :t]
    h_all = h_all[:n, :t]
    if xs_embed is not None:
        xs_embed = np.asarray(xs_embed[:n, :t])
        if not np.array_equal(xs_embed, xs_raw):
            raise ValueError("Embedding xs do not match labyrinth xs after trimming.")
    return xs_raw, h_all


def build_logemit_chunk(
    R_state,
    hoh_chunk,
    trans_probs_j,
    discount=0.95,
    vi_iters=50,
    tau=1.0,
):
    """Build log-emissions for one outer trajectory chunk only."""
    eps = 1e-20
    hs = hoh_chunk[:, :, 0, :]  # (B,T,H)

    def per_t(h_t):
        r_ksa = shaped_rewards_from_h(R_state, h_t, trans_probs_j, discount=discount)
        pi_ksa = jax.vmap(
            lambda r_sa: soft_vi_sa(
                trans_probs_j, r_sa / float(tau), discount=discount, threshold=vi_iters
            )
        )(r_ksa)
        return jnp.log(pi_ksa + eps)

    def per_traj(h_TH):
        return jax.vmap(per_t)(h_TH)  # (T,K,S,A)

    out = jax.vmap(per_traj)(hs)
    return np.array(jax.device_get(out), dtype=np.float32)  # keep on CPU after each chunk


def run_e_step_chunked(
    *,
    R_state,
    inf_state,
    hoh,
    xoh,
    aoh,
    pi0,
    log_Ps,
    Rs,
    trans_probs_j,
    discount,
    vi_iters,
    tau,
    traj_chunk_size,
):
    """
    CHANGED: memory-safe E-step.
    Builds logemit and runs student/teacher inference in outer trajectory chunks.
    """
    N = hoh.shape[0]
    gamma_chunks, xi_chunks, alpha_chunks = [], [], []

    for i in range(0, N, traj_chunk_size):
        sl = slice(i, min(i + traj_chunk_size, N))
        hoh_chunk = hoh[sl]
        xoh_chunk = xoh[sl]
        aoh_chunk = aoh[sl]

        logemit_chunk = build_logemit_chunk(
            R_state, hoh_chunk, trans_probs_j,
            discount=discount, vi_iters=vi_iters, tau=tau,
        )
        gamma_c, xi_c, alpha_c = semi_amortized_e_step_batch(
            inf_state,
            hoh_chunk,
            xoh_chunk,
            aoh_chunk,
            logemit_chunk,
            pi0,
            log_Ps,
            Rs,
            trans_probs_j,
            lam_edge=0.0,
            lam_node=0.0,
            lam_prior=0.0,
            trust_gate=False,
            eta_post=0.0,
        )

        gamma_chunks.append(np.array(jax.device_get(gamma_c), dtype=np.float32))
        xi_chunks.append(np.array(jax.device_get(xi_c), dtype=np.float32))
        alpha_chunks.append(np.array(jax.device_get(alpha_c), dtype=np.float32))

        del logemit_chunk, gamma_c, xi_c, alpha_c
        gc.collect()

    gamma = jnp.array(np.concatenate(gamma_chunks, axis=0), dtype=jnp.float32)
    xi = jnp.array(np.concatenate(xi_chunks, axis=0), dtype=jnp.float32)
    alpha = jnp.array(np.concatenate(alpha_chunks, axis=0), dtype=jnp.float32)
    return gamma, xi, alpha


def mean_mode_mass(gamma: jnp.ndarray) -> np.ndarray:
    g = np.asarray(gamma)
    if g.ndim == 4:
        g = g[:, :, 0, :]
    return g.mean(axis=(0, 1))


def masked_loglik(alpha: jnp.ndarray, valid_len: np.ndarray) -> tuple[float, float]:
    alpha = np.asarray(alpha)
    last_idx = np.maximum(valid_len - 1, 0)
    final = np.array([jax_logsumexp(alpha[n, int(last_idx[n])], axis=-1) for n in range(alpha.shape[0])], dtype=float)
    per_traj = float(np.mean(final))
    denom = float(np.maximum(valid_len.sum(), 1))
    per_step = float(final.sum() / denom)
    return per_traj, per_step


def compute_R_avg_streaming(
    R_state,
    hoh,
    gamma,
    trans_probs_j,
    valid_len,
    discount=0.95,
    time_chunk_size=64,
):
    """
    CHANGED: streaming gamma-weighted average reward.
    Avoids materializing a giant (N,T,K,S,A) tensor.
    """
    hs = np.asarray(hoh, dtype=np.float32)[:, :, 0, :]
    gamma = np.asarray(gamma, dtype=np.float32)
    if gamma.ndim == 4:
        gamma = gamma[:, :, 0, :]

    N, T, K = gamma.shape
    S, A = int(trans_probs_j.shape[0]), int(trans_probs_j.shape[1])

    num = np.zeros((K, S, A), dtype=np.float64)
    den = np.zeros((K,), dtype=np.float64)

    for n in range(N):
        L = int(valid_len[n])
        for t0 in range(0, L, time_chunk_size):
            t1 = min(t0 + time_chunk_size, L)
            h_chunk = jnp.array(hs[n, t0:t1], dtype=jnp.float32)
            g_chunk = gamma[n, t0:t1]  # (C,K)

            r_chunk = jax.vmap(lambda h_t: shaped_rewards_from_h(
                R_state, h_t, trans_probs_j, discount=discount
            ))(h_chunk)  # (C,K,S,A)
            r_chunk = np.array(jax.device_get(r_chunk), dtype=np.float64)

            num += np.sum(g_chunk[:, :, None, None] * r_chunk, axis=0)
            den += np.sum(g_chunk, axis=0)

    den = den[:, None, None] + 1e-12
    R_avg_ksa = (num / den).astype(np.float32)
    R_avg_ks = R_avg_ksa.mean(axis=-1).astype(np.float32)
    return R_avg_ksa, R_avg_ks


def em_train(
    *,
    logpi0,
    log_Ps,
    Rs,
    R_state,
    inf_state,
    train_hoh,
    train_xoh,
    train_aoh,
    train_valid_len,
    trans_probs_j,
    iters=30,
    trans_iters=150,
    emit_iters=50,
    vi_discount=0.95,
    vi_iters=50,
    tau=1.0,
    logemit_chunk_size=2,
    mstep_batch_size=1,
    reward_time_chunk_size=32,
    eta_mstep=0.0,
    seed=0,
):
    LL_list, mode_mass_hist, gamma_entropy_hist = [], [], []

    for it in range(iters):
        pi0 = jnp.exp(logpi0 - jax_logsumexp(logpi0))

        gamma_T, xi_T, alpha_T = run_e_step_chunked(
            R_state=R_state,
            inf_state=inf_state,
            hoh=train_hoh,
            xoh=train_xoh,
            aoh=train_aoh,
            pi0=pi0,
            log_Ps=log_Ps,
            Rs=Rs,
            trans_probs_j=trans_probs_j,
            discount=vi_discount,
            vi_iters=vi_iters,
            tau=tau,
            traj_chunk_size=logemit_chunk_size,
        )

        ll_traj, ll_step = masked_loglik(alpha_T, train_valid_len)
        LL_list.append(ll_step)

        g_np = np.asarray(gamma_T)
        mode_mass_hist.append(g_np.mean(axis=(0, 1)))
        ent = -np.sum(g_np * np.log(np.clip(g_np, 1e-12, 1.0)), axis=-1)
        gamma_entropy_hist.append(float(np.mean(ent)))

        gamma_used, xi_used = gamma_T, xi_T
        if eta_mstep > 0.0:
            gamma_used = (1.0 - eta_mstep) * gamma_T + eta_mstep * gamma_T
            xi_used = (1.0 - eta_mstep) * xi_T + eta_mstep * xi_T

        logpi0 = pi0_m_step(gamma_used)
        log_Ps, Rs = trans_m_step_jax_optax(
            log_Ps, Rs, (gamma_used, xi_used), train_xoh,
            num_iters=trans_iters, learning_rate=5e-4,
            ridge_R=1e-4, dir_Ps=0.0, uni_Ps=0.0,
        )

        R_state = emit_m_step_jaxnet_optax2(
            R_state,
            trans_probs_j,
            gamma_used,
            train_xoh,
            train_aoh,
            train_hoh,
            num_iters=emit_iters,
            batch_size=mstep_batch_size,          # CHANGED
            time_chunk_size=reward_time_chunk_size,  # CHANGED
            discount=vi_discount,
            vi_threshold=vi_iters,
            lr=3e-4,
            weight_decay=5e-3,
            tau=tau,
            seed=seed + it,
            lam_base=1e-3,
            lam_smooth=1e-3,
            lam_phi=1e-4,
            center_phi=True,
            absorb_goal_state=None,
        )

        print(
            f"[it {it:02d}] train_loglik_per_traj={ll_traj:.4f} "
            f"train_loglik_per_step={ll_step:.6f} mode_mass={mode_mass_hist[-1]} "
            f"gamma_entropy={gamma_entropy_hist[-1]:.4f}"
        )
        gc.collect()

    diag_hist = {
        'train_loglik_per_step': np.array(LL_list, dtype=float),
        'mode_mass': np.array(mode_mass_hist, dtype=float),
        'gamma_entropy': np.array(gamma_entropy_hist, dtype=float),
    }
    return logpi0, log_Ps, Rs, R_state, inf_state, diag_hist


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--K', type=int, default=3)
    ap.add_argument('--folder', type=str, default='../data_new')
    ap.add_argument('--emissions', type=str, default='emissions500new.npy')
    ap.add_argument('--trans_probs', type=str, default='trans_probs.npy')
    ap.add_argument('--embed_dir', type=str, default='../results/new_d32_ff128')
    ap.add_argument('--embed_file', type=str, default='h_labyrinth.npz')
    ap.add_argument('--embed_xs_file', type=str, default='xs_labyrinth_for_embed.npz')
    ap.add_argument('--data_npz', type=str, default=None)
    ap.add_argument('--out_dir', type=str, default='../results/swirl_labyrinth_history')
    ap.add_argument('--test_every', type=int, default=5)
    ap.add_argument('--reward_hidden', type=int, default=64)
    ap.add_argument('--inf_lr', type=float, default=1e-4)
    ap.add_argument('--reward_lr', type=float, default=3e-4)
    ap.add_argument('--iters', type=int, default=25)
    ap.add_argument('--trans_iters', type=int, default=150)
    ap.add_argument('--emit_iters', type=int, default=50)
    ap.add_argument('--tau', type=float, default=1.0)
    ap.add_argument('--discount', type=float, default=0.95)
    ap.add_argument('--vi_iters', type=int, default=50)
    ap.add_argument('--logemit_chunk_size', type=int, default=2)   # CHANGED
    ap.add_argument('--mstep_batch_size', type=int, default=1)     # CHANGED
    ap.add_argument('--reward_time_chunk_size', type=int, default=32)  # CHANGED
    ap.add_argument('--reward_avg_time_chunk_size', type=int, default=64)  # CHANGED
    return ap.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.data_npz is not None:
        z = np.load(args.data_npz, allow_pickle=True)
        xs = np.asarray(z['xs'], dtype=np.int64)
        valid_len = np.asarray(z['valid_len'], dtype=np.int64) if 'valid_len' in z.files else infer_valid_len_from_xs(xs)
        if 'acs' in z.files:
            acs = np.asarray(z['acs'], dtype=np.int64)
        else:
            emissions = np.load(Path(args.folder) / args.emissions, allow_pickle=True)
            acs = np.asarray(emissions[:, :, 1], dtype=np.int64)
    else:
        emissions = np.load(Path(args.folder) / args.emissions, allow_pickle=True)
        xs = np.asarray(emissions[:, :, 0], dtype=np.int64)
        acs = np.asarray(emissions[:, :, 1], dtype=np.int64)
        valid_len = infer_valid_len_from_xs(xs)

    trans_probs = np.asarray(np.load(Path(args.folder) / args.trans_probs, allow_pickle=True), dtype=np.float32)
    S, A, _ = trans_probs.shape

    h_npz = np.load(Path(args.embed_dir) / args.embed_file, allow_pickle=True)
    h_all = np.asarray(h_npz['h'] if 'h' in h_npz.files else h_npz[h_npz.files[0]], dtype=np.float32)

    xs_embed = None
    embed_xs_path = Path(args.embed_dir) / args.embed_xs_file
    if embed_xs_path.exists():
        xs_embed_z = np.load(embed_xs_path, allow_pickle=True)
        xs_embed = np.asarray(xs_embed_z['xs'] if 'xs' in xs_embed_z.files else xs_embed_z[xs_embed_z.files[0]], dtype=np.int64)

    xs, h_all = align_embeddings(xs, h_all, xs_embed)
    acs = acs[:xs.shape[0], :xs.shape[1]]
    valid_len = np.minimum(valid_len[:xs.shape[0]], xs.shape[1])

    test_indices = np.arange(0, xs.shape[0], args.test_every).astype(int)
    train_indices = np.setdiff1d(np.arange(xs.shape[0]), test_indices).astype(int)

    train_xs, test_xs = xs[train_indices], xs[test_indices]
    train_acs, test_acs = acs[train_indices], acs[test_indices]
    train_hs, test_hs = h_all[train_indices], h_all[test_indices]
    train_valid_len, test_valid_len = valid_len[train_indices], valid_len[test_indices]

    train_xoh = make_xoh(train_xs, S)
    test_xoh = make_xoh(test_xs, S)
    train_aoh = make_aoh(train_acs, A)
    test_aoh = make_aoh(test_acs, A)
    train_hoh = make_hoh(train_hs)
    test_hoh = make_hoh(test_hs)
    trans_probs_j = jnp.array(trans_probs, dtype=jnp.float32)

    rng = jax.random.PRNGKey(args.seed)
    H = train_hs.shape[-1]
    input_size = S + H

    R_state = create_reward_state(
        rng, input_size=input_size, hidden_size=args.reward_hidden,
        K=args.K, A=A, lr=args.reward_lr, weight_decay=5e-3,
    )
    inf_state = create_inf_state(
        jax.random.PRNGKey(args.seed + 123),
        K=args.K, H=H, S=S, A=A, lr=args.inf_lr,
        d_h=64, d_x=32, d_a=16, d_model=64, n_hidden=2, init_stay_bias=3.0,
    )

    logpi0_start = jnp.log(jnp.ones((args.K,), dtype=jnp.float32) / args.K)
    Ps0 = 0.95 * np.eye(args.K, dtype=np.float32) + 0.05 * np.random.rand(args.K, args.K).astype(np.float32)
    Ps0 /= Ps0.sum(axis=1, keepdims=True)
    log_Ps_start = jnp.log(jnp.array(Ps0, dtype=jnp.float32))
    Rs_start = jnp.zeros((S, 1, args.K), dtype=jnp.float32)

    new_logpi0, new_log_Ps, new_Rs, new_R_state, inf_state, diag_hist = em_train(
        logpi0=logpi0_start,
        log_Ps=log_Ps_start,
        Rs=Rs_start,
        R_state=R_state,
        inf_state=inf_state,
        train_hoh=train_hoh,
        train_xoh=train_xoh,
        train_aoh=train_aoh,
        train_valid_len=train_valid_len,
        trans_probs_j=trans_probs_j,
        iters=args.iters,
        trans_iters=args.trans_iters,
        emit_iters=args.emit_iters,
        vi_discount=args.discount,
        vi_iters=args.vi_iters,
        tau=args.tau,
        logemit_chunk_size=args.logemit_chunk_size,
        mstep_batch_size=args.mstep_batch_size,
        reward_time_chunk_size=args.reward_time_chunk_size,
        eta_mstep=0.0,
        seed=args.seed,
    )

    pi0 = jnp.exp(new_logpi0 - jax_logsumexp(new_logpi0))

    train_gamma, _, train_alpha = run_e_step_chunked(
        R_state=new_R_state, inf_state=inf_state,
        hoh=train_hoh, xoh=train_xoh, aoh=train_aoh,
        pi0=pi0, log_Ps=new_log_Ps, Rs=new_Rs, trans_probs_j=trans_probs_j,
        discount=args.discount, vi_iters=args.vi_iters, tau=args.tau,
        traj_chunk_size=args.logemit_chunk_size,
    )
    test_gamma, _, test_alpha = run_e_step_chunked(
        R_state=new_R_state, inf_state=inf_state,
        hoh=test_hoh, xoh=test_xoh, aoh=test_aoh,
        pi0=pi0, log_Ps=new_log_Ps, Rs=new_Rs, trans_probs_j=trans_probs_j,
        discount=args.discount, vi_iters=args.vi_iters, tau=args.tau,
        traj_chunk_size=args.logemit_chunk_size,
    )

    all_xoh = make_xoh(xs, S)
    all_aoh = make_aoh(acs, A)
    all_hoh = make_hoh(h_all)
    all_gamma, _, all_alpha = run_e_step_chunked(
        R_state=new_R_state, inf_state=inf_state,
        hoh=all_hoh, xoh=all_xoh, aoh=all_aoh,
        pi0=pi0, log_Ps=new_log_Ps, Rs=new_Rs, trans_probs_j=trans_probs_j,
        discount=args.discount, vi_iters=args.vi_iters, tau=args.tau,
        traj_chunk_size=args.logemit_chunk_size,
    )

    train_ll_traj, train_ll_step = masked_loglik(train_alpha, train_valid_len)
    test_ll_traj, test_ll_step = masked_loglik(test_alpha, test_valid_len)

    R_avg_ksa, R_avg_ks = compute_R_avg_streaming(
        new_R_state, all_hoh, all_gamma, trans_probs_j, valid_len,
        discount=args.discount, time_chunk_size=args.reward_avg_time_chunk_size,
    )

    result_npz = out_dir / f'{args.seed}_labyrinth_history_swirl.npz'
    np.savez(
        result_npz,
        new_logpi0=np.array(new_logpi0),
        new_log_Ps=np.array(new_log_Ps),
        new_Rs=np.array(new_Rs),
        train_gamma=np.array(train_gamma),
        test_gamma=np.array(test_gamma),
        all_gamma=np.array(all_gamma),
        R_avg_ksa=R_avg_ksa,
        R_avg_ks=R_avg_ks,
        valid_len=np.array(valid_len),
        train_indices=np.array(train_indices),
        test_indices=np.array(test_indices),
        train_loglik_per_traj=float(train_ll_traj),
        train_loglik_per_step=float(train_ll_step),
        test_loglik_per_traj=float(test_ll_traj),
        test_loglik_per_step=float(test_ll_step),
        mode_mass_all=np.array(mean_mode_mass(all_gamma), dtype=float),
        diag_train_loglik_per_step=np.array(diag_hist['train_loglik_per_step']),
        diag_mode_mass=np.array(diag_hist['mode_mass']),
        diag_gamma_entropy=np.array(diag_hist['gamma_entropy']),
    )

    with open(out_dir / f'{args.seed}_reward_params.msgpack', 'wb') as f:
        f.write(to_bytes(new_R_state.params))
    with open(out_dir / f'{args.seed}_inf_params.msgpack', 'wb') as f:
        f.write(to_bytes(inf_state.params))
    with open(out_dir / f'{args.seed}_config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    its = np.arange(len(diag_hist['train_loglik_per_step']))
    plt.figure(figsize=(7, 5))
    plt.plot(its, diag_hist['train_loglik_per_step'], label='train loglik / step')
    plt.xlabel('EM iteration')
    plt.ylabel('value')
    plt.title('Labyrinth history-SWIRL log-likelihood')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f'{args.seed}_labyrinth_loglik_curve.png', dpi=200)
    plt.close()

    plt.figure(figsize=(7, 5))
    mm = np.asarray(diag_hist['mode_mass'])
    for k in range(mm.shape[1]):
        plt.plot(its, mm[:, k], label=f'mode {k}')
    plt.xlabel('EM iteration')
    plt.ylabel('mean posterior mass')
    plt.title('Labyrinth history-SWIRL mode mass')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f'{args.seed}_labyrinth_mode_mass_curve.png', dpi=200)
    plt.close()

    print(f'[ok] saved results to {result_npz}')
    print(f'[summary] train_loglik_per_step={train_ll_step:.6f} test_loglik_per_step={test_ll_step:.6f}')
    print(f'[summary] mode_mass_all={mean_mode_mass(all_gamma)}')


if __name__ == '__main__':
    main()
