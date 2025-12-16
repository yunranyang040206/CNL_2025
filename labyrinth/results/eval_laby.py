import argparse
import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap
from functools import partial
import sys
import os
import matplotlib.pyplot as plt

# -----------------------------
# Path + imports from project
# -----------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT)

from swirl.swirl_func import vi_temp   # soft value iteration


# -----------------------------
# Helper: load trained S-2 parameters
# -----------------------------
def load_results(results_dir, K, seed):
    """
    Load S-2 results saved by run_labyrinth.py:
      - new_logpi0:    log initial distribution over modes (size K)
      - new_log_Ps:    log transition matrix over modes (K x K)
      - new_Rs:        ARHMM transition network params (unused here)
      - new_reward:    per-mode reward over extended states (K, 1, N_ext)
      - temps:         temperature per mode (K,)
      - new_trans_probs: extended transition kernel (N_ext, A, N_ext)
      - invalid_indices: mask of invalid (base_state, orientation) slots
    """
    path = f"{results_dir}/{K}_{seed}_NM_labyrinth2.npz"
    data = np.load(path, allow_pickle=True)

    logpi0 = np.array(data["new_logpi0"], dtype=float)
    log_Ps = np.array(data["new_log_Ps"], dtype=float)
    Rs = data["new_Rs"]
    reward = data["new_reward"]
    temps = data["temps"]

    new_trans_probs = data["new_trans_probs"] if "new_trans_probs" in data.files else None
    invalid_indices = data["invalid_indices"] if "invalid_indices" in data.files else None

    return logpi0, log_Ps, Rs, reward, temps, new_trans_probs, invalid_indices


# -----------------------------
# Build soft policy π_k(s,a) for each mode k
# -----------------------------
def build_policy(new_trans_probs, reward, temps):
    """
    new_trans_probs: (N_ext, A, N_ext)
    reward:          (K, 1, N_ext)
    temps:           (K,)
    returns:
        pi_modes: (K, N_ext, A)  where pi_modes[k, s, a] = π_k(a | s_ext)
    """
    K = reward.shape[0]
    n_states_ext = new_trans_probs.shape[0]
    n_actions = new_trans_probs.shape[1]

    # R_k(s,a) = R_k_state(s) for all actions a (broadcast over actions)
    rewards_sa = np.expand_dims(reward[:, 0, :], axis=2) * np.ones((K, n_states_ext, n_actions))
    rewards_sa_j = jnp.array(rewards_sa)
    temps_j = jnp.array(temps)

    # vi_temp: (trans_probs, R_sa, temp) -> (pi, V, Q)
    pi, _, _ = vmap(partial(vi_temp, new_trans_probs))(rewards_sa_j, temps_j)
    return np.array(pi)  # (K, N_ext, A)


# -----------------------------
# Utilities: softmax from log-probs
# -----------------------------
def softmax_log(vec):
    """
    vec: 1D array of log-probabilities
    returns: normalized probabilities
    """
    v = np.array(vec, dtype=float)
    v_max = np.max(v)
    exps = np.exp(v - v_max)
    exps_sum = np.sum(exps)
    if exps_sum == 0:
        # fallback: uniform if all underflow
        return np.ones_like(v) / len(v)
    return exps / exps_sum


# -----------------------------
# Roll out trajectories from full SWIRL generative model (Option A)
# -----------------------------
def sample_trajectories_mode_aware(pi_modes, new_trans_probs,
                                   logpi0, log_Ps,
                                   start_states_ext, T, rng):
    """
    Sample trajectories from the full switching model:

      z_0 ~ π0
      for t = 0..T-1:
        a_t ~ π_{z_t}(· | s_t)
        s_{t+1} ~ P(· | s_t, a_t)
        z_{t+1} ~ P_mode(· | z_t)

    Inputs:
      pi_modes:         (K, N_ext, A) soft policies per mode
      new_trans_probs:  (N_ext, A, N_ext)
      logpi0:           (K,) log initial mode distribution
      log_Ps:           (K, K) log mode transition matrix
      start_states_ext: (N_traj,) initial extended state indices
      T:                trajectory length
      rng:              np.random.Generator

    Returns:
      traj_states_ext: (N_traj, T) extended state sequence
      traj_modes:      (N_traj, T) latent mode sequence
    """
    K, N_ext, n_actions = pi_modes.shape
    n_traj = len(start_states_ext)

    traj_states_ext = np.zeros((n_traj, T), dtype=int)
    traj_modes = np.zeros((n_traj, T), dtype=int)

    # Convert log-probs to prob distributions once
    pi0_probs = softmax_log(logpi0)          # (K,)
    mode_trans_probs = np.vstack([softmax_log(row) for row in log_Ps])  # (K, K)

    for i in range(n_traj):
        s = start_states_ext[i]

        # Sample initial mode
        z = rng.choice(K, p=pi0_probs)

        traj_states_ext[i, 0] = s
        traj_modes[i, 0] = z

        for t in range(T - 1):
            # Policy for current mode z at state s
            probs_a = pi_modes[z, s]  # (A,)
            # Guard against numerical issues
            probs_a = np.clip(probs_a, 1e-12, 1.0)
            probs_a = probs_a / probs_a.sum()

            a = rng.choice(n_actions, p=probs_a)

            # Environment transition
            probs_next = new_trans_probs[s, a]  # (N_ext,)
            probs_next = np.clip(probs_next, 1e-12, 1.0)
            probs_next = probs_next / probs_next.sum()
            s_next = rng.choice(N_ext, p=probs_next)

            # Mode transition
            probs_z_next = mode_trans_probs[z]  # (K,)
            z_next = rng.choice(K, p=probs_z_next)

            traj_states_ext[i, t + 1] = s_next
            traj_modes[i, t + 1] = z_next

            s, z = s_next, z_next

    return traj_states_ext, traj_modes


# -----------------------------
# Convert extended states -> (x,y) using maze geometry
# -----------------------------
def trajectories_to_xy(traj_states_ext, m_xc, m_yc):
    # collapse orientation index: base_state = s_ext // 4
    base_states = traj_states_ext // 4
    xs = m_xc[base_states]
    ys = m_yc[base_states]
    return xs, ys


def compute_occupancy(base_states, n_base_states):
    counts = np.bincount(base_states.ravel(), minlength=n_base_states).astype(float)
    probs = counts / counts.sum()
    return probs


def js_divergence(p, q, eps=1e-12):
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return 0.5 * (kl_pm + kl_qm)


# -----------------------------
# Main evaluation: compare generated vs real trajectories
# -----------------------------
def eval_policy(data_dir, results_dir, K, seed, n_eval_traj=100, rng_seed=0):
    # --- Load environment & real data ---
    emissions = np.load(f"{data_dir}/emissions500new.npy")
    xs = emissions[:, :, 0]  # (N_traj, T) base state indices
    # acs = emissions[:, :, 1]  # actions (not used here)
    # trans_probs = np.load(f"{data_dir}/trans_probs.npy")  # original env kernel (not used)

    maze_info = np.load(f"{data_dir}/maze_info.npz", allow_pickle=True)
    m_wa, m_ru, m_xc, m_yc = maze_info["m_wa"], maze_info["m_ru"], maze_info["m_xc"], maze_info["m_yc"]

    # --- Load S-2 params (including new_trans_probs + invalid_indices) ---
    logpi0, log_Ps, Rs, reward, temps, new_trans_probs, invalid_indices = load_results(results_dir, K, seed)
    if new_trans_probs is None:
        raise ValueError(
            "new_trans_probs not found in results .npz. "
            "Make sure you updated run_labyrinth.py to save new_trans_probs and re-ran training."
        )

    n_states_ext = new_trans_probs.shape[0]
    n_base_states = m_xc.shape[0]

    # --- Build per-mode policies π_k (S-2 soft policies) ---
    pi_modes = build_policy(jnp.array(new_trans_probs), reward, temps)  # (K, N_ext, A)

    # --- Select a subset of real trajectories for comparison ---
    n_total, T = xs.shape
    rng = np.random.default_rng(rng_seed)
    n_eval_traj = min(n_eval_traj, n_total)
    idx = rng.choice(n_total, size=n_eval_traj, replace=False)
    xs_eval = xs[idx]
    start_base_states = xs_eval[:, 0]

    # --- Map starting base states to valid extended states (orientation slots) ---
    if invalid_indices is not None:
        start_states_ext = []
        for b in start_base_states:
            valid_orients = np.where(~invalid_indices[b])[0]
            if len(valid_orients) == 0:
                ori = 0
            else:
                ori = rng.choice(valid_orients)
            start_states_ext.append(int(b * 4 + ori))
        start_states_ext = np.array(start_states_ext, dtype=int)
    else:
        # Fallback: assume orientation index 0 for all
        start_states_ext = start_base_states * 4

    # --- Generate trajectories from the full switching model (Option A) ---
    traj_states_ext_gen, traj_modes_gen = sample_trajectories_mode_aware(
        pi_modes, new_trans_probs,
        logpi0, log_Ps,
        start_states_ext, T, rng
    )
    base_states_gen = traj_states_ext_gen // 4

    # --- Occupancy distributions over base states ---
    occ_real = compute_occupancy(xs_eval, n_base_states)
    occ_gen = compute_occupancy(base_states_gen, n_base_states)

    total_variation = 0.5 * np.abs(occ_real - occ_gen).sum()
    js = js_divergence(occ_real, occ_gen)

    # --- Path-wise average position error ---
    real_x = m_xc[xs_eval]
    real_y = m_yc[xs_eval]
    gen_x, gen_y = trajectories_to_xy(traj_states_ext_gen, m_xc, m_yc)
    mse_pos = ((real_x - gen_x) ** 2 + (real_y - gen_y) ** 2).mean()

    metrics = {
        "total_variation": float(total_variation),
        "js_divergence": float(js),
        "mse_position": float(mse_pos),
    }
    return metrics, occ_real, occ_gen, real_x, real_y, gen_x, gen_y, m_wa


# -----------------------------
# Plotting helpers
# -----------------------------
def plot_example_trajectories(real_x, real_y, gen_x, gen_y, m_wa, out_path):
    """
    Plot a few real vs generated trajectories over the maze.
    """
    n_traj, T = real_x.shape
    n_plot = min(10, n_traj)

    plt.figure(figsize=(6, 6))

    # Plot maze walls (assuming m_wa == 0 indicates walls)
    ys, xs = np.where(m_wa == 0)
    plt.scatter(xs, ys, s=1, alpha=0.2)

    # Plot a few real trajectories and generated trajectories
    for i in range(n_plot):
        plt.plot(real_x[i], real_y[i], alpha=0.6, linewidth=1.0,
                 label="real" if i == 0 else "")
        plt.plot(gen_x[i], gen_y[i], alpha=0.6, linestyle="--", linewidth=1.0,
                 label="generated" if i == 0 else "")

    plt.gca().invert_yaxis()  # to match image coordinates
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Real vs Generated Trajectories (mode-aware)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_occupancy_bar(occ_real, occ_gen, out_path, max_states=50):
    """
    Simple bar plot of occupancy over state index.
    For many states, we only show the first max_states.
    """
    n_states = len(occ_real)
    n = min(n_states, max_states)
    idx = np.arange(n)

    plt.figure(figsize=(8, 4))
    plt.bar(idx - 0.2, occ_real[:n], width=0.4, label="real")
    plt.bar(idx + 0.2, occ_gen[:n], width=0.4, label="generated")
    plt.xlabel("State index (truncated)")
    plt.ylabel("Occupancy probability")
    plt.title("Real vs Generated State Occupancy (first {} states)".format(n))
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="../data")
    ap.add_argument("--results_dir", type=str, default="../results")
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n_eval_traj", type=int, default=100)
    ap.add_argument("--rng_seed", type=int, default=0)
    args = ap.parse_args()

    metrics, occ_real, occ_gen, real_x, real_y, gen_x, gen_y, m_wa = eval_policy(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        K=args.K,
        seed=args.seed,
        n_eval_traj=args.n_eval_traj,
        rng_seed=args.rng_seed,
    )

    print("Evaluation metrics for generated vs real trajectories (mode-aware):")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")

    # Save plots
    traj_plot_path = os.path.join(args.results_dir, f"{args.K}_{args.seed}_traj_compare_modeaware.png")
    occ_plot_path = os.path.join(args.results_dir, f"{args.K}_{args.seed}_occupancy_compare_modeaware.png")

    plot_example_trajectories(real_x, real_y, gen_x, gen_y, m_wa, traj_plot_path)
    plot_occupancy_bar(occ_real, occ_gen, occ_plot_path)

    print(f"Saved trajectory comparison to: {traj_plot_path}")
    print(f"Saved occupancy comparison to:  {occ_plot_path}")


if __name__ == "__main__":
    main()
