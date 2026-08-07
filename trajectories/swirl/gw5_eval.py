import numpy as np
from itertools import permutations

import numpy as np
from typing import Dict, Tuple, Optional, Callable

from new_swirl_func import soft_vi_sa

EPS = 1e-12


def make_valid_action_mask(trans_prob: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    trans_prob: (S, A, S) transition probabilities.
    Returns mask_valid: (S, A) where True means action has at least one reachable next state.
    """
    # valid if sum_{s'} P(s'|s,a) > 0
    mask = trans_prob.sum(axis=-1) > eps
    return mask  # (S, A)

def calibrate_reward_affine(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray):
    """
    Fit pred ≈ a * gt + b on masked entries (least squares),
    then return calibrated_pred = (pred - b) / a  or a*pred+b depending on convention.

    Here we want to calibrate pred to gt, so we fit:
        gt ≈ a*pred + b  => pred_cal = a*pred + b

    Returns: pred_cal, (a,b)
    """
    x = pred[mask].reshape(-1).astype(np.float64)
    y = gt[mask].reshape(-1).astype(np.float64)

    if x.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return pred.copy(), (np.nan, np.nan)

    # Solve y = a*x + b
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    pred_cal = a * pred + b
    return pred_cal, (a, b)

def masked_corrcoef(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    """
    Pearson correlation over masked entries. Returns nan if insufficient variance.
    """
    xv = x[mask].reshape(-1).astype(np.float64)
    yv = y[mask].reshape(-1).astype(np.float64)

    if xv.size < 2 or np.std(xv) < 1e-12 or np.std(yv) < 1e-12:
        return np.nan
    return float(np.corrcoef(xv, yv)[0, 1])

def best_perm_corr_KSA(R_pred: np.ndarray, R_gt: np.ndarray, mask_SA: np.ndarray):
    """
    R_pred: (K, S, A)
    R_gt:   (K, S, A)
    mask_SA: (S, A) boolean, valid entries.

    Returns:
      best_corr, best_perm, per_mode_corrs, calib_params_per_mode
    """
    K = R_pred.shape[0]
    assert R_gt.shape[0] == K
    assert mask_SA.shape == R_pred.shape[1:]

    best = (-np.inf, None, None, None)

    for perm in permutations(range(K)):
        # permute predicted modes to match GT modes
        Rp = R_pred[list(perm)]

        per_mode_corr = []
        calib_params = []
        # Calibrate per-mode (this matches typical practice; you can also calibrate jointly)
        for k in range(K):
            Rp_cal, (a, b) = calibrate_reward_affine(Rp[k], R_gt[k], mask_SA)
            corr_k = masked_corrcoef(Rp_cal, R_gt[k], mask_SA)
            per_mode_corr.append(corr_k)
            calib_params.append((a, b))

        # Aggregate (mean ignoring NaNs)
        corr_arr = np.array(per_mode_corr, dtype=np.float64)
        agg = np.nanmean(corr_arr) if np.any(~np.isnan(corr_arr)) else np.nan

        if np.isnan(agg):
            continue
        if agg > best[0]:
            best = (agg, perm, per_mode_corr, calib_params)

    best_corr, best_perm, per_mode_corr, calib_params = best
    if best_perm is None:
        return np.nan, None, [np.nan]*R_pred.shape[0], [(np.nan, np.nan)]*R_pred.shape[0]
    return best_corr, best_perm, per_mode_corr, calib_params

def corr_and_fit_per_mode(R_pred, R_gt, mask_SA, eps=1e-12):
    K = R_pred.shape[0]
    m = mask_SA.astype(bool)
    out = []
    for k in range(K):
        x = R_pred[k][m].reshape(-1)
        y = R_gt[k][m].reshape(-1)
        x = x - x.mean()
        y = y - y.mean()
        corr = float((x @ y) / ((np.linalg.norm(x) * np.linalg.norm(y)) + eps))
        a = float((x @ y) / ((x @ x) + eps))   # slope y ≈ a x + b
        b = float(y.mean() - a * x.mean())
        out.append((corr, a, b, float(x.std()), float(y.std()), x.size))
    return out


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





# -----------------------------
# Policy comparison utilities
# -----------------------------
def _valid_action_mask(trans_prob: np.ndarray) -> np.ndarray:
    """
    Compute a boolean mask valid[s,a] = True if action a from state s
    has at least one reachable next state with nonzero probability.

    Why:
    - Your oracle explore policy should be uniform over *valid* actions.
    - KL/cross-entropy should not be polluted by invalid actions.
    """
    # trans_prob shape: (S, A, S)
    return (trans_prob > 0).any(axis=2)  # (S, A)


def _normalize_policy(pi: np.ndarray, valid: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Normalize pi over actions to be a proper distribution per (K,S).

    If 'valid' is provided, zero out invalid actions before normalizing.

    Why:
    - When you build oracle policies or reconstruct learned policies,
      numerical issues can leave tiny negatives / non-normalized rows.
    - KL requires valid distributions.
    """
    pi = np.asarray(pi, dtype=float)

    if valid is not None:
        # valid: (S,A) -> broadcast to (K,S,A)
        pi = np.where(valid[None, :, :], pi, 0.0)

    # clip negatives (just in case)
    pi = np.clip(pi, 0.0, None)

    Z = pi.sum(axis=-1, keepdims=True)  # (K,S,1)
    # If some state has no valid actions (shouldn't happen), avoid divide-by-zero
    Z = np.where(Z < EPS, 1.0, Z)

    return pi / Z


def kl_divergence_per_mode(pi_p: np.ndarray, pi_q: np.ndarray) -> np.ndarray:
    """
    KL(pi_p || pi_q) per mode (K,).
    pi_p, pi_q shape: (K,S,A)

    Why:
    - KL is invariant to reward scaling/shaping; it compares *behavioral predictions*.
    """
    pi_p = np.asarray(pi_p, dtype=float)
    pi_q = np.asarray(pi_q, dtype=float)

    # add epsilon to avoid log(0); still assumes policies were normalized
    p = np.clip(pi_p, EPS, 1.0)
    q = np.clip(pi_q, EPS, 1.0)

    # sum over S,A
    return np.sum(p * (np.log(p) - np.log(q)), axis=(1, 2))


def cross_entropy_per_mode(pi_p: np.ndarray, pi_q: np.ndarray) -> np.ndarray:
    """
    Cross-entropy H(pi_p, pi_q) per mode (K,):
      H = -E_{a~pi_p}[log pi_q(a)]

    Useful because:
    - For deterministic oracle segments, it becomes near NLL of the correct action.
    """
    p = np.clip(np.asarray(pi_p, dtype=float), EPS, 1.0)
    q = np.clip(np.asarray(pi_q, dtype=float), EPS, 1.0)
    return -np.sum(p * np.log(q), axis=(1, 2))


def entropy_per_mode(pi: np.ndarray) -> np.ndarray:
    """Entropy H(pi) per mode (K,) for diagnostics (e.g., explore should be high-entropy)."""
    p = np.clip(np.asarray(pi, dtype=float), EPS, 1.0)
    return -np.sum(p * np.log(p), axis=(1, 2))

def build_true_behavior_oracle_gw5(
    trans_prob: np.ndarray,   # (S,A,S)
    grid: int = 5,
    water_state: int = 24,
    water_random: float = 0.15,
    water_hesitate: float = 0.05,
    explore_random: float = 0.15,
) -> np.ndarray:
    """
    Reconstruct the *actual* action policy used by data_generator.py.

    Returns:
        pi_true: (2,S,A)
            mode 0 = explore
            mode 1 = water
    """
    trans_prob = np.asarray(trans_prob, dtype=float)
    S, A, S2 = trans_prob.shape
    assert S == S2
    assert S == grid * grid

    def to_xy(s):
        return s // grid, s % grid

    def to_s(x, y):
        return x * grid + y

    # shortest-path distances to water (same as generator)
    from collections import deque
    shortest_dist = np.full(S, np.inf)
    shortest_dist[water_state] = 0
    q = deque([water_state])
    while q:
        s = q.popleft()
        x, y = to_xy(s)
        nbrs = []
        if x > 0: nbrs.append(to_s(x - 1, y))
        if x < grid - 1: nbrs.append(to_s(x + 1, y))
        if y > 0: nbrs.append(to_s(x, y - 1))
        if y < grid - 1: nbrs.append(to_s(x, y + 1))
        for n in nbrs:
            if shortest_dist[n] == np.inf:
                shortest_dist[n] = shortest_dist[s] + 1
                q.append(n)

    pi = np.zeros((2, S, A), dtype=float)

    for s in range(S):
        x, y = to_xy(s)

        # -------------------------------------------------
        # Mode 1: true choose_water_action(s)
        # -------------------------------------------------
        # Generator logic:
        #   if rand < WATER_HESITATE: stay
        #   elif rand < WATER_RANDOM: random over all A actions
        #   else: choose uniformly among "best" actions that reduce shortest_dist,
        #         and if none exist, stay.
        best = []
        for a, (nx, ny) in enumerate([
            (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1), (x, y)
        ]):
            if 0 <= nx < grid and 0 <= ny < grid:
                ns = to_s(nx, ny)
                if shortest_dist[ns] < shortest_dist[s]:
                    best.append(a)

        # first branch: hesitate -> stay
        pi[1, s, 4] += water_hesitate

        # second branch: random over all actions, only reached if first branch failed
        p_random_uncond = (1.0 - water_hesitate) * water_random
        pi[1, s, :] += p_random_uncond / A

        # third branch: best action(s), only reached if neither first nor second triggered
        p_best_uncond = (1.0 - water_hesitate) * (1.0 - water_random)
        if len(best) > 0:
            pi[1, s, best] += p_best_uncond / len(best)
        else:
            pi[1, s, 4] += p_best_uncond

        # -------------------------------------------------
        # Mode 0: true choose_explore_action(s)
        # -------------------------------------------------
        # valid = actions whose deterministic next state is not WATER
        valid = []
        for a in range(A):
            ns = int(np.argmax(trans_prob[s, a]))
            if ns != water_state:
                valid.append(a)
        if len(valid) == 0:
            valid = list(range(A))

        # random valid branch
        pi[0, s, valid] += explore_random / len(valid)

        # remaining 1-explore_random branch
        prem = 1.0 - explore_random

        # perimeter preference
        if x in [0, grid - 1] or y in [0, grid - 1]:
            pi[0, s, valid] += prem / len(valid)
        else:
            # move toward boundary
            if x > 0 and 0 in valid:
                pi[0, s, 0] += prem
            else:
                pi[0, s, valid] += prem / len(valid)

    # numerical cleanup
    pi = np.clip(pi, 0.0, None)
    row_sum = pi.sum(axis=-1, keepdims=True)
    pi = np.divide(pi, np.maximum(row_sum, 1e-12))
    return pi

# -----------------------------
# Learned policy reconstruction from R_avg (recommended)
# -----------------------------
def learned_policy_from_R_avg(
    R_avg: np.ndarray,
    soft_vi_sa: Callable[[np.ndarray, np.ndarray], np.ndarray],
    trans_prob: np.ndarray,
) -> np.ndarray:
    """
    Convert R_avg (K,S,A) into pi_learn (K,S,A) by running soft value iteration per mode.

    Why:
    - In SWIRL, reward -> soft VI -> policy.
    - Your R_avg already summarizes the history-conditioned reward outputs into a stable table.
    - This produces one policy per mode, comparable to oracle pi(a|s,z).
    """
    R_avg = np.asarray(R_avg, dtype=float)
    K, S, A = R_avg.shape

    pi = np.zeros((K, S, A), dtype=float)
    for k in range(K):
        # soft_vi_sa should return pi_sa for all states: shape (S,A)
        pi[k] = soft_vi_sa(R_avg[k], trans_prob)

    valid = _valid_action_mask(trans_prob)
    pi = _normalize_policy(pi, valid=valid)
    return pi

def build_oracle_policy_from_RG(
    trans_prob: np.ndarray,   # (S,A,S)
    RG_sa: np.ndarray,        # (K,S,A)
    discount: float = 0.95,
    vi_iters: int = 50,
    tau: float = 1.0,
) -> np.ndarray:
    """
    Oracle policy computed from the actual ground-truth reward maps RG_sa.npy
    using SoftVI with the same convention as the data generator.

    Returns:
        pi_oracle: (K,S,A)
    """
    trans_prob = np.asarray(trans_prob, dtype=float)
    RG_sa = np.asarray(RG_sa, dtype=float)

    assert trans_prob.ndim == 3, "trans_prob must be (S,A,S)"
    S, A, S2 = trans_prob.shape
    assert S == S2, "trans_prob must be (S,A,S)"
    assert RG_sa.ndim == 3, "RG_sa must be (K,S,A)"
    assert RG_sa.shape[1:] == (S, A), f"RG_sa shape {RG_sa.shape} incompatible with trans_prob {(S,A,S)}"

    K = RG_sa.shape[0]
    pi = np.zeros((K, S, A), dtype=float)

    valid = _valid_action_mask(trans_prob)

    for k in range(K):
        # soft_vi_sa in your codebase expects (trans_prob, reward_sa)
        # and already uses the same SoftVI convention.
        # To match the generator exactly, pass R/tau into SoftVI.
        pi_k = soft_vi_sa(trans_prob, RG_sa[k] / float(tau))
        pi[k] = np.asarray(pi_k, dtype=float)

    pi = _normalize_policy(pi, valid=valid)
    return pi

import numpy as np
from collections import deque

def _policy_get_mode(policy: np.ndarray, mode: int) -> np.ndarray:
    """
    Returns pi_mode as (S,A) from policy in any of these shapes:
      - (K,S,A)
      - (S,A,K)
      - (S,K,A)
    """
    if policy.ndim != 3:
        raise ValueError(f"policy must be 3D, got shape {policy.shape}")

    K, S, A = None, None, None

    # (K,S,A)
    if policy.shape[0] in (2,3,4) and policy.shape[1] == 25:
        return policy[mode]

    # (S,A,K)
    if policy.shape[0] == 25 and policy.shape[2] in (2,3,4):
        return policy[:, :, mode]

    # (S,K,A)
    if policy.shape[0] == 25 and policy.shape[1] in (2,3,4):
        return policy[:, mode, :]

    raise ValueError(f"Unrecognized policy shape: {policy.shape}")

def greedy_to_water_accuracy(policy: np.ndarray,
                            trans_prob: np.ndarray,
                            water_state: int = 24,
                            mode_water: int = 1) -> float:
    """
    Diagnostic: fraction of states where argmax action under mode_water
    moves to a next state with minimal shortest-path distance to water.

    Robust: uses graph shortest-path distances, and handles policy shape.
    """
    S, A, S2 = trans_prob.shape
    assert S == S2

    # deterministic next-state table
    nxt = np.argmax(trans_prob, axis=-1)  # (S,A)

    # compute shortest-path distances to water via reverse BFS
    rev = [[] for _ in range(S)]
    for s in range(S):
        for a in range(A):
            rev[nxt[s, a]].append(s)

    dist = np.full(S, np.inf)
    dist[water_state] = 0.0
    q = deque([water_state])
    while q:
        v = q.popleft()
        for u in rev[v]:
            if np.isinf(dist[u]):
                dist[u] = dist[v] + 1.0
                q.append(u)

    pi_mode = _policy_get_mode(policy, mode_water)  # (S,A)

    hits = 0
    total = 0
    for s in range(S):
        a_hat = int(np.argmax(pi_mode[s]))
        sp_hat = int(nxt[s, a_hat])

        best = np.min(dist[nxt[s]])  # min over actions
        hits += int(dist[sp_hat] == best)
        total += 1

    return hits / max(1, total)



def evaluate_policy_agreement(
    pi_oracle: np.ndarray,
    pi_learn: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Compute core agreement metrics between oracle and learned policies.
    Returns per-mode arrays for KL, cross-entropy, entropy, plus a few summaries.
    """
    # Assumes inputs are normalized.
    out: Dict[str, np.ndarray] = {}
    out["KL_oracle_to_learn"] = kl_divergence_per_mode(pi_oracle, pi_learn)
    out["CE_oracle_vs_learn"] = cross_entropy_per_mode(pi_oracle, pi_learn)
    out["H_oracle"] = entropy_per_mode(pi_oracle)
    out["H_learn"] = entropy_per_mode(pi_learn)
    return out


# =========================
# Time-windowed evaluation
# =========================

from typing import List, Sequence, Tuple, Dict, Any, Optional

def _as_gamma_NTK(gamma: np.ndarray) -> np.ndarray:
    """
    Normalize gamma to shape (N,T,K).
    Accepts (N,T,K), (N,T,1,K), (T,K), or (T,1,K).
    """
    g = np.asarray(gamma)
    if g.ndim == 2:              # (T,K)
        g = g[None, ...]         # (1,T,K)
    elif g.ndim == 3:
        # (N,T,K) ok
        pass
    elif g.ndim == 4:
        # common case: (N,T,1,K)
        if g.shape[2] == 1:
            g = g[:, :, 0, :]
        else:
            raise ValueError(f"gamma has 4 dims but gamma.shape[2]!=1: {g.shape}")
    else:
        raise ValueError(f"gamma must have 2/3/4 dims, got {g.ndim} with shape {g.shape}")
    return g


def _as_R_NTKSA(R_pred: np.ndarray) -> np.ndarray:
    """
    Normalize time-varying reward tensor to shape (N,T,K,S,A).
    Accepts (N,T,K,S,A) or (T,K,S,A).
    """
    R = np.asarray(R_pred)
    if R.ndim == 4:              # (T,K,S,A)
        R = R[None, ...]         # (1,T,K,S,A)
    elif R.ndim == 5:
        pass
    else:
        raise ValueError(f"R_pred must have 4/5 dims, got {R.ndim} with shape {R.shape}")
    return R


def make_time_windows(T: int, win_len: int, stride: Optional[int] = None, drop_last: bool = True) -> List[Tuple[int, int]]:
    """
    Returns list of (t0, t1) windows with t1 exclusive.
    If drop_last=False, includes a final shorter window.
    """
    if stride is None:
        stride = win_len
    if win_len <= 0 or stride <= 0:
        raise ValueError("win_len and stride must be positive")

    windows = []
    t0 = 0
    while t0 < T:
        t1 = t0 + win_len
        if t1 > T and drop_last:
            break
        t1 = min(t1, T)
        windows.append((t0, t1))
        t0 += stride
    return windows


def windowed_mode_mass(gamma: np.ndarray, windows: Sequence[Tuple[int, int]]) -> np.ndarray:
    """
    Returns mode mass per window: (W,K), where each entry is mean gamma over (N,t) in that window.
    """
    g = _as_gamma_NTK(gamma)  # (N,T,K)
    N, T, K = g.shape
    out = np.zeros((len(windows), K), dtype=np.float64)

    for w, (t0, t1) in enumerate(windows):
        gw = g[:, t0:t1, :]  # (N,tw,K)
        out[w] = gw.mean(axis=(0, 1))
    return out


def windowed_reward_map(R_pred: np.ndarray, gamma: np.ndarray, windows: Sequence[Tuple[int, int]],
                        normalize_weights: bool = True) -> np.ndarray:
    """
    Compute time-windowed reward maps using posterior weights gamma.

    R_pred: (N,T,K,S,A) or (T,K,S,A)
    gamma : (N,T,K) or (N,T,1,K) or (T,K)

    Returns: Rw of shape (W,K,S,A), where
        Rw[w,k] = sum_{n,t in win} gamma[n,t,k] * R_pred[n,t,k] / sum gamma (if normalize_weights)
               else just the unnormalized weighted sum.
    """
    R = _as_R_NTKSA(R_pred)     # (N,T,K,S,A)
    g = _as_gamma_NTK(gamma)    # (N,T,K)

    # broadcast-check
    if R.shape[0] != g.shape[0] or R.shape[1] != g.shape[1] or R.shape[2] != g.shape[2]:
        raise ValueError(f"Shape mismatch: R {R.shape} vs gamma {g.shape} (need N,T,K to match)")

    N, T, K, S, A = R.shape
    W = len(windows)
    Rw = np.zeros((W, K, S, A), dtype=np.float64)

    for w, (t0, t1) in enumerate(windows):
        Rw_slice = R[:, t0:t1, :, :, :]              # (N,tw,K,S,A)
        gw_slice = g[:, t0:t1, :]                    # (N,tw,K)

        # weights to (N,tw,K,1,1) for broadcasting over S,A
        wgt = gw_slice[..., None, None]              # (N,tw,K,1,1)
        num = (wgt * Rw_slice).sum(axis=(0, 1))      # (K,S,A)

        if normalize_weights:
            den = gw_slice.sum(axis=(0, 1))          # (K,)
            den = np.maximum(den, EPS)               # avoid /0
            Rw[w] = num / den[:, None, None]
        else:
            Rw[w] = num

    return Rw


def delta_reward_L2(Rw: np.ndarray, per_mode: bool = True) -> np.ndarray:
    """
    Compute window-to-window change magnitude for reward maps.

    Rw: (W,K,S,A)

    Returns:
      - if per_mode: (W-1, K) where each is ||Rw[w+1,k]-Rw[w,k]||_2 / sqrt(S*A)
      - else: (W-1,) averaged over modes
    """
    Rw = np.asarray(Rw, dtype=np.float64)
    if Rw.ndim != 4:
        raise ValueError(f"Rw must be (W,K,S,A), got {Rw.shape}")

    dif = Rw[1:] - Rw[:-1]  # (W-1,K,S,A)
    # normalized RMS change per window/mode
    rms = np.sqrt(np.mean(dif**2, axis=(2, 3)))  # (W-1,K)

    if per_mode:
        return rms
    return rms.mean(axis=1)


def windowed_reward_stats(Rw: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Simple per-window summaries to sanity-check variability.

    Returns dict with:
      - 'std_over_SA': (W,K) std over (S,A) inside each window/mode
      - 'max_abs':     (W,K) max |R| over (S,A)
      - 'mean_abs':    (W,K) mean |R| over (S,A)
    """
    Rw = np.asarray(Rw, dtype=np.float64)
    if Rw.ndim != 4:
        raise ValueError(f"Rw must be (W,K,S,A), got {Rw.shape}")
    std_over_SA = Rw.std(axis=(2, 3))
    max_abs = np.max(np.abs(Rw), axis=(2, 3))
    mean_abs = np.mean(np.abs(Rw), axis=(2, 3))
    return {"std_over_SA": std_over_SA, "max_abs": max_abs, "mean_abs": mean_abs}


def windowed_Rcorr_to_static_GT(Rw: np.ndarray, RG: np.ndarray, valid_mask_SA: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Correlate each window's reward map to a STATIC ground truth RG.

    Rw: (W,K,S,A)
    RG: (K,S,A) or (K,S,A,?) -> should be (K,S,A) for your setup

    valid_mask_SA: optional (S,A) boolean mask to ignore invalid actions (recommended)

    Returns: corr_wk (W,K) Pearson corr per window/mode (nan if degenerate)
    """
    Rw = np.asarray(Rw, dtype=np.float64)
    RG = np.asarray(RG, dtype=np.float64)
    if RG.ndim != 3:
        raise ValueError(f"RG should be (K,S,A), got {RG.shape}")
    if Rw.ndim != 4:
        raise ValueError(f"Rw should be (W,K,S,A), got {Rw.shape}")

    W, K, S, A = Rw.shape
    if RG.shape != (K, S, A):
        raise ValueError(f"RG shape {RG.shape} must match (K,S,A)=({K},{S},{A})")

    if valid_mask_SA is None:
        valid_mask_SA = np.ones((S, A), dtype=bool)
    mask = valid_mask_SA[None, :, :]  # (1,S,A)

    corr = np.full((W, K), np.nan, dtype=np.float64)
    for w in range(W):
        for k in range(K):
            x = Rw[w, k][mask[0]].reshape(-1)
            y = RG[k][mask[0]].reshape(-1)
            if x.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
                corr[w, k] = np.nan
            else:
                corr[w, k] = np.corrcoef(x, y)[0, 1]
    return corr


def summarize_time_windowed(R_pred: np.ndarray, gamma: np.ndarray, T: int,
                            win_len: int = 50, stride: Optional[int] = 25) -> Dict[str, Any]:
    """
    Convenience wrapper that gives you the core objects you’ll want to print/plot.

    Returns dict:
      windows: list[(t0,t1)]
      mode_mass: (W,K)
      Rw: (W,K,S,A)
      dR_rms: (W-1,K)
      stats: dict of per-window summaries
    """
    windows = make_time_windows(T, win_len, stride=stride, drop_last=True)
    mode_mass = windowed_mode_mass(gamma, windows)
    Rw = windowed_reward_map(R_pred, gamma, windows, normalize_weights=True)
    dR = delta_reward_L2(Rw, per_mode=True)
    stats = windowed_reward_stats(Rw)
    return {"windows": windows, "mode_mass": mode_mass, "Rw": Rw, "dR_rms": dR, "stats": stats}

def policy_kl(pi_p, pi_q, eps=1e-12):
    """
    KL(pi_p || pi_q) per mode.
    pi_p, pi_q: (K, S, A) stochastic policies
    returns: (K,) KL averaged over (S) with equal state weight
    """
    p = np.clip(np.asarray(pi_p), eps, 1.0)
    q = np.clip(np.asarray(pi_q), eps, 1.0)
    p = p / p.sum(axis=-1, keepdims=True)
    q = q / q.sum(axis=-1, keepdims=True)

    kl_sa = np.sum(p * (np.log(p) - np.log(q)), axis=-1)  # (K,S)
    return np.mean(kl_sa, axis=-1)                        # (K,)

def learned_policy_from_R_avg(R_avg_KSA, soft_vi_fn, trans_prob_SAS):
    """
    R_avg_KSA: (K,S,A)
    trans_prob_SAS: (S,A,S)
    soft_vi_fn: callable that takes (R_SA, trans_prob_SAS) -> pi_SA
               where pi_SA is (S,A)
    returns pi_KSA: (K,S,A)
    """
    R_avg_KSA = np.asarray(R_avg_KSA)
    K, S, A = R_avg_KSA.shape
    pi = np.zeros((K, S, A), dtype=np.float32)
    for k in range(K):
        pi[k] = soft_vi_fn(R_avg_KSA[k], trans_prob_SAS)
    return pi


def _as_z_NT(z_true: np.ndarray, T: Optional[int] = None) -> np.ndarray:
    """
    Normalize z_true to shape (N,T) int.
    Accepts (T,), (N,T).
    """
    z = np.asarray(z_true)
    if z.ndim == 1:
        z = z[None, :]
    elif z.ndim != 2:
        raise ValueError(f"z_true must have 1/2 dims, got {z.ndim} with shape {z.shape}")
    if T is not None and z.shape[1] != T:
        raise ValueError(f"z_true T mismatch: got {z.shape[1]}, expected {T}")
    return z.astype(int)


def reward_map_from_oracle_z(R_pred: np.ndarray, z_true: np.ndarray, K: int) -> np.ndarray:
    """
    Compute oracle-averaged reward map R_avg_oracle (K,S,A) by hard-assigning each (n,t)
    to its ground-truth mode z_true[n,t].

    R_pred: (N,T,K,S,A) or (T,K,S,A)
    z_true: (N,T) or (T,)
    K: number of modes

    Returns:
      R_avg_oracle: (K,S,A)
    """
    R = _as_R_NTKSA(R_pred)  # (N,T,K,S,A)
    N, T, K2, S, A = R.shape
    assert K2 == K, f"R_pred has K={K2} but K argument is {K}"

    z = _as_z_NT(z_true, T=T)  # (N,T)

    # one-hot weights: (N,T,K)
    W = np.eye(K, dtype=np.float64)[z]  # (N,T,K)

    # weighted average over (N,T)
    num = (W[..., None, None] * R).sum(axis=(0, 1))              # (K,S,A)
    den = np.maximum(W.sum(axis=(0, 1)), EPS)                    # (K,)
    R_avg_oracle = num / den[:, None, None]
    return np.asarray(R_avg_oracle, dtype=np.float64)


def oracle_Rcorr_comparison(
    R_pred: np.ndarray,
    gamma: np.ndarray,
    z_true: np.ndarray,
    RG: np.ndarray,
    trans_prob: np.ndarray,
) -> Dict[str, Any]:
    """
    Compare:
      - Posterior-weighted R_avg from gamma
      - Oracle-weighted R_avg_oracle from z_true

    Returns dict with:
      - R_corr_gamma, per_mode_corrs_gamma, perm_gamma, calib_gamma
      - R_corr_oracle, per_mode_corrs_oracle, perm_oracle, calib_oracle
      - mode_mass_gamma, mode_mass_oracle
    """
    R = _as_R_NTKSA(R_pred)      # (N,T,K,S,A)
    g = _as_gamma_NTK(gamma)     # (N,T,K)
    N, T, K, S, A = R.shape

    mask_SA = make_valid_action_mask(np.asarray(trans_prob))  # (S,A) :contentReference[oaicite:3]{index=3}

    # --- gamma-weighted avg
    num = (g[..., None, None] * R).sum(axis=(0, 1))          # (K,S,A)
    den = np.maximum(g.sum(axis=(0, 1)), EPS)                # (K,)
    R_avg = num / den[:, None, None]

    # --- oracle-weighted avg
    R_avg_oracle = reward_map_from_oracle_z(R, z_true=z_true, K=K)  # (K,S,A)

    # Compute corr with best perm for both
    R_corr_gamma, perm_gamma, per_mode_corrs_gamma, calib_gamma = best_perm_corr_KSA(
        R_pred=np.asarray(R_avg), R_gt=np.asarray(RG), mask_SA=mask_SA
    )  # :contentReference[oaicite:4]{index=4}

    R_corr_oracle, perm_oracle, per_mode_corrs_oracle, calib_oracle = best_perm_corr_KSA(
        R_pred=np.asarray(R_avg_oracle), R_gt=np.asarray(RG), mask_SA=mask_SA
    )

    # mode mass (for interpreting failures)
    mode_mass_gamma = g.mean(axis=(0, 1))  # (K,)
    z = _as_z_NT(z_true, T=T)
    mode_mass_oracle = np.array([(z == k).mean() for k in range(K)], dtype=np.float64)

    return dict(
        R_corr_gamma=R_corr_gamma, perm_gamma=perm_gamma,
        per_mode_corrs_gamma=per_mode_corrs_gamma, calib_gamma=calib_gamma,
        R_corr_oracle=R_corr_oracle, perm_oracle=perm_oracle,
        per_mode_corrs_oracle=per_mode_corrs_oracle, calib_oracle=calib_oracle,
        mode_mass_gamma=mode_mass_gamma, mode_mass_oracle=mode_mass_oracle,
    )

def per_state_policy_metrics(pi_oracle, pi_learn, eps=1e-12):
    """
    pi_oracle, pi_learn: (K,S,A)

    Returns:
        ce_state: (K,S)
        kl_state: (K,S)
        H_oracle_state: (K,S)
        H_learn_state: (K,S)
    """
    po = np.asarray(pi_oracle, dtype=np.float64)
    pl = np.asarray(pi_learn, dtype=np.float64)

    po = np.clip(po, eps, 1.0)
    pl = np.clip(pl, eps, 1.0)

    po = po / po.sum(axis=-1, keepdims=True)
    pl = pl / pl.sum(axis=-1, keepdims=True)

    ce_state = -np.sum(po * np.log(pl), axis=-1)                  # (K,S)
    H_oracle_state = -np.sum(po * np.log(po), axis=-1)            # (K,S)
    H_learn_state  = -np.sum(pl * np.log(pl), axis=-1)            # (K,S)
    kl_state = ce_state - H_oracle_state                          # (K,S)

    return dict(
        ce_state=ce_state,
        kl_state=kl_state,
        H_oracle_state=H_oracle_state,
        H_learn_state=H_learn_state,
    )

def avg_top2_gap(arr):
    """
    arr: (..., K)
    returns mean(top1 - top2)
    """
    arr = np.asarray(arr)
    part = np.partition(arr, -2, axis=-1)
    top2 = part[..., -2]
    top1 = part[..., -1]
    return float(np.mean(top1 - top2))

def realized_lls_from_logemit(logemit, xoh, aoh):
    # squeeze singleton dims
    x = np.asarray(xoh)[:, :, 0, :]   # (N,T,S)
    a = np.asarray(aoh)[:, :, 0, :]   # (N,T,A)
    le = np.asarray(logemit)          # (N,T,K,S,A)

    # pick realized state -> (N,T,K,A)
    le_tka = np.einsum("nts,ntksa->ntka", x, le)
    # pick realized action -> (N,T,K)
    lls = np.einsum("nta,ntka->ntk", a, le_tka)
    return lls

def per_state_action_metrics(pi_oracle, pi_learn, eps=1e-12):
    """
    pi_oracle, pi_learn: (K,S,A)

    Returns:
        tv_state:        (K,S)   total variation per state
        l1_state:        (K,S)   L1 distance per state
        maxprob_gap:     (K,S)   | max_a pi_learn - max_a pi_oracle |
        top1_match:      (K,S)   1 if argmax action matches, else 0
    """
    po = np.asarray(pi_oracle, dtype=np.float64)
    pl = np.asarray(pi_learn, dtype=np.float64)

    po = np.clip(po, eps, 1.0)
    pl = np.clip(pl, eps, 1.0)

    po = po / po.sum(axis=-1, keepdims=True)
    pl = pl / pl.sum(axis=-1, keepdims=True)

    l1_state = np.sum(np.abs(pl - po), axis=-1)              # (K,S)
    tv_state = 0.5 * l1_state                                # (K,S)

    maxprob_gap = np.abs(np.max(pl, axis=-1) - np.max(po, axis=-1))   # (K,S)

    a_oracle = np.argmax(po, axis=-1)                        # (K,S)
    a_learn  = np.argmax(pl, axis=-1)                        # (K,S)
    top1_match = (a_oracle == a_learn).astype(np.float64)    # (K,S)

    return dict(
        tv_state=tv_state,
        l1_state=l1_state,
        maxprob_gap=maxprob_gap,
        top1_match=top1_match,
        a_oracle=a_oracle,
        a_learn=a_learn,
    )


def summarize_action_metrics(action_dbg):
    """
    action_dbg = output of per_state_action_metrics(...)
    Prints compact per-mode summaries.
    """
    for k in range(action_dbg["tv_state"].shape[0]):
        print(
            f"[DEBUG action summary] mode {k}: "
            f"top1_match={action_dbg['top1_match'][k].mean():.4f}  "
            f"TV_mean={action_dbg['tv_state'][k].mean():.4f}  "
            f"L1_mean={action_dbg['l1_state'][k].mean():.4f}  "
            f"maxprob_gap_mean={action_dbg['maxprob_gap'][k].mean():.4f}"
        )


def print_worst_action_states(pi_oracle, pi_learn, action_dbg, top_m=10):
    """
    Print actual oracle vs learned action distributions at worst-TV states.
    """
    po = np.asarray(pi_oracle, dtype=np.float64)
    pl = np.asarray(pi_learn, dtype=np.float64)

    for k in range(po.shape[0]):
        worst = np.argsort(-action_dbg["tv_state"][k])[:top_m]
        print(f"[DEBUG action worst states] mode {k}: {worst.tolist()}")
        for s in worst:
            print(
                f"  state {int(s):2d} | "
                f"TV={action_dbg['tv_state'][k, s]:.4f} "
                f"L1={action_dbg['l1_state'][k, s]:.4f} "
                f"top1(o->l)=({int(action_dbg['a_oracle'][k, s])}->{int(action_dbg['a_learn'][k, s])})"
            )
            print(f"    oracle: {np.round(po[k, s], 4).tolist()}")
            print(f"    learn : {np.round(pl[k, s], 4).tolist()}")


def save_action_metric_heatmaps(action_dbg, grid=5, out_path="action_metric_heatmaps.png"):
    import matplotlib.pyplot as plt

    tv = action_dbg["tv_state"]
    t1 = action_dbg["top1_match"]
    mpg = action_dbg["maxprob_gap"]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    for k in range(2):
        im0 = axes[k, 0].imshow(tv[k].reshape(grid, grid))
        axes[k, 0].set_title(f"mode {k} TV")
        plt.colorbar(im0, ax=axes[k, 0], fraction=0.046, pad=0.04)

        im1 = axes[k, 1].imshow(t1[k].reshape(grid, grid), vmin=0.0, vmax=1.0)
        axes[k, 1].set_title(f"mode {k} top1 match")
        plt.colorbar(im1, ax=axes[k, 1], fraction=0.046, pad=0.04)

        im2 = axes[k, 2].imshow(mpg[k].reshape(grid, grid))
        axes[k, 2].set_title(f"mode {k} maxprob gap")
        plt.colorbar(im2, ax=axes[k, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print("Saved:", out_path)


def inspect_action_semantics(pi_oracle, pi_learn, trans_prob, states_to_check, mode_names=None):
    """
    Print oracle vs learned action semantics for selected states.

    pi_oracle, pi_learn: (K,S,A)
    trans_prob:          (S,A,S)
    states_to_check:     dict like {0: [18,7,11], 1: [24,19,23]}
    mode_names: optional list of mode labels
    """
    po = np.asarray(pi_oracle, dtype=np.float64)
    pl = np.asarray(pi_learn, dtype=np.float64)
    tp = np.asarray(trans_prob)

    K, S, A = po.shape
    if mode_names is None:
        mode_names = [f"mode {k}" for k in range(K)]

    for k, states in states_to_check.items():
        print(f"\n[DEBUG action semantics] {mode_names[k]}")
        for s in states:
            s = int(s)

            oracle_top = int(np.argmax(po[k, s]))
            learn_top  = int(np.argmax(pl[k, s]))

            next_states = [int(np.argmax(tp[s, a])) for a in range(A)]

            print(f"  state {s}")
            print(f"    oracle top action: {oracle_top}")
            print(f"    learn  top action: {learn_top}")
            print(f"    next state per action a=0..{A-1}: {next_states}")
            print(f"    oracle probs: {np.round(po[k, s], 4).tolist()}")
            print(f"    learn  probs: {np.round(pl[k, s], 4).tolist()}")


def learned_policy_from_logemit_avg(logemit, gamma):
    """
    Posterior-weighted average of the actual timestep policies used in training.

    Supports logemit in either:
      (N,T,K,S,A)   [your current build_logemit_list output]
      (N,T,S,K,A)   [older alternative layout]

    gamma:
      (N,T,K) or (N,T,1,K)

    Returns:
      pi_bar: (K,S,A)
    """
    logemit_np = np.asarray(logemit)
    gamma_np = np.asarray(gamma)

    if gamma_np.ndim == 4:   # (N,T,1,K) -> (N,T,K)
        gamma_np = gamma_np[:, :, 0, :]

    # Convert log pi to pi
    pi_t = np.exp(logemit_np)

    # Normalize layout to (N,T,K,S,A)
    if pi_t.ndim != 5:
        raise ValueError(f"logemit must be 5D, got shape {pi_t.shape}")

    # your current script uses (N,T,K,S,A)
    if pi_t.shape[2] == gamma_np.shape[2]:
        pass
    # older alternative: (N,T,S,K,A)
    elif pi_t.shape[3] == gamma_np.shape[2]:
        pi_t = np.transpose(pi_t, (0, 1, 3, 2, 4))
    else:
        raise ValueError(
            f"Cannot align logemit shape {pi_t.shape} with gamma shape {gamma_np.shape}"
        )

    # gamma: (N,T,K) -> (N,T,K,1,1)
    num = np.sum(gamma_np[:, :, :, None, None] * pi_t, axis=(0, 1))   # (K,S,A)
    den = np.sum(gamma_np, axis=(0, 1))[:, None, None] + 1e-12         # (K,1,1)

    pi_bar = num / den
    pi_bar = pi_bar / (pi_bar.sum(axis=-1, keepdims=True) + 1e-12)
    return pi_bar