import numpy as np
from scipy.optimize import linear_sum_assignment

import jax
import jax.numpy as jnp
from jax import vmap
import itertools
from typing import Tuple, Dict, Optional

def get_reward_nm(trans_probs, R_params, apply_fn):
    n_states, n_actions, _ = trans_probs.shape
    def get_reward_single(curr_s):
        one_hot_input = jax.nn.one_hot(curr_s, n_states)
        # Combine one-hot encodings
        return apply_fn({'params': R_params}, one_hot_input)  # Apply the network to get R(hidden, a)
        
    reward_net = vmap(get_reward_single)(jnp.arange(n_states)).transpose(1, 0, 2)
    return reward_net

def get_reward_m(trans_probs, R_params, apply_fn):
    n_states, n_actions, _ = trans_probs.shape
    reshape_func = lambda x: (jnp.tile(jnp.expand_dims(x, axis=-1), (1,) * (x.ndim) + (n_states,)) / n_states).reshape(*x.shape[:-1], x.shape[-1] * x.shape[-1])
    def get_reward_single(curr_s):
        one_hot_input = jax.nn.one_hot(curr_s, n_states)
        # Combine one-hot encodings
        return apply_fn({'params': R_params}, reshape_func(one_hot_input))  # Apply the network to get R(hidden, a)
        
    reward_net = vmap(get_reward_single)(jnp.arange(n_states)).transpose(1, 0, 2)
    return reward_net


def compute_accuracy(zs_pred, zs_true):
    # Flatten the arrays to 1D
    zs_pred_flat = np.array(zs_pred).flatten()
    zs_true_flat = np.array(zs_true).flatten()

    # Get the union of unique labels from both arrays
    labels = np.unique(np.concatenate((zs_true_flat, zs_pred_flat)))
    K = len(labels)

    # Map each label to a unique index
    label_to_index = {label: idx for idx, label in enumerate(labels)}

    # Initialize the confusion matrix
    confusion_matrix = np.zeros((K, K), dtype=int)

    # Populate the confusion matrix
    for t_label, p_label in zip(zs_true_flat, zs_pred_flat):
        idx_true = label_to_index[t_label]
        idx_pred = label_to_index[p_label]
        confusion_matrix[idx_true, idx_pred] += 1

    # Apply the Hungarian algorithm to maximize correct label assignments
    cost_matrix = -confusion_matrix  # Negate for maximization
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Create a mapping from predicted labels to true labels
    mapping = {}
    for idx_true, idx_pred in zip(row_ind, col_ind):
        label_true = labels[idx_true]
        label_pred = labels[idx_pred]
        mapping[label_pred] = label_true

    # Apply the optimal label mapping to the predicted labels
    zs_pred_mapped = np.vectorize(lambda x: mapping.get(x, x))(zs_pred_flat)
    zs_pred_mapped = zs_pred_mapped.reshape(zs_pred.shape)

    # Calculate the accuracy
    accuracy = np.mean(zs_pred_mapped == zs_true)

    return accuracy

def nan_corr(x, y):
    # Mask NaN values
    x = x.flatten()
    y = y.flatten()
    mask = ~np.isnan(x) & ~np.isnan(y)
    
    # Apply mask to both x and y
    x_valid = x[mask]
    y_valid = y[mask]
    
    # Compute the Pearson correlation coefficient for valid values
    if len(x_valid) == 0 or len(y_valid) == 0:
        return np.nan  # Return NaN if no valid values
    corr = np.corrcoef(x_valid, y_valid)[0, 1]
    
    return round(corr, 3)

def best_perm_corr(x, y):
    # Store the best correlation and best permutation
    best_corr = -np.inf
    best_perm = None
    
    # Generate all possible permutations of y along axis 0
    for perm in itertools.permutations(y):
        perm = np.array(perm)
        
        # Compute the correlation for the current permutation
        corr = nan_corr(x, perm)
        
        # Check if this is the best correlation so far
        if corr > best_corr:
            best_corr = corr
            best_perm = perm
    
    return best_corr, best_perm

def calibrate_reward(reward_learnt, RG):
    
    # Compute the mean of the difference along axes (1, 2), keeping the dimensions
    mean_diff = np.nanmean(RG - reward_learnt, axis=(1, 2), keepdims=True)
    
    # Update reward_filtered_net
    updated_reward = reward_learnt + mean_diff
    
    return updated_reward


def _mask_invalid_transitions_with_nan(R: np.ndarray, trans_probs: np.ndarray) -> np.ndarray:
    """
    R: (K, C, C) reward on transitions s->s'
    trans_probs: (C, A, C). invalid transition if for a given (s,s') all actions have prob 0.
    """
    invalid = np.all(trans_probs == 0, axis=1)  # (C, C)
    Rm = np.array(R, copy=True)
    Rm[:, invalid] = np.nan
    return Rm

def reward_recovery_score(
    RG: np.ndarray,
    learned_reward_net: np.ndarray,
    trans_probs: np.ndarray,
    *,
    reduce_actions: str = "mean",
    K: Optional[int] = None,
    C: Optional[int] = None,
    kind: str = "nm",  # "nm" => learned_reward_net already (K, C*C, A) or (K, C, A) depending
    expanded_order: str = "s_prev"
) -> Tuple[float, np.ndarray, Dict]:
    """
    Computes reward recovery vs ground truth RG.

    Inputs
    - RG: ground truth reward tensor, expected shape (K, C, C) for transition rewards.
    - learned_reward_net: output of get_reward_nm/get_reward_m (JAX->numpy),
        typically:
          * get_reward_nm: (K, n_states, n_actions) where n_states=C*C (expanded state space)
          * get_reward_m:  (K, n_states, n_actions) where n_states=C (state-only)
    - trans_probs: (C, A, C) original GW5 transitions (for masking invalid s->s').

    Steps (rigorous)
    1) convert learned reward to comparable (K, C, C) form
    2) mask invalid transitions with NaNs (both RG and learned)
    3) calibrate additive offsets per mode (mean-align under NaNs)
    4) choose best latent-mode permutation (label symmetry)
    5) compute NaN-masked Pearson corr

    Returns
    - best_corr: float
    - learned_aligned: (K, C, C) learned reward after calibration + best permutation
    - info: dict with intermediate arrays
    """
    # infer K, C if not provided
    if K is None:
        K = RG.shape[0]
    if C is None:
        C = RG.shape[1]

    RG = np.asarray(RG)
    if RG.shape != (K, C, C):
        raise ValueError(f"Expected RG shape (K,C,C)={(K,C,C)} but got {RG.shape}")

    # --- Step 1: convert learned reward net to (K, C, C)
    lr = np.asarray(learned_reward_net)

    # If it's the expanded-state version: n_states = C*C
    if lr.ndim != 3 or lr.shape[0] != K:
        raise ValueError(f"learned_reward_net expected (K, n_states, A); got {lr.shape}")

    n_states = lr.shape[1]

    # reduce action dimension (reward net outputs per-action reward-like values)
    if reduce_actions == "mean":
        lr_red = lr.mean(axis=-1)  # (K, n_states)
    elif reduce_actions == "max":
        lr_red = lr.max(axis=-1)
    else:
        raise ValueError("reduce_actions must be 'mean' or 'max'")

    if n_states == C*C:
        learned_R = lr_red.reshape((K, C, C))
        if expanded_order == "s_sprev":   # your run_gw5 construction
            learned_R = learned_R.transpose(0, 2, 1)
    elif n_states == C:
        # state-only reward: tile across next-state to compare with RG (like your S1 block)
        learned_R = np.tile(lr_red[:, :, None], (1, 1, C))  # (K, C, C)
    else:
        raise ValueError(f"Unexpected n_states={n_states}; expected C or C*C")

    # --- Step 2: mask invalid transitions (NaNs)
    RG_masked = _mask_invalid_transitions_with_nan(RG, trans_probs)
    learned_masked = _mask_invalid_transitions_with_nan(learned_R, trans_probs)

    # --- Step 3: calibrate additive offset per mode (IRL invariance)
    learned_cal = calibrate_reward(learned_masked, RG_masked)

    # --- Step 4: best permutation (label symmetry)
    best_corr, learned_perm = best_perm_corr(RG_masked, learned_cal)

    info = {
        "RG_masked": RG_masked,
        "learned_raw": learned_R,
        "learned_masked": learned_masked,
        "learned_cal": learned_cal,
    }
    return best_corr, learned_perm, info


def summarize_heldout_likelihood(
    ll_train_per_step: float,
    ll_test_per_step: float,
    *,
    name: str = "model",
) -> Dict[str, float]:
    """
    Standardizes held-out likelihood reporting.

    Inputs are assumed to already be averaged per-step:
      ll_train_per_step = (1/(N*T)) * log p(a|x) on train
      ll_test_per_step  = (1/(N*T)) * log p(a|x) on test

    Returns a dict with:
      - avg_loglik_train
      - avg_loglik_test
      - avg_nll_train
      - avg_nll_test
      - ppl_train, ppl_test  (exp(NLL), useful for comparisons)
      - generalization_gap (test - train in loglik space)
    """
    ll_tr = float(np.array(ll_train_per_step))
    ll_te = float(np.array(ll_test_per_step))
    out = {
        f"{name}_avg_loglik_train": ll_tr,
        f"{name}_avg_loglik_test": ll_te,
        f"{name}_avg_nll_train": -ll_tr,
        f"{name}_avg_nll_test": -ll_te,
        f"{name}_ppl_train": float(np.exp(-ll_tr)),
        f"{name}_ppl_test": float(np.exp(-ll_te)),
        f"{name}_loglik_gap_test_minus_train": ll_te - ll_tr,
    }
    return out


