"""
Analysis utilities for the Caltech SWIRL experiment.
Adapted from spontda/swirl/da_analysis.py (dopamine-specific parts removed).
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap


def get_reward_m(trans_probs, R_params, apply_fn):
    """
    Extract the learned reward matrix R(hidden, state, action) using the
    second-order (expanded) network: input is the C×C outer-product encoding.
    """
    n_states, n_actions, _ = trans_probs.shape
    reshape_func = lambda x: (
        jnp.tile(jnp.expand_dims(x, axis=-1), (1,) * x.ndim + (n_states,)) / n_states
    ).reshape(*x.shape[:-1], x.shape[-1] * x.shape[-1])

    def get_reward_single(curr_s):
        one_hot_input = jax.nn.one_hot(curr_s, n_states)
        return apply_fn({'params': R_params}, reshape_func(one_hot_input))

    reward_net = vmap(get_reward_single)(jnp.arange(n_states)).transpose(1, 0, 2)
    return reward_net


def get_reward_nm(trans_probs, R_params, apply_fn):
    """
    Extract the learned reward matrix using the first-order network:
    input is a C-dimensional one-hot encoding.
    """
    n_states, n_actions, _ = trans_probs.shape

    def get_reward_single(curr_s):
        one_hot_input = jax.nn.one_hot(curr_s, n_states)
        return apply_fn({'params': R_params}, one_hot_input)

    reward_net = vmap(get_reward_single)(jnp.arange(n_states)).transpose(1, 0, 2)
    return reward_net


def zscore(matrix, axis=1):
    """Z-score normalise along the given axis."""
    mean = np.nanmean(matrix, axis=axis, keepdims=True)
    std  = np.nanstd(matrix,  axis=axis, keepdims=True)
    return (matrix - mean) / std
