import numpy as np
import numpy.random as npr
from scipy.special import logsumexp
import pandas as pd

import jax
import jax.numpy as jnp
from jax import lax, vmap, jit
from functools import partial
from jax.scipy.special import logsumexp as jax_logsumexp
import optax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import sys
if len(sys.argv) < 2:
    seed = 12345
else:
    seed = int(sys.argv[1])

K = 2
D = 1
C = 25
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
# folder = 'labyrinth/data' 
# trans_probs is generated manually now, so folder variable is less relevant unless used elsewhere.
save_folder = os.path.join(script_dir, '../results')
os.makedirs(save_folder, exist_ok=True)

# Load transition probabilities
# data trans_probs.npy is for 127 states (Labyrinth?), but we need 5x5 GW (25 states).
# We generate it manually.
trans_probs = np.zeros((C, 5, C)) # 25 states, 5 actions (up, down, left, right, stay)

# Action map from generate_dataset.py: 0:(-1,0), 1:(1,0), 2:(0,-1), 3:(0,1), 4:(0,0)
actions_map = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0)}

for r in range(5):
    for c in range(5):
        s = r * 5 + c
        for a_idx, (dr, dc) in actions_map.items():
            nr = np.clip(r + dr, 0, 4)
            nc = np.clip(c + dc, 0, 4)
            s_prime = nr * 5 + nc
            trans_probs[s, a_idx, s_prime] = 1.0

# Verify shape
print(f"Generated trans_probs shape: {trans_probs.shape}")

# Load Animal Data
# Load Animal Data
print("Loading animal_data train/val...")
try:
    train_path = os.path.join(script_dir, '../data/animal_data_train.csv')
    val_path = os.path.join(script_dir, '../data/animal_data_val.csv')
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
except FileNotFoundError:
    print(f"Error: train or val csv not found at {train_path} or {val_path}")
    sys.exit(1)

def preprocess_df(df, min_len):
    df['state'] = df['row'] * 5 + df['col']
    xs_list, acs_list = [], []
    for _, group in df.groupby('traj_id'):
        s = group['state'].values[:min_len]
        a = group['action'].values[:min_len]
        if len(s) == min_len:
            xs_list.append(s)
            acs_list.append(a)
    return np.array(xs_list).astype(int), np.array(acs_list).astype(int)

# Group by trajectory and create arrays - Determine lengths from TRAIN
traj_lengths = train_df.groupby('traj_id').size()
target_len = int(traj_lengths.min())
print(f"Truncating to length {target_len}")

train_xs, train_acs = preprocess_df(train_df, target_len)
test_xs, test_acs = preprocess_df(val_df, target_len)

print(f"Train shape: {train_xs.shape}, Test shape: {test_xs.shape}")

n_states, n_actions, _ = trans_probs.shape
def one_hot_jax(z, K):
    z = jnp.atleast_1d(z).astype(int)
    shp = z.shape
    N = z.size
    zoh = jnp.zeros((N, K))
    zoh = zoh.at[jnp.arange(N), jnp.ravel(z)].set(1)
    zoh = jnp.reshape(zoh, shp + (K,))
    return zoh

def one_hot_jax2(z, z_prev, K):
    z = z * K + z_prev
    z = jnp.atleast_1d(z).astype(int)
    K2 = K * K
    shp = z.shape
    N = z.size
    zoh = jnp.zeros((N, K2))
    zoh = zoh.at[jnp.arange(N), jnp.ravel(z)].set(1)
    zoh = jnp.reshape(zoh, shp + (K2,))
    return zoh

def one_hotx_partial(xs):
    return one_hot_jax(xs[:, None], n_states)
def one_hotx2_partial(xs, xs_prev):
    return one_hot_jax2(xs[:, None], xs_prev[:, None], n_states)
def one_hota_partial(acs):
    return one_hot_jax(acs[:, None], n_actions)

train_xohs = vmap(one_hotx_partial)(train_xs)
train_xohs2 = vmap(one_hotx2_partial)(train_xs, jnp.roll(train_xs, 1))
train_aohs = vmap(one_hota_partial)(train_acs)

test_xohs = vmap(one_hotx_partial)(test_xs)
test_xohs2 = vmap(one_hotx2_partial)(test_xs, jnp.roll(test_xs, 1))
test_aohs = vmap(one_hota_partial)(test_acs)

logpi0_start = np.array([0.5, 0.5])
Ps = .95 * np.eye(K) + .05 * npr.rand(K, K)
Ps /= Ps.sum(axis=1, keepdims=True)
log_Ps_start = np.log(Ps)
Rs_start = np.zeros((C, 1, K))

from swirl_func import pi0_m_step, trans_m_step_jax_jaxopt
n_states, n_actions, _ = trans_probs.shape

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state

class MLP(nn.Module):
    subnet_size: int
    hidden_size: int
    output_size: int
    n_hidden: int
    expand: bool

    def setup(self):
        self.subnet = nn.Dense(self.subnet_size)
        self.dense1 = nn.Dense(self.hidden_size)
        self.dense2 = nn.Dense(self.n_hidden)
        if self.expand:
            self.reshape_func = lambda x: jnp.tile(jnp.expand_dims(x, axis=-1), (1,) * (x.ndim) + (C,)) / C
        else:
            self.reshape_func = lambda x: x.reshape(*x.shape[:-1], C, C)

    def __call__(self, x):
        x = self.reshape_func(x)
        x = jax.vmap(self.subnet, in_axes=-1)(x)
        x = x.reshape(*x.shape[1:-1], x.shape[0] * x.shape[-1])
        x = self.dense1(x)
        x = nn.leaky_relu(x)
        x = self.dense2(x) 
        x = nn.leaky_relu(x)
        x = jnp.expand_dims(x, axis=-1)
        x = jnp.tile(x, (1, self.output_size))
        return x

def create_model(rng, subnet_size, n_hidden, input_size, hidden_size, output_size, expand):
    model = MLP(subnet_size=subnet_size, hidden_size=hidden_size, output_size=output_size, n_hidden=n_hidden, expand=expand)
    params = model.init(rng, jnp.ones((1, input_size)))['params']
    return model, params

def create_train_state(rng, subnet_size, learning_rate, n_hidden, input_size, hidden_size, output_size, expand=False):
    model, params = create_model(rng, subnet_size, n_hidden, input_size, hidden_size, output_size, expand)
    tx = optax.adam(learning_rate)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return state

rng = jax.random.PRNGKey(0)
input_size = C*C
subnet_size = 4
hidden_size = 16
output_size = 5
n_hidden = K
learning_rate = 5e-3

R_state = create_train_state(rng, subnet_size, learning_rate, n_hidden, input_size, hidden_size, output_size, expand=False)

n_state, n_action, _ = trans_probs.shape
new_trans_probs = np.zeros((n_state * n_state, n_action, n_state * n_state))

for s_prev in range(n_state):
    for s in range(n_state):
        for a in range(n_action):
            for s_prime in range(n_state):
                if trans_probs[s, a, s_prime] > 0:
                    new_trans_probs[s * n_state + s_prev, a, s_prime * n_state + s] = trans_probs[s, a, s_prime]

from swirl_func import comp_transP, forward, backward, expected_states, comp_ll_jax, vinet, vinet_expand, _viterbi_JAX
from swirl_func import emit_m_step_jaxnet_optax2, emit_m_step_jaxnet_optax2_expand, jaxnet_e_step_batch2, jaxnet_e_step_batch


# Pre-define evaluation function - Refactored to accept data
def compute_viterbi_pi(logpi0, log_Ps, Rs, params, apply_fn, xohs2, xohs, aohs):
    pi0 = jnp.exp(logpi0 - jax_logsumexp(logpi0))
    pi, _, _ = vinet(new_trans_probs, params, apply_fn)
    logemit = jnp.log(pi)
    
    new_lls_jax_vmap = vmap(partial(comp_ll_jax, logemit))(jnp.array(xohs2), jnp.array(aohs))
    new_trans_Ps_vmap = vmap(partial(comp_transP, jnp.array(log_Ps), jnp.array(Rs)))(jnp.array(xohs))
    
    jax_path_vmap = vmap(partial(_viterbi_JAX, jnp.array(pi0)))(jnp.array(new_trans_Ps_vmap), jnp.array(new_lls_jax_vmap))
    return jax_path_vmap, pi

def get_action_predictions(state_seq, z_seq, piG):
    T = state_seq.shape[0]
    s = state_seq
    s_prev = np.roll(s, 1)
    indices = s * 25 + s_prev 
    policy_probs = piG[z_seq, indices, :]
    pred_actions = np.argmax(policy_probs, axis=1)
    return pred_actions[1:]

def evaluate_set(logpi0, log_Ps, Rs, R_state, xs, acs, xohs2, xohs, aohs):
    z_paths, piG = compute_viterbi_pi(logpi0, log_Ps, Rs, R_state.params, R_state.apply_fn, xohs2, xohs, aohs)
    piG = np.array(piG)
    total_correct = 0
    total_count = 0
    for i in range(xs.shape[0]):
        pred_a = get_action_predictions(xs[i], z_paths[i], piG)
        true_a = acs[i][1:]
        total_correct += np.sum(pred_a == true_a)
        total_count += len(true_a)
    return (total_correct / total_count) * 100

def em_train_net2(logpi0, log_Ps, Rs, R_state, iter=100, init=True, trans=True, emit=True):
    LL_list = []
    train_acc_list = []
    val_acc_list = []
    
    pi0 = jnp.exp(logpi0 - jax_logsumexp(logpi0))
    for i in range(iter):
        print(f"Iteration {i}")
        all_gamma_jax, all_xi_jax, all_jax_alphas = jaxnet_e_step_batch2(pi0, log_Ps, Rs, R_state, new_trans_probs, train_xohs, train_xohs2, train_aohs)
        current_ll = jnp.sum(jax_logsumexp(all_jax_alphas[:, -1], axis=-1))
        
        # Evaluate Accuracy on Train and Test
        train_acc = evaluate_set(logpi0, log_Ps, Rs, R_state, train_xs, train_acs, train_xohs2, train_xohs, train_aohs)
        val_acc = evaluate_set(logpi0, log_Ps, Rs, R_state, test_xs, test_acs, test_xohs2, test_xohs, test_aohs)
        
        print(f"Log Likelihood: {current_ll:.2f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)

        if init == True:
            new_logpi0 = pi0_m_step(all_gamma_jax)
        else:
            new_logpi0 = logpi0
        
        pi0 = jnp.exp(new_logpi0 - jax_logsumexp(new_logpi0))

        if trans == True:
            new_log_Ps, new_Rs = trans_m_step_jax_jaxopt(log_Ps, Rs, (all_gamma_jax, all_xi_jax), jnp.array(train_xohs))
        else:
            new_log_Ps, new_Rs = log_Ps, Rs

        if emit == True:
            new_R_state = emit_m_step_jaxnet_optax2_expand(R_state, jnp.array(trans_probs), all_gamma_jax, jnp.array(train_xohs), jnp.array(train_aohs), num_iters=800)
            new_R_state = emit_m_step_jaxnet_optax2(new_R_state, jnp.array(new_trans_probs), all_gamma_jax, jnp.array(train_xohs2), jnp.array(train_aohs), num_iters=200)
        else:
            new_R_state = R_state
            
        LL_list.append(current_ll)
        logpi0, log_Ps, Rs, R_state = new_logpi0, new_log_Ps, new_Rs, new_R_state
        
    return logpi0, log_Ps, Rs, R_state, LL_list, train_acc_list, val_acc_list

# from jax.lib import xla_bridge
# print(f"JAX Platform: {xla_bridge.get_backend().platform}")
print(f"JAX Platform: {jax.default_backend()}")

ITERATIONS = 20
print(f"Training Swirl S-2 for {ITERATIONS} iterations...")
new_logpi0, new_log_Ps, new_Rs, new_R_state, LL_list, train_acc_list, val_acc_list = em_train_net2(jnp.array(logpi0_start), jnp.array(log_Ps_start), jnp.array(Rs_start), R_state, ITERATIONS)

import json
import pickle

stats = {
    'acc': [float(x) for x in train_acc_list], 
    'val_acc': [float(x) for x in val_acc_list], 
    'll': [float(x) for x in LL_list]
}

with open(os.path.join(save_folder, 'swirl_s2_stats.json'), 'w') as f:
    json.dump(stats, f)

# Save Weights
weights = {
    'logpi0': new_logpi0,
    'log_Ps': new_log_Ps,
    'Rs': new_Rs,
    'params': new_R_state.params
}
with open(os.path.join(save_folder, 'swirl_s2_weights.pkl'), 'wb') as f:
    pickle.dump(weights, f)

print(f"Swirl S-2 Stats and Weights saved to {save_folder}/")
