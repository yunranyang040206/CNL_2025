import numpy as np
import argparse
import os
import yaml

# =========================
# Config Loading
# =========================
def load_config(config_path):
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def build_reward_functions(config, n_states):
    """
    Build reward arrays from config.
    Returns two reward vectors for augmented states (prev, curr).
    """
    n_aug = n_states * n_states
    
    # Initialize reward arrays for each mode
    # Mode 0 = HOME, Mode 1 = WATER (or custom names)
    R_mode0 = np.zeros((n_aug,), dtype=float)
    R_mode1 = np.zeros((n_aug,), dtype=float)
    
    def aug(ps, s): 
        return ps * n_states + s
    
    # Process state-based rewards - apply to BOTH modes
    if 'state_based' in config['rewards']:
        for state_str, reward in config['rewards']['state_based'].items():
            state = int(state_str)
            # State-based rewards apply regardless of previous state
            for ps in range(n_states):
                y = aug(ps, state)
                R_mode0[y] = reward
                R_mode1[y] = reward  # Apply to both modes
    
    # Process sequence-based rewards - apply to BOTH modes
    if 'sequence_based' in config['rewards']:
        for seq_reward in config['rewards']['sequence_based']:
            seq_type = seq_reward['sequence_type']
            reward = seq_reward['reward']
            
            if seq_type == 'state_transition':
                from_state = seq_reward.get('from_state')
                to_state = seq_reward.get('to_state')
                not_from_state = seq_reward.get('not_from_state')
                not_to_state = seq_reward.get('not_to_state')
                
                for ps in range(n_states):
                    for s in range(n_states):
                        # Check if this transition matches
                        matches = True
                        
                        if from_state is not None and ps != from_state:
                            matches = False
                        if to_state is not None and s != to_state:
                            matches = False
                        if not_from_state is not None and ps == not_from_state:
                            matches = False
                        if not_to_state is not None and s == not_to_state:
                            matches = False
                        
                        if matches:
                            y = aug(ps, s)
                            R_mode0[y] = reward  # Apply to both modes
                            R_mode1[y] = reward
    
    return R_mode0, R_mode1

# =========================
# Argument Parsing
# =========================
parser = argparse.ArgumentParser(description='Generate trajectories')
parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
args = parser.parse_args()

# Load configuration
config = load_config(args.config)

# Create output directory with config name + timestamp
from datetime import datetime
config_name = os.path.splitext(os.path.basename(args.config))[0]  # Get config filename without extension
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = os.path.join('output', f'{config_name}_{timestamp}')
os.makedirs(output_dir, exist_ok=True)

# =========================
# Extract Config Values
# =========================
GRID_N = config['grid']['size']
N_STATES = GRID_N * GRID_N
A_UP, A_DOWN, A_LEFT, A_RIGHT, A_STAY = range(5)
ACTIONS = np.array([A_UP, A_DOWN, A_LEFT, A_RIGHT, A_STAY])
N_ACTIONS = len(ACTIONS)
HOME_Z, WATER_Z = 0, 1

# MDP params
GAMMA = config['mdp']['gamma']
TAU = config['mdp']['tau']
MAX_ITERS = config['mdp']['max_iters']
TOL = config['mdp']['tol']

# Trajectory params
N_EPISODES = config['trajectories']['n_episodes']
T_STEPS = config['trajectories']['t_steps']
SEED = config['trajectories']['seed']

# Mode switching
P_SWITCH_BASE = config['mode_switching']['base_prob']
P_SWITCH_BONUS = config['mode_switching']['bonus_prob']
TRIGGER_STATES = config['mode_switching']['trigger_states']

rng = np.random.default_rng(SEED)

# =========================
# Helpers
# =========================
def idx(rc):
    r, c = rc
    return r * GRID_N + c

def rc(i):
    return divmod(i, GRID_N)

def step_state(s, a):
    r, c = rc(s)
    if a == A_UP:    r2, c2 = max(0, r-1), c
    elif a == A_DOWN:r2, c2 = min(GRID_N-1, r+1), c
    elif a == A_LEFT:r2, c2 = r, max(0, c-1)
    elif a == A_RIGHT:r2, c2 = r, min(GRID_N-1, c+1)
    else:            r2, c2 = r, c
    return idx((r2, c2))

# Build deterministic transition tensor: T[s, a, s']
T = np.zeros((N_STATES, N_ACTIONS, N_STATES), dtype=float)
for s in range(N_STATES):
    for a in range(N_ACTIONS):
        sp = step_state(s, a)
        T[s, a, sp] = 1.0

# =========================
# Rewards on augmented state (prev, curr)
# =========================
N_AUG = N_STATES * N_STATES
def aug(ps, s): return ps * N_STATES + s
def unaug(y):   return divmod(y, N_STATES)

# Build reward functions from config
R_home, R_water = build_reward_functions(config, N_STATES)

# Augmented transitions: from (ps, s) with action a → (s, sp)
AUG_NEXT = np.zeros((N_AUG, N_ACTIONS), dtype=int)
for ps in range(N_STATES):
    for s in range(N_STATES):
        y = aug(ps, s)
        for a in range(N_ACTIONS):
            sp = np.argmax(T[s, a])
            AUG_NEXT[y, a] = aug(s, sp)

# =========================
# Soft-Q iteration
# =========================
def soft_q_iteration(R_vec):
    Q = np.zeros((N_AUG, N_ACTIONS), dtype=float)
    for it in range(MAX_ITERS):
        m = Q.max(axis=1, keepdims=True)
        lse = m + TAU * np.log(np.exp((Q - m)/TAU).sum(axis=1, keepdims=True))
        V = lse.squeeze(1)

        Q_new = np.empty_like(Q)
        V_next = V[AUG_NEXT]
        Q_new = R_vec[:, None] + GAMMA * V_next

        diff = np.max(np.abs(Q_new - Q))
        Q = Q_new
        if diff < TOL:
            break

    m = Q.max(axis=1, keepdims=True)
    logits = (Q - m) / TAU
    exp_logits = np.exp(logits)
    pi = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    return Q, pi

Q_home, PI_home = soft_q_iteration(R_home)
Q_water, PI_water = soft_q_iteration(R_water)

# =========================
# Mode switching
# =========================
def next_mode(z, s_next):
    # Check if s_next is a trigger state
    trigger_bonus = 0.0
    if s_next in TRIGGER_STATES:
        trigger_type = TRIGGER_STATES[s_next]
        # Apply bonus if appropriate for current mode
        if (z == WATER_Z and trigger_type == "water_to_home") or \
           (z == HOME_Z and trigger_type == "home_to_water"):
            trigger_bonus = P_SWITCH_BONUS
    
    p_switch = P_SWITCH_BASE + trigger_bonus
    
    if z == WATER_Z:
        return HOME_Z if rng.random() < p_switch else WATER_Z
    else:
        return WATER_Z if rng.random() < p_switch else HOME_Z

# =========================
# Rollout
# =========================
xs = np.zeros((N_EPISODES, T_STEPS), dtype=np.int64)
acs = np.zeros((N_EPISODES, T_STEPS), dtype=np.int64)
zs  = np.zeros((N_EPISODES, T_STEPS), dtype=np.int64)
rews = np.zeros((N_EPISODES, T_STEPS), dtype=float)

for e in range(N_EPISODES):
    s = rng.integers(0, N_STATES)
    ps = s
    z = rng.integers(0, 2)

    for t in range(T_STEPS):
        y = aug(ps, s)
        if z == HOME_Z:
            pi = PI_home[y]
        else:
            pi = PI_water[y]

        a = rng.choice(N_ACTIONS, p=pi)
        sp = step_state(s, a)

        xs[e, t] = s
        acs[e, t] = a
        zs[e, t]  = z
        rews[e, t] = R_home[y] if z == HOME_Z else R_water[y]

        ps, s = s, sp
        z = next_mode(z, s)

# =========================
# RG tensor
# =========================
RG = np.zeros((2, N_STATES, N_STATES), dtype=float)
for ps in range(N_STATES):
    for s in range(N_STATES):
        y = aug(ps, s)
        RG[0, ps, s] = R_home[y]
        RG[1, ps, s] = R_water[y]

# Save artifacts
np.save(os.path.join(output_dir, 'xs.npy'), xs)
np.save(os.path.join(output_dir, 'acs.npy'), acs)
np.save(os.path.join(output_dir, 'zs.npy'), zs)
np.save(os.path.join(output_dir, 'RG.npy'), RG)
np.save(os.path.join(output_dir, 'trans_probs.npy'), T.astype(np.float64))

# Save config copy for reproducibility
import shutil
shutil.copy(args.config, os.path.join(output_dir, 'config.yaml'))

print(f'Config: {args.config}')
print(f'Output: {output_dir}')
print('xs:', xs.shape, xs.min(), xs.max())
print('acs:', acs.shape, acs.min(), acs.max())
print('zs:', zs.shape, zs.min(), zs.max())
print('RG:', RG.shape, RG.min(), RG.max(), 'mean', RG.mean())
print('T:', T.shape, 'row-sums ok?', np.allclose(T.sum(axis=2), 1.0))
