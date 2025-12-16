import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import serialization

# --- Config ---
EMBED_DIM = 32
N_ACTIONS = 25
N_MODES = 2
GAMMA = 0.90
GRID_SIZE = 5

# --- Model Definition (Must match training) ---
class QNetwork(nn.Module):
    n_actions: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_actions)(x)
        return x

def main():
    data_dir = "input"
    output_dir = "output"
    
    print("Loading data...")
    # 1. Load Data
    features = np.load(os.path.join(data_dir, "features.npy"))
    xs = np.load(os.path.join(data_dir, "xs.npy")).astype(int)
    
    # Ensure dimensions match
    n_samples, n_timesteps, _ = features.shape
    xs = xs[:n_samples, :n_timesteps]
    
    # gammas = np.load(os.path.join(output_dir, "final_gammas_em.npy")) # Optional if we just want per-mode maps
    
    # 2. Initialize Models & Load Params
    rng = jax.random.PRNGKey(0)
    q_model = QNetwork(n_actions=N_ACTIONS)
    dummy_input = jnp.ones((1, EMBED_DIM))
    dummy_params = q_model.init(rng, dummy_input)['params']
    
    q_params_list = []
    for k in range(N_MODES):
        path = os.path.join(output_dir, f"q_net_{k}.msgpack")
        if not os.path.exists(path):
            print(f"Model {path} not found. Skipping.")
            return

        with open(path, "rb") as f:
            data_bytes = f.read()
            params = serialization.from_bytes(dummy_params, data_bytes)
            q_params_list.append(params)
            
    print("Models loaded. Computing rewards...")
    
    # 3. Compute Rewards
    # r(s, a) = Q(s,a) - gamma * V(s')
    # We will compute this for every transition in the dataset
    
    # Align data: (s_t, a_t, s_{t+1})
    # xs has shape (N, T). xs[:, t] is state at t.
    # features has shape (N, T, D).
    
    current_feat = features[:, :-1, :]   # (N, T-1, D)
    next_feat    = features[:, 1:, :]    # (N, T-1, D)
    # Actions? We don't strictly need actions if we visualize V(s) or average reward.
    # But usually r depends on action.
    # However, to visualize a heat map over STATES, we often just want V(s) or max_a Q(s,a) or avg reward.
    # Let's compute 'Value Map' V(s_k) for each mode.
    # V_k(s) = logsumexp(Q_k(s, .))
    
    # We will accumulate values for each state index.
    
    # Flatten
    states_flat = xs[:, :-1].reshape(-1)
    feat_flat = current_feat.reshape(-1, EMBED_DIM)
    
    # Prepare accumulation grids
    # grid value sum: [Mode, StateIndex]
    value_sums = np.zeros((N_MODES, N_ACTIONS)) 
    counts = np.zeros((N_MODES, N_ACTIONS))
    
    # Batch processing to avoid OOM
    batch_size = 1000
    n_samples = feat_flat.shape[0]
    
    for k in range(N_MODES):
        print(f"  Mode {k}...")
        for i in range(0, n_samples, batch_size):
            batch_feat = feat_flat[i:i+batch_size]
            batch_states = states_flat[i:i+batch_size]
            
            # Predict Q
            q_values = q_model.apply({'params': q_params_list[k]}, batch_feat) # (B, A)
            
            # Compute Value V(s) = logsumexp(Q)
            # This represents "How good is this state in this mode?"
            # which is effectively the potential function.
            # Implicit reward r(s) ~ V(s) - gamma*E[V(s')] roughly. 
            # But just plotting V(s) is            # Compute Value V(s) = logsumexp(Q)
            v_values = jax.scipy.special.logsumexp(q_values, axis=-1)
            
            # Robust conversion
            v_np = np.array(v_values).flatten()
            idx_np = np.array(batch_states).flatten()

            np.add.at(value_sums[k], idx_np, v_np)
            np.add.at(counts[k], batch_states, 1)
            
    # Average
    avg_values = np.divide(value_sums, counts + 1e-8)
    avg_values[counts == 0] = np.nan # Hide unvisited

    # Normalize to [0, 1] per mode
    for k in range(N_MODES):
        vals = avg_values[k]
        valid_mask = ~np.isnan(vals)
        if np.any(valid_mask):
            v_min = np.min(vals[valid_mask])
            v_max = np.max(vals[valid_mask])
            if v_max > v_min:
                avg_values[k][valid_mask] = (vals[valid_mask] - v_min) / (v_max - v_min)
    
    # --- Plotting ---
    print("Plotting...")
    fig, axes = plt.subplots(1, N_MODES, figsize=(6 * N_MODES, 5))
    if N_MODES == 1: axes = [axes]
    
    for k in range(N_MODES):
        ax = axes[k]
        
        # Reshape to grid (5x5)
        # Note: Check if row-major or col-major. Usually row-major 0..4 is row 0.
        grid_map = avg_values[k].reshape(GRID_SIZE, GRID_SIZE)
        
        im = ax.imshow(grid_map, cmap='viridis', origin='upper')
        ax.set_title(f"Mode {k} Recovered Value V(s)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Annotate
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                val = grid_map[r, c]
                if not np.isnan(val):
                    ax.text(c, r, f"{val:.1f}", ha='center', va='center', color='w', fontsize=8)
        
        # Annotate Home (0,0) and Water (4,4)
        # Home: Top Left
        ax.scatter(0, 0, marker='s', s=100, edgecolors='yellow', facecolors='none', linewidth=2, label='Home')
        ax.text(0, 0, 'H', color='yellow', ha='center', va='center', fontweight='bold')

        # Water: Bottom Right
        ax.scatter(GRID_SIZE-1, GRID_SIZE-1, marker='o', s=100, edgecolors='cyan', facecolors='none', linewidth=2, label='Water')
        ax.text(GRID_SIZE-1, GRID_SIZE-1, 'W', color='cyan', ha='center', va='center', fontweight='bold')
        
    plt.tight_layout()
    save_file = os.path.join(output_dir, "recovered_value_maps.png")
    plt.savefig(save_file)
    print(f"✅ Saved reward/value maps to {save_file}")

if __name__ == "__main__":
    main()
