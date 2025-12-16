import sys
import os
import numpy as np
import jax
import jax.numpy as jnp
from flax import serialization
from gtrxl_model import GTrXL

jax.config.update("jax_enable_x64", True)

# --- CONFIGURATION (Must match training!) ---
SEQ_LEN = 20
EMBED_DIM = 32

def extract():
    # 1. Load Data
    data_dir = 'input'
    xs_path = os.path.join(data_dir, 'xs.npy')
    
    if not os.path.exists(xs_path):
        print(f"❌ Error: {xs_path} not found.")
        return

    # Load and force int type
    xs = np.load(xs_path).astype(int)[:200]
    print(f"Loading {len(xs)} trajectories...")

    # 2. Initialize Model Architecture
    model = GTrXL(n_states=25, n_actions=5, embed_dim=EMBED_DIM, seq_len=SEQ_LEN)
    dummy = jnp.ones((1, SEQ_LEN), dtype=jnp.int32)
    # Init dummy params to get the structure
    params = model.init(jax.random.PRNGKey(0), dummy)['params']

    # 3. Load Frozen Weights
    model_path = "output/gtrxl_frozen.msgpack"
    if not os.path.exists(model_path):
        print("❌ Trained model not found. Run training first.")
        return

    with open(model_path, "rb") as f:
        params = serialization.from_bytes(params, f.read())
    print("✅ Frozen GTrXL weights loaded.")

    # 4. Define Fast Extractor (JIT Compiled)
    @jax.jit
    def get_features(p, x):
        # Returns context only
        context, _, _ = model.apply({'params': p}, x, training=False)
        return context

    extracted_context = []
    print(f"Extracting features with Window={SEQ_LEN}...")

    for i, traj in enumerate(xs):
        padding = np.full(SEQ_LEN - 1, traj[0])
        padded = np.concatenate([padding, traj])

        windows = np.lib.stride_tricks.sliding_window_view(
            padded, window_shape=SEQ_LEN
        )
        windows_jax = jnp.array(windows, dtype=jnp.int32)

        # context: (T_steps, embed_dim)
        context_vectors = get_features(params, windows_jax)

        extracted_context.append(np.array(context_vectors))

        if (i+1) % 50 == 0:
            print(f"  Processed {i+1}/{len(xs)} trajectories")

    # Stack into arrays: (N_traj, T_steps, D)
    context_array = np.stack(extracted_context)  # (N, T, embed_dim)

    os.makedirs("output", exist_ok=True)
    np.save("output/features.npy", context_array)

    print(f"\n💾 Context saved to output/features.npy, shape={context_array.shape}")
    print("You are now ready for SWIRL / IRL using these 32-dim features.")    


if __name__ == "__main__":
    extract()