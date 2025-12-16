import sys
import os
import argparse
import time
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from flax import serialization
from functools import partial

from gtrxl_model import GTrXL

jax.config.update("jax_enable_x64", True)

# --- CONFIGURATION ---
BATCH_SIZE = 64
LR = 5e-4
EPOCHS = 10
SEQ_LEN = 20
EMBED_DIM = 32 
HORIZON = 5     
PATIENCE = 5

LAMBDA_FUTURE = 0.2

def create_sliding_windows_with_aux(xs, seq_len, horizon):
    """Creates windows + auxiliary targets."""
    inputs, tar_next, tar_future = [], [], []
    n_states = 25 

    for traj in xs:
        padding = np.full(seq_len - 1, traj[0])
        padded = np.concatenate([padding, traj])
        limit = len(traj) - 1 
        
        for t in range(limit):
            window = padded[t : t + seq_len]
            next_s = traj[t+1]
            end_h = min(t + 1 + horizon, len(traj))
            future_window = traj[t+1 : end_h]
            
            future_multihot = np.zeros(n_states)
            future_multihot[future_window] = 1.0
            
            inputs.append(window)
            tar_next.append(next_s)
            tar_future.append(future_multihot)
            
    return (jnp.array(np.stack(inputs), dtype=jnp.int32), 
            jnp.array(np.stack(tar_next), dtype=jnp.int32), 
            jnp.array(np.stack(tar_future), dtype=jnp.float32))

class FeatureTrainer:
    def __init__(self, rng_key, data_folder):
        self.rng = rng_key
        self.data_folder = data_folder
        
    def load_data(self):
        xs_path = os.path.join(self.data_folder, 'xs.npy')
        if not os.path.exists(xs_path):
             print(f"❌ Error: 'xs.npy' not found in {self.data_folder}")
             sys.exit(1)

        xs = np.load(xs_path).astype(int)[:200]
        
        split_idx = int(len(xs) * 0.8)
        xs_train = xs[:split_idx]
        xs_test = xs[split_idx:]
        
        print(f"Processing Data: {len(xs_train)} Train, {len(xs_test)} Test trajectories...")
        
        self.train_data = create_sliding_windows_with_aux(xs_train, SEQ_LEN, HORIZON)
        self.test_data = create_sliding_windows_with_aux(xs_test, SEQ_LEN, HORIZON)
        
        print(f"Ready: {self.train_data[0].shape[0]} Train Samples | {self.test_data[0].shape[0]} Test Samples")

    def create_state(self):
        model = GTrXL(n_states=25, n_actions=5, embed_dim=EMBED_DIM, seq_len=SEQ_LEN)
        dummy = jnp.ones((1, SEQ_LEN), dtype=jnp.int32)
        params = model.init(self.rng, dummy)['params']
        tx = optax.adamw(learning_rate=LR)
        return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    @partial(jax.jit, static_argnums=(0,))
    def train_step(self, state, batch_x, batch_next, batch_future, rng):
        def loss_fn(params):
            _, logits_next, logits_future = state.apply_fn(
                {'params': params}, batch_x, training=True, rngs={'dropout': rng}
            )
            loss_next = optax.softmax_cross_entropy_with_integer_labels(
                logits_next, batch_next
            ).mean()
            loss_future = optax.sigmoid_binary_cross_entropy(
                logits_future, batch_future
            ).mean()

            loss = loss_next + LAMBDA_FUTURE * loss_future
            return loss, (loss_next, loss_future)

        (loss, (l_n, l_f)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss, l_n, l_f

    @partial(jax.jit, static_argnums=(0,))
    def eval_step(self, state, batch_x, batch_next, batch_future):
        _, logits_next, logits_future = state.apply_fn(
            {'params': state.params}, batch_x, training=False
        )
        loss_next = optax.softmax_cross_entropy_with_integer_labels(
            logits_next, batch_next
        ).mean()
        loss_future = optax.sigmoid_binary_cross_entropy(
            logits_future, batch_future
        ).mean()

        loss_total = loss_next + LAMBDA_FUTURE * loss_future
        return loss_total, (loss_next, loss_future)

    def run(self):
        self.load_data()
        state = self.create_state()
        
        history = {
            'train_loss': [], 'train_next': [], 'train_future': [],
            'test_loss': [], 'test_next': [], 'test_future': []
        }

        # --- EARLY STOPPING VARS ---
        best_val_loss = float('inf')
        patience_counter = 0
        best_params = None
        
        print(f"\nStarting GTrXL Training (Window={SEQ_LEN}, Patience={PATIENCE})...")
        
        for epoch in range(EPOCHS):
            # --- TRAINING ---
            self.rng, rng_perm, rng_drop = jax.random.split(self.rng, 3)
            train_inputs, train_next, train_future = self.train_data
            n_samples = train_inputs.shape[0]
            perm = jax.random.permutation(rng_perm, n_samples)
            
            t_loss, t_next, t_future = 0, 0, 0
            steps = n_samples // BATCH_SIZE
            
            for i in range(steps):
                s = i * BATCH_SIZE
                e = s + BATCH_SIZE
                idx = perm[s:e]
                step_rng = jax.random.fold_in(rng_drop, i)
                state, l, ln, lf = self.train_step(state, train_inputs[idx], train_next[idx], train_future[idx], step_rng)
                t_loss += l; t_next += ln; t_future += lf

            # --- TESTING ---
            test_inputs, test_next, test_future = self.test_data
            n_test = test_inputs.shape[0]
            test_steps = n_test // BATCH_SIZE
            
            v_loss, v_next, v_future = 0, 0, 0
            for i in range(test_steps):
                s = i * BATCH_SIZE
                e = s + BATCH_SIZE
                l, (ln, lf) = self.eval_step(state, test_inputs[s:e], test_next[s:e], test_future[s:e])
                v_loss += l; v_next += ln; v_future += lf

            # Averages
            avg_t_loss = t_loss / steps
            avg_t_next = t_next / steps
            avg_t_fut = t_future / steps
            avg_v_loss = v_loss / test_steps
            avg_v_next = v_next / test_steps
            avg_v_fut = v_future / test_steps
            
            history['train_loss'].append(float(avg_t_loss))
            history['train_next'].append(float(avg_t_next))
            history['train_future'].append(float(avg_t_fut))
            history['test_loss'].append(float(avg_v_loss))
            history['test_next'].append(float(avg_v_next))
            history['test_future'].append(float(avg_v_fut))
            
            print(f"Epoch {epoch+1:02d} | Train: {avg_t_loss:.4f} (N:{avg_t_next:.2f} F:{avg_t_fut:.2f}) | Test: {avg_v_loss:.4f} (N:{avg_v_next:.2f} F:{avg_v_fut:.2f})")

            # --- EARLY STOPPING CHECK ---
            if avg_v_loss < best_val_loss:
                best_val_loss = avg_v_loss
                patience_counter = 0
                best_params = state.params  # Save best weights
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"\n⏹️ Early Stopping Triggered! No improvement for {PATIENCE} epochs.")
                    print(f"Best Test Loss: {best_val_loss:.4f}")
                    state = state.replace(params=best_params) # Restore best weights
                    break

        # --- SAVE ---
        os.makedirs("output", exist_ok=True)
        with open("output/gtrxl_frozen.msgpack", "wb") as f:
            f.write(serialization.to_bytes(state.params))
        np.save("output/gtrxl_training_stats.npy", history)
        print(f"\nStats saved. Run plot_progress.py to generate graphs.")
            
        return state

if __name__ == "__main__":
    trainer = FeatureTrainer(jax.random.PRNGKey(0), 'input')
    trainer.run()