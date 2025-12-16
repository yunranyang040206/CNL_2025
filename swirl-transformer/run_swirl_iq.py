import sys
import os
import numpy as np

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from functools import partial

import optax
from flax import linen as nn
from flax import serialization
from flax.training import train_state

jax.config.update("jax_enable_x64", True)

# -------------------------
# CONFIG
# -------------------------
N_MODES = 2
EMBED_DIM = 32
N_ACTIONS = 25

EM_ITERATIONS = 20        # EM outer iters (Stage 1)
M_STEP_EPOCHS = 2         # epochs per EM M-step (policies + transitions)
BATCH_SIZE = 128

# Stage 2 (per-mode IQ Q-learning)
Q_EPOCHS = 20
Q_BATCH_SIZE = 128
GAMMA = 0.90
TAU = 0.005               # soft update for Q target
VAL_SPLIT = 0.2

# -------------------------
# MODELS
# -------------------------

class PolicyNet(nn.Module):
    """Per-mode policy pi(a | s) via logits."""
    n_actions: int

    @nn.compact
    def __call__(self, x):
        # x: (..., EMBED_DIM)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_actions)(x)  # logits for actions
        return x


class QNetwork(nn.Module):
    """Per-mode Q(s,a) network (for IRL / reward stage)."""
    n_actions: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_actions)(x)  # Q(s,a)
        return x


class TransitionClassifier(nn.Module):
    """P(z_{t+1} | z_t, s_t) as KxK logits conditioned on context."""
    @nn.compact
    def __call__(self, x):
        # x: (B, EMBED_DIM)
        x = nn.Dense(32)(x)
        x = nn.relu(x)
        x = nn.Dense(N_MODES * N_MODES)(x)
        return x.reshape(-1, N_MODES, N_MODES)


class QTrainState(train_state.TrainState):
    target_params: any


# -------------------------
# TRAINER
# -------------------------

class SwirlTrainerMixtureIRL:
    def __init__(self, rng_key, data_dir):
        self.rng = rng_key
        self.data_dir = data_dir
        self.load_data()
        self.init_params()

    # -------------------------
    # Data loading
    # -------------------------
    def load_data(self):
        feat_path = os.path.join(self.data_dir, 'features.npy')
        xs_path = os.path.join(self.data_dir, 'xs.npy')
        zs_path = os.path.join(self.data_dir, 'zs.npy')

        if not os.path.exists(feat_path):
            print(f"Error: {feat_path} not found.")
            sys.exit(1)

        features = np.load(feat_path)  # (N, T, EMBED_DIM)
        if features.shape[-1] != EMBED_DIM:
            print(f"Warning: Expected dim {EMBED_DIM}, got {features.shape[-1]}.")

        n_samples = features.shape[0]

        if os.path.exists(xs_path):
            raw_xs = np.load(xs_path).astype(int)[:n_samples]  # (N, T_raw)
        else:
            print("Error: xs.npy not found")
            sys.exit(1)

        if os.path.exists(zs_path):
            self.has_truth = True
            raw_zs = np.load(zs_path).astype(int)[:n_samples]
        else:
            self.has_truth = False
            raw_zs = None

        # Train/val split
        n_train = int(n_samples * (1 - VAL_SPLIT))

        # Align shapes: we use T-1 transitions (s_t, a_{t+1}, s_{t+1})
        self.X_train       = features[:n_train, :-1, :]   # (N_train, T-1, d)
        self.Y_train       = raw_xs[:n_train, 1:]         # (N_train, T-1)
        self.Next_X_train  = features[:n_train, 1:, :]    # (N_train, T-1, d)

        self.X_val         = features[n_train:, :-1, :]
        self.Y_val         = raw_xs[n_train:, 1:]
        if self.has_truth:
            self.Z_val = raw_zs[n_train:, :-1]            # (N_val, T-1)

        self.T_steps = self.X_train.shape[1]
        self.train_steps = n_train * self.T_steps
        self.val_steps = (n_samples - n_train) * self.T_steps

        print(f"Data Loaded: {n_train} train trajectories, T={self.T_steps}")

    # -------------------------
    # Param / state init
    # -------------------------
    def init_params(self):
        # Per-mode policy nets (for emissions in HMM)
        self.policy_states = []
        policy_model = PolicyNet(n_actions=N_ACTIONS)

        for k in range(N_MODES):
            self.rng, init_rng = jax.random.split(self.rng)
            params = policy_model.init(init_rng, jnp.ones((1, EMBED_DIM)))['params']
            tx = optax.adam(learning_rate=1e-3)

            state = train_state.TrainState.create(
                apply_fn=policy_model.apply,
                params=params,
                tx=tx
            )
            self.policy_states.append(state)

        # Transition classifier
        trans_model = TransitionClassifier()
        self.rng, init_rng = jax.random.split(self.rng)
        trans_params = trans_model.init(init_rng, jnp.ones((1, EMBED_DIM)))['params']
        trans_tx = optax.adam(learning_rate=1e-3)
        self.trans_state = train_state.TrainState.create(
            apply_fn=trans_model.apply,
            params=trans_params,
            tx=trans_tx
        )

        # Initial mode prior pi0
        pi0 = jnp.ones(N_MODES) / N_MODES
        self.log_pi0 = jnp.log(pi0)

        # Stage 2 Q-networks (initialized later after EM if desired)
        self.q_states = None

    # -------------------------
    # E-step: compute gammas, xi, log-likelihood
    # -------------------------
    def run_inference(self, X_batch, Y_batch):
        n_samples = X_batch.shape[0]

        X_flat = X_batch.reshape(-1, EMBED_DIM)
        Y_flat = Y_batch.reshape(-1)

        # Emissions: log p(a | s, z=k) via per-mode policies
        log_emissions_list = []
        for k in range(N_MODES):
            logits = self.policy_states[k].apply_fn(
                {'params': self.policy_states[k].params},
                X_flat
            )  # (N*T, A)
            log_pi = jax.nn.log_softmax(logits, axis=-1)
            chosen = jnp.take_along_axis(
                log_pi, Y_flat[:, None], axis=-1
            ).squeeze(-1)  # (N*T,)
            log_emissions_list.append(chosen)

        log_emissions = jnp.stack(log_emissions_list, axis=1)  # (N*T, K)
        log_emissions = log_emissions.reshape(n_samples, self.T_steps, N_MODES)

        # Transitions: P(z_{t+1} | z_t, s_t)
        trans_input = X_batch[:, :-1, :]                       # (N, T-2, D)
        trans_input_flat = trans_input.reshape(-1, EMBED_DIM)
        trans_logits = self.trans_state.apply_fn(
            {'params': self.trans_state.params},
            trans_input_flat
        )  # (N*(T-2), K, K)
        log_Ps_t = jax.nn.log_softmax(trans_logits, axis=-1)
        log_Ps_t = log_Ps_t.reshape(n_samples, self.T_steps - 1, N_MODES, N_MODES)

        gammas, xi_seq, lls = self.compute_posteriors(
            self.log_pi0, log_Ps_t, log_emissions
        )
        return gammas, xi_seq, jnp.sum(lls)

    @partial(jax.jit, static_argnums=(0,))
    def compute_posteriors(self, log_pi0, log_Ps_t, log_emissions):
        """
        log_emissions: (N, T, K)
        log_Ps_t:      (N, T-1, K, K)
        """
        def process_single(emissions, log_Ps_seq):
            # emissions: (T, K)
            # log_Ps_seq: (T-1, K, K)
            T = emissions.shape[0]

            # Forward
            def scan_fwd(carry, t):
                log_alpha_prev = carry  # (K,)
                # log_Ps_seq[t-1]: (K_prev, K_curr)
                term = log_alpha_prev[:, None] + log_Ps_seq[t - 1]
                log_alpha_t = emissions[t] + logsumexp(term, axis=0)
                return log_alpha_t, log_alpha_t

            log_alpha_0 = emissions[0] + log_pi0
            _, log_alphas_rest = jax.lax.scan(
                scan_fwd, log_alpha_0, jnp.arange(1, T)
            )
            log_alphas = jnp.concatenate(
                [log_alpha_0[None, :], log_alphas_rest], axis=0
            )  # (T, K)

            # Backward
            def scan_bwd(carry, t):
                log_beta_next = carry     # (K,)
                # beta_t(i) = logsum_j P(i->j) + emit_{t+1}(j) + beta_{t+1}(j)
                term = log_Ps_seq[t] + emissions[t + 1] + log_beta_next
                log_beta_t = logsumexp(term, axis=1)  # sum over j
                return log_beta_t, log_beta_t

            log_beta_T = jnp.zeros(N_MODES)
            _, log_betas_rev = jax.lax.scan(
                scan_bwd,
                log_beta_T,
                jnp.arange(T - 2, -1, -1)
            )
            log_betas = jnp.concatenate(
                [log_betas_rev[::-1], log_beta_T[None, :]],
                axis=0
            )  # (T, K)

            # Gammas
            numer = log_alphas + log_betas
            log_norm = logsumexp(numer, axis=1, keepdims=True)
            gammas = jnp.exp(numer - log_norm)  # (T, K)

            # Xi sequence: (T-1, K, K)
            alpha_term = log_alphas[:-1, :, None]    # (T-1, K, 1)
            beta_term  = log_betas[1:, None, :]      # (T-1, 1, K)
            emit_term  = emissions[1:, None, :]      # (T-1, 1, K)
            trans_term = log_Ps_seq                  # (T-1, K, K)

            log_xi = alpha_term + trans_term + emit_term + beta_term
            seq_ll = logsumexp(log_alphas[-1])       # log p(actions | X)

            xi_seq = jnp.exp(log_xi - seq_ll)        # normalized per sequence
            return gammas, xi_seq, seq_ll

        gammas, xi_seq, lls = jax.vmap(process_single)(log_emissions, log_Ps_t)
        return gammas, xi_seq, lls

    # -------------------------
    # Transition M-step
    # -------------------------
    @partial(jax.jit, static_argnums=(0,))
    def train_step_trans(self, state, batch_x, batch_xi):
        """
        batch_x:  (B, EMBED_DIM)
        batch_xi: (B, K, K) posterior transition probs for each (z,z')
        """
        def loss_fn(params):
            logits = state.apply_fn({'params': params}, batch_x)
            log_probs = jax.nn.log_softmax(logits, axis=-1)  # (B, K, K)
            loss = -jnp.sum(batch_xi * log_probs)
            return loss / batch_x.shape[0]

        grads = jax.grad(loss_fn)(state.params)
        return state.apply_gradients(grads=grads)

    # -------------------------
    # Policy M-step (emissions)
    # -------------------------
    @partial(jax.jit, static_argnums=(0,))
    def train_step_policy(self, state, batch_x, batch_y, weights):
        """
        Weighted cross-entropy:
        Loss = - sum_t gamma_{t,k} log pi_k(a_t | s_t)
        """
        def loss_fn(params):
            logits = state.apply_fn({'params': params}, batch_x)  # (B, A)
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            idx = jnp.arange(batch_y.shape[0])
            chosen_logp = log_probs[idx, batch_y]  # (B,)

            weighted = -weights * chosen_logp
            loss = jnp.sum(weighted) / (jnp.sum(weights) + 1e-8)
            return loss

        grads = jax.grad(loss_fn)(state.params)
        return state.apply_gradients(grads=grads)

    # -------------------------
    # EM M-step wrapper
    # -------------------------
    def em_m_step(self, gammas, xi_seq):
        """
        Update:
          - Transition net (using xi_seq)
          - Policy nets (using gammas)
          - Prior pi0
        """
        # ---- Update transitions ----
        # self.X_train: (N, T-1, D), xi_seq: (N, T-1, K, K) (z_t->z_{t+1})
        trans_X = self.X_train[:, :-1, :].reshape(-1, EMBED_DIM)  # (N*(T-2), D)
        trans_xi = xi_seq[:, :-1, :, :].reshape(-1, N_MODES, N_MODES)

        n_trans = trans_X.shape[0]
        self.rng, key = jax.random.split(self.rng)

        # Batch update logic
        # For simplicity, just one epoch or multiple
        perm = jax.random.permutation(key, n_trans)
        for _ in range(M_STEP_EPOCHS):
            # If dataset is large, we should batch.
            # Assuming it fits for now or implementing batching:
            for i in range(n_trans // BATCH_SIZE):
                 idx = perm[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
                 self.trans_state = self.train_step_trans(
                     self.trans_state,
                     trans_X[idx],
                     trans_xi[idx],
                 )

        # ---- Update prior pi0 ----
        # gammas: (N, T, K) -> use t=0
        pi0_counts = jnp.sum(gammas[:, 0, :], axis=0)
        log_pi0 = jnp.log(pi0_counts + 1e-8)
        self.log_pi0 = log_pi0 - logsumexp(log_pi0)

        # ---- Update policy nets ----
        X_flat = self.X_train.reshape(-1, EMBED_DIM)
        Y_flat = self.Y_train.reshape(-1)
        G_flat = gammas.reshape(-1, N_MODES)  # responsibilities per time step

        n_flat = X_flat.shape[0]
        self.rng, key = jax.random.split(self.rng)
        
        perm = jax.random.permutation(key, n_flat)

        for _ in range(M_STEP_EPOCHS):
            for i in range(n_flat // BATCH_SIZE):
                idx = perm[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]

                for k in range(N_MODES):
                    w_k = G_flat[idx, k]
                    if jnp.sum(w_k) < 1e-6:
                        continue
                    self.policy_states[k] = self.train_step_policy(
                        self.policy_states[k],
                        X_flat[idx],
                        Y_flat[idx],
                        w_k,
                    )

    # -------------------------
    # Stage 2: Per-mode Q-learning (IQ-like) to recover rewards
    # -------------------------
    def init_q_states(self):
        """Initialize one Q-network per mode (with targets)."""
        self.q_states = []
        q_model = QNetwork(n_actions=N_ACTIONS)

        for k in range(N_MODES):
            self.rng, init_rng = jax.random.split(self.rng)
            params = q_model.init(init_rng, jnp.ones((1, EMBED_DIM)))['params']
            tx = optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adam(learning_rate=1e-4),
            )
            state = QTrainState.create(
                apply_fn=q_model.apply,
                params=params,
                target_params=params,
                tx=tx,
            )
            self.q_states.append(state)

    @partial(jax.jit, static_argnums=(0,))
    def q_train_step(self, state, batch_x, batch_a, batch_next_x, weights):
        """
        IQ-style loss weighted by mode responsibilities.
        """
        def loss_fn(params):
            q_s = state.apply_fn({'params': params}, batch_x)        # (B, A)
            idx = jnp.arange(batch_a.shape[0])
            q_sa = q_s[idx, batch_a]                                 # (B,)
            V_s = logsumexp(q_s, axis=-1)                            # (B,)

            q_next = state.apply_fn({'params': state.target_params}, batch_next_x)
            V_next = logsumexp(q_next, axis=-1)                      # (B,)

            diff = q_sa - GAMMA * V_next
            iq_loss = (V_s - q_sa) + 0.5 * (diff ** 2)

            # small regularizer to avoid blow-up
            iq_loss += 1e-3 * (q_sa ** 2)

            weighted = iq_loss * weights
            return jnp.sum(weighted) / (jnp.sum(weights) + 1e-8)

        grads = jax.grad(loss_fn)(state.params)
        new_state = state.apply_gradients(grads=grads)

        # Soft update
        new_target = jax.tree_util.tree_map(
            lambda p, tp: p * TAU + tp * (1.0 - TAU),
            new_state.params,
            new_state.target_params,
        )
        return new_state.replace(target_params=new_target)

    def run_q_irl(self, gammas):
        """
        Learn per-mode Q(s,a) using IQ-style loss, weighted by mode responsibilities.
        Then reward estimate per mode is roughly r_k(s,a) ≈ q_sa - γ V_next.
        """
        print("Starting per-mode Q-learning (Stage 2: reward recovery)...")
        self.init_q_states()

        X_flat      = self.X_train.reshape(-1, EMBED_DIM)
        Y_flat      = self.Y_train.reshape(-1)
        Next_X_flat = self.Next_X_train.reshape(-1, EMBED_DIM)
        G_flat      = gammas.reshape(-1, N_MODES)

        n_flat = X_flat.shape[0]
        self.rng, key = jax.random.split(self.rng)

        for epoch in range(Q_EPOCHS):
            perm = jax.random.permutation(key, n_flat)
            key, _ = jax.random.split(key)

            for i in range(n_flat // Q_BATCH_SIZE):
                idx = perm[i * Q_BATCH_SIZE:(i + 1) * Q_BATCH_SIZE]

                for k in range(N_MODES):
                    w_k = G_flat[idx, k]
                    if jnp.sum(w_k) < 1e-6:
                        continue
                    self.q_states[k] = self.q_train_step(
                        self.q_states[k],
                        X_flat[idx],
                        Y_flat[idx],
                        Next_X_flat[idx],
                        w_k,
                    )

            print(f"  Q Epoch {epoch+1}/{Q_EPOCHS} done")

        print("Per-mode Q-learning completed.")

    # -------------------------
    # Helper: compute val mode accuracy if ground truth modes exist
    # -------------------------
    def mode_accuracy(self, gammas_val):
        if not self.has_truth:
            return None

        z_pred = jnp.argmax(gammas_val, axis=-1)      # (N_val, T-1)
        acc1 = jnp.mean(z_pred == self.Z_val)
        acc2 = jnp.mean((1 - z_pred) == self.Z_val)   # swap labels
        return float(jnp.maximum(acc1, acc2))

    # -------------------------
    # Main run: EM + Q IRL
    # -------------------------
    def run(self):
        print("Starting Stage 1: EM for modes + policies...")
        os.makedirs("output", exist_ok=True)

        final_train_gammas = None
        stats_history = {
            't_ll': [], 'v_ll': [], 'mode_acc': [], 'avg_pz': []
        }

        for it in range(EM_ITERATIONS):
            # E-step
            train_gammas, train_xi, t_ll = self.run_inference(
                self.X_train, self.Y_train
            )
            val_gammas, _, v_ll = self.run_inference(
                self.X_val, self.Y_val
            )

            final_train_gammas = train_gammas  # keep last

            msg = f"Iter {it+1:02d} | T_LL: {float(t_ll)/self.train_steps:.4f} | " \
                  f"V_LL: {float(v_ll)/self.val_steps:.4f}"

            if self.has_truth:
                acc = self.mode_accuracy(val_gammas)
                msg += f" | Mode Acc: {acc*100:.1f}%"

            # Check for Pz stats
            sample_feat = self.X_train[:100, 0, :]
            sample_logits = self.trans_state.apply_fn({'params': self.trans_state.params}, sample_feat)
            avg_Pz = jnp.exp(jax.nn.log_softmax(sample_logits, axis=-1)).mean(axis=0)
            msg += f" | Avg Pz(0->.): {avg_Pz[0]}"
            
            # Store stats
            stats_history['t_ll'].append(float(t_ll)/self.train_steps)
            stats_history['v_ll'].append(float(v_ll)/self.val_steps)
            if self.has_truth:
                stats_history['mode_acc'].append(acc)
            else:
                 stats_history['mode_acc'].append(0.0)
            stats_history['avg_pz'].append(np.array(avg_Pz))

            print(msg)

            # M-step
            self.em_m_step(train_gammas, train_xi)

        # Save EM results
        print("Saving EM results...")
        np.save("output/final_gammas_em.npy", np.array(final_train_gammas))
        np.savez(
            "output/hmm_params.npz",
            log_pi0=np.array(self.log_pi0),
        )

        # Stage 2: per-mode Q-learning (reward recovery)
        self.run_q_irl(final_train_gammas)

        # Save stats
        print("Saving training stats...")
        np.savez("output/swirl_stats.npz", 
                 t_ll=np.array(stats_history['t_ll']),
                 v_ll=np.array(stats_history['v_ll']),
                 mode_acc=np.array(stats_history['mode_acc']),
                 avg_pz=np.array(stats_history['avg_pz']))

        # Save Q-network params (simple serialization)
        print("Saving Q-network parameters...")
        for k in range(N_MODES):
             q_params = self.q_states[k].params
             # Flatten or save as dictionary via flax serialization
             # For simple reloading, we can use flax.serialization.to_bytes
             # But let's save as a pickle/msgpack or just rely on the fact user wants "files"
             # Let's use simple numpy save for the weights if possible, or just optax state
             # For simpler usage let's just save the bytes of the full state
             with open(f"output/q_net_{k}.msgpack", "wb") as f:
                 f.write(serialization.to_bytes(self.q_states[k].params))

        print("All done.")


if __name__ == "__main__":
    SwirlTrainerMixtureIRL(jax.random.PRNGKey(42), 'input').run()
