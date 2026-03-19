'''
Helper functions for SWIRL + Transformer
With reward ambiguity regularization experiments + diagnostics.
'''
import numpy as np
from flax import linen as nn
from flax.training import train_state

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import logsumexp as jax_logsumexp
import optax
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_platform_name", "cpu")


def make_goal_absorbing(trans_probs, goal_state=24):
    """Return a copy of trans_probs with the goal state made absorbing.

    For the goal state, every action deterministically transitions back to the
    goal state. This only changes planning dynamics; it does not alter rewards.
    """
    tp = jnp.array(trans_probs)
    S, A, _ = tp.shape
    if goal_state is None:
        return tp
    if goal_state < 0 or goal_state >= S:
        raise ValueError(f"goal_state={goal_state} out of bounds for S={S}")

    goal_row = jnp.zeros((A, S), dtype=tp.dtype).at[:, goal_state].set(1.0)
    return tp.at[goal_state, :, :].set(goal_row)

def soft_vi_sa(trans_probs, reward_sa, discount=0.95, threshold=100):
    # trans_probs: (S,A,S), reward_sa: (S,A)
    S, A, _ = trans_probs.shape
    V = jnp.zeros((S,))

    def scan_iter(V, _):
        Q = reward_sa + discount * jnp.einsum("sas,s->sa", trans_probs, V)
        V_new = jax.scipy.special.logsumexp(Q, axis=1)
        return V_new, None

    V, _ = lax.scan(scan_iter, V, jnp.arange(threshold))
    Q = reward_sa + discount * jnp.einsum("sas,s->sa", trans_probs, V)
    pi = jnp.exp(Q - jax.scipy.special.logsumexp(Q, axis=1, keepdims=True))
    return pi

def soft_vi_fn(R_SA, trans_prob_SAS, discount=0.95, vi_iters=50, tau=1.0):
    R_SA = R_SA / tau
    return soft_vi_sa(trans_prob_SAS, R_SA, discount=discount, threshold=vi_iters)

def reward_and_phi_from_h(R_state, h_t, center_phi=True):
    """
    Shared AIRL-style decomposition helper.

    Inputs
    ------
    R_state : flax TrainState
    h_t     : (H,)
    center_phi : bool

    Returns
    -------
    r_ska : (S,K,A)
    phi_sk: (S,K)
    """
    params = R_state.params
    apply_fn = R_state.apply_fn

    S_plus_H = R_state.params  # dummy to keep interface note-free
    del S_plus_H
    raise RuntimeError("Use _reward_and_phi_from_h_from_tp(...) instead.")
def _reward_and_phi_from_h_from_tp(R_state, h_t, trans_probs, center_phi=True):
    """
    Shared AIRL-style decomposition helper.

    Returns
    -------
    r_ska : (S,K,A)
    phi_sk: (S,K)
    """
    tp = jnp.array(trans_probs)
    S, A = tp.shape[0], tp.shape[1]

    eyeS = jnp.eye(S)
    h_rep = jnp.repeat(h_t[None, :], S, axis=0)         # (S,H)
    inp = jnp.concatenate([eyeS, h_rep], axis=1)        # (S,S+H)

    out = R_state.apply_fn({"params": R_state.params}, inp)

    if out.ndim != 2:
        raise ValueError(f"[AIRL helper] expected reward net output (S, K*(A+1)); got {out.shape}")

    if out.shape[0] != S:
        raise ValueError(f"[AIRL helper] first dim should equal S={S}; got {out.shape}")

    if out.shape[1] % (A + 1) != 0:
        raise ValueError(
            f"[AIRL helper] output last dim {out.shape[1]} is not divisible by A+1={A+1}"
        )

    K = out.shape[1] // (A + 1)
    out = out.reshape(S, K, A + 1)

    r_ska = out[:, :, :A]     # (S,K,A)
    phi_sk = out[:, :, A]     # (S,K)

    if center_phi:
        phi_sk = phi_sk - jnp.mean(phi_sk, axis=0, keepdims=True)

    return r_ska, phi_sk


def shaped_rewards_from_h(R_state, h_t, trans_probs, discount=0.95, center_phi=True):
    """
    Shared AIRL-style shaped reward.

    Returns
    -------
    shaped_ksa : (K,S,A)
    """
    tp = jnp.array(trans_probs)

    r_ska, phi_sk = _reward_and_phi_from_h_from_tp(
        R_state, h_t, tp, center_phi=center_phi
    )  # (S,K,A), (S,K)

    r_ksa = jnp.transpose(r_ska, (1, 0, 2))            # (K,S,A)
    phi_ks = jnp.transpose(phi_sk, (1, 0))             # (K,S)

    exp_next_phi = jnp.einsum("sas,ks->ksa", tp, phi_ks)   # (K,S,A)
    shaped_ksa = r_ksa + discount * exp_next_phi - phi_ks[:, :, None]

    return shaped_ksa

def comp_ll_jax(logits, one_hot_x, one_hot_a):
    if one_hot_x.ndim == 3:
        one_hot_x = one_hot_x[:, 0, :]   # (T,S)
    if one_hot_a.ndim == 3:
        one_hot_a = one_hot_a[:, 0, :]   # (T,A)

    logits = logits - jax_logsumexp(logits, axis=-1, keepdims=True)

    T = one_hot_x.shape[0]

    if logits.ndim == 4:
        # (T,K,S,A)
        logits_tka = jnp.einsum("ts,tksa->tka", one_hot_x, logits)
        return jnp.sum(one_hot_a[:, None, :] * logits_tka, axis=-1)  # (T,K)

    if logits.ndim == 3:
        # Either (K,S,A) or (T,S,A)
        if logits.shape[0] == T:
            # (T,S,A)
            logits_ta = jnp.einsum("ts,tsa->ta", one_hot_x, logits)
            return jnp.sum(one_hot_a * logits_ta, axis=-1)  # (T,)
        else:
            # (K,S,A)
            logits_tka = jnp.einsum("ts,ksa->tka", one_hot_x, logits)
            return jnp.sum(one_hot_a[:, None, :] * logits_tka, axis=-1)  # (T,K)

    raise ValueError(f"Unsupported logits shape: {logits.shape}")

def comp_log_transP(log_Ps, Rs, one_hot_x, ctx_bias=None):
    """
    Computes log transition probabilities for z.
    Original: log P(z_{t+1}|z_t,s_t) ∝ log_Ps + state_bias(s_t)

    With history embedding: add ctx_bias(t) derived from h_t:
      log P ∝ log_Ps + state_bias + ctx_bias(t)

    Args:
      log_Ps: (K, K) base logits
      Rs:     (S, 1, K) or compatible with dot below
      one_hot_x: (T, 1, S)
      ctx_bias: optional (T-1, K, K)

    Returns:
      (T-1, K, K) normalized log transition probs
    """
    T = one_hot_x.shape[0]

    log_Ps_t = jnp.tile(log_Ps[None, :, :], (T-1, 1, 1))  # (T-1,K,K)

    # state-dependent bias term
    if one_hot_x.ndim == 2:
            x = one_hot_x[:-1, :]
    else:
        x = one_hot_x[:-1, 0, :]
    if Rs.ndim != 3 or Rs.shape[0] != x.shape[1]:
        raise ValueError(f"Expected Rs shape (S,1,K) with S={x.shape[1]}, got {Rs.shape}")
    state_bias = jnp.dot(x, Rs[:, 0, :])

    log_Ps_t = log_Ps_t + state_bias[:, None, :]             # broadcast to (T-1,K,K)

    # NEW: context bias from h_t (must be computed upstream without peeking future)
    if ctx_bias is not None:
        log_Ps_t = log_Ps_t + ctx_bias

    # normalize over next-mode
    return log_Ps_t - jax_logsumexp(log_Ps_t, axis=2, keepdims=True)

    
def comp_transP(log_Ps, Rs, one_hot_x, ctx_bias=None):
    return jnp.exp(comp_log_transP(log_Ps, Rs, one_hot_x, ctx_bias=ctx_bias))

def comp_ll_jax_timevary(logits_tksa, one_hot_x, one_hot_a):
    """
    Time-varying action log-likelihoods.

    logits_tksa: (T, K, S, A)   (can be logits or log-probs; we normalize over A)
    one_hot_x:   (T, S) or (T, 1, S)
    one_hot_a:   (T, A) or (T, 1, A)

    Returns:
      lls: (T, K) where lls[t,k] = log p(a_t | x_t, h_t, z=k)
    """
    # squeeze singleton dims if present
    if one_hot_x.ndim == 3:
        one_hot_x = one_hot_x[:, 0, :]
    if one_hot_a.ndim == 3:
        one_hot_a = one_hot_a[:, 0, :]

    # normalize across actions (stable)
    logits_tksa = logits_tksa - jax_logsumexp(logits_tksa, axis=-1, keepdims=True)

    # pick logits for the realized state x_t: (T,K,S,A) x (T,S) -> (T,K,A)
    logits_tka = jnp.einsum('ts,tksa->tka', one_hot_x, logits_tksa)

    # pick the realized action a_t: (T,K,A) x (T,A) -> (T,K)
    lls_tk = jnp.sum(logits_tka * one_hot_a[:, None, :], axis=-1)
    return lls_tk
   

def forward(pi0, Ps, log_likes):
    T = log_likes.shape[0]
    K = log_likes.shape[1]
    alpha0 = jnp.log(pi0) + log_likes[0]
    def scan_body(carry, inputs):
        alpha_prev = carry
        Ps_t, log_like_t = inputs
        m = jnp.max(alpha_prev)
        alpha_t = jnp.log(jnp.dot(jnp.exp(alpha_prev - m), Ps_t)) + m + log_like_t
        return alpha_t, alpha_t
    _, alphas = lax.scan(scan_body, alpha0, (Ps, log_likes[1:]))
    return jnp.concatenate([alpha0[None, :], alphas])

def backward(Ps, log_likes):
    T = log_likes.shape[0]
    K = log_likes.shape[1]
    betaT = jnp.zeros((K))

    def scan_body(carry, inputs):
        beta_next = carry
        Ps_t, log_like_next = inputs
        tmp = log_like_next + beta_next
        m = jnp.max(tmp)
        beta_t = jnp.log(jnp.dot(Ps_t, jnp.exp(tmp - m))) + m

        return beta_t, beta_t
        
    _, betas = lax.scan(scan_body, betaT, (Ps[::-1], log_likes[1:][::-1]))

    return jnp.concatenate([betas[::-1], betaT[None, :]])

def expected_states(alphas, betas, Ps, ll):
    T, K = ll.shape

    expected_states = alphas + betas
    expected_states -= jax_logsumexp(expected_states, axis=1, keepdims=True)
    expected_states = jnp.exp(expected_states)

    log_Ps = jnp.log(Ps)

    expected_joints = alphas[:-1,:,None] + betas[1:,None,:] + ll[1:,None,:] + log_Ps
    expected_joints -= expected_joints.max((1,2))[:,None, None]
    expected_joints = jnp.exp(expected_joints)
    expected_joints /= expected_joints.sum((1,2))[:,None,None]

    return expected_states, expected_joints

def expected_states_logP(alphas, betas, logP, ll):
    # gamma
    log_gamma = alphas + betas
    log_gamma -= jax_logsumexp(log_gamma, axis=1, keepdims=True)
    gamma = jnp.exp(log_gamma)

    # xi
    log_xi = alphas[:-1, :, None] + logP + ll[1:, None, :] + betas[1:, None, :]
    log_xi -= jax_logsumexp(log_xi.reshape(log_xi.shape[0], -1), axis=1)[:, None, None]
    xi = jnp.exp(log_xi)
    return gamma, xi


class InferenceNet(nn.Module):
    """
    Structured inference network for q(z_1:T | h_1:T, x_1:T, a_1:T)
    """
    K: int
    H: int
    S: int
    A: int

    # embedding sizes
    d_h: int = 64
    d_x: int = 32
    d_a: int = 16
    d_model: int = 64

    # MLP depth
    n_hidden: int = 2
    dropout: float = 0.0

    # init bias toward staying in same mode (helps prevent rapid switching collapse)
    init_stay_bias: float = 3.0

    # clamp logits to avoid overflow/underflow in exp/log downstream
    clamp: float = 20.0

    @nn.compact
    def __call__(self, h_TH, x_TS, a_TA, *, train: bool = True):
        if h_TH.ndim == 3 and h_TH.shape[1] == 1:
            h_TH = h_TH[:, 0, :]

        # ---- embed each modality ----
        h_emb = nn.Dense(self.d_h, name="h_proj")(h_TH)          # (T, d_h)
        x_emb = nn.Dense(self.d_x, name="x_proj")(x_TS)          # (T, d_x)
        a_emb = nn.Dense(self.d_a, name="a_proj")(a_TA)          # (T, d_a)

        feat = jnp.concatenate([h_emb, x_emb, a_emb], axis=-1)   # (T, d_h+d_x+d_a)
        feat = nn.tanh(nn.Dense(self.d_model, name="feat_proj")(feat))

        # ---- optional MLP trunk ----
        for i in range(self.n_hidden):
            feat = nn.tanh(nn.Dense(self.d_model, name=f"mlp_{i}")(feat))
            if self.dropout > 0:
                feat = nn.Dropout(rate=self.dropout)(feat, deterministic=not train)

        # ---- node logits: (T,K) ----
        node_logits = nn.Dense(self.K, name="node_out")(feat)
        node_logits = jnp.clip(node_logits, -self.clamp, self.clamp)

        # ---- edge logits: (T-1,K,K) ----
        # We produce edge logits from feat[t] for t=0..T-2.
        feat_edge = feat[:-1]                                   # (T-1, d_model)

        # Base edge logits (full K*K, time-varying)
        edge_raw = nn.Dense(self.K * self.K, name="edge_out")(feat_edge)
        edge_raw = edge_raw.reshape((edge_raw.shape[0], self.K, self.K))

        # Add a learned (or fixed-initialized) stay bias to diagonal to encourage persistence
        stay_bias = self.param(
            "stay_bias",
            lambda key, shape: jnp.ones(shape) * self.init_stay_bias,
            (self.K,)
        )  # (K,)

        edge_logits = edge_raw + jnp.eye(self.K)[None, :, :] * stay_bias[None, :, None]
        edge_logits = jnp.clip(edge_logits, -self.clamp, self.clamp)

        return node_logits, edge_logits

def create_inf_state(rng, K, H, S, A,
                     lr=1e-4, d_h=64, d_x=32, d_a=16, d_model=64,
                     n_hidden=2, dropout=0.0, init_stay_bias=3.0):

    model = InferenceNet(
        K=K, H=H, S=S, A=A,
        d_h=d_h, d_x=d_x, d_a=d_a, d_model=d_model,
        n_hidden=n_hidden, dropout=dropout,
        init_stay_bias=init_stay_bias
    )

    # minimal dummy inputs for init
    h0 = jnp.zeros((1, H))
    x0 = jnp.zeros((1, S))
    a0 = jnp.zeros((1, A))

    params = model.init(rng, h0, x0, a0, train=True)["params"]
    tx = optax.adamw(lr, weight_decay=1e-4)

    return train_state.TrainState.create(apply_fn=model.apply,params=params,tx=tx)



def structured_q_marginals(node_logits_tk, edge_logits_tkk):
    """
    Compute q marginals for chain potentials from InferenceNet without numerical blow-ups.

    node_logits_tk: (T,K)       unnormalized node potentials φ_t(k)
    edge_logits_tkk: (T-1,K,K)  unnormalized edge potentials ψ_t(k,k')
    """
    T, K = node_logits_tk.shape
    assert edge_logits_tkk.shape == (T-1, K, K)

    # 1) Convert edge potentials to log-transition logP (row log-softmax)
    #    logP_t(k,k') = ψ_t(k,k') - logsumexp_{k'} ψ_t(k,k')
    row_logZ = jax_logsumexp(edge_logits_tkk, axis=-1, keepdims=True)   # (T-1,K,1)
    logP = edge_logits_tkk - row_logZ                                   # (T-1,K,K)
    Ps = jnp.exp(logP)                                                  # (T-1,K,K)

    # 2) Compensation: add row_logZ (squeezed) to the node term at time t for t=0..T-2
    #    so the induced distribution matches the original CRF potentials.
    ll = node_logits_tk
    ll = ll - jax_logsumexp(ll, axis=-1, keepdims=True)                 # optional per-time stabilization
    ll = ll.at[:-1].add(row_logZ.squeeze(-1))                           # (T,K)

    # 3) Run forward/backward on normalized transitions.
    pi0 = jnp.ones((K,), dtype=ll.dtype) / K
    alpha = forward(pi0, Ps, ll)
    beta  = backward(Ps, ll)

    # 4) Compute marginals using logP directly (no log(Ps))
    gamma, xi = expected_states_logP(alpha, beta, logP, ll)
    return gamma, xi, alpha

@jax.jit
def amortized_e_step_batch(inf_state, hoh_batch, xoh_batch, aoh_batch):
    '''Student E-step using structured q(z|h,x,a).'''
    hoh = jnp.array(hoh_batch)
    if hoh.ndim == 4 and hoh.shape[2] == 1:
        hoh = hoh[:, :, 0, :]

    xoh = jnp.array(xoh_batch)
    if xoh.ndim == 4 and xoh.shape[2] == 1:
        xoh = xoh[:, :, 0, :]

    aoh = jnp.array(aoh_batch)
    if aoh.ndim == 4 and aoh.shape[2] == 1:
        aoh = aoh[:, :, 0, :]

    def per_traj(h_TH, x_TS, a_TA):
        node_logits_tk, edge_logits_tkk = inf_state.apply_fn(
            {'params': inf_state.params},
            h_TH, x_TS, a_TA,
            train=False
        )
        gamma, xi, alpha = structured_q_marginals(node_logits_tk, edge_logits_tkk)
        return gamma, xi, alpha

    return jax.vmap(per_traj)(hoh, xoh, aoh)

def semi_amortized_e_step_batch(
    inf_state,
    hoh_batch, xoh_batch, aoh_batch,
    logemit_batch,
    pi0, log_Ps, Rs, trans_probs_j,
    lam_edge=0.5,
    lam_node=0.0,
    lam_prior=0.0,
    mode_prior=None,
    # --- Step F knobs ---
    trust_gate=True,          # use entropy-based trust gating
    trust_temp=1.0,           # >1 makes gate softer; <1 sharper
    trust_floor=0.0,          # minimum trust (0.0 = can fully ignore student)
    trust_cap=1.0,            # maximum trust
    eta_post=0.0,             # optional posterior blending strength (0 disables)
    eps=1e-8
):
    """
    Semi-amortized E-step (EDGE + NODE + optional MODE PRIOR) with Step F:

      - EDGE: bias transitions using student q_phi(z_t | z_{t-1}, h,x,a)
      - NODE: bias node evidence using student q_phi(z_t | h,x,a)
      - PRIOR: bias node evidence toward a target marginal over modes
      - Step F: trust-weight student guidance; optional blending of teacher posterior with student posterior

    Returns:
      gamma: (N,T,K)
      xi:    (N,T-1,K,K)
      alpha: (N,T,K)
    """
    hoh = jnp.array(hoh_batch)
    if hoh.ndim == 4 and hoh.shape[2] == 1:
        hoh = hoh[:, :, 0, :]

    xoh = jnp.array(xoh_batch)      # (N,T,1,S) expected by comp_transP/comp_ll
    aoh = jnp.array(aoh_batch)
    logemit = jnp.array(logemit_batch)

    # actions for ll: (N,T,A)
    aoh_ll = aoh
    if aoh_ll.ndim == 4 and aoh_ll.shape[2] == 1:
        aoh_ll = aoh_ll[:, :, 0, :]

    pi0 = jnp.array(pi0)
    log_Ps = jnp.array(log_Ps)
    Rs = jnp.array(Rs)

    # infer K from logemit (N,T,K,S,A)
    K = logemit.shape[2]
    if mode_prior is None:
        mode_prior = jnp.ones((K,), dtype=logemit.dtype) / K
    mode_prior = jnp.array(mode_prior, dtype=logemit.dtype)
    mode_prior = mode_prior / (jnp.sum(mode_prior) + eps)
    log_mode_prior = jnp.log(mode_prior + eps)  # (K,)
    logK = jnp.log(jnp.array(K, dtype=logemit.dtype) + eps)

    lam_edge_j = jnp.array(lam_edge, dtype=logemit.dtype)
    lam_node_j = jnp.array(lam_node, dtype=logemit.dtype)
    lam_prior_j = jnp.array(lam_prior, dtype=logemit.dtype)
    eta_post_j = jnp.array(eta_post, dtype=logemit.dtype)

    trust_temp_j = jnp.array(trust_temp, dtype=logemit.dtype)
    trust_floor_j = jnp.array(trust_floor, dtype=logemit.dtype)
    trust_cap_j = jnp.array(trust_cap, dtype=logemit.dtype)

    def _entropy_from_logp(logp, axis=-1):
        p = jnp.exp(logp)
        return -jnp.sum(p * logp, axis=axis)

    def _trust_from_ent(ent):
        # normalized entropy in [0,1] using logK, then invert => confidence
        # confidence = 1 - H/logK, then soften with trust_temp
        conf = 1.0 - ent / (logK + eps)
        # soften/sharpen
        conf = jnp.clip(conf, 0.0, 1.0)
        conf = conf ** (1.0 / (trust_temp_j + eps))
        # clamp
        return jnp.clip(conf, trust_floor_j, trust_cap_j)

    def per_traj(h_TH, x_T1S, a_TA, logemit_TKSA):
        # --- teacher transitions P_theta(z_t|z_{t-1}, x_{t-1})
        Ps = comp_transP(log_Ps, Rs, x_T1S)  # (T-1,K,K)

        # --- teacher log-likelihood log p(a_t | x_t, z_t, h_t)
        log_likes = comp_ll_jax_timevary(logemit_TKSA, x_T1S, a_TA)  # (T,K)

        # --- student logits (depend on h_TH!)
        x_TS = x_T1S[:, 0, :]  # (T,S)
        node_logits_tk, edge_logits_tkk = inf_state.apply_fn(
            {'params': inf_state.params}, h_TH, x_TS, a_TA, train=False
        )

        # student posteriors
        log_qnode = jax.nn.log_softmax(node_logits_tk, axis=-1)      # (T,K)
        qnode = jnp.exp(log_qnode)

        log_qtrans = jax.nn.log_softmax(edge_logits_tkk, axis=-1)    # (T-1,K,K)
        qtrans = jnp.exp(log_qtrans)

        # --- Step F trust gating (per time-step)
        if trust_gate:
            ent_node = _entropy_from_logp(log_qnode, axis=-1)        # (T,)
            trust_node = _trust_from_ent(ent_node)                   # (T,)

            # edge: entropy over next-state distribution for each prev-state
            ent_edge = _entropy_from_logp(log_qtrans, axis=-1)        # (T-1,K)
            trust_edge = _trust_from_ent(jnp.mean(ent_edge, axis=-1)) # (T-1,)
        else:
            trust_node = jnp.ones((log_qnode.shape[0],), dtype=logemit.dtype)
            trust_edge = jnp.ones((log_qtrans.shape[0],), dtype=logemit.dtype)

        # EDGE: bias transitions (trust-weighted)
        if lam_edge != 0.0:
            logPs_tilde = jnp.log(Ps + eps) + (lam_edge_j * trust_edge)[:, None, None] * log_qtrans
            Ps = jax.nn.softmax(logPs_tilde, axis=-1)

        # NODE: bias node evidence (trust-weighted)
        if lam_node != 0.0:
            log_likes = log_likes + (lam_node_j * trust_node)[:, None] * log_qnode

        # PRIOR: bias node evidence toward target marginal (anti-collapse)
        if lam_prior != 0.0:
            log_likes = log_likes + lam_prior_j * log_mode_prior[None, :]

        # forward-backward (teacher)
        alpha = forward(pi0, Ps, log_likes)
        beta = backward(Ps, log_likes)
        gamma, xi = expected_states(alpha, beta, Ps, log_likes)

        # --- Step F (optional): posterior blending with student posteriors
        # Blend only if eta_post > 0, and use trust_node/trust_edge as safety.
        if eta_post != 0.0:
            eta_t = jnp.clip(eta_post_j * trust_node, 0.0, 1.0)              # (T,)
            gamma = (1.0 - eta_t)[:, None] * gamma + eta_t[:, None] * qnode
            gamma = gamma / (jnp.sum(gamma, axis=-1, keepdims=True) + eps)

            # xi blend with qtrans * gamma_prev approx (keeps shape consistent)
            # build a "student xi" proxy: xiS[t,i,j] = gamma[t,i] * qtrans[t,i,j]
            xiS = gamma[:-1, :, None] * qtrans                              # (T-1,K,K)
            eta_e = jnp.clip(eta_post_j * trust_edge, 0.0, 1.0)              # (T-1,)
            xi = (1.0 - eta_e)[:, None, None] * xi + eta_e[:, None, None] * xiS
            xi = xi / (jnp.sum(xi, axis=(-2, -1), keepdims=True) + eps)

        return gamma, xi, alpha

    # For aoh: need (N,T,A) for per_traj
    a_for_net = aoh
    if a_for_net.ndim == 4 and a_for_net.shape[2] == 1:
        a_for_net = a_for_net[:, :, 0, :]

    gamma, xi, alpha = jax.vmap(per_traj)(hoh, xoh, a_for_net, logemit)
    return gamma, xi, alpha


@jax.jit
def distill_step(
    inf_state,
    hoh, xoh, aoh,
    gamma_T, xi_T,
    clip_norm=1.0,
    ent_w=0.0,
    class_w=None,
    mi_ent_w=0.0,
    mi_temp=1.0,
    eps=1e-8,
):
    """
    Distill student inference net to match teacher posteriors + E2 MI regularization.

    JIT-safe: no Python `if` on traced values. Always computes entropy terms and
    weights them by ent_w and mi_ent_w as JAX scalars.
    """

    # ---- squeeze to (N,T,*) ----
    hoh_ = jnp.array(hoh)
    xoh_ = jnp.array(xoh)
    aoh_ = jnp.array(aoh)
    if hoh_.ndim == 4 and hoh_.shape[2] == 1: hoh_ = hoh_[:, :, 0, :]
    if xoh_.ndim == 4 and xoh_.shape[2] == 1: xoh_ = xoh_[:, :, 0, :]
    if aoh_.ndim == 4 and aoh_.shape[2] == 1: aoh_ = aoh_[:, :, 0, :]

    gammaT = jnp.array(gamma_T)
    xiT = jnp.array(xi_T)
    if gammaT.ndim == 4:  # (N,T,1,K) -> (N,T,K)
        gammaT = gammaT[:, :, 0, :]

    N, T, K = gammaT.shape

    if class_w is None:
        class_w = jnp.ones((K,), dtype=jnp.float32)
    class_w = jnp.array(class_w, dtype=jnp.float32)

    # Make weights JAX scalars (JIT-safe)
    ent_w_j = jnp.asarray(ent_w, dtype=jnp.float32)
    mi_ent_w_j = jnp.asarray(mi_ent_w, dtype=jnp.float32)
    mi_temp_j = jnp.asarray(mi_temp, dtype=jnp.float32)

    def loss_fn(params):
        # ---- vmap over trajectories (model expects (T,*) per traj) ----
        def f_one_traj(h_TH, x_TS, a_TA):
            return inf_state.apply_fn({"params": params}, h_TH, x_TS, a_TA, train=True)

        node_logits_ntk, edge_logits_ntkk = jax.vmap(f_one_traj, in_axes=(0, 0, 0))(hoh_, xoh_, aoh_)
        # node_logits_ntk: (N,T,K)
        # edge_logits_ntkk: (N,T-1,K,K)

        # ---- NODE q(z|c) ----
        node_logits = node_logits_ntk / (mi_temp_j + 1e-8)
        log_qnode = jax.nn.log_softmax(node_logits, axis=-1)    # (N,T,K)
        qnode = jnp.exp(log_qnode)                              # (N,T,K)
        # ---- KL(gamma_T || gamma_S) diagnostic ----
        log_gammaT = jnp.log(gammaT + eps)                     # (N,T,K)
        kl_T_S = jnp.sum(
            gammaT * (log_gammaT - log_qnode),
            axis=-1                                            # sum over K
        )                                                      # (N,T)

        kl_T_S_mean = jnp.mean(kl_T_S)                         # scalar

        # ---- Entropy diagnostics ----
        ent_T = -jnp.sum(gammaT * log_gammaT, axis=-1)      # (N,T)
        ent_S = -jnp.sum(qnode  * log_qnode,  axis=-1)      # (N,T)

        ent_T_mean = jnp.mean(ent_T)
        ent_S_mean = jnp.mean(ent_S)


        # weighted node CE: -sum_k w_k * gamma * log q
        w = class_w[None, None, :]                              # (1,1,K)
        node_ce = -jnp.sum(w * gammaT * log_qnode, axis=-1)     # (N,T)
        node_loss = jnp.mean(node_ce)

        # ---- EDGE q(z_t->z_{t+1}|c) ----
        log_qedge = jax.nn.log_softmax(edge_logits_ntkk, axis=-1)   # (N,T-1,K,K)
        edge_ce = -jnp.sum(xiT * log_qedge, axis=(-2, -1))          # (N,T-1)
        edge_loss = jnp.mean(edge_ce)

        # ---- local entropy (always computed) ----
        ent_local = -jnp.sum(qnode * log_qnode, axis=-1)         # (N,T)
        ent_bonus = jnp.mean(ent_local)

        # ---- marginal entropy (always computed) ----
        qbar = jnp.mean(qnode.reshape(-1, K), axis=0)            # (K,)
        ent_marg = -jnp.sum(qbar * jnp.log(qbar + eps))

        # total loss (minimize)
        total = node_loss + edge_loss - ent_w_j * ent_bonus - mi_ent_w_j * ent_marg

        aux = {
            "node_loss": node_loss,
            "edge_loss": edge_loss,
            "ent_local": ent_bonus,
            "ent_marg": ent_marg,
            "qbar": qbar,
            "kl_T_S_mean": kl_T_S_mean,
            "ent_T_mean": ent_T_mean,
            "ent_S_mean": ent_S_mean
        }
        return total, aux

    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(inf_state.params)

    updates, new_opt_state = inf_state.tx.update(grads, inf_state.opt_state, inf_state.params)
    new_params = optax.apply_updates(inf_state.params, updates)

    inf_state = inf_state.replace(params=new_params, opt_state=new_opt_state)
    return inf_state, loss, aux



def jaxnet_e_step_logpi2(pi0, log_Ps, Rs, logemit, trans_probs, xoh, xoh2, aoh):
    """
    Same as jaxnet_e_step_logpi, but keeps the xoh/xoh2 split as in the original code.

    logemit can be (K,S,A) or (T,K,S,A).
    """
    Ps_jax = comp_transP(jnp.array(log_Ps), jnp.array(Rs), jnp.array(xoh))

    logemit = jnp.array(logemit)
    if logemit.ndim == 3:
        log_likes_jax = comp_ll_jax(logemit, jnp.array(xoh2), jnp.array(aoh))
    elif logemit.ndim == 4:
        log_likes_jax = comp_ll_jax_timevary(logemit, jnp.array(xoh2), jnp.array(aoh))
    else:
        raise ValueError(f"logemit must have ndim 3 or 4, got {logemit.ndim}")

    alpha_jax = forward(pi0, Ps_jax, log_likes_jax)
    beta_jax  = backward(Ps_jax, log_likes_jax)
    gamma_jax, xi_jax = expected_states(alpha_jax, beta_jax, Ps_jax, log_likes_jax)
    return gamma_jax, xi_jax, alpha_jax

def jaxnet_e_step_batch2(pi0, log_Ps, Rs, trans_probs, xoh_list, xoh_list2, aoh_list, logemit_list):
    """
    Time-varying E-step batch for h_t model.

    logemit_list: (N,T,K,S,A) or (N,K,S,A)
    returns:
      gamma: (N,T,K)
      xi:    (N,T-1,K,K)
      alpha: (N,T,K)
    """
    pi0 = jnp.array(pi0)
    log_Ps = jnp.array(log_Ps)
    Rs = jnp.array(Rs)
    trans_probs = jnp.array(trans_probs)

    xoh_list = jnp.array(xoh_list)
    xoh_list2 = jnp.array(xoh_list2)
    aoh_list = jnp.array(aoh_list)
    logemit_list = jnp.array(logemit_list)

    gamma_list, xi_list, alpha_list = jax.vmap(
        lambda xoh, xoh2, aoh, logemit: jaxnet_e_step_logpi2(
            pi0, log_Ps, Rs, logemit, trans_probs, xoh, xoh2, aoh
        )
    )(xoh_list, xoh_list2, aoh_list, logemit_list)

    return gamma_list, xi_list, alpha_list


def trans_m_step_jax_optax(
    log_Ps,
    Rs,
    expectations,
    one_hot_xs,
    num_iters=1000,
    learning_rate=5e-3,
    clip_logPs=8.0,
    clip_Rs=10.0,
    eps=1e-8,
    **kwargs
):
    """
    Update (log_Ps, Rs) by maximizing:
        sum_{n,t} E_q[ log p(z_{t+1}|z_t, x_t) ].

    This version adds small, targeted regularization / projection meant to reduce
    transition collapse and reward gauge drift without changing the overall
    architecture.

    Regularization choices:
      1) row-normalize log_Ps after every step
      2) clip log_Ps / Rs to avoid runaway logits
      3) center Rs across modes k for each state s (gauge fixing)
      4) small L2 penalty on Rs
      5) Dirichlet-style pseudo-count smoothing on learned Ps
      6) entropy-floor mixing of Ps toward uniform

    Expected shapes:
      log_Ps : (K, K)
      Rs     : (S, 1, K)
      xi_arr : (N, T-1, K, K)
      one_hot_xs : (N, T, 1, S) or equivalent list/array
    """

    # -----------------------------
    # hyperparameters
    # -----------------------------
    ridge_R = float(kwargs.get("ridge_R", 1e-4))
    dir_Ps  = float(kwargs.get("dir_Ps", 0.05))
    uni_Ps  = float(kwargs.get("uni_Ps", 0.02))

    # ---- normalize expectations into arrays
    if (
        isinstance(expectations, (list, tuple))
        and len(expectations) > 0
        and isinstance(expectations[0], (tuple, list))
    ):
        gamma_arr = jnp.stack([jnp.array(e[0]) for e in expectations], axis=0)
        xi_arr    = jnp.stack([jnp.array(e[1]) for e in expectations], axis=0)
    else:
        gamma_arr = jnp.array(expectations[0])
        xi_arr    = jnp.array(expectations[1])

    xoh = jnp.array(one_hot_xs)

    if xoh.ndim == 4 and xoh.shape[2] != 1:
        raise ValueError(f"one_hot_xs expected (N,T,1,S) or list; got {xoh.shape}")

    # -----------------------------
    # helpers
    # -----------------------------
    def _project_params(params):
        log_Ps_p, Rs_p = params

        # stabilize / normalize transition logits
        log_Ps_p = jnp.clip(log_Ps_p, -clip_logPs, clip_logPs)
        log_Ps_p = log_Ps_p - jax_logsumexp(log_Ps_p, axis=1, keepdims=True)

        # stabilize / identify reward offsets
        Rs_p = jnp.clip(Rs_p, -clip_Rs, clip_Rs)

        # gauge fix: for each state, remove the mean across modes
        # Rs is (K,1,S)
        Rs_p = Rs_p - jnp.mean(Rs_p, axis=2, keepdims=True)

        return (log_Ps_p, Rs_p)

    def _expected_log_joint(params):
        log_Ps_p, Rs_p = params

        def per_traj(xoh_i, xi_i):
            log_trans = comp_log_transP(log_Ps_p, Rs_p, xoh_i)  # (T-1,K,K)
            return jnp.sum(xi_i * log_trans)

        return jnp.sum(jax.vmap(per_traj)(xoh, xi_arr))

    def _loss(params):
        log_Ps_p, Rs_p = params
        neg_elbo = -_expected_log_joint((log_Ps_p, Rs_p))
        reg_R = ridge_R * jnp.mean(Rs_p ** 2)
        return neg_elbo + reg_R

    params = (jnp.array(log_Ps), jnp.array(Rs))
    params = _project_params(params)

    opt = optax.adam(learning_rate)
    opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state):
        loss_val, grads = jax.value_and_grad(_loss)(params)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        params = _project_params(params)
        return params, opt_state, loss_val

    for _ in range(num_iters):
        params, opt_state, _ = step(params, opt_state)

    # -----------------------------
    # final anti-collapse smoothing on Ps
    # -----------------------------
    new_log_Ps, new_Rs = params
    P = jnp.exp(new_log_Ps)  # row-stochastic by construction

    # implied expected transition counts from xi
    counts = jnp.sum(xi_arr, axis=(0, 1))  # (K,K)

    # Dirichlet-style pseudo-count smoothing
    counts_sm = counts + dir_Ps

    # row-normalize, then combine with optimized P
    P_counts = counts_sm / jnp.clip(jnp.sum(counts_sm, axis=1, keepdims=True), a_min=eps)
    P_mix = 0.5 * P + 0.5 * P_counts

    # entropy floor: mix slightly with uniform
    K = P_mix.shape[1]
    U = jnp.ones_like(P_mix) / K
    P_mix = (1.0 - uni_Ps) * P_mix + uni_Ps * U
    P_mix = P_mix / jnp.clip(jnp.sum(P_mix, axis=1, keepdims=True), a_min=eps)

    new_log_Ps = jnp.log(jnp.clip(P_mix, a_min=eps))
    new_log_Ps = new_log_Ps - jax_logsumexp(new_log_Ps, axis=1, keepdims=True)

    # one more gauge-fix on Rs before returning
    new_Rs = new_Rs - jnp.mean(new_Rs, axis=2, keepdims=True)

    return new_log_Ps, new_Rs

def emit_m_step_jaxnet_optax2(
    R_state,
    trans_probs,
    expectations,
    one_hot_xs,
    one_hot_acs,
    one_hot_hs,
    num_iters=1000,
    batch_size=16,
    discount=0.95,
    vi_threshold=50,
    tau=1.0,
    return_metrics=False,
    **kwargs
):
    """
    Teacher-only M-step for reward net parameters theta.

    Maximizes:
        sum_{n,t,k} gamma[n,t,k] * log pi_{theta,t,k}(a_{n,t} | x_{n,t})

    where pi_{theta,t,k} is induced by SoftVI on reward R_{theta,t,k}(s,a)
    produced from (h_t, mode k).

    IMPORTANT:
      This function must use the SAME reward->policy map as build_logemit_list,
      i.e. SoftVI on (reward / tau), not on raw reward.
    """
    apply_fn = R_state.apply_fn
    eps = 1e-20

    # -----------------------
    # Parse expectations -> gamma: (N,T,K)
    # -----------------------
    gamma_arr = None

    # Case 1: expectations is output of batch e-step: (gamma, xi, alpha)
    if isinstance(expectations, (tuple, list)) and len(expectations) >= 1:
        cand0 = jnp.array(expectations[0])
        if cand0.ndim >= 2:
            gamma_arr = cand0

    # Case 2: expectations is list of per-traj tuples [(gamma, xi), ...]
    if gamma_arr is None and isinstance(expectations, (list, tuple)) and len(expectations) > 0:
        if isinstance(expectations[0], (tuple, list)):
            gamma_arr = jnp.stack([jnp.array(e[0]) for e in expectations], axis=0)

    # Case 3: expectations is directly an array
    if gamma_arr is None:
        gamma_arr = jnp.array(expectations)

    # Common squeeze: (N,T,1,K) -> (N,T,K)
    if gamma_arr.ndim == 4:
        gamma_arr = gamma_arr[:, :, 0, :]

    if gamma_arr.ndim != 3:
        raise ValueError(f"[emit_m_step] gamma must be (N,T,K); got {gamma_arr.shape}")

    N, T, K = gamma_arr.shape

    # -----------------------
    # Squeeze one-hots to (N,T,*)
    # -----------------------
    xoh = jnp.array(one_hot_xs)
    aoh = jnp.array(one_hot_acs)
    hoh = jnp.array(one_hot_hs)

    if xoh.ndim == 4:
        xoh = xoh[:, :, 0, :]
    if aoh.ndim == 4:
        aoh = aoh[:, :, 0, :]
    if hoh.ndim == 4:
        hoh = hoh[:, :, 0, :]

    if xoh.shape[:2] != (N, T):
        raise ValueError(
            f"[emit_m_step] xoh must have shape (N,T,S); got {xoh.shape}, "
            f"expected first dims {(N, T)}"
        )
    if aoh.shape[:2] != (N, T):
        raise ValueError(
            f"[emit_m_step] aoh must have shape (N,T,A); got {aoh.shape}, "
            f"expected first dims {(N, T)}"
        )
    if hoh.shape[:2] != (N, T):
        raise ValueError(
            f"[emit_m_step] hoh must have shape (N,T,H); got {hoh.shape}, "
            f"expected first dims {(N, T)}"
        )

    # -----------------------
    # Hyperparameters
    # -----------------------
    tau = float(tau)
    lr = float(kwargs.get("lr", 3e-4))
    wd = float(kwargs.get("weight_decay", 0.0))
    grad_clip = float(kwargs.get("grad_clip", 1.0))
    seed = int(kwargs.get("seed", 0))

    # ---- structural identifiability penalties ----
    # Keep h_t-dependent reward close to a stable mode-specific base reward
    lam_base = float(kwargs.get("lam_base", 1e-3))

    # Encourage adjacent timesteps to have similar reward maps
    lam_smooth = float(kwargs.get("lam_smooth", 1e-3))

    # Planning with an absorbing goal helps distinguish "reach" from "reach-and-stay".
    absorb_goal_state = kwargs.get("absorb_goal_state", 24)

    tp = make_goal_absorbing(trans_probs, goal_state=absorb_goal_state)  # (S,A,S)
    S, A = tp.shape[0], tp.shape[1]
    eyeS = jnp.eye(S, dtype=tp.dtype)

    # ---- AIRL-style decomposition hyperparams ----
    lam_phi = float(kwargs.get("lam_phi", 1e-4))          # mild L2 on potential
    center_phi = bool(kwargs.get("center_phi", True))     # fix gauge: mean_s phi = 0

    def per_traj_expected_loglik(params, gamma_TK, x_Ts, a_Ta, h_TH):
        """
        Returns:
            sum_{t,k} gamma[t,k] * log pi_{t,k}(a_t | x_t)
        """
        def per_t(h_t):
            tmp_state = R_state.replace(params=params)
            shaped_ksa = shaped_rewards_from_h(
                tmp_state,
                h_t,
                tp,
                discount=discount,
                center_phi=center_phi
            )  # (K,S,A)

            pi_ksa = jax.vmap(
                lambda r_sa: soft_vi_sa(
                    tp,
                    r_sa / tau,
                    discount=discount,
                    threshold=vi_threshold
                )
            )(shaped_ksa)                                            # (K,S,A)

            return jnp.log(pi_ksa + eps)                             # (K,S,A)

        # (T,K,S,A)
        logemit_tksa = jax.vmap(per_t)(h_TH)

        # (T,K)
        lls_tk = comp_ll_jax_timevary(logemit_tksa, x_Ts, a_Ta)

        return jnp.sum(gamma_TK * lls_tk)
    
    def per_traj_airl_penalty(params, h_TH):
        """
        Mild structural regularizer:
          - penalize potential magnitude so the model cannot dump everything into phi
          - keep base reward smoother over time (optional but useful for h_t setup)
        """

        def per_t(h_t):
            tmp_state = R_state.replace(params=params)
            r_ska, phi_sk = _reward_and_phi_from_h_from_tp(
                tmp_state,
                h_t,
                tp,
                center_phi=center_phi
            )
            return r_ska, phi_sk

        r_t_ska, phi_t_sk = jax.vmap(per_t)(h_TH)   # (T,S,K,A), (T,S,K)

        # phi magnitude penalty
        phi_pen = jnp.mean(phi_t_sk ** 2)

        # keep h_t-dependent reward close to a trajectory-level base reward
        # base reward = time-average reward map over this trajectory
        r_base_ska = jnp.mean(r_t_ska, axis=0, keepdims=True)   # (1,S,K,A)
        base_pen = jnp.mean((r_t_ska - r_base_ska) ** 2)

        # temporal smoothness on reward
        if h_TH.shape[0] > 1:
            dr = r_t_ska[1:] - r_t_ska[:-1]
            smooth_pen = jnp.mean(dr ** 2)
        else:
            smooth_pen = 0.0

        return lam_phi * phi_pen + lam_base * base_pen + lam_smooth * smooth_pen

    # -----------------------
    # Optimizer
    # -----------------------
    opt = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adamw(learning_rate=lr, weight_decay=wd),
    )
    opt_state = opt.init(R_state.params)
    key = jax.random.PRNGKey(seed)

    def loss_on_batch(params, idx):
        gamma_b = gamma_arr[idx]
        xoh_b = xoh[idx]
        aoh_b = aoh[idx]
        hoh_b = hoh[idx]

        traj_logps = jax.vmap(
            per_traj_expected_loglik,
            in_axes=(None, 0, 0, 0, 0)
        )(params, gamma_b, xoh_b, aoh_b, hoh_b)

        traj_pen = jax.vmap(
            per_traj_airl_penalty,
            in_axes=(None, 0)
        )(params, hoh_b)

        return -jnp.sum(traj_logps) + jnp.sum(traj_pen)
    

    @jax.jit
    def step(params, opt_state, key):
        key, subk = jax.random.split(key)
        replace = batch_size > N
        idx = jax.random.choice(subk, N, (batch_size,), replace=replace)
        loss, grads = jax.value_and_grad(loss_on_batch)(params, idx)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, key, loss

    params = R_state.params
    last_loss = None
    for _ in range(num_iters):
        params, opt_state, key, last_loss = step(params, opt_state, key)

    new_state = R_state.replace(params=params)

    if not return_metrics:
        return new_state

    metrics = {
        "neg_expected_loglik_batch": float(last_loss) if last_loss is not None else None,
        "N": int(N),
        "T": int(T),
        "K": int(K),
        "tau": float(tau),
        "lr": float(lr),
        "weight_decay": float(wd),
        "grad_clip": float(grad_clip),
        "lam_base": float(lam_base),
        "lam_smooth": float(lam_smooth),
        "lam_phi": float(lam_phi),
    }
    return new_state, metrics


def pi0_m_step(all_gamma, eps=1e-8):
    # Convert lists / nested structures to a JAX array
    gamma = jnp.array(all_gamma)

    # gamma should be (N, T, K) or (T, K)
    if gamma.ndim == 3:
        pi0 = jnp.mean(gamma[:, 0, :], axis=0)
    elif gamma.ndim == 2:
        pi0 = gamma[0, :]
    else:
        raise ValueError(f"Unexpected gamma shape in pi0_m_step: {gamma.shape}")

    # Safe normalization + clipping to avoid log(0), inf, nan
    pi0 = pi0 / (jnp.sum(pi0) + eps)
    pi0 = jnp.clip(pi0, eps, 1.0)
    pi0 = pi0 / jnp.sum(pi0)

    return jnp.log(pi0)
