import numpy as np
import numpy.random as npr
from scipy.special import logsumexp
from flax import linen as nn
from flax.training import train_state


#JAX
import jax
import jax.numpy as jnp
from jax import lax, vmap, jit
from functools import partial
from jax.scipy.special import logsumexp as jax_logsumexp
import optax
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_platform_name", "cpu")


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
      Rs:     (K, 1, S) or compatible with dot below (same as your current code)
      one_hot_x: (T, 1, S)
      ctx_bias: optional (T-1, K, K)

    Returns:
      (T-1, K, K) normalized log transition probs
    """
    T = one_hot_x.shape[0]

    log_Ps_t = jnp.tile(log_Ps[None, :, :], (T-1, 1, 1))  # (T-1,K,K)

    # state-dependent bias (your original term)
    if one_hot_x.ndim == 2:
            x = one_hot_x[:-1, :]
    else:
        x = one_hot_x[:-1, 0, :]
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
   
def one_hot(z, K):
    z = np.atleast_1d(z).astype(int)
    assert np.all(z >= 0) and np.all(z < K)
    shp = z.shape
    N = z.size
    zoh = np.zeros((N, K))
    zoh[np.arange(N), np.arange(K)[np.ravel(z)]] = 1
    zoh = np.reshape(zoh, shp + (K,))
    return zoh

def one_hot2(z, z_prev, K):
    z = z * K + z_prev
    z = np.atleast_1d(z).astype(int)
    K2 = K * K
    assert np.all(z >= 0) and np.all(z < K2)
    shp = z.shape
    N = z.size
    zoh = np.zeros((N, K2))
    zoh[np.arange(N), np.arange(K2)[np.ravel(z)]] = 1
    zoh = np.reshape(zoh, shp + (K2,))
    return zoh

def comp_ll_transP(xs, acs, logemit_learnt, model):
    lls = []
    trans_Ps = []
    for (x, ac) in zip(xs, acs):
        variational_mean = x.astype(int)[:, np.newaxis]
        n_ac = ac.astype(int)[:, np.newaxis]
        trans = model.transitions.transition_matrices(variational_mean, None, None, None)
        log_likes = model.dynamics.log_likelihoods(variational_mean, n_ac, None, np.ones_like(variational_mean, dtype=bool), None)
        lls.append(log_likes)
        trans_Ps.append(trans)
    return np.array(lls), np.array(trans_Ps)

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


import flax.linen as nn
import jax
import jax.numpy as jnp

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


def _log_softmax(x, axis=-1):
    return x - jax_logsumexp(x, axis=axis, keepdims=True)

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

    # 3) Run your existing forward/backward (safe because Ps is normalized)
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

    logemit_list: (N,T,K,S,A) (time-varying) OR (N,K,S,A) if you ever want stationary baseline
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


from jaxopt import BFGS, LBFGS

def trans_m_step_jax_optax(log_Ps, Rs, expectations, one_hot_xs, num_iters=1000, learning_rate=5e-3, **kwargs):
    """
    Update (log_Ps, Rs) by maximizing sum_{n,t} E_q[ log p(z_{t+1}|z_t, x_t) ].

    expectations can be:
      - list of tuples: [(gamma_i, xi_i), ...] where gamma_i:(T,K), xi_i:(T-1,K,K)
      - or (gamma_arr, xi_arr): gamma_arr:(N,T,K), xi_arr:(N,T-1,K,K)
    one_hot_xs: (N,T,1,S) or list of (T,1,S)
    """

    # ---- normalize expectations into arrays
    if isinstance(expectations, (list, tuple)) and len(expectations) > 0 and isinstance(expectations[0], (tuple, list)):
        gamma_arr = jnp.stack([jnp.array(e[0]) for e in expectations], axis=0)
        xi_arr    = jnp.stack([jnp.array(e[1]) for e in expectations], axis=0)
    else:
        gamma_arr = jnp.array(expectations[0])
        xi_arr    = jnp.array(expectations[1])

    xoh = jnp.array(one_hot_xs)

    # squeeze (N,T,1,S) -> (N,T,1,S) kept because comp_log_transP expects [:,0,:] in your file
    if xoh.ndim == 4 and xoh.shape[2] != 1:
        raise ValueError(f"one_hot_xs expected (N,T,1,S) or list; got {xoh.shape}")

    def expected_log_joint(params):
        log_Ps_p, Rs_p = params  # shapes consistent with your comp_log_transP
        def per_traj(xoh_i, xi_i):
            # log_trans: (T-1,K,K)
            log_trans = comp_log_transP(log_Ps_p, Rs_p, xoh_i)
            return jnp.sum(xi_i * log_trans)

        return jnp.sum(jax.vmap(per_traj)(xoh, xi_arr))

    def loss(params):
        return -expected_log_joint(params)

    params = (jnp.array(log_Ps), jnp.array(Rs))
    opt = optax.adam(learning_rate)
    opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state):
        l, grads = jax.value_and_grad(loss)(params)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, l

    for _ in range(num_iters):
        params, opt_state, _ = step(params, opt_state)

    return params  # (new_log_Ps, new_Rs)


import optax

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
    return_metrics=False,
    **kwargs
):
    """
    Teacher-only Step-F / M-step for reward net parameters theta.

    This maximizes the teacher-posterior expected action log-likelihood:

        max_theta  sum_{n,t,k} gamma_T[n,t,k] * log pi_{theta,t,k}(a_{n,t} | x_{n,t})

    where:
      - gamma_T is produced by the TEACHER (forward-backward) E-step
      - pi_{theta,t,k} is induced by SoftVI on reward R_{theta,t,k}(s,a) produced from (h_t, mode k)

    Notes:
      - This is "teacher-only" because the only posterior weights used are gamma_T.
      - The entropy term -E_q[log q] in the ELBO does not depend on theta, so it is irrelevant here.
      - The mode prior term log p(z) also does not depend on theta (unless you couple it to theta),
        so this update is exactly the correct teacher-only reward M-step.

    """
    apply_fn = R_state.apply_fn

    tp = jnp.array(trans_probs)  # (S,A,S)
    S, A = tp.shape[0], tp.shape[1]
    eyeS = jnp.eye(S, dtype=tp.dtype)
    eps = 1e-20

    # -----------------------
    # Parse expectations -> gamma_T: (N,T,K)
    # -----------------------
    gamma_arr = None

    # Case 1: expectations is output of teacher batch e-step: (gamma, xi, alpha)
    if isinstance(expectations, (tuple, list)) and len(expectations) >= 1:
        # If first element looks like gamma batch
        cand0 = jnp.array(expectations[0])
        if cand0.ndim >= 2:
            gamma_arr = cand0

    # Case 2: expectations is list of per-traj tuples like [(gamma, xi), ...]
    if gamma_arr is None and isinstance(expectations, (list, tuple)) and len(expectations) > 0:
        if isinstance(expectations[0], (tuple, list)):
            gamma_arr = jnp.stack([jnp.array(e[0]) for e in expectations], axis=0)

    # Case 3: expectations is directly an array
    if gamma_arr is None:
        gamma_arr = jnp.array(expectations)

    # Squeeze common singleton dims: (N,T,1,K) -> (N,T,K)
    if gamma_arr.ndim == 4:
        gamma_arr = gamma_arr[:, :, 0, :]
    if gamma_arr.ndim != 3:
        raise ValueError(f"[emit_m_step] gamma must be (N,T,K); got {gamma_arr.shape}")

    N, T, K = gamma_arr.shape

    # -----------------------
    # Squeeze one-hots to (N,T,*) consistently
    # -----------------------
    xoh = jnp.array(one_hot_xs)
    aoh = jnp.array(one_hot_acs)
    hoh = jnp.array(one_hot_hs)

    if xoh.ndim == 4: xoh = xoh[:, :, 0, :]
    if aoh.ndim == 4: aoh = aoh[:, :, 0, :]
    if hoh.ndim == 4: hoh = hoh[:, :, 0, :]

    if xoh.shape[:2] != (N, T):
        raise ValueError(f"[emit_m_step] xoh must have shape (N,T,S); got {xoh.shape}, expected first dims {(N,T)}")
    if aoh.shape[:2] != (N, T):
        raise ValueError(f"[emit_m_step] aoh must have shape (N,T,A); got {aoh.shape}, expected first dims {(N,T)}")
    if hoh.shape[:2] != (N, T):
        raise ValueError(f"[emit_m_step] hoh must have shape (N,T,H); got {hoh.shape}, expected first dims {(N,T)}")

    # -----------------------
    # SoftVI temperature
    # -----------------------
    tau = float(kwargs.get("tau", 1.0))

    def rewards_from_h(params, h_t):
        """
        Reward net is expected to output either:
          - (S, K*A)  -> reshape to (S,K,A)
          - (S, K, A) -> use directly
        Input to reward net is [one_hot_state || h_t repeated].
        """
        h_rep = jnp.repeat(h_t[None, :], S, axis=0)   # (S,H)
        inp = jnp.concatenate([eyeS, h_rep], axis=1)  # (S,S+H)

        out = apply_fn({'params': params}, inp)

        if out.ndim == 2:
            if out.shape[1] != K * A:
                raise ValueError(
                    f"[emit_m_step] Reward net output {out.shape} incompatible with K={K}, A={A}. "
                    f"Expected (S, K*A) = ({S}, {K*A})."
                )
            return out.reshape(S, K, A)

        if out.ndim == 3:
            if out.shape != (S, K, A):
                raise ValueError(f"[emit_m_step] Reward net output must be (S,K,A); got {out.shape}, expected {(S,K,A)}")
            return out

        raise ValueError(f"[emit_m_step] Reward net output must be (S,K*A) or (S,K,A); got {out.shape}")

    def per_traj_expected_loglik(params, gamma_TK, x_Ts, a_Ta, h_TH):
        """
        Returns: sum_{t,k} gamma[t,k] * log pi_{t,k}(a_t | x_t)
        """
        def per_t(h_t):
            r_ska = rewards_from_h(params, h_t)            # (S,K,A)
            r_ksa = jnp.transpose(r_ska, (1, 0, 2))        # (K,S,A)

            # SoftVI per mode, reward scaled by tau
            def vi_on_mode(r_sa):
                return soft_vi_sa(tp, r_sa / tau, discount=discount, threshold=vi_threshold)

            pi_ksa = jax.vmap(vi_on_mode)(r_ksa)           # (K,S,A)
            return jnp.log(pi_ksa + eps)                   # (K,S,A)

        # logemit_tksa: (T,K,S,A)
        logemit_tksa = jax.vmap(per_t)(h_TH)

        # lls_tk: (T,K)
        lls_tk = comp_ll_jax_timevary(logemit_tksa, x_Ts, a_Ta)

        return jnp.sum(gamma_TK * lls_tk)

    # -----------------------
    # Optimizer: AdamW + grad clip
    # -----------------------
    lr = float(kwargs.get("lr", 3e-4))
    wd = float(kwargs.get("weight_decay", 0.0))
    grad_clip = float(kwargs.get("grad_clip", 1.0))

    opt = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adamw(learning_rate=lr, weight_decay=wd),
    )
    opt_state = opt.init(R_state.params)

    # PRNG for minibatching
    seed = int(kwargs.get("seed", 0))
    key = jax.random.PRNGKey(seed)

    def loss_on_batch(params, idx):
        gamma_b = gamma_arr[idx]
        xoh_b   = xoh[idx]
        aoh_b   = aoh[idx]
        hoh_b   = hoh[idx]
        traj_logps = jax.vmap(per_traj_expected_loglik, in_axes=(None,0,0,0,0))(params, gamma_b, xoh_b, aoh_b, hoh_b)
        # Negative because we minimize
        return -jnp.sum(traj_logps)

    @jax.jit
    def step(params, opt_state, key):
        key, subk = jax.random.split(key)
        # If batch_size > N, fall back to sampling with replacement
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

    # Report avg negative expected log-likelihood per trajectory for the final minibatch loss scale
    # (This is mainly for sanity checking / monitoring.)
    metrics = {
        "neg_expected_loglik_batch": float(last_loss) if last_loss is not None else None,
        "N": int(N),
        "T": int(T),
        "K": int(K),
        "tau": float(tau),
        "lr": float(lr),
        "weight_decay": float(wd),
    }
    return new_state, metrics

import optax

def emit_m_step_jaxnet_optax2_student(
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
    return_metrics=False,
    **kwargs
):
    """
    STUDENT-ONLY M-step for reward net parameters theta.

    This maximizes the amortized-posterior expected action log-likelihood:

        max_theta  sum_{n,t,k} gamma_S[n,t,k] * log pi_{theta,t,k}(a_{n,t} | x_{n,t})

    where:
      - gamma_S comes from the STUDENT amortized E-step:
            gamma_S, xi_S, alpha_S = amortized_e_step_batch(inf_state, h, x, a)
      - pi_{theta,t,k} is induced by SoftVI on reward R_{theta,t,k}(s,a) produced from (h_t, mode k)

    """
    apply_fn = R_state.apply_fn

    tp = jnp.array(trans_probs)  # (S,A,S)
    S, A = tp.shape[0], tp.shape[1]
    eyeS = jnp.eye(S, dtype=tp.dtype)
    eps = 1e-20

    # -----------------------
    # Parse expectations -> gamma_S: (N,T,K)
    # -----------------------
    gamma_arr = None

    # Preferred: amortized_e_step_batch returns (gamma, xi, alpha)
    if isinstance(expectations, (tuple, list)) and len(expectations) >= 1:
        cand0 = jnp.array(expectations[0])
        if cand0.ndim >= 2:
            gamma_arr = cand0

    # list of per-traj tuples [(gamma_i, xi_i, ...), ...]
    if gamma_arr is None and isinstance(expectations, (list, tuple)) and len(expectations) > 0:
        if isinstance(expectations[0], (tuple, list)):
            gamma_arr = jnp.stack([jnp.array(e[0]) for e in expectations], axis=0)

    # direct array
    if gamma_arr is None:
        gamma_arr = jnp.array(expectations)

    # Squeeze common singleton dims: (N,T,1,K) -> (N,T,K)
    if gamma_arr.ndim == 4:
        gamma_arr = gamma_arr[:, :, 0, :]
    if gamma_arr.ndim != 3:
        raise ValueError(f"[emit_m_step_student] gamma must be (N,T,K); got {gamma_arr.shape}")

    # IMPORTANT: M-step treats q fixed (variational EM)
    gamma_arr = jax.lax.stop_gradient(gamma_arr)

    N, T, K = gamma_arr.shape

    # -----------------------
    # Squeeze one-hots to (N,T,*) consistently
    # -----------------------
    xoh = jnp.array(one_hot_xs)
    aoh = jnp.array(one_hot_acs)
    hoh = jnp.array(one_hot_hs)

    if xoh.ndim == 4: xoh = xoh[:, :, 0, :]
    if aoh.ndim == 4: aoh = aoh[:, :, 0, :]
    if hoh.ndim == 4: hoh = hoh[:, :, 0, :]

    if xoh.shape[:2] != (N, T):
        raise ValueError(f"[emit_m_step_student] xoh must be (N,T,S); got {xoh.shape}, expected first dims {(N,T)}")
    if aoh.shape[:2] != (N, T):
        raise ValueError(f"[emit_m_step_student] aoh must be (N,T,A); got {aoh.shape}, expected first dims {(N,T)}")
    if hoh.shape[:2] != (N, T):
        raise ValueError(f"[emit_m_step_student] hoh must be (N,T,H); got {hoh.shape}, expected first dims {(N,T)}")

    # -----------------------
    # SoftVI temperature
    # -----------------------
    tau = float(kwargs.get("tau", 1.0))

    def rewards_from_h(params, h_t):
        """
        Reward net outputs either:
          - (S, K*A)  -> reshape to (S,K,A)
          - (S, K, A) -> use directly
        Input is [one_hot_state || h_t repeated].
        """
        h_rep = jnp.repeat(h_t[None, :], S, axis=0)   # (S,H)
        inp = jnp.concatenate([eyeS, h_rep], axis=1)  # (S,S+H)

        out = apply_fn({'params': params}, inp)

        if out.ndim == 2:
            if out.shape[1] != K * A:
                raise ValueError(
                    f"[emit_m_step_student] Reward net output {out.shape} incompatible with K={K}, A={A}. "
                    f"Expected (S, K*A)=({S}, {K*A})."
                )
            return out.reshape(S, K, A)

        if out.ndim == 3:
            if out.shape != (S, K, A):
                raise ValueError(f"[emit_m_step_student] Reward net output must be (S,K,A); got {out.shape}, expected {(S,K,A)}")
            return out

        raise ValueError(f"[emit_m_step_student] Reward net output must be (S,K*A) or (S,K,A); got {out.shape}")

    def per_traj_expected_loglik(params, gamma_TK, x_Ts, a_Ta, h_TH):
        """
        Returns: sum_{t,k} gamma[t,k] * log pi_{t,k}(a_t | x_t)
        Note: pi_{t,k}(a|x) == pi(a|x,h_t,z_t=k) since h_t enters through R_{t,k}.
        """
        def per_t(h_t):
            r_ska = rewards_from_h(params, h_t)            # (S,K,A)
            r_ksa = jnp.transpose(r_ska, (1, 0, 2))        # (K,S,A)

            def vi_on_mode(r_sa):
                return soft_vi_sa(tp, r_sa / tau, discount=discount, threshold=vi_threshold)

            pi_ksa = jax.vmap(vi_on_mode)(r_ksa)           # (K,S,A)
            return jnp.log(pi_ksa + eps)                   # (K,S,A)

        logemit_tksa = jax.vmap(per_t)(h_TH)               # (T,K,S,A)
        lls_tk = comp_ll_jax_timevary(logemit_tksa, x_Ts, a_Ta)  # (T,K)
        return jnp.sum(gamma_TK * lls_tk)

    # -----------------------
    # Optimizer: AdamW + grad clip
    # -----------------------
    lr = float(kwargs.get("lr", 3e-4))
    wd = float(kwargs.get("weight_decay", 0.0))
    grad_clip = float(kwargs.get("grad_clip", 1.0))

    opt = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adamw(learning_rate=lr, weight_decay=wd),
    )
    opt_state = opt.init(R_state.params)

    # PRNG for minibatching
    seed = int(kwargs.get("seed", 0))
    key = jax.random.PRNGKey(seed)

    def loss_on_batch(params, idx):
        gamma_b = gamma_arr[idx]
        xoh_b   = xoh[idx]
        aoh_b   = aoh[idx]
        hoh_b   = hoh[idx]
        traj_logps = jax.vmap(per_traj_expected_loglik, in_axes=(None,0,0,0,0))(params, gamma_b, xoh_b, aoh_b, hoh_b)
        return -jnp.sum(traj_logps)  # minimize negative expected loglik

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
        "posterior": "student",
    }
    return new_state, metrics


def pi0_m_step(all_gamma):
    # Convert lists / nested structures to a JAX array
    gamma = jnp.array(all_gamma)

    # gamma should be (N, T, K) or (T, K)
    if gamma.ndim == 3:
        # (N, T, K): average gamma at time 0 over trajectories
        # gamma[:, 0, :] -> (N, K)
        pi0 = jnp.mean(gamma[:, 0, :], axis=0)
    elif gamma.ndim == 2:
        # (T, K): single trajectory; use t = 0
        pi0 = gamma[0, :]
    else:
        raise ValueError(f"Unexpected gamma shape in pi0_m_step: {gamma.shape}")

    # Normalize to sum to 1 (stay as a vector, not a scalar)
    pi0 = pi0 / jnp.sum(pi0)
    return jnp.log(pi0)

