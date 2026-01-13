import numpy as np
import numpy.random as npr
from scipy.special import logsumexp

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

def _viterbi_JAX(pi0, Ps, ll):
    T, K = ll.shape

    # Check if the transition matrices are stationary or time-varying (hetero)
    hetero = (Ps.shape[0] == T-1)

    def score_fn(carry, inputs):
        score_next_t = carry
        t = inputs

        vals = jnp.log(Ps[t * hetero]) + score_next_t + ll[t+1]
        
        def comp_arg_score_over_K(vals):
            return jnp.argmax(vals), jnp.max(vals)
        
        arg_next_t, score_t = vmap(comp_arg_score_over_K)(vals)

        return score_t[None, :], (score_t, arg_next_t)
    
    _, (scores, args) = lax.scan(score_fn, jnp.zeros((1, K)), jnp.arange(T-2,-1,-1))

    z0 = jnp.array([(scores[-1] + jnp.log(pi0) + ll[0]).argmax()])
    
    def scan_z(carry, inputs):
        z_prev_t = carry
        arg_next_t = inputs

        z_t = arg_next_t[z_prev_t.astype(int)]

        return z_t, z_t

    _, z = lax.scan(scan_z, z0, args[::-1])

    return jnp.concatenate([z0, z.squeeze()]).astype(int)

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

def jaxnet_e_step_logpi(pi0, log_Ps, Rs, logemit, trans_probs, xoh, aoh, ctx_bias=None):
    """
    E-step for ONE trajectory.

    logemit can be:
      (K, S, A)     stationary emissions (original SWIRL)
      (T, K, S, A)  time-varying emissions (via h_t)

    ctx_bias can be:
      None or (T-1, K, K)  time-varying transition bias (via h_t)
    """
    # transitions over z
    Ps_jax = comp_transP(jnp.array(log_Ps), jnp.array(Rs), jnp.array(xoh), ctx_bias=ctx_bias)

    # emissions / action likelihoods
    logemit = jnp.array(logemit)
    if logemit.ndim == 3:
        log_likes_jax = comp_ll_jax(logemit, jnp.array(xoh), jnp.array(aoh))  # (T,K)
    elif logemit.ndim == 4:
        log_likes_jax = comp_ll_jax_timevary(logemit, jnp.array(xoh), jnp.array(aoh))  # (T,K)
    else:
        raise ValueError(f"logemit must have ndim 3 or 4, got {logemit.ndim}")

    alpha_jax = forward(pi0, Ps_jax, log_likes_jax)
    beta_jax  = backward(Ps_jax, log_likes_jax)
    gamma_jax, xi_jax = expected_states(alpha_jax, beta_jax, Ps_jax, log_likes_jax)
    return gamma_jax, xi_jax, alpha_jax


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
    **kwargs
):
    apply_fn = R_state.apply_fn
    tp = jnp.array(trans_probs)
    S, A = tp.shape[0], tp.shape[1]

    # gamma to (N,T,K)
    if isinstance(expectations, (list, tuple)) and len(expectations) > 0 and isinstance(expectations[0], (tuple, list)):
        gamma_arr = jnp.stack([jnp.array(e[0]) for e in expectations], axis=0)
    else:
        gamma_arr = jnp.array(expectations)

    if gamma_arr.ndim == 4:  # (N,T,1,K)
        gamma_arr = gamma_arr[:, :, 0, :]
    if gamma_arr.ndim != 3:
        raise ValueError(f"gamma must be (N,T,K); got {gamma_arr.shape}")

    N, T, K = gamma_arr.shape

    # squeeze inputs to (N,T,S/A/H)
    xoh = jnp.array(one_hot_xs)
    aoh = jnp.array(one_hot_acs)
    hoh = jnp.array(one_hot_hs)

    if xoh.ndim == 4: xoh = xoh[:, :, 0, :]
    if aoh.ndim == 4: aoh = aoh[:, :, 0, :]
    if hoh.ndim == 4: hoh = hoh[:, :, 0, :]

    eyeS = jnp.eye(S)

    def rewards_from_h(params, h_t):
        h_rep = jnp.repeat(h_t[None, :], S, axis=0)           # (S,H)
        inp = jnp.concatenate([eyeS, h_rep], axis=1)          # (S,S+H)

        out = apply_fn({'params': params}, inp)               # (S,K,A) or (S,K*A)
        if out.ndim == 2:
            if out.shape[1] != K * A:
                raise ValueError(f"Net output {out.shape} not compatible with K*A={K*A}")
            out = out.reshape(S, K, A)
        elif out.ndim != 3:
            raise ValueError(f"Net output must be (S,K,A) or (S,K*A); got {out.shape}")
        return out                                            # (S,K,A)

    def per_traj_logp(params, gamma_TK, x_Ts, a_Ta, h_TH):
        def per_t(h_t):
            r_ska = rewards_from_h(params, h_t)               # (S,K,A)
            r_ksa = jnp.transpose(r_ska, (1, 0, 2))           # (K,S,A)
            pi_ksa = vmap(lambda r_sa: soft_vi_sa(tp, r_sa, discount=discount, threshold=vi_threshold))(r_ksa)
            return jnp.log(pi_ksa + 1e-20)                    # (K,S,A)

        logemit_tksa = vmap(per_t)(h_TH)                      # (T,K,S,A)
        lls_tk = comp_ll_jax_timevary(logemit_tksa, x_Ts, a_Ta)  # (T,K)
        return jnp.sum(gamma_TK * lls_tk)

    lr = kwargs.get("lr", 3e-4)
    wd = kwargs.get("weight_decay", 0.0)
    grad_clip = kwargs.get("grad_clip", 1.0)

    opt = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adamw(learning_rate=lr, weight_decay=wd),
    )
    opt_state = opt.init(R_state.params)

    key = jax.random.PRNGKey(kwargs.get("seed", 0))

    def loss_on_batch(params, idx):
        gamma_b = gamma_arr[idx]
        xoh_b   = xoh[idx]
        aoh_b   = aoh[idx]
        hoh_b   = hoh[idx]
        traj_logps = jax.vmap(per_traj_logp, in_axes=(None,0,0,0,0))(params, gamma_b, xoh_b, aoh_b, hoh_b)
        return -jnp.sum(traj_logps)

    @jax.jit
    def step(params, opt_state, key):
        key, subk = jax.random.split(key)
        idx = jax.random.choice(subk, N, (batch_size,), replace=False)
        loss, grads = jax.value_and_grad(loss_on_batch)(params, idx)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, key, loss

    params = R_state.params
    for _ in range(num_iters):
        params, opt_state, key, _ = step(params, opt_state, key)

    return R_state.replace(params=params)



def emit_m_step_jaxnet_optax2_expand(
    R_state,
    trans_probs,
    expectations,
    one_hot_xs,
    one_hot_acs,
    one_hot_hs,
    num_iters=1000,
    **kwargs
):
    # Just reuse the corrected implementation
    return emit_m_step_jaxnet_optax2(
        R_state=R_state,
        trans_probs=trans_probs,
        expectations=expectations,
        one_hot_xs=one_hot_xs,
        one_hot_acs=one_hot_acs,
        one_hot_hs=one_hot_hs,
        num_iters=num_iters,
        **kwargs
    )


# def pi0_m_step(gammas):
#     pi0 = sum(gammas[:, 0]) + 1e-8
#     return jnp.log(pi0 / pi0.sum())

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

