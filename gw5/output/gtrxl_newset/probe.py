#!/usr/bin/env python3
"""
Probe whether h_t encodes "thirst" (time-since-last-water), AND whether posterior gamma uses it.

Inputs (NPZ):
  - h_gw5.npz with key 'h'
      supported shapes:
        (N, T, H)           e.g., (200, 500, 32)
        (N, T, 1, H)        sometimes saved with an extra singleton axis
  - xs_gw5_for_embed.npz with key 'xs'
      shape (N, T), integer state ids
  - OPTIONAL: gamma_T_latest.npz with key 'gamma'
      shape (N_train, T, K), typically train subset only.

Key additions:
  - Handles your split logic: "every 5th episode is test" by default.
  - Aligns xs/h/tsld to the same trajectories used in gamma (train set).
  - Diagnostics:
      1) corr(tsld, gamma_k)
      2) corr(probe_prob(thirsty|h_t), gamma_k)
      3) binned mean gamma vs tsld bins (plus probe prob per bin)

Usage:
  python probe.py \
    --h_npz h_gw5.npz \
    --xs_npz xs_gw5_for_embed.npz \
    --water_state 24 \
    --tau 20 \
    --gamma_npz gamma_T_latest.npz
"""

import argparse
import numpy as np

def load_h(path: str) -> np.ndarray:
    data = np.load(path)
    if "h" not in data:
        raise KeyError(f"{path} missing key 'h'. Keys found: {list(data.keys())}")
    h = data["h"]
    if h.ndim == 3:
        return h
    if h.ndim == 4:
        if h.shape[2] == 1:      # (N,T,1,H)
            return h[:, :, 0, :]
        if h.shape[3] == 1:      # (N,T,H,1)
            return h[:, :, :, 0]
    raise ValueError(f"Unsupported h shape {h.shape}. Expected (N,T,H) or (N,T,1,H).")

def load_xs(path: str) -> np.ndarray:
    data = np.load(path)
    if "xs" not in data:
        raise KeyError(f"{path} missing key 'xs'. Keys found: {list(data.keys())}")
    xs = data["xs"]
    if xs.ndim != 2:
        raise ValueError(f"Expected xs shape (N,T), got {xs.shape}")
    return xs.astype(np.int64)

def compute_tsld(xs: np.ndarray, water_state: int) -> np.ndarray:
    N, T = xs.shape
    tsld = np.zeros((N, T), dtype=np.int64)
    for i in range(N):
        last = -1
        for t in range(T):
            if xs[i, t] == water_state:
                last = t
            tsld[i, t] = t if last == -1 else (t - last)
    return tsld

def split_every_kth(N: int, k: int = 5):
    """Your dataset split: every kth episode is test."""
    test_idx = np.arange(0, N, k, dtype=int)
    train_idx = np.setdiff1d(np.arange(N, dtype=int), test_idx).astype(int)
    return train_idx, test_idx

def bin_stats(tsld_flat: np.ndarray, gamma_flat: np.ndarray, prob_flat: np.ndarray,
              bin_width: int, max_bin: int):
    """
    Compute mean gamma (and mean probe prob) per tsld bin.
    Bins: [0,bin_width), [bin_width,2*bin_width), ..., [max_bin, +inf)
    """
    edges = list(range(0, max_bin + bin_width, bin_width))
    edges = np.array(edges, dtype=np.int64)

    # Assign bins, last bin is overflow
    b = np.minimum(tsld_flat // bin_width, max_bin // bin_width)

    nbins = (max_bin // bin_width) + 1
    out = []
    for bi in range(nbins):
        mask = (b == bi)
        cnt = int(mask.sum())
        if cnt == 0:
            out.append((bi, cnt, np.nan, np.nan))
        else:
            out.append((bi, cnt, float(gamma_flat[mask].mean()), float(prob_flat[mask].mean())))
    return out, edges

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h_npz", type=str, required=True)
    ap.add_argument("--xs_npz", type=str, required=True)
    ap.add_argument("--gamma_npz", type=str, default=None,
                    help="Optional gamma npz with key 'gamma' (train posterior).")
    ap.add_argument("--water_state", type=int, default=24)
    ap.add_argument("--tau", type=int, default=20)
    ap.add_argument("--split_k", type=int, default=5, help="Every kth episode is test (default 5).")
    ap.add_argument("--test_frac", type=float, default=0.2, help="Used only for probe split (internal).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_tsld_clip", type=int, default=200)
    ap.add_argument("--bin_width", type=int, default=5)
    ap.add_argument("--max_bin", type=int, default=60, help="Last bin is overflow >= max_bin.")
    args = ap.parse_args()

    # ---- Load base data ----
    h = load_h(args.h_npz)
    xs = load_xs(args.xs_npz)

    print(f"[load] h: shape={h.shape} dtype={h.dtype}")
    print(f"[load] xs: shape={xs.shape} dtype={xs.dtype}")

    # Align h/xs to common N,T (these two should already match)
    N = min(h.shape[0], xs.shape[0])
    T = min(h.shape[1], xs.shape[1])
    h = h[:N, :T, :]
    xs = xs[:N, :T]
    H = h.shape[-1]

    # ---- Compute tsld and thirsty labels on ALL episodes ----
    tsld = compute_tsld(xs, args.water_state)
    thirsty = (tsld >= args.tau).astype(np.int64)
    print(f"[labels] water_state={args.water_state} tau={args.tau}")
    print(f"[labels] thirsty frac={thirsty.mean():.4f} (over all traj/timesteps)")

    # ---- Train probe on all data (split by trajectory) ----
    # NOTE: This is only for diagnosing embedding quality; it doesn't change gamma alignment.
    rng = np.random.default_rng(args.seed)
    idx = np.arange(N)
    rng.shuffle(idx)
    n_test = int(round(N * args.test_frac))
    probe_test = idx[:n_test]
    probe_train = idx[n_test:]

    X_train = h[probe_train].reshape(-1, H)
    y_train = thirsty[probe_train].reshape(-1)
    X_test  = h[probe_test].reshape(-1, H)
    y_test  = thirsty[probe_test].reshape(-1)

    # Standardize
    mu = X_train.mean(axis=0, keepdims=True)
    sig = X_train.std(axis=0, keepdims=True) + 1e-8
    X_train_z = (X_train - mu) / sig
    X_test_z  = (X_test  - mu) / sig

    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import accuracy_score, roc_auc_score, r2_score, mean_absolute_error

    clf = LogisticRegression(max_iter=400, solver="lbfgs", class_weight="balanced")
    clf.fit(X_train_z, y_train)

    y_pred = clf.predict(X_test_z)
    y_prob = clf.predict_proba(X_test_z)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else float("nan")

    print("\n=== Logistic probe: thirsty(tsld>=tau) from h_t ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC:      {auc:.4f}")

    # Regression probe (optional)
    tsld_clip = np.clip(tsld, 0, args.max_tsld_clip).astype(np.float32)
    yreg_train = tsld_clip[probe_train].reshape(-1)
    yreg_test  = tsld_clip[probe_test].reshape(-1)
    reg = Ridge(alpha=1.0)
    reg.fit(X_train_z, yreg_train)
    yreg_pred = reg.predict(X_test_z)
    print("\n=== Ridge probe: clipped tsld from h_t ===")
    print(f"R^2: {r2_score(yreg_test, yreg_pred):.4f}")
    print(f"MAE: {mean_absolute_error(yreg_test, yreg_pred):.4f}  (tsld clipped to [0,{args.max_tsld_clip}])")

    # Compute probe probability for EVERY (traj,time), using the same normalization
    X_all = h.reshape(-1, H)
    X_all_z = (X_all - mu) / sig
    prob_all = clf.predict_proba(X_all_z)[:, 1].reshape(N, T)  # p(thirsty|h_t)

    # ---- If gamma provided, align properly and compute diagnostics ----
    if args.gamma_npz is None:
        print("\n[gamma] No --gamma_npz provided; skipping gamma alignment diagnostics.")
        return

    gdat = np.load(args.gamma_npz)
    if "gamma" not in gdat:
        raise KeyError(f"{args.gamma_npz} missing key 'gamma'. Keys found: {list(gdat.keys())}")
    gamma = gdat["gamma"]  # expected (N_train, T, K) OR (N, T, K)

    print(f"\n[gamma] gamma shape={gamma.shape} dtype={gamma.dtype}")

    # Your split logic: every kth episode test.
    # If gamma was computed on train set only, it should match len(train_indices).
    train_indices, test_indices = split_every_kth(N, args.split_k)
    print(f"[split-every-{args.split_k}] train={len(train_indices)} test={len(test_indices)}")

    # Decide how to align gamma:
    #  - If gamma has N dimension equal to full N: use all episodes directly.
    #  - If gamma has N dimension equal to len(train_indices): use train_indices to slice xs/tsld/prob.
    if gamma.shape[0] == N:
        use_idx = np.arange(N, dtype=int)
    elif gamma.shape[0] == len(train_indices):
        use_idx = train_indices
    else:
        raise ValueError(
            f"Cannot align gamma N={gamma.shape[0]} with data N={N} or train N={len(train_indices)}. "
            f"Best fix: save traj indices inside gamma NPZ."
        )

    # Align T
    T_use = min(T, gamma.shape[1])
    gamma = gamma[:, :T_use, :]
    xs_use = xs[use_idx, :T_use]
    tsld_use = tsld[use_idx, :T_use]
    prob_use = prob_all[use_idx, :T_use]

    # Flatten
    tsld_flat = tsld_use.reshape(-1).astype(np.float64)
    prob_flat = prob_use.reshape(-1).astype(np.float64)

    print(f"[aligned] use_idx N={len(use_idx)} T={T_use} flat={tsld_flat.size}")

    print("\n=== Correlation diagnostics ===")
    K = gamma.shape[-1]
    for k in range(K):
        gamma_k = gamma[:, :, k].reshape(-1).astype(np.float64)
        c_tsld = np.corrcoef(tsld_flat, gamma_k)[0, 1]
        c_prob = np.corrcoef(prob_flat, gamma_k)[0, 1]
        print(f"Mode {k}: corr(tsld, gamma)={c_tsld:+.4f}   corr(prob_thirsty, gamma)={c_prob:+.4f}")

        # Binned curve (mean gamma and mean prob per tsld bin)
        stats, edges = bin_stats(tsld_flat, gamma_k, prob_flat, args.bin_width, args.max_bin)
        print(f"\n  [Mode {k}] mean gamma by tsld bins (bin_width={args.bin_width}, overflow >= {args.max_bin})")
        print("  bin  range        count   mean_gamma   mean_prob_thirsty")
        for bi, cnt, mg, mp in stats:
            lo = bi * args.bin_width
            hi = (bi + 1) * args.bin_width
            rng = f"[{lo:>2},{hi:>2})" if bi < (args.max_bin // args.bin_width) else f"[{args.max_bin:>2},+inf)"
            print(f"  {bi:>3}  {rng:<10}  {cnt:>6}   {mg:>9.4f}      {mp:>9.4f}")

    print("\nDone. If mean_gamma curves are flat across bins, posterior is ignoring thirst semantics.")

if __name__ == "__main__":
    main()
