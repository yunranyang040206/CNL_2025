import os
import json
import numpy as np
from collections import deque
from scipy.special import logsumexp
from sklearn.metrics import roc_auc_score

# ==========================================================
# CONFIGURATION (fixed, reproducible)
# ==========================================================

SEED = 42
np.random.seed(SEED)

N = 200
T = 500
GRID = 5
S = GRID * GRID
A = 5
WATER = 24

# Drive dynamics
DELTA = 0.03
THRESHOLD = 0.5

# Noise
WATER_RANDOM = 0.15
WATER_HESITATE = 0.05
EXPLORE_RANDOM = 0.15

# SoftVI validation params
TAU = 1.0
DISCOUNT = 0.95
VI_ITERS = 50

OUTDIR = "generated_dataset_gw5"
os.makedirs(OUTDIR, exist_ok=True)

# ==========================================================
# Load transition matrix
# ==========================================================

if not os.path.exists("trans_prob.npy"):
    raise FileNotFoundError("trans_prob.npy must be present.")

trans_prob = np.load("trans_prob.npy")  # shape (S,A,S)

# ==========================================================
# Utilities
# ==========================================================

def to_xy(s):
    return s // GRID, s % GRID

def to_s(x, y):
    return x * GRID + y

# ==========================================================
# Shortest path distances to water
# ==========================================================

def compute_shortest_dist():
    dist = np.full(S, np.inf)
    dist[WATER] = 0
    queue = deque([WATER])

    while queue:
        s = queue.popleft()
        x, y = to_xy(s)

        neighbors = []
        if x > 0: neighbors.append(to_s(x-1,y))
        if x < GRID-1: neighbors.append(to_s(x+1,y))
        if y > 0: neighbors.append(to_s(x,y-1))
        if y < GRID-1: neighbors.append(to_s(x,y+1))

        for n in neighbors:
            if dist[n] == np.inf:
                dist[n] = dist[s] + 1
                queue.append(n)

    return dist

shortest_dist = compute_shortest_dist()

# ==========================================================
# Behavior policies
# ==========================================================

def choose_water_action(s):
    if np.random.rand() < WATER_HESITATE:
        return 4

    if np.random.rand() < WATER_RANDOM:
        return np.random.randint(A)

    best = []
    x, y = to_xy(s)

    for a, (nx, ny) in enumerate([
        (x-1,y),(x+1,y),(x,y-1),(x,y+1),(x,y)
    ]):
        if 0 <= nx < GRID and 0 <= ny < GRID:
            ns = to_s(nx, ny)
            if shortest_dist[ns] < shortest_dist[s]:
                best.append(a)

    if best:
        return np.random.choice(best)
    return 4


def choose_explore_action(s):
    """
    IMPORTANT FIX:
    Exploration AVOIDS WATER so drive can accumulate.
    """
    x, y = to_xy(s)

    actions = list(range(A))

    # remove actions that lead to water
    valid = []
    for a in actions:
        probs = trans_prob[s, a]
        ns = np.argmax(probs)
        if ns != WATER:
            valid.append(a)

    if not valid:
        valid = actions

    if np.random.rand() < EXPLORE_RANDOM:
        return np.random.choice(valid)

    # perimeter preference
    if x in [0, GRID-1] or y in [0, GRID-1]:
        return np.random.choice(valid)

    # move toward boundary
    if x > 0:
        if 0 in valid: return 0
    return np.random.choice(valid)

# ==========================================================
# Generate trajectories
# ==========================================================

xs = np.zeros((N, T+1), dtype=np.int64)
acs = np.zeros((N, T), dtype=np.int64)
zs = np.zeros((N, T), dtype=np.int64)

for n in range(N):
    s = np.random.randint(S)
    d = 0.0
    xs[n,0] = s

    for t in range(T):

        z = 1 if d >= THRESHOLD else 0
        zs[n,t] = z

        if z == 1:
            a = choose_water_action(s)
        else:
            a = choose_explore_action(s)

        acs[n,t] = a

        probs = trans_prob[s,a]
        s_next = np.random.choice(S, p=probs)

        xs[n,t+1] = s_next

        if s_next == WATER:
            d = 0.0
        else:
            d = min(1.0, d + DELTA)

        s = s_next

xs_fixed = xs.copy()

# ==========================================================
# Construct reward maps (approximate heuristics)
# ==========================================================

RG_sa = np.zeros((2,S,A))

for s in range(S):
    for a in range(A):
        probs = trans_prob[s,a]
        ns = np.argmax(probs)

        # Mode 1 reward (water-seeking)
        if ns == WATER:
            RG_sa[1,s,a] += 20
        if shortest_dist[ns] < shortest_dist[s]:
            RG_sa[1,s,a] += 5
        RG_sa[1,s,a] -= 0.1

        # Mode 0 reward (perimeter preference)
        x,y = to_xy(ns)
        if x in [0,GRID-1] or y in [0,GRID-1]:
            RG_sa[0,s,a] += 2
        RG_sa[0,s,a] -= 0.1

RG = np.zeros((2,S,S))  # compatibility

# ==========================================================
# SoftVI for validation
# ==========================================================

def soft_vi(R):
    V = np.zeros(S)
    for _ in range(VI_ITERS):
        Q = np.zeros((S,A))
        for s in range(S):
            for a in range(A):
                Q[s,a] = R[s,a] + DISCOUNT*np.sum(trans_prob[s,a]*V)
        V = TAU*logsumexp(Q/TAU, axis=1)
    policy = np.exp(Q/TAU - logsumexp(Q/TAU, axis=1, keepdims=True))
    return policy

pi0 = soft_vi(RG_sa[0])
pi1 = soft_vi(RG_sa[1])

# ==========================================================
# Per-timestep Δ separability
# ==========================================================

scores = []
labels = []

for n in range(N):
    for t in range(T):
        s = xs[n,t+1]
        a = acs[n,t]
        delta = np.log(pi1[s,a]+1e-12) - np.log(pi0[s,a]+1e-12)
        scores.append(delta)
        labels.append(zs[n,t])

scores = np.array(scores)
labels = np.array(labels)

print("Fraction z=1:", labels.mean())

if len(np.unique(labels)) < 2:
    raise RuntimeError("Only one latent class present in dataset.")

auc = roc_auc_score(labels, scores)
print("Per-timestep Δ AUC:", auc)

if auc < 0.7:
    raise RuntimeError("Dataset not sufficiently separable.")

# ==========================================================
# Save dataset
# ==========================================================

np.save(os.path.join(OUTDIR,"xs.npy"), xs)
np.save(os.path.join(OUTDIR,"xs_fixed.npy"), xs_fixed)
np.save(os.path.join(OUTDIR,"acs.npy"), acs)
np.save(os.path.join(OUTDIR,"zs.npy"), zs)
np.save(os.path.join(OUTDIR,"trans_prob.npy"), trans_prob)
np.save(os.path.join(OUTDIR,"RG_sa.npy"), RG_sa)
np.save(os.path.join(OUTDIR,"RG.npy"), RG)

manifest = {
    "seed": SEED,
    "N": N,
    "T": T,
    "grid_size": GRID,
    "water_state": WATER,
    "delta": DELTA,
    "threshold": THRESHOLD,
    "tau": TAU,
    "discount": DISCOUNT,
    "vi_iters": VI_ITERS,
    "alignment": "a_t moves x_t to x_{t+1}"
}

with open(os.path.join(OUTDIR,"manifest.json"),"w") as f:
    json.dump(manifest,f,indent=4)

print("Dataset successfully generated in:", OUTDIR)
