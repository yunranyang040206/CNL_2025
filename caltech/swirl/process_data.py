"""
Process the Caltech Mouse Social Behavior dataset (CalMS21 Task 1) into SWIRL format.

The dataset has 4 behavior classes:
  0: attack
  1: investigation
  2: mount
  3: other

Following the spontda convention:
  - state  = current behavior label (0-3)
  - action = next behavior label (0-3)
  - P(s' | s, a) = 1 if s' == a else 0  (deterministic environment)

Outputs saved to ../data/:
  seqs.npy              shape (n_videos, MAX_LEN)
  trans_probs.npy       shape (C, C, C)
  K_seed_arhmm_caltech.npz  initialisation parameters for SWIRL

Usage:
  python process_data.py [K] [seed]
"""

import sys
import json
import numpy as np
import numpy.random as npr
from pathlib import Path

# ── configuration ────────────────────────────────────────────────────────────
K    = int(sys.argv[1]) if len(sys.argv) > 1 else 4
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 30

C       = 4        # number of behavior classes
MIN_LEN = 2000     # discard videos shorter than this
MAX_LEN = 2000     # truncate videos to this length

data_dir   = Path('../data/task1_classic_classification')
output_dir = Path('../data')
output_dir.mkdir(parents=True, exist_ok=True)

# ── load raw data ─────────────────────────────────────────────────────────────
print("Loading calms21_task1_train.json ...")
with open(data_dir / 'calms21_task1_train.json', 'r') as f:
    train_data = json.load(f)

annotator = list(train_data.keys())[0]
videos    = list(train_data[annotator].keys())
print(f"  annotator : {annotator}")
print(f"  videos    : {len(videos)}")

# vocab (from metadata of first video)
meta  = train_data[annotator][videos[0]]['metadata']
vocab = meta.get('vocab', {})
print(f"  vocab     : {vocab}")

# ── extract sequences ─────────────────────────────────────────────────────────
seqs_list = []
for video in videos:
    anns = np.array(train_data[annotator][video]['annotations'], dtype=np.int64)
    if len(anns) >= MIN_LEN:
        seqs_list.append(anns[:MAX_LEN])
    else:
        print(f"  skipped {video} (length {len(anns)} < {MIN_LEN})")

seqs = np.array(seqs_list)          # (n_videos, MAX_LEN)
print(f"\nSequences kept : {seqs.shape[0]} × {seqs.shape[1]} frames")
counts = np.bincount(seqs.ravel(), minlength=C)
print(f"Class counts   : { {i: int(counts[i]) for i in range(C)} }")

np.save(output_dir / 'seqs.npy', seqs)
print(f"Saved seqs.npy  {seqs.shape}")

# ── transition probabilities ──────────────────────────────────────────────────
# Since action = next state, the environment is deterministic:
#   P(s' | s, a) = 1  iff  s' == a
trans_probs = np.zeros((C, C, C))
for a in range(C):
    for s in range(C):
        trans_probs[s, a, a] = 1.0

np.save(output_dir / 'trans_probs.npy', trans_probs)
print(f"Saved trans_probs.npy  {trans_probs.shape}")

# ── ARHMM initialisation parameters ──────────────────────────────────────────
# Following gw5: near-identity hidden-state transitions, zero reward modulation.
npr.seed(seed)

logpi0_start = np.log(np.ones(K) / K)           # uniform initial distribution

Ps            = 0.95 * np.eye(K) + 0.05 * npr.rand(K, K)
Ps           /= Ps.sum(axis=1, keepdims=True)
log_Ps_start  = np.log(Ps)                       # (K, K)

Rs_start      = np.zeros((C, 1, K))              # (n_states, 1, n_hidden)
init_start    = np.ones(K) / K                   # uniform

out_path = output_dir / f'{K}_{seed}_arhmm_caltech.npz'
np.savez(out_path,
         init_start    = init_start,
         logpi0_start  = logpi0_start,
         log_Ps_start  = log_Ps_start,
         Rs_start      = Rs_start)

print(f"\nSaved {out_path.name}")
print(f"  init_start  : {init_start.shape}")
print(f"  logpi0_start: {logpi0_start.shape}")
print(f"  log_Ps_start: {log_Ps_start.shape}")
print(f"  Rs_start    : {Rs_start.shape}")
print("\nDone.")
