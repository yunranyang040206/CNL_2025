"""
Process the Caltech Mouse Social Behavior dataset (CalMS21 Task 1) into
SWIRL format using COMPRESSED sequences.

Compression: consecutive self-transitions are removed, keeping only frames
where the behavior actually changes.  This collapses the high-frequency
self-transition noise caused by the fine time resolution and lets SWIRL
focus on genuine behavior switches.

Outputs saved to ../data/ (separate from the non-compressed pipeline):
  compressed_seqs.npy              shape (n_videos, t_traj)
  compressed_trans_probs.npy       shape (C, C, C)   (same deterministic structure)
  K_seed_arhmm_caltech_compressed.npz   SWIRL initialisation parameters

Usage:
  python process_compressed_data.py [K] [seed] [t_traj]

  K       number of hidden states                          (default 4)
  seed    random seed                                      (default 30)
  t_traj  fixed trajectory length; sequences shorter than
          this are discarded, longer ones are truncated    (default 20)
"""

import sys
import json
import numpy as np
import numpy.random as npr
from pathlib import Path

# ── CLI args ──────────────────────────────────────────────────────────────────
K      = int(sys.argv[1]) if len(sys.argv) > 1 else 2
seed   = int(sys.argv[2]) if len(sys.argv) > 2 else 30
T_TRAJ = int(sys.argv[3]) if len(sys.argv) > 3 else 20

C = 4
BEHAVIOR_NAMES = ['Attack', 'Investigation', 'Mount', 'Other']

data_dir   = Path('../data/task2_classic_classification')
output_dir = Path('../data')
output_dir.mkdir(parents=True, exist_ok=True)

# ── load raw data ─────────────────────────────────────────────────────────────
print("Loading calms21_task2_train.json ...")
with open(data_dir / 'calms21_task2_train.json', 'r') as f:
    train_data = json.load(f)

annotator = list(train_data.keys())[0]
videos    = list(train_data[annotator].keys())
print(f"  annotator : {annotator}")
print(f"  videos    : {len(videos)}")

# ── compress sequences ────────────────────────────────────────────────────────
def compress(seq, C_=4):
    """Remove consecutive self-transitions; keep only genuine state changes."""
    out = [seq[0]]
    for s in seq[1:]:
        if s < C_ and s != out[-1]:
            out.append(s)
    return out

raw_seqs = []
for video in videos:
    anns = list(train_data[annotator][video]['annotations'])
    raw_seqs.append(anns)

compressed = [compress(seq) for seq in raw_seqs]
comp_lens  = [len(c) for c in compressed]

print(f"\nCompressed length stats (all {len(compressed)} videos):")
print(f"  min={min(comp_lens)}  max={max(comp_lens)}  "
      f"mean={np.mean(comp_lens):.1f}  median={np.median(comp_lens):.1f}")

# ── filter and truncate to T_TRAJ ─────────────────────────────────────────────
print(f"\nUsing T_traj = {T_TRAJ}  (sequences shorter than this are discarded)")

final = [c for c in compressed if len(c) >= T_TRAJ]
print(f"Sequences kept  : {len(final)} / {len(compressed)}")
if len(final) == 0:
    raise ValueError(f"No sequences of length >= {T_TRAJ}. Lower t_traj.")

seqs_arr = np.array([c[:T_TRAJ] for c in final], dtype=np.int64)
print(f"Saved shape     : {seqs_arr.shape}")

counts = np.bincount(seqs_arr.ravel(), minlength=C)
print(f"\nBehavior counts (compressed): { {BEHAVIOR_NAMES[i]: int(counts[i]) for i in range(C)} }")
print(f"Behavior fracs              : { {BEHAVIOR_NAMES[i]: round(counts[i]/counts.sum(), 3) for i in range(C)} }")

np.save(output_dir / 'compressed_seqs2.npy', seqs_arr)
print(f"\nSaved compressed_seqs2.npy  {seqs_arr.shape}")

# ── transition probabilities (same deterministic structure) ───────────────────
trans_probs = np.zeros((C, C, C))
for a in range(C):
    for s in range(C):
        trans_probs[s, a, a] = 1.0

np.save(output_dir / 'compressed_trans_probs2.npy', trans_probs)
print(f"Saved compressed_trans_probs2.npy  {trans_probs.shape}")

# ── ARHMM initialisation parameters ──────────────────────────────────────────
npr.seed(seed)

logpi0_start = np.log(np.ones(K) / K)

Ps           = 0.95 * np.eye(K) + 0.05 * npr.rand(K, K)
Ps          /= Ps.sum(axis=1, keepdims=True)
log_Ps_start = np.log(Ps)

Rs_start     = np.zeros((C, 1, K))
init_start   = np.ones(K) / K

out_path = output_dir / f'{K}_{seed}_arhmm_caltech_compressed.npz'
np.savez(out_path,
         init_start   = init_start,
         logpi0_start = logpi0_start,
         log_Ps_start = log_Ps_start,
         Rs_start     = Rs_start)

print(f"\nSaved {out_path.name}")
print(f"  init_start  : {init_start.shape}")
print(f"  logpi0_start: {logpi0_start.shape}")
print(f"  log_Ps_start: {log_Ps_start.shape}")
print(f"  Rs_start    : {Rs_start.shape}")
print("\nDone.")
