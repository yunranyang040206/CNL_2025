## CalMS21 S-2 Analysis Update (K = 2)

### 1. New Analysis Scripts (`swirl/`)
Added interpretability evaluation modules:
- `social_drive_profiles.py` → Method 1: latent social drive profiles  
- `escalation_potential.py` → Method 2: escalation potential  
- `persistence_commitment.py` → Method 3: persistence / commitment  

---

### 2. Task 2 Dataset (`data/`)
⚠️ Not included in the repo due to size limits.

- Dataset: `task2_annotation_styles.zip`  
- Download link: https://data.caltech.edu/records/s0vdx-0k302  

After downloading:
- Extract into `data/`
- Folder used: `task2_classic_classification/`

Details:
- Same behavior categories as Task 1  
- Includes **6 annotators** (used for annotation-style analysis)

---

### 3. Task 2 Processed / Model Files
All files with suffix `2` correspond to **Task 2**:

- ARHMM outputs:
  - `*_caltech2.npz`
  - `*_caltech_compressed2.npz`

- Processed data:
  - `compressed_seqs2.npy`
  - `compressed_trans_probs2.npy`

---

### 4. Results (`results/`)
- New files correspond to:
  - Methods 1–3 (interpretability analyses)
  - Task 2 dataset
  - K = 2 only

---

### 5. Notes
- No changes to existing Task 1 pipeline  
- All experiments currently restricted to **K = 2** for consistency and interpretability focus  
