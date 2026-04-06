Update on CalMS21 S-2 analysis (K = 2 only):

1. In swirl/:
Added new interpretability analysis scripts:
- social_drive_profiles.py → Method 1 (latent social drive profiles)
- escalation_potential.py → Method 2 (escalation potential)
- persistence_commitment.py → Method 3 (persistence / commitment)

2. In data/:
- Added task2_classic_classification/, which corresponds to CalMS21 Task 2
- Same behavior categories as Task 1, but includes 6 annotators (annotation-style variation)

3. New Task 2 processed / model files:
- All ARHMM outputs ending with “2” (e.g. *_caltech2.npz, *_caltech_compressed2.npz) are for Task 2
- compressed_seqs2.npy and compressed_trans_probs2.npy are also Task 2 versions

4. Results:
- Any new files in results/ correspond to the new analyses (Methods 1–3) on Task 2

5. Notes:
- No modification to existing Task 1 pipeline
- All analysis currently run only for K = 2
