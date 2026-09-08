# CASR Progress Log — Session 2026-08-19

Workspace: `E:\Saad_audi deepfakes\_casr\` · venv: `E:\Saad_audi deepfakes\venv`
GPU: RTX 4000 Ada 20GB · ffmpeg 8.1.2 (libx264/libx265/libsvtav1/aac/opus/mp3)

## Completed this session

### 0. Datasets
- **FF++ c23 acquired** (7,000 videos, 17.9GB): `E:\Datasets\FaceForensicsC23\FaceForensics++_C23\`
  - `fake/{DeepFakeDetection,Deepfakes,Face2Face,FaceShifter,FaceSwap,NeuralTextures}` (6,000)
  - `real/` (1,000) — mirrors bitmind/FaceForensicsC23
- **DFDC Preview: BLOCKED** — official route needs AWS account + IAM keys (ai.meta.com/datasets/dfdc);
  Kaggle mirror needs API credentials. User decision needed.

### 1. Step 1 — Compression-state estimation (VALIDATED)
`step1_estimator.py` → `results/step1/` · 60 videos, 583 rows
- Features: blockiness (ρ=0.67), zero_dct_ratio (ρ=0.68), grid_std, brisque(ρ=0.35)
- Per-codec Ridge: JPEG Q ρ=0.94/R²=0.62 · H.264 CRF ρ=0.74/R²=0.30
- **Gate (heavy CRF≥33 vs light ≤23): CV AUC 0.93 ± 0.04** → `models/gate_logreg.npz`
- Key: DCT-deterministic features beat learned NIQE/BRISQUE on CG content.
- `casr_estimator.py` = reusable reliability-branch module. → STEP1_FINDINGS.md

### 2. Step 2 — Multi-codec corpus + CNN baseline
`step2_harness.py` → 5,400/5,400 encodes OK · `step2_eval.py` (ResNet50 WildDeepfake ckpt)
- Bitrate stats (CRF not codec-equivalent!): AV1 CRF40≈48kbps ≈ H.264 CRF28≈46kbps;
  H.264 CRF38≈18kbps. (paper figure)
- **CNN AUC on LAV-DF: ~0.51 on ALL conditions (clean 0.513)** — detector is at chance on
  LAV-DF; domain shift dominates. AUC-drop curves flat because baseline floor = chance.
  → Codec-transfer measurement must move to FF++ (in-domain) — TODO Step 2b-v2.

### 3. Step 3 — CASR v1 prototype (WORKING, positive)
`step3_casr.py` → `results/step3_auc.json` (200-video FER corpus, honest video-stratified clean split)

| condition | cnn | sem | fixed | gated | oracle |
|---|---|---|---|---|---|
| clean | 0.645 | 0.505 | 0.640 | 0.643 | 0.645 |
| h264_crf18 | 0.514 | 0.592 | 0.520 | 0.521 | 0.592 |
| h264_crf28 | 0.553 | 0.516 | 0.558 | 0.570 | 0.566 |
| h264_crf38 | 0.572 | 0.533 | 0.576 | **0.605** | 0.598 |

- Gated fusion **beats fixed fusion and even the per-condition oracle at CRF28/38** (per-video
  adaptive weights). Gate heavy-prob rises monotonically with compression (0.56→0.63).
- Semantic branch is weak but complementary; trained on clean, transfers to compression.
- Surprising: sem AUC higher at CRF18 (0.59) than clean (0.51) — compression spreads FER
  confidence, restoring feature variance. Paper-worthy observation.

### 4. Step 4 — Audio compression sweep (provenance artifact QUANTIFIED)
`step4_audio_sweep.py` → `results/step4/` · 80 RAVDESS + 30 LAV-DF
- RAVDESS (true clean): SER decision agreement 0.75–0.91 (AAC32k–128k) — audio IS sensitive
- LAV-DF (source already AAC≈73kbps): AAC re-encode agreement **1.000 at ALL bitrates**;
  MP3 re-encode collapse to 0.467 — codec-transfer failure on pre-compressed source
- Confirms: "SER invariant to AAC" in the earlier KB experiment was a provenance artifact.

## Open items / next

1. **Step 2b-v2: FF++ codec-transfer matrix** — train CNN on FF++ c23 (real vs Deepfakes),
   re-encode FF++ sample (H.264/HEVC/AV1 CRF sweep), produce AUC-drop curves + transfer
   matrix. This becomes the paper's headline measurement (in-domain, community standard).
2. **Step 3-v2**: strengthen semantic branch (add SER features, richer FER time-stats),
   and evaluate on FF++ (need FER on FF++ frames or reuse emotion model + LAV-DF 200).
3. **DFDC**: user decision (AWS account / Kaggle creds / skip).
4. **Step 5**: QAD/PLADA/FreqDebias reimplementation for baselines; manuscript.

## Scripts (all in _casr\)
- step1_estimator.py (features + analysis, --analyze-only)
- train_gate.py (saves models/gate_logreg.npz)
- casr_estimator.py (reusable gate features + gate_score)
- step2_harness.py (corpus gen), step2_eval.py (CNN eval + per-video probs)
- step3_casr.py (CASR v1 fusion experiment)
- step4_audio_sweep.py (audio codec sweep)
