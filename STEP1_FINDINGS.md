# CASR Step 1 — Compression-State Estimation: Findings

**Date:** 2026-08-19 · **Run:** 60 LAV-DF test videos, 583 (video, condition) rows
**Conditions:** clean, H.264 CRF 18/23/28/33/38/43, JPEG Q 30/60/90 · 1 frame @ 50% duration, 512×512 central crop (native frames are 224×224)

## Results (step1_report.json)

| Feature | Spearman vs all | Spearman vs compressed-only |
|---|---|---|
| zero_dct_ratio (8×8 DCT-II matmul) | **0.68** | **0.66** |
| blockiness (Wang-style) | **0.67** | 0.57 |
| brisque (piq) | 0.35 | 0.35 |
| grid_std | −0.28 | −0.22 |

| Regression (log1p features, Ridge) | ρ(pred,true) | R² |
|---|---|---|
| H.264 CRF (n=360) | **0.74** | 0.30 |
| JPEG Q (n=163) | **0.94** | 0.62 |
| Mixed-scale (all) | 0.54 | 0.25 |

| Decision task | Metric |
|---|---|
| Heavy (CRF≥33) vs light (CRF≤23) H.264 — single frame | **AUC 0.93** |
| 3-class ordinal gate light/mid/heavy | 63.9% acc (33% chance) |

## Conclusions

1. **Premise validated**: a single central frame cheaply predicts whether
   content is so compressed that semantic features are unreliable (AUC 0.93
   for the binary gate). CASR's gating mechanism is feasible without codec
   metadata.
2. **Deterministic DCT features beat learned quality metrics** on CG face
   content: zero_dct_ratio and blockiness are the strongest signals;
   BRISQUE (trained on natural images) is weak. Supports the
   "exploit determinism of compression" thesis.
3. **JPEG quality is near-linearly recoverable** (ρ=0.94) — relevant for the
   photo-side of the multimodal story; H.264 CRF estimation is coarser but
   adequate for a reliability gate.
4. Mixed-scale regression (CRF⊕Q in one target) is invalid; per-codec
   analysis is the honest protocol (kept in report for transparency).

## Next

- The estimator becomes the CASR **reliability branch**: (i) extract frame →
  (ii) 4 features → (iii) logistic gate → (iv) weight for semantic branch at
  fusion. Weights tuned in Step 3.
- Feature code lives in `step1_estimator.py` (`blockiness`, `zero_dct_ratio`,
  `grid_std`, `brisque_score`); refactor into `_casr/casr_estimator.py`
  module for reuse.