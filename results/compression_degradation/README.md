# Compression Degradation Experiment — KB Pipeline

**Date:** 2026-08-17
**Scripts:** `scripts/compression_degradation.py` (data collection), `scripts/analyze_compression.py` (metrics), `scripts/plot_compression.py` (plots)
**Dataset:** LAV-DF test split, first 200 videos (sorted by path) — identical subset to the ablation study (n=200) for direct comparability
**Models:** FER ResNet50 (68.46%) + FER VGG19 (63.78%), each paired with SER MLP (~65.6%)

## Experiment Design

| Condition family | Levels | Method |
|---|---|---|
| `clean` | — | Original LAV-DF video |
| `jpeg_{Q}` | Q = 10,20,30,40,50,60,70,80,90,95 | Full frames JPEG-encoded/decoded in-memory (cv2), **then** face detection + FER |
| `h264_{CRF}_keepaudio` | CRF = 18,23,28,33,38,43 | ffmpeg libx264 re-encode, original audio stream copied (`-c:a copy`) |
| `h264_{CRF}_keepaudio` (OSN) | same CRFs | ffmpeg libx264 re-encode **+ audio re-encoded AAC 96k** (full OSN pipeline simulation) |

Note: OSN runs reuse the `_keepaudio` condition keys in JSON (audio differs, keys identical by design).

## Files

| File | Contents |
|---|---|
| `results_{model}_{keepaudio\|osn}.json` | Full per-video, per-condition results incl. VT/AT/IM probs, predictions, n_faces |
| `results_{model}_{keepaudio\|osn}.npz` | Per-face FER prob arrays + per-segment SER prob arrays (for entropy/agreement analysis) |
| `results_{model}_{keepaudio\|osn}.csv` | One row per (video, condition): prediction, GT, VT/AT/IM, n_faces |
| `long_{model}_{keepaudio\|osn}.csv` | Enriched: + FER/SER confidence & entropy per video-condition |
| `per_frame_agreement.json` | FER per-frame argmax agreement vs clean, per condition |
| `plots/*.png` | Degradation curves |

## Headline Results (n=200)

### 1. Video-level KB predictions never change under ANY compression
- Prediction flips vs clean: **0 of 200 for every JPEG Q and every CRF** (both models, both audio modes)
- Accuracy is a constant **72.0% = always-fake baseline** at every condition
- Cause: the KB voting classifier output saturates (always "fake") on LAV-DF; aggregate accuracy is uninformative

### 2. But FER per-frame emotion labels degrade substantially (ResNet50)

| Condition | FER per-frame agreement vs clean |
|---|---|
| clean | 1.000 |
| JPEG Q95 | 0.844 |
| JPEG Q50 | 0.774 |
| JPEG Q10 | **0.530** |
| H.264 CRF18 | 0.790 |
| H.264 CRF43 | **0.351** |

### 3. VGG19 FER is MORE compression-robust at frame level
| Condition | VGG19 agreement | ResNet50 agreement |
|---|---|---|
| JPEG Q10 | 0.666 | 0.530 |
| H.264 CRF18 | 0.853 | 0.790 |
| H.264 CRF43 | 0.468 | 0.351 |

Reason: VGG19 operates on 48×48 grayscale crops — downsampling pre-filters most compression artifacts before the model sees them.

### 4. SER / AT signals are compression-invariant here
- SER per-frame agreement = 1.000 at every condition
- **LAV-DF audio is already AAC-compressed (~73 kbps)**; re-encoding AAC→AAC 96k is near-lossless (feature corr 0.999998). SER robustness to AAC is thus expected, not evidence of robustness — the source audio was never high-quality.

### 5. KB fake probabilities drift modestly
- VT: clean→CRF43 Cohen's d = **+0.85 (ResNet50)**, +0.47 (VGG19) — meaningful shift
- JPEG Q10: IM d = +0.42 (ResNet50)
- AT: d = 0.00 everywhere (audio unchanged)

## Interpretation

- **The KB's decision layer is too degenerate (always-fake) to reveal compression sensitivity** — video-level metrics are useless for this study.
- **The underlying semantic features (FER emotions) ARE compression-sensitive**: H.264 CRF43 destroys ~65% of frame-level emotion labels (ResNet50), JPEG Q10 destroys ~47%.
- **SER is invariant to AAC re-encode only because LAV-DF audio is pre-compressed** — testing audio robustness on LAV-DF requires synthetic uncompressed sources or higher-bitrate originals.
- **VGG19's robustness is an artifact of input resolution**, not of the emotion signal itself.

## Reproducibility Commands

```powershell
# Pass 1: original audio preserved
python -u scripts/compression_degradation.py --limit 200 --workers 4 --collect_probs

# Pass 2: full OSN (AAC audio re-encode)
python -u scripts/compression_degradation.py --limit 200 --workers 4 --collect_probs --use_full_osn

# Analysis + plots
python -u scripts/analyze_compression.py
python -u scripts/plot_compression.py
```
