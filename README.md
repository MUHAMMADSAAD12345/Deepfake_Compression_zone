# CASR: Compression-Aware Semantic Reliability for Robust Deepfake Detection

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**CASR (Compression-Aware Semantic Reliability)** is a deepfake detection framework that dynamically gates a compression-robust semantic branch against a high-capacity CNN branch using a learned compression-state estimator. The estimator predicts heavy-compression probability from hand-crafted features (blockiness, zero-DCT ratio, grid standard deviation, BRISQUE) and computes a mixing weight `w_sem = 1 - P(heavy)`.

## 🎯 Key Results (FF++ c23 Test Split)

| Condition | CNN (compress-aug) | **CASR (gate + clean proxy)** | Δ AUC |
|---|---|---|---|
| clean | 0.9602 | 0.9602 | — |
| h264_crf28 | 0.8897 | **0.9174** | +0.028 |
| **h264_crf38** | **0.7077** | **0.7755** | **+0.068** |
| hevc_crf38 | 0.6410 | **0.8112** | **+0.170** |
| av1_crf40 | 0.8827 | 0.9207 | +0.038 |

*All CASR gains at heavy compression (CRF38/40) are statistically significant (bootstrap 1000 resamples, p≈1.0).*

## 🏗️ Architecture

```
Video → Compression Features → Gate (Logistic Regression) → w_sem = 1 - P(heavy)
          ↓                                              ↓
    CNN Branch (ResNet50)                          Semantic Branch
    p_cnn                                          p_sem (clean proxy)
          ↓                                              ↓
          └──────────────→ p_casr = w_sem·p_sem + (1-w_sem)·p_cnn
```

## 📁 Project Structure

```
_casr/
├── step1_estimator.py        # Compression-state estimator (gate features)
├── train_gate.py             # Gate logistic regression training
├── casr_estimator.py         # Reusable gate features + gate_score()
├── step2_harness.py          # LAV-DF corpus generation
├── step2_eval.py             # CNN eval on LAV-DF
├── step2b_ffpp_harness.py    # FF++ multi-codec corpus (6,300 encodes)
├── step2b_eval.py            # FF++ CNN eval
├── step3_ffpp.py             # CASR on FF++ test split
├── step4_audio_sweep.py      # Audio provenance analysis
├── step5_baselines.py        # QAD/PLADA/FreqDebias baselines
├── step5_deepdive.py         # Fixed/learned fusion baselines
├── step6_semantic.py         # TRUE FER semantic branch
├── train_ffpp_cnn.py         # FF++ CNN training (compression-aug)
├── MANUSCRIPT.md             # Paper draft
├── .gitignore
├── results/                  # All JSON results
├── checkpoints/              # Model weights
├── models/                   # Gate model
├── data/                     # Face/FER caches
└── requirements.txt
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- PyTorch 2.0+ with CUDA
- FFmpeg (with libx264, libx265, libsvtav1, libmp3lame, libopus)
- 20GB+ GPU memory (RTX 4000 Ada recommended)

### Installation
```bash
cd _casr
pip install -r requirements.txt
# Or manually: pip install torch torchvision torchaudio scikit-learn numpy opencv-python librosa piq tqdm
```

### Data Preparation
Datasets are **not included** (excluded by `.gitignore`). You need:

1. **FF++ c23** → `E:\Datasets\FaceForensicsC23\FaceForensics++_C23\`
   - Download from HuggingFace: `bitmind/FaceForensicsC23`
   - Structure: `fake/{Deepfakes,Face2Face,FaceSwap,NeuralTextures}/` + `real/`

2. **LAV-DF** → `E:\Saad_audi deepfakes\data\LAV-DF\test\` + `lavdf_ground_truth.json`

3. **RAVDESS** → `E:\Saad_audi deepfakes\data\RAVDESS\`

4. **Celeb-DF v2** (optional) → `E:\Downloads\Celeb-DF-v2.zip` (9.3 GB)

### Running the Pipeline

#### Step 1: Compression-State Estimator (Gate)
```bash
python step1_estimator.py          # Analyze compression features
python train_gate.py               # Train logistic regression gate
# Output: models/gate_logreg.npz (AUC 0.93 on H.264 heavy vs light)
```

#### Step 2: LAV-DF Corpus + CNN Baseline
```bash
# Generate multi-codec corpus (background task)
schtasks /Create /TN "CASR_STEP2_GEN" /TR "step2_gen.cmd" /SC ONCE /ST 23:59 /F
schtasks /Run /TN "CASR_STEP2_GEN"

# Evaluate CNN on corpus
python step2_eval.py
```

#### Step 2b: FF++ Corpus + CNN
```bash
# Generate FF++ multi-codec test corpus (6,300 encodes)
python step2b_ffpp_harness.py --methods Deepfakes Face2Face FaceSwap NeuralTextures --include_real --split test --workers 8

# Train CNN on FF++ c23 (compression-augmented)
python train_ffpp_cnn.py --compress-aug --epochs 20 --batch 128

# Evaluate on all 10 codec conditions
python train_ffpp_cnn.py --eval clean h264_crf18 h264_crf28 h264_crf38 hevc_crf18 hevc_crf28 hevc_crf38 av1_crf20 av1_crf30 av1_crf40 --ckpt checkpoints/ffpp_cnn_best.pt
```

#### Step 3: CASR Evaluation on FF++
```bash
python step3_ffpp.py
# Output: results/step3_ffpp.json (CASR AUC per condition)
```

#### Step 4: Audio Provenance (Optional)
```bash
python step4_audio_sweep.py
```

#### Step 5/6: Baselines & True Semantic Branch
```bash
python step5_deepdive.py      # Fixed/learned fusion baselines
python step6_semantic.py --extract-train --extract-eval
python step6_semantic.py --run  # TRUE FER semantic branch
```

### Background Jobs (Windows)
All long-running tasks use `schtasks` for background execution:
```bash
# Example: Create and run background task
schtasks /Create /TN "CASR_STEP2B_GEN" /TR "step2b_gen.cmd" /SC ONCE /ST 23:59 /F
schtasks /Run /TN "CASR_STEP2B_GEN"

# Check status
schtasks /Query /TN "CASR_STEP2B_GEN" /FO LIST | findstr Status
```

## 📊 Key Outputs

| File | Description |
|---|---|
| `results/step2b_eval.json` | Codec-transfer matrix (CNN AUC per condition) |
| `results/step2b_probs/` | Per-video CNN probabilities per condition |
| `results/step3_ffpp.json` | CASR AUC per condition (gate + clean proxy) |
| `results/step3_auc.json` | LAV-DF CASR results (original) |
| `results/step5_deepdive.json` | Fixed/learned fusion baselines |
| `results/step6_semantic.json` | TRUE FER semantic branch results |
| `checkpoints/ffpp_cnn_best.pt` | Best compression-augmented CNN |
| `models/gate_logreg.npz` | Compression-state gate (logistic regression) |

## 🔬 Reproducing Paper Results

The manuscript (`MANUSCRIPT.md`) contains all tables and figures. To reproduce:

1. **Table 1 (Codec-Transfer Matrix)** → `results/step2b_eval.json`
2. **Table 2 (CASR vs CNN)** → `results/step3_ffpp.json`
3. **Table 3 (Baselines)** → `results/step5_deepdive.json`
4. **Table 4 (True Semantic Branch)** → `results/step6_semantic.json`
5. **Bootstrap CIs** → Run `python bootstrap_final.py`

## ⚙️ Configuration

Key paths are defined at the top of each script:
```python
FFPP_ROOT = r"E:\Datasets\FaceForensicsC23\FaceForensics++_C23"
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(ROOT, "data", "ffpp_compressed")
```
Modify these if your data lives elsewhere.

## 📝 Citation

If you use CASR in your research, please cite:
```bibtex
@article{casr2024,
  title={CASR: Compression-Aware Semantic Reliability for Robust Deepfake Detection},
  author={...},
  journal={...},
  year={2024}
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 📧 Contact

For questions or collaborations, please open an issue or email: ...

## 🙏 Acknowledgments

- FF++ dataset: `bitmind/FaceForensicsC23` on HuggingFace
- LAV-DF, RAVDESS, Celeb-DF datasets
- FER2013 for emotion recognition
- Psychological knowledge base from [Ren, 2009]