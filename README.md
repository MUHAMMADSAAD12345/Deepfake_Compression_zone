# Deepfake Detection Research Workspace

This repository contains the **original paper reproduction** project and shared infrastructure. The CASR project lives in `_casr/` (separate git repo).

## 📁 Repository Structure

```
Saad_audi deepfakes/
├── src/                      # Shared library (both projects)
│   ├── models/
│   │   ├── fer_model.py      # FER CNN (VGG19/ResNet50)
│   │   └── ser_model.py      # SER MLP
│   ├── utils/
│   │   └── preprocessing.py  # Face detection, audio extraction
│   └── pipeline.py           # Inference pipeline
│
├── scripts/                  # Training/eval scripts (original paper)
│   ├── train_fer.py
│   ├── train_ser.py
│   └── evaluate.py
│
├── data/                     # Raw datasets (gitignored)
│   ├── LAV-DF/
│   └── RAVDESS/
│
├── checkpoints/              # Shared weights (gitignored)
│   ├── fer_resnet50_best.pt
│   ├── fer2013_vgg19.pt
│   ├── ser_weights.pt
│   └── cnn_baseline_best.pt
│
├── _casr/                    # CASR project (SEPARATE GIT REPO)
│   ├── step1_estimator.py    # Compression-state estimator
│   ├── step2_*.py            # LAV-DF & FF++ corpus generation + eval
│   ├── step3_ffpp.py         # CASR on FF++
│   ├── step4_audio_sweep.py  # Audio provenance
│   ├── step5_*.py            # Baseline comparisons
│   ├── step6_semantic.py     # TRUE FER semantic branch
│   ├── train_ffpp_cnn.py     # Compression-augmented CNN training
│   ├── MANUSCRIPT.md         # CASR paper draft
│   ├── README.md             # CASR project documentation
│   ├── requirements.txt
│   └── .gitignore
│
├── data/                     # Raw datasets (gitignored)
│   ├── LAV-DF/
│   └── RAVDESS/
│
├── checkpoints/              # Shared weights (gitignored)
│   ├── fer_resnet50_best.pt
│   ├── fer2013_vgg19.pt
│   ├── ser_weights.pt
│   └── cnn_baseline_best.pt
│
├── AGENTS.md                 # Original paper reproduction guide
├── paper_text.txt            # Paper full text
├── 3624748.pdf               # Original paper (ACM TOMM 2024)
└── requirements.txt          # Shared dependencies
```

## 🎯 Project: Original Paper Reproduction

**Paper**: "Multimodal Neurosymbolic Approach for Explainable Deepfake Detection"  
**Venue**: ACM TOMM 2024, DOI: 10.1145/3624748  
**Authors**: Ijaz Ul Haq, Khalid Mahmood Malik, Khan Muhammad  

### Approach
1. **FER**: CNN (4 conv+pool blocks → 1024 FC → 7-way softmax) on FER2013
2. **SER**: MLP (180-dim MFCC+Chroma+MEL → 300×2 hidden → 7-way softmax) on RAVDESS
3. **Intra-modality reasoning**: KB transition probability table (psychological study)
4. **Inter-modality reasoning**: Arousal-valence quadrants
5. **Classification**: Per-modality fake ratio > 15% → fake; majority vote (VT, AT, IM)
6. **Explanation**: Timestamps + fake transition indices + modality probabilities

### Expected Results (from paper)
| Dataset | Accuracy |
|---|---|
| PDD (voting) | 93.7% |
| PDD (visual intra) | 87.5% |
| PDD (aural intra) | 84.3% |
| WLD | 75.34% |

### Key Features
- **Zero deepfake training data** — only emotion models trained
- **Explainable**: Textual explanations with timestamps
- **Generalizable**: No deepfake training data required

## 🚀 Quick Start

### Dependencies
```bash
pip install -r requirements.txt
# PyTorch with CUDA: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Train Emotion Models
```bash
cd scripts
python train_fer.py    # Train FER on FER2013 (50 epochs, lr=0.001, batch=128)
python train_ser.py    # Train SER on RAVDESS (500 epochs, batch=256)
```

### Evaluate
```bash
python evaluate.py --dataset ../data/PDD --gt gt.json
```

## 📦 Datasets (not in repo)

| Dataset | Expected Location | Size |
|---|---|---|
| FER2013 | Kaggle (msambare/fer2013) | 35k images |
| RAVDESS | `data/RAVDESS/` | ~1 GB |
| LAV-DF | `data/LAV-DF/` | ~10 GB |
| PDD / WLD | From authors | ~500 MB |

## 📄 Documentation

| File | Description |
|---|---|
| `AGENTS.md` | Reproduction guide & architecture details |
| `paper_text.txt` | Paper full text |
| `3624748.pdf` | Original paper PDF (ACM TOMM 2024) |
| `_casr/README.md` | CASR project documentation |
| `_casr/MANUSCRIPT.md` | CASR paper draft |

## 🔬 CASR Project (Separate)

The **CASR (Compression-Aware Semantic Reliability)** project is in `_casr/` — a separate git repo with its own `README.md` and `MANUSCRIPT.md`.

**CASR Key Idea**: Gate CNN branch with clean semantic branch using learned compression-state estimator.

| CASR Result | FF++ Test |
|---|---|
| CNN (compress-aug) @ h264_crf38 | 0.7077 |
| **CASR (gate + clean proxy)** | **0.7755** (+0.068) |
| hevc_crf38 | 0.8112 (+0.170) |

See `_casr/README.md` and `_casr/MANUSCRIPT.md` for full CASR documentation.

## 📦 Datasets (not in repo)

| Dataset | Expected Location | Size |
|---|---|---|
| FER2013 | Kaggle (msambare/fer2013) | 35k images |
| RAVDESS | `data/RAVDESS/` | ~1 GB |
| LAV-DF | `data/LAV-DF/` | ~10 GB |
| Celeb-DF v2 | `E:\Datasets\Celeb-DF-v2\` | 9.3 GB |

## 📄 Key Files

| File | Description |
|---|---|
| `AGENTS.md` | Reproduction guide & architecture |
| `3624748.pdf` | Original paper PDF |
| `_casr/README.md` | CASR project docs |
| `_casr/MANUSCRIPT.md` | CASR paper draft |

## 📄 License

MIT License — see LICENSE files in respective repos.