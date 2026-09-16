# AGENTS.md — Multimodal Neurosymbolic Explainable Deepfake Detection

## Paper
- **Title:** "Multimodal Neurosymbolic Approach for Explainable Deepfake Detection"
- **Authors:** Ijaz Ul Haq, Khalid Mahmood Malik, Khan Muhammad
- **Venue:** ACM TOMCCAP, 20(11), Nov 2024
- **DOI:** `10.1145/3624748`
- **Path:** `3624748.pdf` in repo root

## Project goal
Implement the paper faithfully, then use as benchmark for journal research publication.

## No official code or models
Authors (SMILES Lab, Oakland / U. Michigan-Flint) never released implementation or weights. This is a from-scratch reimplementation.

## Framework architecture
1. **Preprocessing:** MTCNN face detection → 48×48 grayscale; audio → 0.5s segments @16kHz with 0.25s overlap; 3 frames + 3 segments per second.
2. **FER:** CNN (4 conv+pool blocks → 1024 FC → 7-way softmax). Trained on FER2013 (50 epochs, lr=0.001, batch=128).
3. **SER:** MLP (180-dim MFCC+Chroma+MEL → 300×2 hidden → 7-way softmax). Trained on RAVDESS (500 epochs, batch=256).
4. **Intra-modality reasoning:** KB transition probability table (Table 2 in paper). Threshold = row mean. If prob(current→next) ≤ threshold → fake.
5. **Inter-modality reasoning:** Arousal-valence quadrants (Fig 2). If vis & aur emotions in different quadrants → fake.
6. **Classification:** Per-modality fake ratio > 15% → modality=fake. Majority vote (VT, AT, IM) → final verdict.
7. **Explanation:** Timestamps + fake transition indices + modality probabilities.

## Implementation
- Language: Python 3.11, TensorFlow 2.21 / Keras 3
- Virtual env: `venv\` at repo root
- Training scripts: `scripts/train_fer.py`, `scripts/train_ser.py`
- Inference: `python -m src.pipeline <video>`
- Evaluation: `python scripts/evaluate.py --dataset <dir> --gt <json>`
- Datasets needed: FER2013 (Kaggle), RAVDESS (Zenodo), PDD/WLD (from authors)
- Key threshold (paper): 15% for per-modality fake classification
- Voting: 2-of-3 majority (VT, AT, IM)

## Expected results (from paper)
- PDD: 93.7% (voting), 87.5% (visual intra only), 84.3% (aural intra only), 87.5% (inter only)
- WLD: 75.34%

## Dependencies
`requirements.txt` — install with `venv\Scripts\pip install -r requirements.txt`

## Notes
- Paper used TF 2.0.5 / Keras 2.0.0 / Python 3.7; current code uses TF 2.21 / Keras 3.
- FER model is a custom CNN (not ResNet50 despite paper naming — paper's ref [26] is WuJie1010's PyTorch repo which uses a 4-block CNN).
- SER feature dim = 40 (MFCC) + 12 (Chroma) + 128 (MEL) = 180.
- No deepfake training required — only emotion models need training. The reasoning module is rule-based.
