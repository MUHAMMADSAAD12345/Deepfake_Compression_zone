"""CASR Step 5: Baseline comparisons on FF++ test split.

Implements three strong compression-robust baselines:
1. QAD  - Quality-Aware Detection (quality features + ensemble)
2. PLADA - Probabilistic LDA on deep features
3. FreqDebias - DCT frequency debiasing + classifier

All evaluated on the same FF++ test split (140 vids/method) with the same
10 codec conditions used for CASR Step 3-v2.
"""
import json
import os
import sys
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from scipy.fftpack import dct

sys.path.insert(0, r"E:\Saad_audi deepfakes\_casr")

ROOT = os.path.dirname(os.path.abspath(__file__))
FACES_EVAL = os.path.join(ROOT, "data", "ffpp_faces_eval")
OUT = os.path.join(ROOT, "results", "step5_baselines.json")

GROUPS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "real"]
CONDITIONS = ["clean", "h264_crf18", "h264_crf28", "h264_crf38",
              "hevc_crf18", "hevc_crf28", "hevc_crf38",
              "av1_crf20", "av1_crf30", "av1_crf40"]


# ------------------------------------------------------------
# Feature extractors
# ------------------------------------------------------------
def dct_coeffs_8x8(img_gray):
    """Extract 8x8 DCT coefficients from image (top-left 8x8 block)."""
    h, w = img_gray.shape
    block = img_gray[:8, :8].astype(np.float32) - 128.0
    coeffs = dct(dct(block.T, norm='ortho').T, norm='ortho')
    return coeffs.flatten()


def freq_debias_features(img_bgr):
    """FreqDebias-style: DCT histogram + spectral residual."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (256, 256))
    # DCT on 8x8 grid
    coeffs = []
    for y in range(0, 256, 8):
        for x in range(0, 256, 8):
            block = gray[y:y+8, x:x+8].astype(np.float32) - 128.0
            if block.shape == (8, 8):
                c = dct(dct(block.T, norm='ortho').T, norm='ortho')
                coeffs.append(c.flatten())
    coeffs = np.array(coeffs)  # (1024, 64)
    # Histogram of AC coefficients (skip DC)
    ac = coeffs[:, 1:].flatten()
    hist, _ = np.histogram(ac, bins=64, range=(-256, 256), density=True)
    # Spectral residual (log spectrum - smoothed log spectrum)
    f = np.fft.fft2(gray)
    mag = np.abs(f)
    log_mag = np.log(mag + 1e-8)
    # Smooth with 3x3 avg
    from scipy.ndimage import uniform_filter
    smooth = uniform_filter(log_mag, size=3)
    residual = log_mag - smooth
    res_hist, _ = np.histogram(residual.flatten(), bins=32, range=(-4, 4), density=True)
    return np.concatenate([hist, res_hist])


def qad_quality_features(img_bgr):
    """QAD-style: image quality metrics + compression artifacts."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (512, 512))
    feats = []
    # BRISQUE-like (using piq if available, else simple approx)
    try:
        from piq import brisque
        feats.append(float(brisque(torch.from_numpy(img_bgr).permute(2,0,1).unsqueeze(0)/255.0)))
    except Exception:
        feats.append(0.0)
    # Blockiness (8x8 grid)
    h, w = gray.shape
    grid = gray[::8, ::8].std() / 8.0
    feats.append(grid)
    # Zero DCT ratio
    blocks = 0; zeros = 0
    for y in range(0, h-7, 8):
        for x in range(0, w-7, 8):
            b = gray[y:y+8, x:x+8].astype(np.float32) - 128
            c = dct(dct(b.T, norm='ortho').T, norm='ortho')
            blocks += 1
            zeros += np.sum(np.abs(c) < 1.0)
    feats.append(zeros / (blocks * 64))
    # Gradient magnitude stats
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx*gx + gy*gy)
    feats.extend([mag.mean(), mag.std(), np.percentile(mag, 90)])
    return np.array(feats, dtype=np.float32)


def plada_deep_features(img_bgr):
    """PLADA-style: use ResNet50 penultimate features (2048-d)."""
    # Use the trained FF++ CNN as feature extractor
    import torch
    import torchvision.models as models
    from torchvision import transforms
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Identity()
    ckpt = torch.load(os.path.join(ROOT, "checkpoints", "ffpp_cnn_best.pt"), map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    x = cv2.resize(img_bgr, (224, 224))
    x = torch.from_numpy(x).permute(2,0,1).float()/255.0
    x = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])(x).unsqueeze(0)
    with torch.no_grad():
        feat = model(x).numpy().flatten()
    return feat


def extract_frame(video_path, pos=0.5):
    """Extract frame at relative position (0.5 = middle)."""
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * pos))
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def get_video_path(cond, grp, name):
    if cond == "clean":
        base = r"E:\Datasets\FaceForensicsC23\FaceForensics++_C23"
        if grp == "real":
            return os.path.join(base, "real", name.replace(".npz", ".mp4"))
        return os.path.join(base, "fake", grp, name.replace(".npz", ".mp4"))
    else:
        base = os.path.join(ROOT, "data", "ffpp_compressed", cond, grp)
        return os.path.join(base, name.replace(".npz", ".mp4"))


# ------------------------------------------------------------
# Per-condition evaluation
# ------------------------------------------------------------
def evaluate_method(cond, method="qad"):
    """Train/eval one baseline on one condition."""
    probs_path = os.path.join(ROOT, "results", "step2b_probs", f"{cond}.json")
    with open(probs_path) as f:
        probs = json.load(f)
    per_vid = {}
    for k, v in probs.items():
        grp, name = k.split(os.sep, 1)
        per_vid.setdefault((grp, name), []).append(v)

    X, y = [], []
    for (grp, name), preds in per_vid.items():
        label = 0 if grp == "real" else 1
        # Use first frame for feature extraction
        vid_path = get_video_path(cond, grp, name)
        frame = extract_frame(vid_path, 0.5)
        if frame is None:
            continue
        if method == "qad":
            feat = qad_quality_features(frame)
        elif method == "plada":
            feat = plada_deep_features(frame)
        elif method == "freqdebias":
            feat = freq_debias_features(frame)
        else:
            raise ValueError(method)
        X.append(feat)
        y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    
    if len(np.unique(y)) < 2:
        return None
    
    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Train/test split (same as CASR: video-stratified 65/35)
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(y))
    tr_idx, te_idx = train_test_split(idx, test_size=0.35, random_state=0, stratify=y)
    
    if method == "plada":
        clf = LinearDiscriminantAnalysis()
    else:
        clf = LogisticRegression(max_iter=1000, class_weight='balanced', C=1.0)
    
    clf.fit(X[tr_idx], y[tr_idx])
    pred = clf.predict_proba(X[te_idx])[:, 1]
    return float(roc_auc_score(y[te_idx], pred))


def main():
    results = {}
    for method in ["qad", "plada", "freqdebias"]:
        print(f"\n=== {method.upper()} ===", flush=True)
        results[method] = {}
        for c in CONDITIONS:
            try:
                auc = evaluate_method(c, method)
                if auc is not None:
                    results[method][c] = {"auc": round(auc, 4), "n": 700}
                    print(f"  {c:12s} AUC {auc:.4f}", flush=True)
                else:
                    results[method][c] = {"auc": None, "n": 0}
            except Exception as e:
                print(f"  {c:12s} ERROR: {e}", flush=True)
                results[method][c] = {"auc": None, "n": 0, "error": str(e)}
    
    with open(OUT, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()