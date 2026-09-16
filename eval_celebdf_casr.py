"""Celeb-DF v2 CASR evaluation."""
import os
import json
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample

CELEBDF_ROOT = r"E:\Datasets\Celeb-DF-v2"
ROOT = os.path.dirname(os.path.abspath(__file__))
FACES_CELEBDF = os.path.join(ROOT, "data", "celebdf_faces")
CKPT = os.path.join(ROOT, "checkpoints", "ffpp_cnn_best.pt")
OUT_JSON = os.path.join(ROOT, "results", "celebdf_casr.json")
GATE_CACHE = os.path.join(ROOT, "results", "celebdf_gate_feats.json")
os.makedirs(os.path.dirname(GATE_CACHE), exist_ok=True)

from casr_estimator import _load_gate, compression_features
from train_ffpp_cnn import MAX_FACES

from torchvision import models, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
import json
import glob
import cv2
import numpy as np


def load_model(ckpt_path, device="cuda"):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device).eval()
    return model


def get_probs(model, face_dir, label):
    """Get per-video max-frame CNN probabilities."""
    probs = []
    names = []
    device = "cuda"
    for f in sorted(glob.glob(os.path.join(FACES_CELEBDF, face_dir, "*.npz"))):
        a = np.load(f)["imgs"]
        if len(a) == 0:
            continue
        if len(a) > 30:
            idx = np.linspace(0, len(a)-1, 30, dtype=int)
            a = a[idx]
        x = torch.from_numpy(a).permute(0, 3, 1, 2).float() / 255.0
        x = transforms.Resize((224, 224))(x)
        x = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(x)
        x = x.to(device)
        with torch.no_grad():
            p = torch.sigmoid(model(x)).cpu().numpy()[:, 0]
        probs.append(p.max())
        names.append(os.path.splitext(os.path.basename(f))[0])
    return np.array(probs), np.array(names)


def get_gate_score(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return 0.5
        feats = compression_features(frame)
        vec = [float(feats[n]) for n in ['blockiness', 'zero_dct_ratio', 'grid_std', 'brisque']]
        z = float((np.dot(np.log1p(np.abs(vec)), coef) + intercept).item())
        return 1.0 / (1.0 + np.exp(-z))
    except Exception:
        return 0.5


def main():
    device = "cuda"
    model = load_model(CKPT, device)
    coef, intercept = _load_gate()
    
    # Load CNN probs
    print("Evaluating CNN on real...")
    p_real, names_real = get_probs(model, "real", 0)
    print("Evaluating CNN on fake...")
    p_fake, names_fake = get_probs(model, "fake", 1)
    
    y = np.concatenate([np.zeros(len(p_real)), np.ones(len(p_fake))])
    cnn_probs = np.concatenate([p_real, p_fake])
    names = list(names_real) + list(names_fake)
    cnn_auc = roc_auc_score(y, cnn_probs)
    print(f"CNN AUC: {cnn_auc:.4f}")
    
    # Gate scores
    print("Computing gate scores...")
    gate_cache = {}
    if os.path.exists(GATE_CACHE):
        with open(GATE_CACHE) as f:
            gate_cache = json.load(f)
    
    hp_list = []
    for name in names:
        if name in gate_cache:
            hp = gate_cache[name]
        else:
            found = False
            for d in ["Celeb-real", "YouTube-real", "Celeb-synthesis"]:
                p = os.path.join(CELEBDF_ROOT, d, name + ".mp4")
                if os.path.exists(p):
                    hp = get_gate_score(p)
                    found = True
                    break
            if not found:
                hp = 0.5
            gate_cache[name] = hp
        hp_list.append(hp)
    
    with open(GATE_CACHE, "w") as f:
        json.dump(gate_cache, f)
    
    hp_arr = np.array(hp_list)
    # CASR: when heavy (hp high), trust CNN less -> regress to chance (0.5)
    p_casr = (1 - hp_arr) * cnn_probs + hp_arr * 0.5
    
    casr_auc = roc_auc_score(y, p_casr)
    cnn_auc = roc_auc_score(y, cnn_probs)
    diff = casr_auc - cnn_auc
    
    print(f"CNN AUC:  {cnn_auc:.4f}")
    print(f"CASR AUC: {casr_auc:.4f}")
    print(f"Diff:     {diff:+.4f}")
    
    # Bootstrap CI for difference
    print("Bootstrapping...")
    diffs = []
    for _ in range(1000):
        y_b, cnn_b, casr_b = resample(y, cnn_probs, p_casr, random_state=None)
        if len(np.unique(y_b)) < 2:
            continue
        diffs.append(roc_auc_score(y_b, casr_b) - roc_auc_score(y_b, cnn_b))
    
    if diffs:
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        p_val = np.mean(np.array(diffs) > 0)
    else:
        lo, hi, p_val = 0, 0, 0
    
    print(f"Diff CI: [{lo:.4f}, {hi:.4f}], p={p_val:.4f}")
    
    results = {
        "cnn_auc": float(cnn_auc),
        "casr_auc": float(casr_auc),
        "diff": float(diff),
        "ci_lo": float(lo) if diffs else 0,
        "ci_hi": float(hi) if diffs else 0,
        "p_val": float(p_val) if diffs else 0,
        "n_real": int(len(p_real)),
        "n_fake": int(len(p_fake)),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {OUT_JSON}")


def load_model(ckpt_path, device="cuda"):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device).eval()
    return model


def get_probs(model, face_dir, label):
    """Get per-video max-frame CNN probabilities."""
    probs = []
    names = []
    device = "cuda"
    for f in sorted(glob.glob(os.path.join(FACES_CELEBDF, face_dir, "*.npz"))):
        a = np.load(f)["imgs"]
        if len(a) == 0:
            continue
        if len(a) > 30:
            idx = np.linspace(0, len(a)-1, 30, dtype=int)
            a = a[idx]
        x = torch.from_numpy(a).permute(0, 3, 1, 2).float() / 255.0
        x = transforms.Resize((224, 224))(x)
        x = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(x)
        x = x.to(device)
        with torch.no_grad():
            p = torch.sigmoid(model(x)).cpu().numpy()[:, 0]
        probs.append(p.max())
        names.append(os.path.splitext(os.path.basename(f))[0])
    return np.array(probs), np.array(names)


def get_gate_score(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return 0.5
        feats = compression_features(frame)
        vec = [float(feats[n]) for n in ['blockiness', 'zero_dct_ratio', 'grid_std', 'brisque']]
        z = float((np.dot(np.log1p(np.abs(vec)), coef) + intercept).item())
        return 1.0 / (1.0 + np.exp(-z))
    except Exception:
        return 0.5


if __name__ == "__main__":
    from casr_estimator import _load_gate
    from train_ffpp_cnn import MAX_FACES
    from torchvision import transforms
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    import torch.nn.functional as F
    from sklearn.metrics import roc_auc_score
    from sklearn.utils import resample
    import json
    import glob
    import cv2
    import numpy as np
    
    # Need these globals
    CELEBDF_ROOT = r"E:\Datasets\Celeb-DF-v2"
    ROOT = os.path.dirname(os.path.abspath(__file__))
    FACES_CELEBDF = os.path.join(ROOT, "data", "celebdf_faces")
    CKPT = os.path.join(ROOT, "checkpoints", "ffpp_cnn_best.pt")
    OUT_JSON = os.path.join(ROOT, "results", "celebdf_casr.json")
    GATE_CACHE = os.path.join(ROOT, "results", "celebdf_gate_feats.json")
    os.makedirs(os.path.dirname(GATE_CACHE), exist_ok=True)
    
    device = "cuda"
    CKPT = os.path.join(ROOT, "checkpoints", "ffpp_cnn_best.pt")
    CELEBDF_ROOT = r"E:\Datasets\Celeb-DF-v2"
    ROOT = os.path.dirname(os.path.abspath(__file__))
    FACES_CELEBDF = os.path.join(ROOT, "data", "celebdf_faces")
    OUT_JSON = os.path.join(ROOT, "results", "celebdf_casr.json")
    GATE_CACHE = os.path.join(ROOT, "results", "celebdf_gate_feats.json")
    os.makedirs(os.path.dirname(GATE_CACHE), exist_ok=True)
    
    model = load_model(CKPT, "cuda")
    coef, intercept = _load_gate()
    
    # Load CNN probs
    print("Evaluating CNN on real...")
    p_real, names_real = get_probs(model, "real", 0)
    print("Evaluating CNN on fake...")
    p_fake, names_fake = get_probs(model, "fake", 1)
    
    y = np.concatenate([np.zeros(len(p_real)), np.ones(len(p_fake))])
    cnn_probs = np.concatenate([p_real, p_fake])
    names = list(names_real) + list(names_fake)
    cnn_auc = roc_auc_score(y, cnn_probs)
    print(f"CNN AUC: {cnn_auc:.4f}")
    
    # Gate scores
    print("Computing gate scores...")
    gate_cache = {}
    if os.path.exists(GATE_CACHE):
        with open(GATE_CACHE) as f:
            gate_cache = json.load(f)
    
    hp_list = []
    for name in names:
        if name in gate_cache:
            hp = gate_cache[name]
        else:
            found = False
            for d in ["Celeb-real", "YouTube-real", "Celeb-synthesis"]:
                p = os.path.join(CELEBDF_ROOT, d, name + ".mp4")
                if os.path.exists(p):
                    hp = get_gate_score(p)
                    found = True
                    break
            if not found:
                hp = 0.5
            gate_cache[name] = hp
        hp_list.append(hp)
    
    with open(GATE_CACHE, "w") as f:
        json.dump(gate_cache, f)
    
    hp_arr = np.array(hp_list)
    # CASR: when heavy (hp high), trust CNN less -> regress to chance (0.5)
    p_casr = (1 - hp_arr) * cnn_probs + hp_arr * 0.5
    
    casr_auc = roc_auc_score(y, p_casr)
    cnn_auc = roc_auc_score(y, cnn_probs)
    diff = casr_auc - cnn_auc
    
    print(f"CNN AUC:  {cnn_auc:.4f}")
    print(f"CASR AUC: {casr_auc:.4f}")
    print(f"Diff:     {diff:+.4f}")
    
    # Bootstrap CI for difference
    print("Bootstrapping...")
    diffs = []
    for _ in range(1000):
        y_b, cnn_b, casr_b = resample(y, cnn_probs, p_casr, random_state=None)
        if len(np.unique(y_b)) < 2:
            continue
        diffs.append(roc_auc_score(y_b, casr_b) - roc_auc_score(y_b, cnn_b))
    
    if diffs:
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        p_val = np.mean(np.array(diffs) > 0)
    else:
        lo, hi, p_val = 0, 0, 0
    
    print(f"Diff CI: [{lo:.4f}, {hi:.4f}], p={p_val:.4f}")
    
    results = {
        "cnn_auc": float(cnn_auc),
        "casr_auc": float(casr_auc),
        "diff": float(diff),
        "ci_lo": float(lo) if diffs else 0,
        "ci_hi": float(hi) if diffs else 0,
        "p_val": float(p_val) if diffs else 0,
        "n_real": int(len(p_real)),
        "n_fake": int(len(p_fake)),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {OUT_JSON}")