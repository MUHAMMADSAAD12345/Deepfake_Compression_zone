"""CASR Step 3-v2: reliability-gated fusion on FF++ test split (all 10 conditions)."""
import cv2
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, r"E:\Saad_audi deepfakes\src")
sys.path.insert(0, r"E:\Saad_audi deepfakes\_casr")

from casr_estimator import FEAT_NAMES, compression_features, gate_score

ROOT = os.path.dirname(os.path.abspath(__file__))
PROBS_DIR = os.path.join(ROOT, "results", "step2b_probs")
EVAL_JSON = os.path.join(ROOT, "results", "step3_ffpp.json")

GROUPS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "real"]
CONDITIONS = ["clean", "h264_crf18", "h264_crf28", "h264_crf38",
              "hevc_crf18", "hevc_crf28", "hevc_crf38",
              "av1_crf20", "av1_crf30", "av1_crf40"]


def load_cnn_probs(cond):
    p = os.path.join(PROBS_DIR, f"{cond}.json")
    with open(p) as f:
        d = json.load(f)
    return d


def get_video_path(cond, grp, name):
    """Map (cond, group, filename.npz) -> actual video path on disk."""
    if cond == "clean":
        base = r"E:\Datasets\FaceForensicsC23\FaceForensics++_C23"
        if grp == "real":
            return os.path.join(base, "real", name.replace(".npz", ".mp4"))
        return os.path.join(base, "fake", grp, name.replace(".npz", ".mp4"))
    else:
        base = os.path.join(ROOT, "data", "ffpp_compressed", cond, grp)
        return os.path.join(base, name.replace(".npz", ".mp4"))


def compute_gate_feats(video_path):
    """Extract gate features from a video file (first frame)."""
    try:
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None
        return compression_features(frame)
    except Exception:
        return None


def eval_condition(cond):
    probs = load_cnn_probs(cond)
    per_vid = {}
    for k, v in probs.items():
        grp, name = k.split(os.sep, 1)
        per_vid.setdefault((grp, name), []).append(v)

    # build jobs
    jobs = []
    for (grp, name), preds in per_vid.items():
        p_max = max(preds)
        label = 0 if grp == "real" else 1
        jobs.append((grp, name, p_max, label))

    # parallel feature extraction
    videos = [get_video_path(cond, g, n) for g, n, _, _ in jobs]
    with ProcessPoolExecutor(max_workers=8) as ex:
        feats_list = list(ex.map(compute_gate_feats, videos))

    y, cnn, sem, casr = [], [], [], []
    for (grp, name, p_max, label), feats in zip(jobs, feats_list):
        if feats is None:
            continue
        hp = gate_score(feats)  # heavy prob
        w_sem = 1.0 - hp
        # semantic = clean CNN prob for this video
        # We don't have clean prob per video here; approximate with cnn at clean
        # For now: sem = clean AUC level per method (or just cnn_clean video prob)
        # Simpler: sem = cnn at clean condition for same video
        y.append(label)
        cnn.append(p_max)
        # need clean prob for this exact video
        casr.append(w_sem * p_max + (1 - w_sem) * p_max)  # placeholder

    return y, cnn, casr


def main():
    # load clean probs as semantic reference
    clean_probs = load_cnn_probs("clean")
    clean_vid = {}
    for k, v in clean_probs.items():
        grp, name = k.split(os.sep, 1)
        clean_vid.setdefault((grp, name), []).append(v)
    clean_vid = {k: max(v) for k, v in clean_vid.items()}

    rows = {}
    for c in CONDITIONS:
        probs = load_cnn_probs(c)
        per_vid = {}
        for k, v in probs.items():
            grp, name = k.split(os.sep, 1)
            per_vid.setdefault((grp, name), []).append(v)

        jobs = []
        for (grp, name), preds in per_vid.items():
            p_max = max(preds)
            label = 0 if grp == "real" else 1
            p_clean = clean_vid.get((grp, name), p_max)
            jobs.append((grp, name, p_max, p_clean, label))

        videos = [get_video_path(c, g, n) for g, n, _, _, _ in jobs]
        with ProcessPoolExecutor(max_workers=8) as ex:
            feats_list = list(ex.map(compute_gate_feats, videos))

        y, cnn, sem, casr = [], [], [], []
        for (grp, name, p_max, p_clean, label), feats in zip(jobs, feats_list):
            if feats is None:
                continue
# gate_score returns heavy prob; it expects features dict
            hp = gate_score(feats)
            if isinstance(hp, np.ndarray):
                hp = float(hp.item())
            w_sem = 1.0 - hp
            y.append(label)
            cnn.append(p_max)
            sem.append(p_clean)
            casr.append(w_sem * p_max + (1 - w_sem) * p_clean)

        y = np.array(y)
        rows[c] = {
            "cnn": float(roc_auc_score(y, cnn)),
            "sem": float(roc_auc_score(y, sem)),
            "casr": float(roc_auc_score(y, casr)),
            "n": len(y)
        }
        print(f"{c:12s}  CNN {rows[c]['cnn']:.4f}  SEM {rows[c]['sem']:.4f}  CASR {rows[c]['casr']:.4f}")

    with open(EVAL_JSON, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"wrote {EVAL_JSON}")


if __name__ == "__main__":
    main()