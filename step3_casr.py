"""CASR Step 3: Compression-Aware Semantic Reliability prototype.

Headline experiment: does gating the semantic (emotion) branch by estimated
compression state beat fixed fusion / single branches along the
compression-AUC-drop curve?

Pipeline per (video, condition) over the 200-video FER corpus:
  1. gate features  -> gate_score (heavy-compression prob), from _casr/casr_estimator
  2. semantic score -> ridge classifier on 7-class FER prob stats, TRAINED ON
     CLEAN CONDITIONS ONLY (honest out-of-distribution transfer to compression)
  3. cnn score      -> step2_probs/<cond>.json (WildDeepfake ResNet50 baseline)
  4. fusion:
       fixed  : 0.5 p_cnn + 0.5 p_sem
       gated  : w(cond) p_cnn + (1-w) p_sem,  w = gate-driven
       oracle : ideal w per condition (upper bound)

Outputs AUC per method per condition -> results/step3_auc.json + csv.
"""
import csv
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from casr_estimator import FEAT_NAMES, blockiness, brisque_score, grid_std, zero_dct_ratio

ROOT = os.path.dirname(os.path.abspath(__file__))
NPZ = r"E:\Saad_audi deepfakes\results\compression_degradation\results_resnet50_keepaudio.npz"
GT = r"E:\Saad_audi deepfakes\data\lavdf_ground_truth.json"
PROBS_DIR = os.path.join(ROOT, "results", "step2_probs")
GATE_FEATS = os.path.join(ROOT, "results", "step3_gate_feats.json")
OUT = os.path.join(ROOT, "results", "step3_auc.json")
OUT_CSV = os.path.join(ROOT, "results", "step3_auc.csv")

CONDITIONS = ["clean"] + [f"h264_crf{c}" for c in (18, 28, 38)]
NPZ_COND = {"clean": "clean",
            "h264_crf18": "h264_18_keepaudio",
            "h264_crf28": "h264_28_keepaudio",
            "h264_crf38": "h264_38_keepaudio"}


def sem_features(vp):
    """Statistics over per-face 7-class FER probs (n_faces, 7)."""
    vp = np.asarray(vp, dtype=np.float64)
    m = vp.mean(axis=0)
    m = m / m.sum() if m.sum() > 0 else m
    ent = float(-(m * np.log(m + 1e-12)).sum())
    conf = float(vp.max(axis=1).mean())
    conf_iqr = float(np.percentile(vp.max(axis=1), 75) - np.percentile(vp.max(axis=1), 25))
    dom = float((m == m.max()).mean())
    return [conf, ent / np.log(7), conf_iqr, dom]


def load_vp(npz, vid, cond):
    """_vp array for (video, condition) or None."""
    k = f"{vid}_{NPZ_COND[cond]}_vp"
    if k in npz.files:
        return npz[k]
    return None


def gate_features_from_cond(video, cond, corpus_root):
    """Single frame at 50% duration -> 4 gate features (mirrors step1)."""
    import subprocess
    import cv2
    if cond == "clean":
        p = rf"E:\Saad_audi deepfakes\data\LAV-DF\test\{video}.mp4"
    else:
        p = os.path.join(corpus_root, cond, "test", video + ".mp4")
    if not os.path.exists(p):
        return None
    dur = float(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", p]).decode().strip())
    tmp = os.path.join(ROOT, "results", "step3_tmp.png")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-ss", f"{dur / 2:.2f}", "-i", p, "-frames:v", "1", tmp],
                   check=True, capture_output=True)
    img = cv2.imread(tmp)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return {k: blockiness(gray) if k == "blockiness"
            else zero_dct_ratio(gray) if k == "zero_dct_ratio"
            else grid_std(gray) if k == "grid_std"
            else brisque_score(img)
            for k in FEAT_NAMES}


def main():
    npz = np.load(NPZ, allow_pickle=True)
    gt = json.load(open(GT))
    videos = sorted({k.split("_")[0] for k in npz.files})
    videos = [v for v in videos if f"test/{v}.mp4" in gt]
    print(f"{len(videos)} videos with FER probs")

    # ---- labels + semantic features ----
    n_faces_per_vid = {}
    rows_sem = {c: [] for c in CONDITIONS}
    labels = {c: [] for c in CONDITIONS}
    for v in videos:
        lab = 1 if gt[f"test/{v}.mp4"]["label"] == "fake" else 0
        for c in CONDITIONS:
            vp = load_vp(npz, v, c)
            if vp is None or len(vp) == 0:
                continue
            rows_sem[c].append(sem_features(vp))
            labels[c].append(lab)

    # ---- semantic classifier: video-stratified train/test on clean ----
    from sklearn.model_selection import train_test_split
    Xtr_all = np.array(rows_sem["clean"])
    ytr_all = np.array(labels["clean"])
    tr_idx, te_idx = train_test_split(
        np.arange(len(Xtr_all)), test_size=0.35, random_state=0,
        stratify=ytr_all)
    reg = Ridge(alpha=10.0)
    reg.fit(Xtr_all[tr_idx], ytr_all[tr_idx])
    sem_scores = {}
    for c in CONDITIONS:
        X = np.array(rows_sem[c])
        sem_scores[c] = reg.predict(X)
    # clean-condition honest split masks for reporting
    clean_test_mask = np.zeros(len(videos), dtype=bool)
    clean_test_mask[te_idx] = True

    # ---- cnn scores from step2_probs ----
    cnn_scores = {}
    ok = {}
    for c in CONDITIONS:
        pf = os.path.join(PROBS_DIR, c + ".json")
        if not os.path.exists(pf):
            continue
        probs = {r["rel_path"].split("/")[-1]: r["mean_fake_prob"]
                 for r in json.load(open(pf))}
        cnn_scores[c] = np.array([probs.get(v + ".mp4", np.nan) for v in videos])
        ok[c] = ~np.isnan(cnn_scores[c])
        cnn_scores[c] = np.nan_to_num(cnn_scores[c])

    # ---- gate scores (computed once, cached) ----
    if os.path.exists(GATE_FEATS):
        gates = json.load(open(GATE_FEATS))
    else:
        gates = {}
        for c in CONDITIONS:
            for v in videos:
                gates.setdefault(c, {})[v] = gate_features_from_cond(
                    v, c, os.path.join(ROOT, "data", "lavdf_compressed"))
        json.dump(gates, open(GATE_FEATS, "w"))
    default_feats = {k: 0.0 for k in FEAT_NAMES}
    gate_vals = {}
    for c in CONDITIONS:
        feats = gates.get(c, {})
        gate_vals[c] = np.array(
            [[{**default_feats, **(feats.get(v) or {})}[k] for k in FEAT_NAMES]
             for v in videos], dtype=np.float64)
    from casr_estimator import _load_gate
    coef, intercept = _load_gate()
    gz = {c: np.dot(np.log1p(np.abs(gate_vals[c])), coef) + intercept
          for c in CONDITIONS}
    w_sem = {c: 1.0 / (1.0 + np.exp(-gz[c])) for c in CONDITIONS}  # gate prob
    w_sem = {c: 1.0 - w_sem[c] for c in w_sem}  # semantic reliability = 1 - heavy prob

    # ---- fusion + AUC table ----
    report = {"n_videos": len(videos), "conditions": {}}
    csv_rows = []
    for c in CONDITIONS:
        y = np.array(labels[c])
        if c not in cnn_scores or ok[c].sum() < 20:
            continue
        m = ok[c].copy()
        if c == "clean":
            m &= clean_test_mask  # honest clean evaluation (video-stratified)
        y, cn, sm, ws = y[m], cnn_scores[c][m], sem_scores[c][m], w_sem[c][m]
        auc_cnn = float(roc_auc_score(y, cn))
        auc_sem = float(roc_auc_score(y, sm))
        fixed = 0.5 * cn + 0.5 * sm
        auc_fixed = float(roc_auc_score(y, fixed))
        gated = ws * cn + (1 - ws) * sm
        auc_gated = float(roc_auc_score(y, gated))

        # oracle: grid-search w on THIS condition (upper bound)
        best = -1.0
        for w in np.linspace(0, 1, 21):
            a = float(roc_auc_score(y, w * cn + (1 - w) * sm))
            best = max(best, a)
        auc_oracle = best

        entry = {"n": int(m.sum()), "auc_cnn": round(auc_cnn, 4),
                 "auc_sem": round(auc_sem, 4), "auc_fixed": round(auc_fixed, 4),
                 "auc_gated": round(auc_gated, 4), "auc_oracle": round(auc_oracle, 4),
                 "mean_w_sem": round(float(ws.mean()), 4),
                 "mean_gate_heavy_p": round(float((1 - ws).mean()), 4)}
        report["conditions"][c] = entry
        csv_rows.append({"condition": c, **entry})
        print(c, entry, flush=True)

    json.dump(report, open(OUT, "w"), indent=2)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()