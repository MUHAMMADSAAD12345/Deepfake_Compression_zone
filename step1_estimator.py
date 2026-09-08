"""CASR Step 1: Compression-state estimator validation.

Measures how well lightweight hand-crafted + off-the-shelf image features
predict the ACTUAL compression applied to LAV-DF videos (ground truth we
generated), justifying the "we can estimate compression state at inference"
premise of the CASR method.

Conditions per video (mirrors the FER degradation experiment):
  - clean (original)
  - h264 CRF 18, 23, 28, 33, 38, 43  (ffmpeg libx264, video-only)
  - jpeg Q 30, 60, 90                 (in-memory cv2 encode/decode of the frame)

Features per extracted frame (fixed 512x512 central ROI):
  - blockiness        (Wang-style grid-boundary gradient ratio)
  - zero_dct_ratio    (fraction of ~zero AC DCT coefficients in 8x8 blocks)
  - grid_std          (std of gradient pattern with 8px periodicity)
  - brisque           (piq implementation)

Output:
  - results/step1_estimates.csv        per (video, condition) rows
  - results/step1_report.json          Spearman corr of each feature vs ground
                                       truth, Ridge-regression R2 + rank corr,
                                       binary "heavily compressed?" AUC
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
LAVDF_TEST = r"E:\Saad_audi deepfakes\data\LAV-DF\test"
OUT_DIR = os.path.join(ROOT, "results", "step1")
TMP = os.path.join(OUT_DIR, "tmp")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TMP, exist_ok=True)

FFMPEG = "ffmpeg"
H264_CRFS = [18, 23, 28, 33, 38, 43]
JPEG_QS = [30, 60, 90]


def run_ffmpeg(args):
    r = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y"] + args,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg failed: " + r.stderr[-500:])


def video_duration(video_path):
    dur = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", video_path]).decode().strip()
    return float(dur)


def mid_frame_path(video_path, tag=""):
    t = video_duration(video_path) / 2.0
    out = os.path.join(TMP, os.path.basename(video_path)[:-4] + tag + ".png")
    run_ffmpeg(["-ss", f"{t:.2f}", "-i", video_path, "-frames:v", "1", out])
    return out


def central_crop_gray(img, size=512):
    h, w = img.shape[:2]
    x0, y0 = max((w - size) // 2, 0), max((h - size) // 2, 0)
    cs = img[y0:y0 + size, x0:x0 + size]
    return cv2.cvtColor(cs, cv2.COLOR_BGR2GRAY)


def blockiness(gray):
    """Wang-style blockiness: boundary-adjacent diffs / in-block diffs."""
    g = gray.astype(np.float64)
    h, w = g.shape
    bc = np.arange(8, w, 8)          # boundary columns (diff pos: bc-1)
    db = np.abs(g[:, bc - 1] - g[:, bc]).sum()
    inner = np.abs(g[:, 1:] - g[:, :-1]).sum(axis=0)
    mask = np.ones(w - 1, dtype=bool)
    mask[bc - 1] = False
    bh = db / max(inner[mask].sum(), 1e-9)
    br = np.arange(8, h, 8)          # boundary rows
    dbv = np.abs(g[br - 1, :] - g[br, :]).sum()
    innerv = np.abs(g[1:, :] - g[:-1, :]).sum(axis=1)
    maskv = np.ones(h - 1, dtype=bool)
    maskv[br - 1] = False
    bv = dbv / max(innerv[maskv].sum(), 1e-9)
    return float((bh + bv) / 2.0)


_DCT8 = None


def _dct8_matrix():
    global _DCT8
    if _DCT8 is None:
        n = 8
        k = np.arange(n)
        f = np.pi * (k[:, None] * (2 * k[None, :] + 1)) / (2 * n)
        M = np.cos(f)
        M[0] *= np.sqrt(1.0 / n)
        M[1:] *= np.sqrt(2.0 / n)
        _DCT8 = M
    return _DCT8


def zero_dct_ratio(gray, eps=1.0):
    """Fraction of ~zero AC DCT coefficients over 8x8 blocks (DCT-II via matmul)."""
    h, w = gray.shape
    hb, wb = h // 8, w // 8
    blocks = (gray[:hb * 8, :wb * 8].astype(np.float64) - 128.0)\
        .reshape(hb, 8, wb, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)
    M = _dct8_matrix()
    d = np.einsum("ij,njk,lk->nil", M, blocks, M)
    return float((np.abs(d) < eps).mean())


def grid_std(gray):
    g = gray.astype(np.float64)
    col_edges = np.abs(np.diff(g, axis=1)).mean(axis=0)
    row_edges = np.abs(np.diff(g, axis=0)).mean(axis=1)
    if len(col_edges) < 16 or len(row_edges) < 16:
        return 0.0
    return float(np.std(col_edges[::8]) + np.std(row_edges[::8]))


def brisque_score(img_bgr):
    import torch
    from piq import brisque
    t = torch.from_numpy(img_bgr.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    with torch.no_grad():
        return float(brisque(t).item())


def make_conditions(video):
    video_path = os.path.join(LAVDF_TEST, video)
    if not os.path.exists(video_path):
        return []
    jobs = [(video, "clean", video_path, None, 0)]
    for crf in H264_CRFS:
        out = os.path.join(TMP, f"{video}__crf{crf}.mp4")
        run_ffmpeg(["-i", video_path, "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", str(crf), "-an", out])
        jobs.append((video, f"h264_crf{crf}", video_path, out, crf))
    for q in JPEG_QS:
        jobs.append((video, f"jpeg_q{q}", video_path, None, q))
    return jobs


def measure(img):
    gray = central_crop_gray(img)
    return {
        "blockiness": round(blockiness(gray), 6),
        "zero_dct_ratio": round(zero_dct_ratio(gray), 6),
        "grid_std": round(grid_std(gray), 6),
        "brisque": round(brisque_score(img), 4),
    }


def worker(job):
    video, cond_name, video_path, tmp_video, true_val = job
    try:
        if cond_name.startswith("jpeg"):
            src = mid_frame_path(video_path)
            img = cv2.imread(src)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, true_val])
            if not ok:
                return None
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        else:
            img = cv2.imread(mid_frame_path(tmp_video if tmp_video else video_path))
        row = {"video": video, "condition": cond_name, "true_val": true_val}
        row.update(measure(img))
        return row
    except Exception as e:  # noqa: BLE001
        print(f"[ERR] {video} {cond_name}: {e}", file=sys.stderr)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40, help="videos to process")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()

    csvpath = os.path.join(OUT_DIR, "step1_estimates.csv")
    if args.analyze_only and os.path.exists(csvpath):
        with open(csvpath) as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            for k in r:
                if k not in ("video", "condition"):
                    r[k] = float(r[k])
    else:
        videos = sorted(os.listdir(LAVDF_TEST))[args.offset:args.offset + args.limit]
        jobs = []
        for v in videos:
            jobs.extend(make_conditions(v))

        rows = []
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, r in enumerate(ex.map(worker, jobs)):
                if r is not None:
                    rows.append(r)
                if (i + 1) % 50 == 0:
                    print(f"{i + 1}/{len(jobs)}", flush=True)

        with open(csvpath, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    from scipy.stats import spearmanr
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    feat_names = ["blockiness", "zero_dct_ratio", "grid_std", "brisque"]
    X = np.array([[r[k] for k in feat_names] for r in rows])
    true = np.array([r["true_val"] for r in rows])
    valid = true > 0
    Xv, yv = X[valid], true[valid]

    report = {"n_rows": len(rows), "features": {}}
    for j, k in enumerate(feat_names):
        r_all = spearmanr(X[:, j], true)
        r_codec = spearmanr(Xv[:, j], yv)
        report["features"][k] = {
            "spearman_vs_all": round(float(r_all.statistic), 4),
            "spearman_vs_compressed_only": round(float(r_codec.statistic), 4),
        }

    Xl = np.log1p(np.abs(X[valid]))
    Xtr, Xte, ytr, yte = train_test_split(Xl, yv, test_size=0.25, random_state=0)
    reg = Ridge(alpha=1.0).fit(Xtr, ytr)
    pred = reg.predict(Xte)
    rho_pred, _ = spearmanr(pred, yte)
    ss_res = float(((yte - pred) ** 2).sum())
    ss_tot = float(((yte - yte.mean()) ** 2).sum())
    report["ridge"] = {
        "test_n": int(len(yte)),
        "spearman_pred_vs_true": round(float(rho_pred), 4),
        "r2": round(1.0 - ss_res / max(ss_tot, 1e-12), 4),
    }

    bin_mask = (yv >= 33) | (yv <= 23)
    ybin = (yv[bin_mask] >= 33).astype(int)
    if ybin.sum() > 0 and (ybin == 0).sum() > 0:
        report["binary_heavy_vs_light_auc"] = round(
            float(roc_auc_score(ybin, reg.predict(Xl[bin_mask]))), 4)

    # ---- per-family regressions (separate QP/CRF scales) ----
    report["per_codec_ridge"] = {}
    for fam, lb, ub in (("h264", 18, 43), ("jpeg", 30, 90)):
        idx = np.array([i for i, r in enumerate(rows)
                        if r["condition"].startswith(fam)])
        if len(idx) < 20:
            continue
        Xf, yf = np.log1p(np.abs(X[idx])), true[idx]
        Xa, Xb, ya, yb = train_test_split(Xf, yf, test_size=0.3, random_state=0)
        rf = Ridge(alpha=1.0).fit(Xa, ya)
        pf = rf.predict(Xb)
        rr = spearmanr(pf, yb).statistic
        rsq = 1.0 - float(((yb - pf) ** 2).sum()) / float(((yb - yb.mean()) ** 2).sum())
        report["per_codec_ridge"][fam] = {
            "n": int(len(yf)),
            "spearman_pred_vs_true": round(float(rr), 4),
            "r2": round(float(rsq), 4),
        }

    # ---- ordinal gate: clean / light / mid / heavy from ALL features ----
    from sklearn.linear_model import LogisticRegression
    classes = ["clean", "light", "mid", "heavy"]
    lab = []
    for r in rows:
        c = r["condition"]
        if c == "clean":
            lab.append(0)
        elif c.startswith("h264"):
            v = r["true_val"]
            lab.append(1 if v <= 23 else (2 if v <= 28 else 3))
        else:
            lab.append(0)  # jpeg excluded from gate (single-frame re-encode)
    lab = np.array(lab)
    keep = lab > 0
    Xg = np.log1p(np.abs(X[keep]))
    Xa, Xb, ya, yb = train_test_split(
        Xg, lab[keep], test_size=0.3, random_state=0, stratify=lab[keep])
    lg = LogisticRegression(max_iter=2000).fit(Xa, ya)
    report["ordinal_gate"] = {
        "n": int(len(yb)),
        "acc_3class": round(float(lg.score(Xb, yb)), 4),
    }

    with open(os.path.join(OUT_DIR, "step1_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()