"""CASR Step 6: TRUE semantic branch (FER-based) + gated fusion on FF++.

Replaces the clean-CNN proxy with a genuine FER model (ResNet50, trained on
FER2013, 7 emotions) so no clean reference is needed at test time.

Modes:
  --extract-train : FER probs over data/ffpp_faces/{train,val} -> data/ffpp_fer/train.npz|val.npz
  --extract-eval  : FER probs over data/ffpp_faces_eval/<cond>  -> data/ffpp_fer/<cond>.npz
  --run           : train semantic clf on CLEAN train only; evaluate per condition:
                    cnn / sem(true FER) / fixed0.5 / CASR-gated / oracle

Outputs results/step6_semantic.json (+ gate-feature cache).
"""
import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, r"E:\Saad_audi deepfakes\src")
sys.path.insert(0, r"E:\Saad_audi deepfakes\_casr")

ROOT = os.path.dirname(os.path.abspath(__file__))
FACES_TRAIN = os.path.join(ROOT, "data", "ffpp_faces")
FACES_EVAL = os.path.join(ROOT, "data", "ffpp_faces_eval")
FFPP_ROOT = r"E:\Datasets\FaceForensicsC23\FaceForensics++_C23"
CORPUS = os.path.join(ROOT, "data", "ffpp_compressed")
FER_OUT = os.path.join(ROOT, "data", "ffpp_fer")
PROBS_DIR = os.path.join(ROOT, "results", "step2b_probs")
GATE_CACHE = os.path.join(ROOT, "results", "step6_gate_feats.json")
OUT_JSON = os.path.join(ROOT, "results", "step6_semantic.json")

GROUPS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "real"]
CONDITIONS = ["clean", "h264_crf18", "h264_crf28", "h264_crf38",
              "hevc_crf18", "hevc_crf28", "hevc_crf38",
              "av1_crf20", "av1_crf30", "av1_crf40"]


# ---------------------------------------------------------------- extraction
def fer_over_dir(model, src_dir, out_dict, prefix):
    for f in sorted(glob.glob(os.path.join(src_dir, "*.npz"))):
        a = np.load(f)["imgs"]
        if len(a) == 0:
            continue
        _, probs = model.predict_batch(a, batch_size=64)
        out_dict[f"{prefix}/{os.path.splitext(os.path.basename(f))[0]}"] = \
            probs.astype(np.float16)


def do_extract(split_name, root, workers=None):
    os.makedirs(FER_OUT, exist_ok=True)
    from models.fer_model import FERModel
    model = FERModel(
        model_path=r"E:\Saad_audi deepfakes\checkpoints\fer_resnet50_best.pt",
        model_type="resnet50")
    out = {}
    n_done = 0
    for g in GROUPS:
        d = os.path.join(root, g)
        if not os.path.isdir(d):
            continue
        before = len(out)
        fer_over_dir(model, d, out, g)
        n_done += len(out) - before
        print(f"[fer] {split_name}/{g}: {len(out)} total", flush=True)
    dst = os.path.join(FER_OUT, f"{split_name}.npz")
    np.savez_compressed(dst, **out)
    print(f"[fer] wrote {dst} ({len(out)} videos)", flush=True)


# ------------------------------------------------------------- sem features
def sem_features(vp):
    """Stats over a (n_frames, 7) emotion-prob sequence."""
    vp = np.asarray(vp, dtype=np.float64)
    m = vp.mean(axis=0)
    s = m.sum()
    m = m / s if s > 0 else m
    conf = float(vp.max(axis=1).mean())
    ent = float(-(m * np.log(m + 1e-12)).sum()) / np.log(7)
    q1, q3 = np.percentile(vp.max(axis=1), [25, 75])
    dom = float((vp.argmax(axis=1) == m.argmax()).mean())
    trans = float((np.diff(vp.argmax(axis=1)) != 0).mean())
    return [conf, ent, q3 - q1, dom, trans]


FEATS = ["conf", "ent", "conf_iqr", "dom", "trans"]


def load_fer(name):
    z = np.load(os.path.join(FER_OUT, f"{name}.npz"))
    return {k: z[k].astype(np.float32) for k in z.files}


# ------------------------------------------------------------ gate features
def gate_one(video_path):
    try:
        from casr_estimator import compression_features
        cap = cv2.VideoCapture(video_path)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return None
        f = compression_features(frame)
        return {k: float(f[k]) for k in
                ("blockiness", "zero_dct_ratio", "grid_std", "brisque")}
    except Exception:  # noqa: BLE001
        return None


def gate_feats_for(cond):
    """4 raw gate features per video for a condition (cached)."""
    from casr_estimator import FEAT_NAMES
    if os.path.exists(GATE_CACHE) and cond in json.load(open(GATE_CACHE)):
        return json.load(open(GATE_CACHE))[cond]
    cache = json.load(open(GATE_CACHE)) if os.path.exists(GATE_CACHE) else {}
    jobs = []
    if cond == "clean":
        for g in GROUPS:
            src = os.path.join(FFPP_ROOT, "real") if g == "real" \
                else os.path.join(FFPP_ROOT, "fake", g)
            names = sorted(os.listdir(src))
            if g == "real":
                names = names[len(names) - 140:]
            else:
                names = names[len(names) - 140:]
            for n in names:
                jobs.append((f"{g}/{os.path.splitext(n)[0]}",
                             os.path.join(src, n)))
    else:
        for g in GROUPS:
            d = os.path.join(CORPUS, cond, g)
            if not os.path.isdir(d):
                continue
            for n in sorted(os.listdir(d)):
                jobs.append((f"{g}/{os.path.splitext(n)[0]}",
                             os.path.join(d, n)))
    with open(os.devnull, "w") as devnull:  # noqa: SIM115
        pass
    results = []
    for i, (_, p) in enumerate(jobs):
        r = gate_one(p)
        if r is None and i < 5:
            print(f"[gate] WARN null {p}", flush=True)
        results.append(r)
        if (i + 1) % 200 == 0:
            print(f"[gate] {cond} {i + 1}/{len(jobs)}", flush=True)
    feats = {j[0]: r for j, r in zip(jobs, results)}
    cache[cond] = feats
    json.dump(cache, open(GATE_CACHE, "w"))
    return feats


# --------------------------------------------------------------------- main
def run():
    from casr_estimator import _load_gate, FEAT_NAMES
    coef, intercept = _load_gate()

    tr_fer = load_fer("train")
    y_tr, X_tr, keys_tr = [], [], []
    for k, vp in tr_fer.items():
        grp = k.split("/")[0]
        y_tr.append(0 if grp == "real" else 1)
        X_tr.append(sem_features(vp))
        keys_tr.append(k)
    sc = StandardScaler().fit(X_tr)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(sc.transform(X_tr), y_tr)
    tr_auc = roc_auc_score(y_tr, clf.predict_proba(sc.transform(X_tr))[:, 1])
    print(f"[sem] trained on clean train ({len(y_tr)} vids), train-AUC "
          f"{tr_auc:.4f} (in-sample reference)", flush=True)

    rows = {}
    for c in CONDITIONS:
        try:
            fer = load_fer(c)
        except FileNotFoundError:
            print(f"[run] missing fer cache for {c}, skipping", flush=True)
            continue
        cnn_raw = json.load(open(os.path.join(PROBS_DIR, f"{c}.json")))
        gf = gate_feats_for(c)

        y, cn, sm, hp = [], [], [], []
        for k, vp in fer.items():
            cn_p = cnn_raw.get(k.replace("/", "\\") + ".npz",
                               cnn_raw.get(k + ".npz"))
            if cn_p is None:
                continue
            f = gf.get(k)
            if f is None:
                continue
            vec = [float(f[n]) for n in FEAT_NAMES]
            z = float((np.dot(np.log1p(np.abs(vec)), coef) + intercept).item())
            h = 1.0 / (1.0 + np.exp(-z))
            y.append(0 if k.split("/")[0] == "real" else 1)
            cn.append(float(cn_p))
            sm.append(float(clf.predict_proba(
                sc.transform([sem_features(vp)]))[0, 1]))
            hp.append(h)
        y, cn, sm, hp = map(np.array, (y, cn, sm, hp))
        if len(y) < 100 or len(np.unique(y)) < 2:
            print(f"[run] {c}: insufficient data", flush=True)
            continue
        w_sem = 1.0 - hp
        fixed = 0.5 * cn + 0.5 * sm
        gated = w_sem * cn + (1.0 - w_sem) * sm
        best = -1.0
        for w in np.linspace(0, 1, 41):
            best = max(best, float(roc_auc_score(y, w * cn + (1 - w) * sm)))
        rows[c] = {"n": int(len(y)),
                   "auc_cnn": round(float(roc_auc_score(y, cn)), 4),
                   "auc_sem_true": round(float(roc_auc_score(y, sm)), 4),
                   "auc_fixed": round(float(roc_auc_score(y, fixed)), 4),
                   "auc_casr": round(float(roc_auc_score(y, gated)), 4),
                   "auc_oracle": round(best, 4),
                   "mean_heavy_p": round(float(hp.mean()), 4)}
        print(f"[run] {c:12s} CNN {rows[c]['auc_cnn']:.4f}  "
              f"SEM* {rows[c]['auc_sem_true']:.4f}  "
              f"CASR {rows[c]['auc_casr']:.4f}  "
              f"(oracle {rows[c]['auc_oracle']:.4f})", flush=True)

    json.dump(rows, open(OUT_JSON, "w"), indent=1)
    print(f"wrote {OUT_JSON}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract-train", action="store_true")
    ap.add_argument("--extract-eval", nargs="*", default=None)
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()
    if args.extract_train:
        do_extract("train", os.path.join(FACES_TRAIN, "train"))
        do_extract("val", os.path.join(FACES_TRAIN, "val"))
    if args.extract_eval is not None:
        conds = args.extract_eval or CONDITIONS
        for cnd in conds:
            do_extract(cnd, os.path.join(FACES_EVAL, cnd))
    if args.run:
        run()


if __name__ == "__main__":
    main()
