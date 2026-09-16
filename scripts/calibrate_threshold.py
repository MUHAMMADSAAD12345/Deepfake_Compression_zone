import os, sys, json, time, traceback

os.environ["PATH"] = (
    r"E:\audi deepfakes\venv\Lib\site-packages\imageio_ffmpeg\binaries;"
    + r"E:\audi deepfakes\venv\Scripts;"
    + os.environ.get("PATH", "")
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from src.pipeline import DeepfakeDetector

LAVDF_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "LAV-DF")
CHK_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints")
GT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "lavdf_ground_truth.json")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=LAVDF_DIR)
    parser.add_argument("--fer_weights", default=os.path.join(CHK_DIR, "fer2013_vgg19.pt"))
    parser.add_argument("--ser_weights", default=os.path.join(CHK_DIR, "ser_weights.pt"))
    parser.add_argument("--output", default="results/calibration.json")
    parser.add_argument("--split", default="dev", choices=["dev", "val"])
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--model_type", default="vgg19", choices=["vgg19", "resnet50"])
    args = parser.parse_args()

    with open(GT_PATH) as f:
        all_gt = json.load(f)

    items = [(k, v) for k, v in all_gt.items() if v["split"] == args.split]
    if args.limit:
        items = items[: args.limit]

    print(f"Calibrating on {args.split} split: {len(items)} videos")

    detector = DeepfakeDetector(
        fer_weights=args.fer_weights if os.path.exists(args.fer_weights) else None,
        ser_weights=args.ser_weights if os.path.exists(args.ser_weights) else None,
        model_type=args.model_type,
    )

    probs_list = []
    labels_list = []
    start_time = time.time()
    for i, (rel_path, meta) in enumerate(items):
        video_path = os.path.join(args.dataset, rel_path)
        if not os.path.exists(video_path):
            continue
        try:
            result = detector.predict_video(video_path)
        except Exception as e:
            continue

        pp = result.get("probabilities", {})
        probs_list.append({
            "VT": pp.get("VT", {}).get("probability", 0),
            "AT": pp.get("AT", {}).get("probability", 0),
            "IM": pp.get("IM", {}).get("probability", 0),
        })
        labels_list.append(1 if meta["label"] == "fake" else 0)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            print(f"  [{i+1}/{len(items)}] {elapsed:.0f}s elapsed, ~{elapsed/(i+1):.1f}s/vid")

    probs = np.array([[p["VT"], p["AT"], p["IM"]] for p in probs_list])
    labels = np.array(labels_list)

    print(f"\nProcessed {len(probs)} videos")

    from sklearn.metrics import accuracy_score, precision_recall_curve, roc_auc_score

    # Score = sum of VT + AT + IM probabilities (higher = more fake)
    scores = probs.mean(axis=1)
    auc = roc_auc_score(labels, scores)
    print(f"Mean-prob AUC: {auc:.4f}")

    # Find best threshold for each modality individually and for ensemble
    best_thresh = {}
    for idx, name in enumerate(["VT", "AT", "IM"]):
        prec, rec, thresh = precision_recall_curve(labels, probs[:, idx])
        f1 = 2 * prec * rec / (prec + rec + 1e-10)
        best_idx = np.argmax(f1[:-1])
        best_thresh[name] = float(thresh[best_idx])
        print(f"  {name}: best_threshold={thresh[best_idx]:.3f}, best_F1={f1[best_idx]:.4f}")

    # Grid search on ensemble mean threshold
    best_acc = 0
    best_t = 0
    for t in np.arange(0.05, 0.95, 0.01):
        preds = (scores > t).astype(int)
        acc = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc = acc
            best_t = t

    print(f"\nEnsemble (mean): best_threshold={best_t:.2f}, best_acc={best_acc:.4f}")

    # Voting classifier grid search on per-modality threshold
    print("\nVoting classifier grid search (shared threshold across VT/AT/IM):")
    best_vote_acc = 0
    best_vote_t = 0
    for t in np.arange(0.05, 0.95, 0.01):
        preds = []
        for p in probs:
            votes = sum(1 for v in [p[0], p[1], p[2]] if v > t)
            preds.append(1 if votes >= 2 else 0)
        acc = accuracy_score(labels, preds)
        if acc > best_vote_acc:
            best_vote_acc = acc
            best_vote_t = t
    print(f"  best_threshold={best_vote_t:.2f}, best_acc={best_vote_acc:.4f}")

    calib = {
        "auc": auc,
        "best_mean_threshold": best_t,
        "best_mean_accuracy": best_acc,
        "best_vote_threshold": best_vote_t,
        "best_vote_accuracy": best_vote_acc,
        "per_modality_thresholds": best_thresh,
        "n_samples": len(probs),
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(calib, f, indent=2)
    print(f"\nCalibration saved to {args.output}")


if __name__ == "__main__":
    main()
