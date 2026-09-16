import os, sys, json, time, traceback

os.environ["PATH"] = (
    r"E:\audi deepfakes\venv\Lib\site-packages\imageio_ffmpeg\binaries;"
    + r"E:\audi deepfakes\venv\Scripts;"
    + os.environ.get("PATH", "")
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from src.models.fer_model import FERModel
from src.models.ser_model import SERModel
from src.utils.preprocessing import preprocess_video
from src.reasoning.intra_modality import intra_modality_reasoning
from src.reasoning.inter_modality import inter_modality_reasoning
from src.reasoning.classifier import compute_fake_probabilities, classify_modality, voting_classifier

LAVDF_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "LAV-DF")
CHK_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints")
GT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "lavdf_ground_truth.json")


def run_ablation(video_path, fer, ser, threshold=0.47):
    data = preprocess_video(video_path)
    face_images = data["face_images"]
    segments = data["audio_segments"]
    sync_map = data["sync_map"]
    if not face_images or not segments:
        return None

    visual_emotions, _ = fer.predict_batch(face_images)
    aural_emotions, _ = ser.predict_batch(segments)

    vt_results, vt_fake = intra_modality_reasoning(visual_emotions)
    at_results, at_fake = intra_modality_reasoning(aural_emotions)
    im_results, im_fake = inter_modality_reasoning(visual_emotions, aural_emotions, sync_map)

    prob_dict = compute_fake_probabilities(vt_results, at_results, im_results)
    p = prob_dict

    # All combinations of VT, AT, IM
    vt_flag = p["VT"]["probability"] > threshold
    at_flag = p["AT"]["probability"] > threshold
    im_flag = p["IM"]["probability"] > threshold

    combos = {
        "VT_only": vt_flag,
        "AT_only": at_flag,
        "IM_only": im_flag,
        "VT_AT": vt_flag or at_flag,
        "VT_IM": vt_flag or im_flag,
        "AT_IM": at_flag or im_flag,
        "VT_AT_IM": (vt_flag and at_flag) or (vt_flag and im_flag) or (at_flag and im_flag),
    }

    results = {}
    for name, verdict in combos.items():
        results[name] = "fake" if verdict else "real"

    results["VT_prob"] = p["VT"]["probability"]
    results["AT_prob"] = p["AT"]["probability"]
    results["IM_prob"] = p["IM"]["probability"]

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=LAVDF_DIR)
    parser.add_argument("--fer_weights", default=os.path.join(CHK_DIR, "fer2013_vgg19.pt"))
    parser.add_argument("--ser_weights", default=os.path.join(CHK_DIR, "ser_weights.pt"))
    parser.add_argument("--output", default="results/ablation.json")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--threshold", type=float, default=0.47)
    parser.add_argument("--model_type", default="vgg19", choices=["vgg19", "resnet50"])
    args = parser.parse_args()

    with open(GT_PATH) as f:
        all_gt = json.load(f)

    items = [(k, v) for k, v in all_gt.items() if v["split"] == args.split]
    if args.limit:
        items = items[: args.limit]

    print(f"Ablation on {args.split}: {len(items)} videos, threshold={args.threshold}, model={args.model_type}")

    fer = FERModel(args.fer_weights if os.path.exists(args.fer_weights) else None, model_type=args.model_type)
    ser = SERModel(args.ser_weights if os.path.exists(args.ser_weights) else None)

    all_results = []
    start_time = time.time()
    for i, (rel_path, meta) in enumerate(items):
        video_path = os.path.join(args.dataset, rel_path)
        if not os.path.exists(video_path):
            continue
        try:
            r = run_ablation(video_path, fer, ser, args.threshold)
        except Exception as e:
            continue
        if r is None:
            continue

        r["rel_path"] = rel_path
        r["ground_truth"] = meta["label"]
        all_results.append(r)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            print(f"  [{i+1}/{len(items)}] {elapsed:.0f}s, ~{elapsed/(i+1):.1f}s/vid")

    print(f"\nProcessed {len(all_results)} videos")

    combo_names = ["VT_only", "AT_only", "IM_only", "VT_AT", "VT_IM", "AT_IM", "VT_AT_IM"]
    stats = {}
    for combo in combo_names:
        correct = sum(1 for r in all_results if r[combo] == r["ground_truth"])
        total = len(all_results)
        tp = sum(1 for r in all_results if r[combo] == "fake" and r["ground_truth"] == "fake")
        fp = sum(1 for r in all_results if r[combo] == "fake" and r["ground_truth"] == "real")
        tn = sum(1 for r in all_results if r[combo] == "real" and r["ground_truth"] == "real")
        fn = sum(1 for r in all_results if r[combo] == "real" and r["ground_truth"] == "fake")
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        stats[combo] = {
            "accuracy": correct / total,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        }
        print(f"  {combo:12s}: acc={stats[combo]['accuracy']:.2%}  "
              f"p={stats[combo]['precision']:.2%}  r={stats[combo]['recall']:.2%}  "
              f"f1={stats[combo]['f1']:.4f}")

    mean_real_probs = {
        "VT": np.mean([r["VT_prob"] for r in all_results if r["ground_truth"] == "real"]),
        "AT": np.mean([r["AT_prob"] for r in all_results if r["ground_truth"] == "real"]),
        "IM": np.mean([r["IM_prob"] for r in all_results if r["ground_truth"] == "real"]),
    }
    mean_fake_probs = {
        "VT": np.mean([r["VT_prob"] for r in all_results if r["ground_truth"] == "fake"]),
        "AT": np.mean([r["AT_prob"] for r in all_results if r["ground_truth"] == "fake"]),
        "IM": np.mean([r["IM_prob"] for r in all_results if r["ground_truth"] == "fake"]),
    }
    print(f"\n  Mean REAL probs: VT={mean_real_probs['VT']:.1%}  AT={mean_real_probs['AT']:.1%}  IM={mean_real_probs['IM']:.1%}")
    print(f"  Mean FAKE probs: VT={mean_fake_probs['VT']:.1%}  AT={mean_fake_probs['AT']:.1%}  IM={mean_fake_probs['IM']:.1%}")

    output = {
        "n_samples": len(all_results),
        "threshold": args.threshold,
        "split": args.split,
        "results": stats,
        "mean_real_probs": mean_real_probs,
        "mean_fake_probs": mean_fake_probs,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
