import os, sys, json, time, traceback

os.environ["PATH"] = (
    r"E:\audi deepfakes\venv\Lib\site-packages\imageio_ffmpeg\binaries;"
    + r"E:\audi deepfakes\venv\Scripts;"
    + os.environ.get("PATH", "")
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.pipeline import DeepfakeDetector
from src.reasoning.classifier import FAKE_PROB_THRESHOLD

LAVDF_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "LAV-DF")
CHK_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints")
GT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "lavdf_ground_truth.json")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=LAVDF_DIR)
    parser.add_argument("--fer_weights", default=os.path.join(CHK_DIR, "fer2013_vgg19.pt"))
    parser.add_argument("--ser_weights", default=os.path.join(CHK_DIR, "ser_weights.pt"))
    parser.add_argument("--output", default="results/lavdf_eval.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split", default="test", choices=["test", "dev", "train"])
    parser.add_argument("--threshold", type=float, default=FAKE_PROB_THRESHOLD)
    parser.add_argument("--model_type", default="vgg19", choices=["vgg19", "resnet50"])
    args = parser.parse_args()

    if not os.path.exists(GT_PATH):
        print(f"Ground truth not found at {GT_PATH}")
        return
    if not os.path.exists(args.dataset):
        print(f"Dataset not found at {args.dataset}")
        return

    with open(GT_PATH) as f:
        all_gt = json.load(f)

    items = [(k, v) for k, v in all_gt.items() if v["split"] == args.split]
    if args.limit:
        items = items[: args.limit]

    print(f"Split: {args.split}, Testing: {len(items)} videos")
    print(f"Threshold: {args.threshold}")
    print(f"Device: cuda (available={torch.cuda.is_available()})")

    detector = DeepfakeDetector(
        fer_weights=args.fer_weights if os.path.exists(args.fer_weights) else None,
        ser_weights=args.ser_weights if os.path.exists(args.ser_weights) else None,
        model_type=args.model_type,
    )

    results = {}
    correct, total = 0, 0
    start_time = time.time()
    for i, (rel_path, meta) in enumerate(items):
        video_path = os.path.join(args.dataset, rel_path)
        if not os.path.exists(video_path):
            print(f"  SKIP {rel_path}: not found")
            continue
        expected = meta["label"]

        try:
            result = detector.predict_video(video_path)
        except Exception as e:
            tb = traceback.format_exc()
            print(f"  ERROR {rel_path}: {e}")
            print(f"  TRACE: {tb[:200]}")
            results[rel_path] = {"ground_truth": expected, "prediction": "error", "error": str(e)}
            continue

        pred = result.get("prediction", "unknown")
        probs = result.get("probabilities", {})
        results[rel_path] = {
            "ground_truth": expected,
            "prediction": pred,
            "VT_prob": probs.get("VT", {}).get("probability", 0),
            "AT_prob": probs.get("AT", {}).get("probability", 0),
            "IM_prob": probs.get("IM", {}).get("probability", 0),
        }
        if pred == expected:
            correct += 1
        total += 1
        mark = "[OK]" if pred == expected else "[X ]"
        elapsed = time.time() - start_time
        pace = elapsed / (i + 1)
        remaining = pace * (len(items) - i - 1)
        print(
            f"  [{i+1}/{len(items)}] {mark} {rel_path} "
            f"gt={expected} pred={pred} "
            f"VT={probs.get('VT',{}).get('probability',0):.0%} "
            f"AT={probs.get('AT',{}).get('probability',0):.0%} "
            f"IM={probs.get('IM',{}).get('probability',0):.0%} "
            f"elapsed={elapsed:.0f}s pace={pace:.1f}s/vid eta={remaining:.0f}s"
        )

    accuracy = correct / total if total > 0 else 0
    print(f"\nAccuracy: {accuracy:.2%} ({correct}/{total})")
    print(f"Total time: {time.time()-start_time:.0f}s")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"accuracy": accuracy, "correct": correct, "total": total, "details": results}, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
