import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from src.pipeline import DeepfakeDetector

REAL_THRESHOLD = 0.15

def evaluate_on_dataset(detector, dataset_dir, ground_truth):
    results = {}
    correct = 0
    total = 0
    for video_name, expected_label in ground_truth.items():
        video_path = os.path.join(dataset_dir, video_name)
        if not os.path.exists(video_path):
            print(f"Skipping {video_name}: not found")
            continue
        result = detector.predict_video(video_path)
        pred = result.get("prediction", "unknown")
        results[video_name] = {
            "ground_truth": expected_label,
            "prediction": pred,
            "probabilities": result.get("probabilities", {}),
        }
        if pred == expected_label:
            correct += 1
        total += 1
        status = "✓" if pred == expected_label else "✗"
        print(f"  {status} {video_name}: gt={expected_label}, pred={pred}")
    accuracy = correct / total if total > 0 else 0
    return {"accuracy": accuracy, "correct": correct, "total": total, "details": results}

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to dataset directory")
    parser.add_argument("--gt", required=True, help="Path to ground truth JSON")
    parser.add_argument("--fer_weights", default=None)
    parser.add_argument("--ser_weights", default=None)
    parser.add_argument("--output", default=None, help="Save results JSON")
    args = parser.parse_args()

    with open(args.gt, "r") as f:
        ground_truth = json.load(f)

    detector = DeepfakeDetector(fer_weights=args.fer_weights, ser_weights=args.ser_weights)
    eval_results = evaluate_on_dataset(detector, args.dataset, ground_truth)

    print(f"\nAccuracy: {eval_results['accuracy']:.2%} ({eval_results['correct']}/{eval_results['total']})")
    if args.output:
        with open(args.output, "w") as f:
            json.dump(eval_results, f, indent=2)
        print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
