"""Evaluate CNN baseline on LAV-DF test videos (video-level mean-prob aggregation)."""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models

sys.path.insert(0, "E:\\Saad_audi deepfakes")
from src.utils.preprocessing import extract_frames, detect_face_color

def main():
    torch.multiprocessing.freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--ckpt", default="E:\\Saad_audi deepfakes\\checkpoints\\cnn_baseline_best.pt")
    parser.add_argument("--output", default="E:\\Saad_audi deepfakes\\results\\cnn_lavdf_results.json")
    parser.add_argument("--max_frames", type=int, default=12)
    args = parser.parse_args()

    DATA_DIR = "E:\\Saad_audi deepfakes\\data\\LAV-DF"
    GT_PATH = "E:\\Saad_audi deepfakes\\data\\lavdf_ground_truth.json"

    with open(GT_PATH) as f:
        gt = json.load(f)
    items = [(k, v) for k, v in sorted(gt.items()) if v["split"] == args.split][: args.limit]

    device = torch.device("cuda")
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(args.ckpt, weights_only=True))
    model.to(device).eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if os.path.exists(args.output):
        with open(args.output) as f:
            done = {r["rel_path"]: r for r in json.load(f)}
    else:
        done = {}

    results, skipped = [], 0
    t0 = time.time()
    for i, (rel_path, meta) in enumerate(items):
        if rel_path in done:
            results.append(done[rel_path])
            continue
        video_path = os.path.join(DATA_DIR, rel_path)
        if not os.path.exists(video_path):
            skipped += 1
            continue
        frames, ts, fps = extract_frames(video_path, max_frames=args.max_frames)
        faces = [detect_face_color(f) for f in frames]
        faces = [f for f in faces if f is not None]
        if not faces:
            skipped += 1
            print(f"  [{i+1}/{len(items)}] {rel_path}: NO FACES skipped", flush=True)
            continue
        probs = []
        with torch.no_grad():
            for j in range(0, len(faces), 64):
                batch = torch.stack([transform(f) for f in faces[j:j+64]]).to(device)
                with torch.amp.autocast("cuda"):
                    out = model(batch)
                probs.append(torch.softmax(out, 1)[:, 1].cpu().numpy())
        probs = np.concatenate(probs)
        results.append({
            "rel_path": rel_path,
            "ground_truth": meta["label"],
            "n_frames": len(frames),
            "n_faces": len(faces),
            "mean_fake_prob": float(probs.mean()),
            "max_fake_prob": float(probs.max()),
        })
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(items)}] {elapsed:.0f}s, ~{elapsed/(i+1):.2f}s/vid", flush=True)

        with open(args.output + ".tmp", "w") as f:
            json.dump(results, f)
    os.replace(args.output + ".tmp", args.output)

    print(f"\nProcessed {len(results)} videos, skipped {skipped}")

    for threshold in [0.5]:
        correct = sum(1 for r in results if (r["mean_fake_prob"] > threshold) == (r["ground_truth"] == "fake"))
        tp = sum(1 for r in results if r["mean_fake_prob"] > threshold and r["ground_truth"] == "fake")
        fp = sum(1 for r in results if r["mean_fake_prob"] > threshold and r["ground_truth"] == "real")
        tn = sum(1 for r in results if r["mean_fake_prob"] <= threshold and r["ground_truth"] == "real")
        fn = sum(1 for r in results if r["mean_fake_prob"] <= threshold and r["ground_truth"] == "fake")
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        print(f"Threshold {threshold}: acc={correct/len(results):.4f} ({correct}/{len(results)}) "
              f"p={precision:.4f} r={recall:.4f} f1={f1:.4f}")
        fake_ratio = sum(1 for r in results if r["ground_truth"] == "fake") / len(results)
        always_fake = fake_ratio
        print(f"  Always-fake baseline: {always_fake:.4f}")
        mean_real = np.mean([r["mean_fake_prob"] for r in results if r["ground_truth"] == "real"])
        mean_fake = np.mean([r["mean_fake_prob"] for r in results if r["ground_truth"] == "fake"])
        print(f"  Mean fake prob: real={mean_real:.4f} fake={mean_fake:.4f}")

    print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()