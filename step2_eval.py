"""CASR Step 2b: CNN baseline AUC-drop curves + codec-transfer matrix.

Runs the WildDeepfake-trained ResNet50 baseline on every compressed
condition corpus produced by step2_harness.py, computing per-condition
video-level AUC (and AUC drop vs clean), plus the codec-transfer
implications (train=clean/online distribution, test=any codec).

Usage:  python -u _casr/step2_eval.py --sample 600
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from sklearn.metrics import roc_auc_score

sys.path.insert(0, r"E:\Saad_audi deepfakes")
from src.utils.preprocessing import extract_frames, detect_face_color

ROOT = os.path.dirname(os.path.abspath(__file__))
CKPT = r"E:\Saad_audi deepfakes\checkpoints\cnn_baseline_best.pt"
GT_PATH = r"E:\Saad_audi deepfakes\data\lavdf_ground_truth.json"
COMPRESSED = os.path.join(ROOT, "data", "lavdf_compressed")
OUT = os.path.join(ROOT, "results", "step2_auc.json")

AV1_CRFS = [20, 30, 40]
CONDITIONS = ["clean"] + \
    [f"h264_crf{c}" for c in (18, 28, 38)] + \
    [f"hevc_crf{c}" for c in (18, 28, 38)] + \
    [f"av1_crf{c}" for c in AV1_CRFS]

MAX_FRAMES = 12


def load_model():
    device = torch.device("cuda")
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(CKPT, weights_only=True))
    model.to(device).eval()
    return model, device


def eval_condition(model, device, transform, videos, cond_dir, tag):
    """videos: list of (filename, gt_key) pairs."""
    rows = []
    t0 = time.time()
    for i, (vf, key) in enumerate(videos):
        vp = os.path.join(cond_dir, vf)
        if not os.path.exists(vp):
            continue
        frames, _, fps = extract_frames(vp, max_frames=MAX_FRAMES)
        faces = [f for f in (detect_face_color(f) for f in frames) if f is not None]
        if not faces:
            continue
        probs = []
        with torch.no_grad():
            for j in range(0, len(faces), 64):
                batch = torch.stack([transform(f) for f in faces[j:j + 64]]).to(device)
                with torch.amp.autocast("cuda"):
                    out = model(batch)
                probs.append(torch.softmax(out, 1)[:, 1].cpu().numpy())
        probs = float(np.concatenate(probs).mean())
        rows.append({"rel_path": key, "mean_fake_prob": probs})
        if (i + 1) % 100 == 0:
            print(f"    [{i + 1}/{len(videos)}] {time.time() - t0:.0f}s", flush=True)
    dump_dir = os.path.join(ROOT, "results", "step2_probs")
    os.makedirs(dump_dir, exist_ok=True)
    with open(os.path.join(dump_dir, tag + ".json"), "w") as f:
        json.dump(rows, f)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=600)
    ap.add_argument("--offset", type=int, default=0)
    args = ap.parse_args()

    with open(GT_PATH) as f:
        gt = json.load(f)
    keys = sorted(k for k, v in gt.items() if v["split"] == "test")
    keys = keys[args.offset:args.offset + args.sample]
    video_files = [k.split("/", 1)[-1] for k in keys]  # gt keys carry "test/" prefix
    videos = list(zip(video_files, keys))

    model, device = load_model()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    report = {"sample": args.sample, "conditions": {}, "aucs": {}}
    for cond in CONDITIONS:
        if cond == "clean":
            base = r"E:\Saad_audi deepfakes\data\LAV-DF\test"
        else:
            base = os.path.join(COMPRESSED, cond, "test")
        if not os.path.isdir(base):
            print(f"skip {cond}: not found", flush=True)
            continue
        print(f"Evaluating {cond} ...", flush=True)
        rows = eval_condition(model, device, transform, videos, base, cond)
        if not rows:
            print(f"  no rows for {cond}", flush=True)
            continue
        probs = np.array([r["mean_fake_prob"] for r in rows])
        ys = np.array([1 if gt[r["rel_path"]]["label"] == "fake" else 0
                       for r in rows])
        acc = float(((probs > 0.5) == ys).mean())
        auc = float(roc_auc_score(ys, probs))
        report["conditions"][cond] = {
            "n": int(len(rows)), "acc": round(acc, 4), "auc": round(auc, 4),
        }
        report["aucs"][cond] = round(auc, 4)
        print(f"  {cond}: acc={acc:.4f} auc={auc:.4f}", flush=True)

    base_auroc = report["aucs"].get("clean")
    if base_auroc:
        report["auc_drop_vs_clean"] = {
            c: round(base_auroc - a, 4) for c, a in report["aucs"].items() if c != "clean"
        }
    with open(OUT, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()