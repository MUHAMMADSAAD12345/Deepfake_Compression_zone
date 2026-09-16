import sys, os, json, time, numpy as np

os.environ["PATH"] = (
    r"E:\audi deepfakes\venv\Scripts;" + os.environ.get("PATH", "")
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.models.fer_model import FERModel
from src.models.ser_model import SERModel
from src.utils.preprocessing import preprocess_video
from src.reasoning.intra_modality import intra_modality_reasoning
from src.reasoning.inter_modality import inter_modality_reasoning
from src.reasoning.classifier import (
    compute_fake_probabilities,
    classify_modality,
    voting_classifier,
)

device = torch.device("cuda")
print(f"Device: {device}")

fer = FERModel(
    model_path=r"E:\audi deepfakes\checkpoints\fer_resnet50_best.pt",
    device=device,
    model_type="resnet50",
)
print(f"FER: {fer.model_type}")

ser = SERModel(
    model_path=r"E:\audi deepfakes\checkpoints\ser_weights.pt",
    device=device,
)

gt = json.load(open(r"E:\audi deepfakes\data\lavdf_ground_truth.json"))
items = [(k, v) for k, v in gt.items() if v["split"] == "test"][:10]

correct = 0
total = 0
for rel_path, meta in items:
    video_path = os.path.join(r"E:\audi deepfakes\data\LAV-DF", rel_path)
    if not os.path.exists(video_path):
        continue
    expected = meta["label"]
    try:
        data = preprocess_video(video_path)
        faces = data["face_images"]
        segments = data["audio_segments"]
        sync_map = data["sync_map"]
        if not faces or not segments:
            continue
        ve, _ = fer.predict_batch(faces)
        ae, _ = ser.predict_batch(segments)
        vt_r, vt_f = intra_modality_reasoning(ve)
        at_r, at_f = intra_modality_reasoning(ae)
        im_r, im_f = inter_modality_reasoning(ve, ae, sync_map)
        prob = compute_fake_probabilities(vt_r, at_r, im_r)
        mod_r = classify_modality(prob, threshold=0.47)
        pred = voting_classifier(mod_r)
        mark = "OK" if pred == expected else "X "
        print(
            f"  [{mark}] {rel_path} gt={expected} pred={pred} "
            f"VT={prob['VT']['probability']:.0%} "
            f"AT={prob['AT']['probability']:.0%} "
            f"IM={prob['IM']['probability']:.0%}"
        )
        if pred == expected:
            correct += 1
        total += 1
    except Exception as e:
        print(f"  ERROR {rel_path}: {e}")
        continue

print(f"\nResult: {correct}/{total} = {correct/total:.2%}")
