"""Evaluate trained CNN baseline on WLD videos (frame-level, video-majority aggregation)."""
import os, sys
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models

sys.path.insert(0, "E:\\Saad_audi deepfakes")
from src.utils.preprocessing import extract_frames, detect_face_color

WLD_DIR = "E:\\Saad_audi deepfakes\\data\\WLD"
CKPT = "E:\\Saad_audi deepfakes\\checkpoints\\cnn_baseline_best.pt"

def main():
    torch.multiprocessing.freeze_support()
    device = torch.device("cuda")
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(CKPT, weights_only=True))
    model.to(device).eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    videos = sorted([f for f in os.listdir(WLD_DIR) if f.endswith(".mp4")])
    for v in videos:
        path = os.path.join(WLD_DIR, v)
        frames, ts, fps = extract_frames(path)
        faces = [detect_face_color(f) for f in frames]
        faces = [f for f in faces if f is not None]
        if not faces:
            print(f"{v}: NO FACES ({len(frames)} frames)")
            continue
        probs = []
        with torch.no_grad():
            for i in range(0, len(faces), 32):
                batch = torch.stack([transform(f) for f in faces[i:i+32]]).to(device)
                with torch.amp.autocast("cuda"):
                    out = model(batch)
                probs.append(torch.softmax(out, 1)[:, 1].cpu().numpy())
        probs = np.concatenate(probs)
        mean_fake = float(probs.mean())
        pred = "FAKE" if mean_fake > 0.5 else "REAL"
        print(f"{v}: {len(faces)} faces, mean_fake_prob={mean_fake:.4f}, pred={pred}")

if __name__ == "__main__":
    main()