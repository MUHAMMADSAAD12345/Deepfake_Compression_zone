"""Celeb-DF v2 evaluation for CASR models."""
import os
import json
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.metrics import roc_auc_score
from concurrent.futures import ProcessPoolExecutor

# Celeb-DF v2 paths
CELEBDF_ROOT = r"E:\Datasets\Celeb-DF-v2"
ROOT = os.path.dirname(os.path.abspath(__file__))
FACES_CELEBDF = os.path.join(ROOT, "data", "celebdf_faces")
CKPT = os.path.join(ROOT, "checkpoints", "ffpp_cnn_best.pt")
OUT_JSON = os.path.join(ROOT, "results", "celebdf_eval.json")
PROBS_DIR = os.path.join(ROOT, "results", "celebdf_probs")
os.makedirs(PROBS_DIR, exist_ok=True)

from train_ffpp_cnn import MAX_FACES

# Model
def load_model(ckpt_path, device="cuda"):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device).eval()
    return model

# Face extraction (self-contained for multiprocessing)
def extract_faces_video(path):
    """Extract face crops from video using haarcascade (self-contained for multiprocessing)."""
    face_cascade = cv2.CascadeClassifier(
        r"E:\Saad_audi deepfakes\haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        cap.release()
        return None
    stride = max(1, n // 60)
    sel = set(np.linspace(0, n - 1, min(MAX_FACES, n), dtype=int))
    out = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in sel:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            if len(faces) > 0:
                x, y, w, h = faces[0]
                x, y = max(0, x), max(0, y)
                face = frame[y:y+h, x:x+w]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                out.append(face)
                if len(out) >= MAX_FACES:
                    break
        frame_idx += 1
    cap.release()
    return out if out else None


def extract_faces_split(split_name, video_list, workers=8):
    """Extract faces for a list of videos."""
    # Determine subdir based on split_name
    subdir = "real" if "real" in split_name else "fake"
    dst_dir = os.path.join(FACES_CELEBDF, subdir)
    os.makedirs(dst_dir, exist_ok=True)
    
    jobs = []
    for vpath in video_list:
        name = os.path.splitext(os.path.basename(vpath))[0]
        out = os.path.join(dst_dir, f"{name}.npz")
        if os.path.exists(out):
            continue
        jobs.append((vpath, out))
    print(f"[{split_name}] {len(jobs)} videos pending", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        done = 0
        for (src, out), faces in zip(jobs, ex.map(extract_faces_video, [j[0] for j in jobs])):
            if faces:
                np.savez_compressed(out, imgs=np.stack(faces))
            done += 1
            if done % 100 == 0:
                print(f"[{split_name}] {done}/{len(jobs)}", flush=True)
    print(f"[{split_name}] done", flush=True)


def load_fer_model():
    from models.fer_model import FERModel
    return FERModel(
        model_path=r"E:\Saad_audi deepfakes\checkpoints\fer_resnet50_best.pt",
        model_type="resnet50")


def eval_model(model, face_dir, label):
    """Evaluate model on all npz files in a directory."""
    probs = []
    labels = []
    device = next(model.parameters()).device
    for f in sorted(glob.glob(os.path.join(face_dir, "*.npz"))):
        a = np.load(f)["imgs"]
        if len(a) == 0:
            continue
        if len(a) > 30:
            idx = np.linspace(0, len(a)-1, 30, dtype=int)
            a = a[idx]
        x = torch.from_numpy(a).permute(0, 3, 1, 2).float() / 255.0
        x = transforms.Resize((224, 224))(x)
        x = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(x)
        x = x.to(device)
        with torch.no_grad():
            p = torch.sigmoid(model(x)).cpu().numpy()[:, 0]
        probs.append(p.max())
        labels.append(label)
    return np.array(probs), np.array(labels)


def main():
    # Load models
    device = "cuda"
    cnn = load_model(CKPT, device)
    
    # Collect video paths
    real_videos = []
    for d in ["Celeb-real", "YouTube-real"]:
        real_videos += glob.glob(os.path.join(CELEBDF_ROOT, d, "*.mp4"))
    fake_videos = glob.glob(os.path.join(CELEBDF_ROOT, "Celeb-synthesis", "*.mp4"))
    
    print(f"Real videos: {len(real_videos)}")
    print(f"Fake videos: {len(fake_videos)}")
    
    # Extract faces (resumable)
    print("Extracting faces...")
    extract_faces_split("celebdf_real", real_videos, workers=8)
    extract_faces_split("celebdf_fake", fake_videos, workers=8)
    
    # Evaluate CNN
    print("Evaluating CNN...")
    real_probs, real_labels = eval_model(cnn, os.path.join(FACES_CELEBDF, "real"), 0)
    fake_probs, fake_labels = eval_model(cnn, os.path.join(FACES_CELEBDF, "fake"), 1)
    
    y = np.concatenate([real_labels, fake_labels])
    p = np.concatenate([real_probs, fake_probs])
    cnn_auc = roc_auc_score(y, p)
    print(f"CNN AUC: {cnn_auc:.4f}")
    
    # Save results
    results = {
        "cnn_auc": float(cnn_auc),
        "n_real": len(real_probs),
        "n_fake": len(fake_probs),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {OUT_JSON}")


if __name__ == "__main__":
    main()