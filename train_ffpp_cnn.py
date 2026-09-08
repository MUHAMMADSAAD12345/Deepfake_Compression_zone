"""CASR Step 2b-v2: FF++ c23 CNN + codec-transfer eval.

Modes:
  --extract : face crops (haarcascade) for train (idx 0-719) + val (720-859)
              per group (real/Deepfakes/Face2Face/FaceSwap/NeuralTextures).
  (default) : train ResNet50 fakeness classifier on extracted crops.
  --eval    : per-video fake-prob + AUC for each given condition dir
              (clean = FF++ source dirs; else data/ffpp_compressed/<cond>/).

Outputs:
  data/ffpp_faces/<split>/<group>/<video>.npz
  checkpoints/ffpp_cnn_best.pt
  results/step2b_eval.json, results/step2b_probs/<cond>.json, results/step2b_train.log
"""
import argparse
import glob
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, r"E:\Saad_audi deepfakes\src")
import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

FFPP_ROOT = r"E:\Datasets\FaceForensicsC23\FaceForensics++_C23"
ROOT = os.path.dirname(os.path.abspath(__file__))
FACES = os.path.join(ROOT, "data", "ffpp_faces")
FACES_EVAL = os.path.join(ROOT, "data", "ffpp_faces_eval")
CKPT = os.path.join(ROOT, "checkpoints", "ffpp_cnn_best.pt")
EVAL_JSON = os.path.join(ROOT, "results", "step2b_eval.json")
PROBS_DIR = os.path.join(ROOT, "results", "step2b_probs")
LOG = os.path.join(ROOT, "results", "step2b_train.log")
os.makedirs(os.path.join(ROOT, "checkpoints"), exist_ok=True)
os.makedirs(PROBS_DIR, exist_ok=True)

GROUPS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "real"]
SPLITS = {"train": (0, 719), "val": (720, 859)}
MAX_FACES = 30


def list_videos(group, split):
    d = os.path.join(FFPP_ROOT, "real") if group == "real" else \
        os.path.join(FFPP_ROOT, "fake", group)
    lo, hi = SPLITS[split]
    names = sorted(os.listdir(d))[lo:hi + 1]
    return [os.path.join(d, n) for n in names]


FACE_CASCADE = None


def detect_face_rgb(image_rgb):
    global FACE_CASCADE
    if FACE_CASCADE is None:
        FACE_CASCADE = cv2.CascadeClassifier(
            r"E:\Saad_audi deepfakes\haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1,
                                          minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    x, y = max(0, x), max(0, y)
    crop = image_rgb[y:y + h, x:x + w]
    return cv2.resize(crop, (224, 224))


def extract_one(path):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        cap.release()
        return None
    stride = max(1, n // 40)
    step = min(12, n)
    sel = set(np.linspace(0, n - 1, step, dtype=int))
    out = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in sel:
            try:
                crop = detect_face_rgb(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            except Exception:  # noqa: BLE001  (cv2 OOM/alloc hiccup)
                crop = None
            if crop is not None:
                out.append(crop)
                if len(out) >= MAX_FACES:
                    break
        frame_idx += 1
    cap.release()
    return out


def ensure_faces(jobs, workers, min_frames=None):
    """jobs: [(src_video, dst_npz)]; skip if dst exists with >= min_frames."""
    todo = []
    for s, d in jobs:
        if os.path.exists(d):
            if min_frames:
                try:
                    if np.load(d)["imgs"].shape[0] < min_frames:
                        os.remove(d)
                    else:
                        continue
                except Exception:  # noqa: BLE001  corrupt cache: redo
                    os.remove(d)
            else:
                continue
        todo.append((s, d))
    print(f"[faces] {len(jobs)} videos, {len(todo)} pending", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        done = 0
        for (src, out), faces in zip(todo, ex.map(extract_one, [j[0] for j in todo])):
            if faces:
                try:
                    np.savez_compressed(out, imgs=np.stack(faces))
                except Exception as e:  # noqa: BLE001
                    print(f"[faces] SAVE-ERR {out}: {str(e)[:120]}", flush=True)
            done += 1
            if done % 100 == 0:
                print(f"[faces] {done}/{len(todo)}", flush=True)
    print("[faces] done", flush=True)


def do_extract(split, workers):
    jobs = []
    for g in GROUPS:
        dst = os.path.join(FACES, split, g)
        os.makedirs(dst, exist_ok=True)
        for src in list_videos(g, split):
            name = os.path.splitext(os.path.basename(src))[0]
            jobs.append((src, os.path.join(dst, f"{name}.npz")))
    ensure_faces(jobs, workers, min_frames=MAX_FACES)


def do_extract_eval(conds, workers):
    for cond in conds:
        if cond == "clean":
            root = FFPP_ROOT
        else:
            root = os.path.join(ROOT, "data", "ffpp_compressed", cond)
        jobs = []
        for g in GROUPS:
            if cond == "clean":
                src = os.path.join(root, "real") if g == "real" \
                    else os.path.join(root, "fake", g)
            else:
                src = os.path.join(root, g)
            if not os.path.isdir(src):
                continue
            names = sorted(os.listdir(src))
            if cond == "clean":
                names = names[len(names) - 140:]
            dst = os.path.join(FACES_EVAL, cond, g)
            os.makedirs(dst, exist_ok=True)
            for f in names:
                jobs.append((os.path.join(src, f),
                             os.path.join(dst, os.path.splitext(f)[0] + ".npz")))
        ensure_faces(jobs, workers)


class FaceDS(Dataset):
    """Preloads all face npz (train/val split) into one uint8 array.
    __getitem__ returns a random frame of a random video."""

    def __init__(self, split, compress_aug=False):
        self.label_map = {g: 0 if g == "real" else 1 for g in GROUPS}
        self.items = []
        self.compress_aug = compress_aug
        self.train = (split == "train")
        self.counts, self.offsets, blocks = [], [0], []
        self.lbl = []
        self.vids = []
        for g in GROUPS:
            d = os.path.join(FACES, split, g)
            for f in glob.glob(os.path.join(d, "*.npz")):
                a = np.load(f)["imgs"].astype(np.uint8)
                blocks.append(a)
                self.counts.append(len(a))
                self.offsets.append(self.offsets[-1] + len(a))
                self.lbl.append(self.label_map[g])
                self.vids.append(f)
        self.all = np.concatenate(blocks) if blocks else np.zeros((1, 1, 1, 3),
                                                                  np.uint8)
        self.n_vid = len(self.lbl)

    def __len__(self):
        return self.n_vid

    def __getitem__(self, i):
        if self.n_vid == 0:
            return torch.zeros(3, 224, 224), torch.tensor(0.0)
        v = random.randrange(self.n_vid)
        f = random.randrange(self.counts[v])
        x = torch.from_numpy(
            self.all[self.offsets[v] + f].copy()).permute(2, 0, 1).float() / 255.0
        if self.compress_aug:
            s = random.randint(192, 224)
            x = transforms.Resize((s, s))(x)
            x = transforms.RandomHorizontalFlip()(x)
            x = transforms.RandomErasing(p=0.25, scale=(0.02, 0.1))(x)
            # Compression augmentation (JPEG artifacts on face crop)
            if self.compress_aug:
                q = random.randint(30, 95)
                # Encode-decode via OpenCV JPEG to simulate I-frame compression
                img_np = (x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                _, enc = cv2.imencode('.jpg', cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR),
                                      [cv2.IMWRITE_JPEG_QUALITY, q])
                dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
                if dec is not None:
                    x = torch.from_numpy(cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0
        x = transforms.Resize((224, 224))(x)
        x = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(x)
        return x, torch.tensor(self.lbl[v], dtype=torch.float32)


def video_score(model, all_imgs, counts, offsets, device="cuda"):
    """max-frame sigmoid per video."""
    model.eval()
    out = np.zeros(len(counts))
    with torch.no_grad():
        for v in range(len(counts)):
            a = all_imgs[offsets[v]:offsets[v] + counts[v]]
            x = torch.from_numpy(a.copy()).permute(0, 3, 1, 2).float() / 255.0
            x = transforms.Resize((224, 224))(x)
            x = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])(x)
            p = torch.sigmoid(model(x.to(device))).cpu().numpy()[:, 0]
            out[v] = p.max()
    return out


def train(args):
    import torch.optim as optim
    from torch.optim.lr_scheduler import CosineAnnealingLR
    tr = FaceDS("train", compress_aug=args.compress_aug)
    va = FaceDS("val")
    tr_frames = int(tr.offsets[-1])
    va_frames = int(va.offsets[-1])
    print(f"[train] videos train={tr.n_vid} val={va.n_vid} "
          f"frames {tr_frames}/{va_frames} compress_aug={args.compress_aug}", flush=True)
    model = models.resnet50(weights=None)
    # First replace fc with our 1-output head (matches fine-tuned checkpoints)
    model.fc = nn.Linear(model.fc.in_features, 1)
    # If resuming, skip ImageNet weights (already in checkpoint)
    if not (args.resume and os.path.exists(args.resume)):
        pth = r"C:\Users\i232084\.cache\torch\hub\checkpoints\resnet50-0676ba61.pth"
        if os.path.exists(pth):
            model.load_state_dict(torch.load(pth, map_location="cpu"), strict=False)
            print("[train] loaded cached ImageNet weights", flush=True)
    # Resume from checkpoint if specified
    if args.resume and os.path.exists(args.resume):
        model.load_state_dict(torch.load(args.resume, map_location="cpu"))
        print(f"[train] resumed from {args.resume}", flush=True)
    model = model.to(args.gpu)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    sched = CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.BCEWithLogitsLoss()
    dl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=0,
                    drop_last=True)
    best = -1.0
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for x, y in dl:
            x, y = x.to(args.gpu), y.to(args.gpu)
            opt.zero_grad()
            loss = lossf(model(x).squeeze(1), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
        sched.step()
        vp = video_score(model, va.all, va.counts, va.offsets)
        auc = roc_auc_score(va.lbl, vp)
        line = f"[train] ep {ep} loss {tot / tr_frames:.4f} valAUC {auc:.4f}"
        print(line, flush=True)
        with open(LOG, "a") as f:
            f.write(line + "\n")
        if auc > best:
            best = auc
            torch.save(model.state_dict(), CKPT)
            print(f"[train] saved {CKPT} (val AUC {auc:.4f})", flush=True)
    print(f"[train] done best val AUC {best:.4f}", flush=True)


def eval_cond(model, cond):
    root = os.path.join(FACES_EVAL, cond)
    counts, offsets, lbl, names, blocks = [], [0], [], [], []
    for g in GROUPS:
        src = os.path.join(root, g)
        if not os.path.isdir(src):
            continue
        for f in sorted(os.listdir(src)):
            if f.endswith(".npz"):
                a = np.load(os.path.join(src, f))["imgs"].astype(np.uint8)
                blocks.append(a)
                counts.append(len(a))
                offsets.append(offsets[-1] + len(a))
                lbl.append(0 if g == "real" else 1)
                names.append(os.path.join(g, f))
    all_imgs = np.concatenate(blocks) if blocks else np.zeros((1, 1, 1, 3),
                                                              np.uint8)
    probs = video_score(model, all_imgs, counts, offsets)
    auc = roc_auc_score(lbl, probs)
    per_g = {}
    real_idx = [i for i, n in enumerate(names) if n.startswith("real" + os.sep)]
    for g in GROUPS:
        if g == "real":
            continue
        idx = [i for i, n in enumerate(names) if n.startswith(g + os.sep)]
        if idx and real_idx:
            y = [lbl[i] for i in idx] + [lbl[i] for i in real_idx]
            p = [probs[i] for i in idx] + [probs[i] for i in real_idx]
            per_g[g] = float(roc_auc_score(y, p))
    return {"auc": float(auc), "n": len(names), "groups": per_g,
            "prob": {n: float(p) for n, p in zip(names, probs)}}


def do_eval(model, conds):
    rows, probs = {}, {}
    for c in conds:
        r = eval_cond(model, c)
        rows[c] = {"auc": r["auc"], "n": r["n"], "groups": r["groups"]}
        probs[c] = r["prob"]
        print(f"[eval] {c}: AUC {rows[c]['auc']:.4f} (n={r['n']})", flush=True)
    with open(EVAL_JSON, "w") as f:
        json.dump(rows, f, indent=1)
    for c, p in probs.items():
        with open(os.path.join(PROBS_DIR, f"{c}.json"), "w") as f:
            json.dump(p, f)
    print(f"[eval] wrote {EVAL_JSON} + probs in {PROBS_DIR}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--eval-extract", nargs="+",
                    help="extract eval-face cache for condition dirs")
    ap.add_argument("--split", default="all", choices=["all", "train", "val"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--gpu", default="cuda")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--compress-aug", action="store_true",
                    help="apply random JPEG compression (Q30-95) augmentation during training")
    ap.add_argument("--resume", type=str, default="",
                    help="checkpoint path to resume training from")
    ap.add_argument("--ckpt")
    ap.add_argument("--eval", nargs="+",
                    help="condition dir names under data/ffpp_compressed; "
                         "'clean' = original FF++ videos")
    args = ap.parse_args()
    if args.extract:
        for s in ("train", "val"):
            if args.split in ("all", s):
                do_extract(s, args.workers)
        return
    if args.eval_extract:
        do_extract_eval(args.eval_extract, args.workers)
        return
    if args.eval:
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 1)
        model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
        model = model.to(args.gpu).eval()
        do_eval(model, args.eval)
        return
    train(args)


if __name__ == "__main__":
    main()