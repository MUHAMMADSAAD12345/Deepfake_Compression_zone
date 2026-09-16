import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import librosa
from src.models.ser_model import SERMLP, extract_audio_features, RAVDESS_EMOTIONS

EMOTION_MAP = {e: i for i, e in enumerate(RAVDESS_EMOTIONS)}

class RAVDESSDataset(Dataset):
    def __init__(self, features, labels):
        self.x = torch.from_numpy(features).float()
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def load_ravdess(data_path):
    features_list, labels_list = [], []
    if not os.path.exists(data_path):
        print(f"RAVDESS not found at {data_path}")
        return None, None, None, None
    for root, dirs, files in os.walk(data_path):
        for fname in files:
            if not fname.endswith(".wav"):
                continue
            parts = fname.split("-")
            if len(parts) < 3:
                continue
            emotion_code = int(parts[2])
            emotion_map = {
                1: "neutral", 2: "calm", 3: "happy", 4: "sad",
                5: "angry", 6: "fearful", 7: "disgust", 8: "surprised",
            }
            emotion = emotion_map.get(emotion_code)
            if emotion not in EMOTION_MAP:
                continue
            audio, sr = librosa.load(os.path.join(root, fname), sr=16000)
            features = extract_audio_features(audio, sr)
            features_list.append(features)
            labels_list.append(EMOTION_MAP[emotion])
    if not features_list:
        print("No audio files found")
        return None, None, None, None
    X = np.array(features_list)
    y = np.array(labels_list)
    n = len(X)
    perm = np.random.permutation(n)
    X, y = X[perm], y[perm]
    split = int(n * 0.8)
    return (RAVDESSDataset(X[:split], y[:split]),
            RAVDESSDataset(X[split:], y[split:]))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/ravdess")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--output", default="checkpoints/ser_weights.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = load_ravdess(args.data)
    if data[0] is None:
        return
    train_ds, test_ds = data

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              pin_memory=(device.type == "cuda"), num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             pin_memory=(device.type == "cuda"), num_workers=2)

    model = SERMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    patience_counter = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == y).sum().item()
            train_total += y.size(0)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                outputs = model(x)
                loss = criterion(outputs, y)
                val_loss += loss.item() * x.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        print(f"Epoch {epoch:3d}/{args.epochs} | Train Loss: {train_loss/train_total:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss/val_total:.4f} Acc: {val_acc:.4f}", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), args.output)
            print(f"  -> Saved best model (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= 30:
                print(f"Early stopping at epoch {epoch}")
                break

    torch.save(model.state_dict(), args.output)
    print(f"Weights saved to {args.output}")


if __name__ == "__main__":
    main()
