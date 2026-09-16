import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from src.models.fer_model import VGG19FER, EMOTIONS

class FER2013Dataset(Dataset):
    def __init__(self, pixels, emotions):
        self.x = torch.from_numpy(np.array(pixels, dtype=np.float32).reshape(-1, 1, 48, 48) / 255.0)
        self.y = torch.tensor(emotions, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def load_fer2013(data_path):
    csv_file = os.path.join(data_path, "fer2013.csv")
    import csv
    pixels_train, emotions_train = [], []
    pixels_val, emotions_val = [], []
    pixels_test, emotions_test = [], []
    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            usage = row[2].strip()
            emotion = int(row[0])
            pixel_data = np.fromstring(row[1], dtype=int, sep=" ")
            if usage == "Training":
                emotions_train.append(emotion); pixels_train.append(pixel_data)
            elif usage == "PublicTest":
                emotions_val.append(emotion); pixels_val.append(pixel_data)
            elif usage == "PrivateTest":
                emotions_test.append(emotion); pixels_test.append(pixel_data)

    return (FER2013Dataset(pixels_train, emotions_train),
            FER2013Dataset(pixels_val, emotions_val),
            FER2013Dataset(pixels_test, emotions_test))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/fer2013")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--output", default="checkpoints/fer2013_vgg19.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds, val_ds, test_ds = load_fer2013(args.data)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              pin_memory=(device.type == "cuda"), num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            pin_memory=(device.type == "cuda"), num_workers=2)

    model = VGG19FER().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, min_lr=1e-6)
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
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                outputs = model(x)
                loss = criterion(outputs, y)
                val_loss += loss.item() * x.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        scheduler.step(val_loss / val_total)

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

    model.load_state_dict(torch.load(args.output, map_location=device, weights_only=True))
    model.eval()
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, pin_memory=(device.type == "cuda"))
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            outputs = model(x)
            _, preds = torch.max(outputs, 1)
            test_correct += (preds == y).sum().item()
            test_total += y.size(0)
    test_acc = test_correct / test_total
    print(f"PrivateTest accuracy: {test_acc:.4f}")

    torch.save(model.state_dict(), args.output)
    print(f"Final weights saved to {args.output}")


if __name__ == "__main__":
    main()
