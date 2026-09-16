import os, argparse, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "fer2013", "fer2013.csv")
CHK_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints")
os.makedirs(CHK_DIR, exist_ok=True)

device = torch.device("cuda")
print(f"Device: {device}")

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


class FER2013Dataset(Dataset):
    def __init__(self, df, usage, transform=None):
        self.df = df[df["Usage"].str.strip() == usage].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        pixels = self.df.loc[idx, "pixels"]
        pixels = np.array(pixels.split(), dtype=np.uint8).reshape(48, 48)
        img = Image.fromarray(pixels).convert("RGB").resize((224, 224))  # ResNet50 needs 224x224 RGB
        label = int(self.df.loc[idx, "emotion"])
        if self.transform:
            img = self.transform(img)
        return img, label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    args = parser.parse_args()

    df = pd.read_csv(DATA_PATH)
    print(f"Total samples: {len(df)}")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = FER2013Dataset(df, "Training", train_transform)
    val_ds = FER2013Dataset(df, "PublicTest", test_transform)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, pin_memory=True, num_workers=2)

    model = models.resnet50(weights=None, num_classes=7)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    best_val_acc = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            train_correct += preds.eq(labels).sum().item()
            train_total += inputs.size(0)
        train_acc = train_correct / train_total

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = outputs.max(1)
                val_correct += preds.eq(labels).sum().item()
                val_total += inputs.size(0)
        val_acc = val_correct / val_total

        scheduler.step(val_acc)
        lr_now = optimizer.param_groups[0]["lr"]

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(CHK_DIR, "fer_resnet50_best.pt"))
            marker = " *"

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss/train_total:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss/val_total:.4f} Acc: {val_acc:.4f} | "
              f"LR: {lr_now:.6f}{marker}")

    print(f"Best val acc: {best_val_acc:.4f}")

    # Final evaluation on all three splits
    model.load_state_dict(torch.load(os.path.join(CHK_DIR, "fer_resnet50_best.pt")))
    model.eval()
    for usage in ["Training", "PublicTest", "PrivateTest"]:
        ds = FER2013Dataset(df, usage, test_transform)
        dl = DataLoader(ds, batch_size=args.batch_size, pin_memory=True, num_workers=2)
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in dl:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum().item()
                total += inputs.size(0)
        print(f"  {usage}: {correct}/{total} = {correct/total:.4f}")

    # Compute full-dataset accuracy (training + test combined)
    full_ds = FER2013Dataset(df, "Training", test_transform)
    full_loader = DataLoader(full_ds, batch_size=args.batch_size, pin_memory=True, num_workers=2)
    # Also include public+private test
    for usage in ["PublicTest", "PrivateTest"]:
        ds = FER2013Dataset(df, usage, test_transform)
        dl = DataLoader(ds, batch_size=args.batch_size, pin_memory=True, num_workers=2)
        full_loader = full_loader.dataset + dl.dataset  # won't work, skip
    correct_all, total_all = 0, 0
    for usage in ["Training", "PublicTest", "PrivateTest"]:
        ds = FER2013Dataset(df, usage, test_transform)
        dl = DataLoader(ds, batch_size=args.batch_size, pin_memory=True, num_workers=2)
        with torch.no_grad():
            for inputs, labels in dl:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = outputs.max(1)
                correct_all += preds.eq(labels).sum().item()
                total_all += inputs.size(0)
    print(f"Full dataset accuracy: {correct_all}/{total_all} = {correct_all/total_all:.4f}")


if __name__ == "__main__":
    main()
