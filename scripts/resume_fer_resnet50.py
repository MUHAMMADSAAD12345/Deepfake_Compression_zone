import os, argparse
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
BEST_PATH = os.path.join(CHK_DIR, "fer_resnet50_best.pt")
os.makedirs(CHK_DIR, exist_ok=True)

device = torch.device("cuda")
print(f"Device: {device}")


class FER2013Dataset(Dataset):
    def __init__(self, df, usage, transform=None):
        self.df = df[df["Usage"].str.strip() == usage].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        pixels = self.df.loc[idx, "pixels"]
        pixels = np.array(pixels.split(), dtype=np.uint8).reshape(48, 48)
        img = Image.fromarray(pixels).convert("RGB").resize((224, 224))
        label = int(self.df.loc[idx, "emotion"])
        if self.transform:
            img = self.transform(img)
        return img, label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    args = parser.parse_args()

    df = pd.read_csv(DATA_PATH)
    print(f"Total samples: {len(df)}")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = FER2013Dataset(df, "Training", train_transform)
    val_ds = FER2013Dataset(df, "PrivateTest", test_transform)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)

    model = models.resnet50(weights=None, num_classes=7).to(device)

    # Load existing checkpoint if available
    start_epoch = 1
    if os.path.exists(BEST_PATH):
        model.load_state_dict(torch.load(BEST_PATH, map_location=device))
        print(f"Loaded existing checkpoint {BEST_PATH}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = float("-inf")

    # Log file
    log_file = os.path.join(os.path.dirname(DATA_PATH), "fer_resnet50_v2_log.txt")

    for epoch in range(start_epoch, args.epochs + 1):
        # Train
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            train_correct += preds.eq(labels).sum().item()
            train_total += inputs.size(0)
        train_acc = train_correct / train_total

        # Val
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

        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), BEST_PATH)
            marker = " *"

        msg = (f"Epoch {epoch:3d}/{args.epochs} | "
               f"Train Loss: {train_loss/train_total:.4f} Acc: {train_acc:.4f} | "
               f"Val Loss: {val_loss/val_total:.4f} Acc: {val_acc:.4f} | "
               f"LR: {lr_now:.6f}{marker}")
        print(msg)
        with open(log_file, "a") as f:
            f.write(msg + "\n")

    print(f"\nBest val acc: {best_val_acc:.4f}")

    # Final evaluation
    model.load_state_dict(torch.load(BEST_PATH, map_location=device))
    model.eval()
    for usage in ["Training", "PublicTest", "PrivateTest"]:
        ds = FER2013Dataset(df, usage, test_transform)
        dl = DataLoader(ds, batch_size=args.batch_size, num_workers=0)
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in dl:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum().item()
                total += inputs.size(0)
        print(f"  {usage}: {correct}/{total} = {correct/total:.4f}")


if __name__ == "__main__":
    main()
