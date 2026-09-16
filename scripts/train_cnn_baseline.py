"""Train frame-level ResNet50 CNN baseline on WildDeepfake."""
import os, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models


def main():
    DATA_DIR = "D:\\WildDeepfake_resized"
    BATCH_SIZE = 128
    EPOCHS = 5
    LR = 1e-4
    VAL_SPLIT = 0.05
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT = "E:\\Saad_audi deepfakes\\checkpoints\\cnn_baseline_best.pt"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_train = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=transform)
    val_len = int(len(full_train) * VAL_SPLIT)
    train_len = len(full_train) - val_len
    train_ds, val_ds = random_split(full_train, [train_len, val_len], generator=torch.Generator().manual_seed(42))
    test_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=transform)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, persistent_workers=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, persistent_workers=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, persistent_workers=False)

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

    best_val_acc = 0.0
    t_start = time.time()
    for epoch in range(EPOCHS):
        model.train()
        train_correct, train_total = 0, 0
        t0 = time.time()
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            if (batch_idx + 1) % 500 == 0:
                print(f"  Epoch {epoch+1} batch {batch_idx+1}/{len(train_loader)}: acc={train_correct/train_total:.4f} ({time.time()-t0:.0f}s)", flush=True)
        train_acc = train_correct / train_total

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                    outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / val_total
        scheduler.step()

        elapsed = time.time() - t_start
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Elapsed: {elapsed:.0f}s", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), OUTPUT)
            print(f"  -> Saved best model (val_acc={val_acc:.4f})", flush=True)

    model.load_state_dict(torch.load(OUTPUT, weights_only=True))
    model.eval()
    test_correct, test_total = 0, 0
    t0 = time.time()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)
    test_acc = test_correct / test_total
    print(f"\nTest Accuracy: {test_acc:.4f} ({test_correct}/{test_total}) in {time.time()-t0:.0f}s", flush=True)
    print(f"Total time: {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
