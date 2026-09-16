"""Foreground diagnostic: mirror training DataLoader exactly, print per-batch timing."""
import time, torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

mp.freeze_support()

def main():
    DATA_DIR = "E:\\Saad_audi deepfakes\\data\\WildDeepfake_resized"
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    full_train = datasets.ImageFolder(DATA_DIR + "\\train", transform=transform)
    val_len = int(len(full_train) * 0.05)
    train_len = len(full_train) - val_len
    train_ds, val_ds = random_split(full_train, [train_len, val_len], generator=torch.Generator().manual_seed(42))
    print(f"Train: {len(train_ds)}", flush=True)

    loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=8, persistent_workers=True)
    t0 = time.time()
    for i, (x, y) in enumerate(loader):
        print(f"batch {i}: {x.shape} ({time.time()-t0:.1f}s)", flush=True)
        if i >= 5:
            break
    print("DONE", flush=True)

if __name__ == "__main__":
    main()