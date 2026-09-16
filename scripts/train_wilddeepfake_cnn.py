"""Train a simple CNN baseline on WildDeepfake face images (real vs fake)"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "E:\\Saad_audi deepfakes")
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

WD_DIR = "E:\\Saad_audi deepfakes\\data\\WildDeepfake_extracted"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

class WildDeepfakeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        for label_name, label in [("real", 0), ("fake", 1)]:
            label_dir = os.path.join(root_dir, label_name)
            if not os.path.exists(label_dir):
                continue
            for seq in os.listdir(label_dir):
                seq_dir = os.path.join(label_dir, seq)
                if not os.path.isdir(seq_dir):
                    continue
                for fname in os.listdir(seq_dir):
                    if fname.endswith(".png"):
                        self.samples.append((os.path.join(seq_dir, fname), label))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Load data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
dataset = WildDeepfakeDataset(WD_DIR, transform=transform)
print(f"Total samples: {len(dataset)}")
n_real = sum(1 for _, l in dataset.samples if l == 0)
n_fake = sum(1 for _, l in dataset.samples if l == 1)
print(f"  Real: {n_real}, Fake: {n_fake}")

if len(dataset) < 50:
    print("Too few samples, skipping training")
    sys.exit(0)

# Train/test split
from sklearn.model_selection import train_test_split
indices = list(range(len(dataset)))
train_idx, test_idx = train_test_split(indices, test_size=0.3, random_state=42, stratify=[l for _, l in dataset.samples])
train_ds = torch.utils.data.Subset(dataset, train_idx)
test_ds = torch.utils.data.Subset(dataset, test_idx)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=32, num_workers=0)

model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\nTraining CNN baseline...")
for epoch in range(20):
    model.train()
    train_loss, correct, total = 0, 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    train_acc = correct / total

    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            val_correct += preds.eq(labels).sum().item()
            val_total += labels.size(0)
    val_acc = val_correct / val_total
    print(f"  Epoch {epoch+1:2d}/20 | Train: {train_acc:.4f} | Val: {val_acc:.4f}")

print(f"\nBaseline CNN on WildDeepfake (subset): {val_acc*100:.1f}% accuracy")
print(f"(Always-fake baseline: {n_fake/len(dataset)*100:.1f}%)")
