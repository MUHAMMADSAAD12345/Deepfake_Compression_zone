import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
NUM_CLASSES = 7


class VGG19FER(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(512),
            nn.MaxPool2d(2),
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(512),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, NUM_CLASSES)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def _resnet50_fer(num_classes=NUM_CLASSES):
    return models.resnet50(weights=None, num_classes=num_classes)


def preprocess_face_48(face_pixel_array):
    face_pixel_array = face_pixel_array.astype(np.float32) / 255.0
    if face_pixel_array.ndim == 2:
        face_pixel_array = np.expand_dims(face_pixel_array, axis=-1)
    orig_dtype = face_pixel_array.dtype
    if face_pixel_array.shape[:2] != (48, 48):
        if face_pixel_array.ndim == 3 and face_pixel_array.shape[2] >= 3:
            face_pixel_array = cv2.resize(face_pixel_array, (48, 48)).astype(orig_dtype)
        else:
            face_pixel_array = cv2.resize(face_pixel_array[:, :, 0] if face_pixel_array.ndim == 3 else face_pixel_array, (48, 48))
            face_pixel_array = np.expand_dims(face_pixel_array, axis=-1).astype(orig_dtype)
    return face_pixel_array.astype(np.float32)


def preprocess_face_224(face_pixel_array):
    if face_pixel_array.dtype != np.uint8:
        face_pixel_array = (face_pixel_array * 255).astype(np.uint8)
    if face_pixel_array.ndim == 2:
        face_rgb = cv2.cvtColor(face_pixel_array, cv2.COLOR_GRAY2RGB)
    else:
        face_rgb = cv2.cvtColor(face_pixel_array, cv2.COLOR_BGR2RGB)
    face_rgb = cv2.resize(face_rgb, (224, 224)).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    face_rgb = (face_rgb - mean) / std
    return face_rgb


class FERModel:
    def __init__(self, model_path=None, device=None, model_type="vgg19"):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.model_type = model_type
        if model_type == "resnet50":
            self.model = _resnet50_fer().to(self.device)
            self.preprocess = preprocess_face_224
            self.input_size = 224
        else:
            self.model = VGG19FER().to(self.device)
            self.preprocess = preprocess_face_48
            self.input_size = 48
        self.model.eval()
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            print(f"[FER] Loaded weights from {model_path}")

    def predict(self, face_image):
        preprocessed = self.preprocess(face_image)
        if self.model_type == "resnet50":
            tensor = torch.from_numpy(preprocessed).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        else:
            batch = np.expand_dims(preprocessed, axis=0)
            tensor = torch.from_numpy(batch).float().permute(0, 3, 1, 2).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        emotion_idx = int(np.argmax(probs))
        return EMOTIONS[emotion_idx], float(probs[emotion_idx])

    def predict_batch(self, face_images, batch_size=32):
        all_probs = []
        for i in range(0, len(face_images), batch_size):
            batch = np.array([self.preprocess(f) for f in face_images[i:i+batch_size]], dtype=np.float32)
            tensor = torch.from_numpy(batch).float().permute(0, 3, 1, 2).to(self.device)
            with torch.no_grad():
                logits = self.model(tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
        all_probs = np.concatenate(all_probs, axis=0)
        indices = np.argmax(all_probs, axis=1)
        return [EMOTIONS[i] for i in indices], all_probs
