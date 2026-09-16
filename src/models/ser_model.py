import os
import numpy as np
import librosa
import torch
import torch.nn as nn

EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
RAVDESS_EMOTIONS = ["neutral", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
NUM_CLASSES = 7

def extract_audio_features(audio, sr=16000, n_mfcc=40):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr)
    mel_mean = np.mean(mel.T, axis=0)
    features = np.concatenate([mfcc_mean, chroma_mean, mel_mean])
    return features

FEATURE_DIM = 40 + 12 + 128

class SERMLP(nn.Module):
    def __init__(self, input_dim=FEATURE_DIM, hidden_units=300):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_units),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_units),
            nn.Dropout(0.3),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_units),
            nn.Dropout(0.3),
            nn.Linear(hidden_units, NUM_CLASSES),
        )

    def forward(self, x):
        return self.net(x)


class SERModel:
    def __init__(self, model_path=None, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.model = SERMLP().to(self.device)
        self.model.eval()
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            print(f"[SER] Loaded weights from {model_path}")

    def predict(self, audio, sr=16000):
        features = extract_audio_features(audio, sr)
        batch = np.expand_dims(features, axis=0)
        tensor = torch.from_numpy(batch).float().to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        emotion_idx = int(np.argmax(probs))
        return RAVDESS_EMOTIONS[emotion_idx], float(probs[emotion_idx])

    def predict_batch(self, audio_segments, sr=16000):
        features_batch = np.array([extract_audio_features(a, sr) for a in audio_segments])
        tensor = torch.from_numpy(features_batch).float().to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        indices = np.argmax(probs, axis=1)
        return [RAVDESS_EMOTIONS[i] for i in indices], probs
