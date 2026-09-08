"""CASR Step 4: audio compression robustness sweep.

Quantifies how lossy audio re-encoding (the OSN pipeline) degrades
speech-emotion features and SER decisions, on TWO source populations:
  - RAVDESS WAVs (uncompressed PCM source)
  - LAV-DF video audio (already AAC ~73 kbps source — benchmark provenance)

Conditions: aac 32/64/128k, opus 32/64/128k, mp3 64/128/256k
Metrics per (file, condition) vs clean:
  - normalized feature L2 distance (180-dim SER features)
  - mel cosine similarity
  - SER prediction agreement (7-class argmax)
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import librosa

sys.path.insert(0, r"E:\Saad_audi deepfakes")
from src.models.ser_model import extract_audio_features  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
RAVDESS = r"E:\Saad_audi deepfakes\data\ravdess"
LAVDF = r"E:\Saad_audi deepfakes\data\LAV-DF\test"
OUT_DIR = os.path.join(ROOT, "results", "step4")
os.makedirs(OUT_DIR, exist_ok=True)

CONDITIONS = [("aac", 32), ("aac", 64), ("aac", 128),
              ("opus", 32), ("opus", 64), ("opus", 128),
              ("mp3", 64), ("mp3", 128), ("mp3", 256)]
SR = 16000
SEG_S = 4.0


def ffmpeg_run(args):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"] + args,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg: " + r.stderr[-300:])


def encode_decode(src_wav, codec, kbps, tmp_prefix):
    """WAV -> encoded container -> decoded 16k mono WAV (in tmp)."""
    ext = {"aac": ".m4a", "opus": ".opus", "mp3": ".mp3"}[codec]
    enc = os.path.join(tmp_prefix, f"{codec}{kbps}{ext}")
    dec = os.path.join(tmp_prefix, f"{codec}{kbps}.wav")
    if codec == "aac":
        ffmpeg_run(["-i", src_wav, "-c:a", "aac", "-b:a", f"{kbps}k", enc])
    elif codec == "opus":
        ffmpeg_run(["-i", src_wav, "-c:a", "libopus", "-b:a", f"{kbps}k", enc])
    else:
        ffmpeg_run(["-i", src_wav, "-c:a", "libmp3lame", "-b:a", f"{kbps}k", enc])
    ffmpeg_run(["-i", enc, "-ar", str(SR), "-ac", "1", "-c:a", "pcm_s16le", dec])
    return dec


def seg(audio, sr):
    n = int(SEG_S * sr)
    return audio[:n] if len(audio) >= n else np.pad(audio, (0, n - len(audio)))


def load_ser():
    sys.path.insert(0, r"E:\Saad_audi deepfakes")
    from src.models.ser_model import SERMLP  # noqa: E402
    import torch
    m = SERMLP()
    m.load_state_dict(torch.load(r"E:\Saad_audi deepfakes\checkpoints\ser_weights.pt",
                                 map_location="cpu", weights_only=True))
    m.eval()
    return m


def ser_argmax(a, sr, model):
    import torch
    f = extract_audio_features(a, sr)
    with torch.no_grad():
        logits = model(torch.from_numpy(np.expand_dims(f, 0)).float())
        return int(torch.softmax(logits, 1).argmax(1).item())


def process_one(item, tmp_prefix, ser_model):
    name, src = item
    try:
        if src.endswith(".mp4"):
            wav = os.path.join(tmp_prefix, "src.wav")
            ffmpeg_run(["-i", src, "-ar", str(SR), "-ac", "1", wav])
        else:
            wav = src
        a0, sr = librosa.load(wav, sr=SR, mono=True)
        a0 = seg(a0, sr)
        f0 = extract_audio_features(a0, sr)
        mel0 = librosa.feature.melspectrogram(y=a0, sr=sr)
        mel0 = (mel0 / (mel0.sum() + 1e-12)).flatten()
        pred0 = ser_argmax(a0, sr, ser_model)
        rows = []
        for codec, kbps in CONDITIONS:
            dec = encode_decode(wav, codec, kbps, tmp_prefix)
            a1, _ = librosa.load(dec, sr=SR, mono=True)
            a1 = seg(a1, sr)
            f1 = extract_audio_features(a1, sr)
            d = float(np.linalg.norm(f1 - f0) / (np.linalg.norm(f0) + 1e-12))
            mel1 = librosa.feature.melspectrogram(y=a1, sr=sr)
            mel1 = (mel1 / (mel1.sum() + 1e-12)).flatten()
            cos = float(np.dot(mel0, mel1) / (np.linalg.norm(mel0) * np.linalg.norm(mel1) + 1e-12))
            pred1 = ser_argmax(a1, sr, ser_model)
            rows.append({"source": name, "codec": codec, "kbps": kbps,
                         "feat_l2_rel": round(d, 6), "mel_cos": round(cos, 6),
                         "ser_agree": int(pred1 == pred0)})
            if os.path.exists(dec):
                os.remove(dec)
        return name, rows
    except Exception as e:  # noqa: BLE001
        print(f"[ERR] {name}: {str(e)[:200]}", file=sys.stderr)
        return name, []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ravdess_n", type=int, default=80)
    ap.add_argument("--lavdf_n", type=int, default=30)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    items = []
    ravd = sorted(os.path.join(RAVDESS, a, f)
                  for a in os.listdir(RAVDESS) if os.path.isdir(os.path.join(RAVDESS, a))
                  for f in os.listdir(os.path.join(RAVDESS, a)) if f.endswith(".wav"))
    items += [(os.path.basename(p), p) for p in ravd[:args.ravdess_n]]
    lav = sorted(os.path.join(LAVDF, f) for f in os.listdir(LAVDF) if f.endswith(".mp4"))
    items += [(os.path.basename(p), p) for p in lav[:args.lavdf_n]]

    tmp_prefix = os.path.join(OUT_DIR, "tmp")
    os.makedirs(tmp_prefix, exist_ok=True)
    ser_model = load_ser()

    all_rows = []
    workers = args.workers

    def run_one(it):
        import threading
        tid = threading.get_ident()
        sub_tmp = os.path.join(tmp_prefix, str(tid))
        os.makedirs(sub_tmp, exist_ok=True)
        return process_one(it, sub_tmp, ser_model)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for name, rows in ex.map(run_one, items):
            if rows:
                all_rows.extend(rows)

    csvp = os.path.join(OUT_DIR, "step4_audio_sweep.csv")
    with open(csvp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    # summary: per (codec,kbps) median across sources
    from collections import defaultdict
    g = defaultdict(list)
    for r in all_rows:
        g[(r["codec"], r["kbps"])].append(r)
    summary = {}
    for k in sorted(g):
        rows = g[k]
        summary[f"{k[0]}{k[1]}k"] = {
            "n": len(rows),
            "median_feat_l2_rel": round(float(np.median([r["feat_l2_rel"] for r in rows])), 6),
            "median_mel_cos": round(float(np.median([r["mel_cos"] for r in rows])), 6),
            "ser_agree_rate": round(float(np.mean([r["ser_agree"] for r in rows])), 4),
        }
    with open(os.path.join(OUT_DIR, "step4_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()