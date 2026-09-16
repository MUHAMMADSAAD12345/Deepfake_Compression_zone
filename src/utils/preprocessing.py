import cv2
import numpy as np
import librosa

face_cascade = cv2.CascadeClassifier("E:\\Saad_audi deepfakes\\haarcascade_frontalface_default.xml")

AUDIO_SR = 16000
SEGMENT_DURATION = 0.5
HOP_DURATION = 0.25
FRAMES_PER_SECOND = 3

def extract_frames(video_path, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
    frame_interval = int(fps / FRAMES_PER_SECOND)
    frames = []
    timestamps = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
            timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
        frame_idx += 1
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return frames, timestamps, fps

def detect_face(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        x, y, w, h = faces[0]
        x, y = max(0, x), max(0, y)
        face = gray[y : y + h, x : x + w]
        resized = cv2.resize(face, (48, 48))
        return resized
    return None

def detect_face_color(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        x, y, w, h = faces[0]
        x, y = max(0, x), max(0, y)
        face = image_rgb[y : y + h, x : x + w]
        resized = cv2.resize(face, (224, 224))
        return resized
    return None

def extract_audio_segments(video_path):
    audio, sr = librosa.load(video_path, sr=AUDIO_SR, mono=True)
    segment_samples = int(SEGMENT_DURATION * AUDIO_SR)
    hop_samples = int(HOP_DURATION * AUDIO_SR)
    segments = []
    timestamps = []
    start = 0
    while start + segment_samples <= len(audio):
        segment = audio[start : start + segment_samples]
        segments.append(segment)
        timestamps.append(start / AUDIO_SR)
        start += hop_samples
    return segments, timestamps, audio, sr

def sync_frames_and_segments(frames, frame_ts, segments, segment_ts):
    synced = []
    for seg_ts in segment_ts:
        closest_idx = min(range(len(frame_ts)), key=lambda i: abs(frame_ts[i] - seg_ts))
        synced.append(closest_idx)
    return synced

def preprocess_video(video_path):
    frames, frame_ts, fps = extract_frames(video_path)
    segments, segment_ts, raw_audio, sr = extract_audio_segments(video_path)
    face_data = [(detect_face(f), ts) for f, ts in zip(frames, frame_ts)]
    face_data = [(f, ts) for f, ts in face_data if f is not None]
    if not face_data:
        return {"error": "No faces detected in any frame"}
    face_images, frame_ts_filtered = zip(*face_data)
    face_images, frame_ts_filtered = list(face_images), list(frame_ts_filtered)
    sync_map = sync_frames_and_segments(frames, frame_ts_filtered, segments, segment_ts)
    return {
        "face_images": face_images,
        "frame_timestamps": frame_ts_filtered,
        "audio_segments": segments,
        "segment_timestamps": segment_ts,
        "sync_map": sync_map,
        "raw_audio": raw_audio,
        "fps": fps,
    }
