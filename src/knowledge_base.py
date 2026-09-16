import numpy as np

EMOTIONS = ["normal", "happy", "sad", "angry", "disgust", "fear", "surprise"]

EMOTION_ALIASES = {
    "neutral": "normal",
    "fearful": "fear",
    "surprised": "surprise",
    "calm": "normal",
}


def normalize_emotion(name: str) -> str:
    return EMOTION_ALIASES.get(name, name)

KB_EMOTION_TRANSITION = np.array([
    [0.421, 0.213, 0.084, 0.190, 0.056, 0.050, 0.047],
    [0.061, 0.090, 0.320, 0.091, 0.123, 0.137, 0.092],
    [0.362, 0.509, 0.296, 0.264, 0.262, 0.244, 0.252],
    [0.060, 0.055, 0.058, 0.243, 0.075, 0.101, 0.056],
    [0.027, 0.039, 0.108, 0.086, 0.293, 0.096, 0.164],
    [0.034, 0.051, 0.064, 0.076, 0.069, 0.279, 0.075],
    [0.032, 0.042, 0.068, 0.048, 0.121, 0.092, 0.313],
], dtype=np.float32)

EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOTIONS)}
IDX_TO_EMOTION = {i: e for i, e in enumerate(EMOTIONS)}

def compute_thresholds():
    return KB_EMOTION_TRANSITION.mean(axis=1)

THRESHOLDS = compute_thresholds()

AROUSAL_VALENCE_QUADRANTS = {
    "happy": 1,
    "surprise": 1,
    "angry": 2,
    "fear": 3,
    "sad": 3,
    "disgust": 4,
    "normal": 4,
}

def get_transition_prob(current_emotion: str, next_emotion: str) -> float:
    ci = EMOTION_TO_IDX[normalize_emotion(current_emotion)]
    ni = EMOTION_TO_IDX[normalize_emotion(next_emotion)]
    return float(KB_EMOTION_TRANSITION[ci, ni])

def is_normal_transition(current_emotion: str, next_emotion: str, modality: str = "visual") -> bool:
    ci = EMOTION_TO_IDX[normalize_emotion(current_emotion)]
    prob = KB_EMOTION_TRANSITION[ci, EMOTION_TO_IDX[normalize_emotion(next_emotion)]]
    threshold = THRESHOLDS[ci]
    return prob > threshold

def get_quadrant(emotion: str) -> int:
    return AROUSAL_VALENCE_QUADRANTS.get(normalize_emotion(emotion), 4)

def same_quadrant(emotion_a: str, emotion_b: str) -> bool:
    return get_quadrant(emotion_a) == get_quadrant(emotion_b)
