from src.knowledge_base import is_normal_transition, EMOTIONS, THRESHOLDS

def intra_modality_reasoning(emotion_sequence):
    results = []
    fake_indices = []
    for i in range(len(emotion_sequence) - 1):
        current = emotion_sequence[i]
        next_em = emotion_sequence[i + 1]
        is_normal = is_normal_transition(current, next_em)
        results.append({
            "timestamp_index": i,
            "from_emotion": current,
            "to_emotion": next_em,
            "is_normal": is_normal,
            "label": "real" if is_normal else "fake",
        })
        if not is_normal:
            fake_indices.append(i)
    return results, fake_indices
