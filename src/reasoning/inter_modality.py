from src.knowledge_base import same_quadrant

def inter_modality_reasoning(visual_emotions, aural_emotions, sync_map):
    results = []
    fake_indices = []
    for seg_idx, frame_idx in enumerate(sync_map):
        if frame_idx >= len(visual_emotions):
            continue
        v_em = visual_emotions[frame_idx]
        a_em = aural_emotions[seg_idx] if seg_idx < len(aural_emotions) else None
        if a_em is None:
            continue
        match = same_quadrant(v_em, a_em)
        results.append({
            "timestamp_index": seg_idx,
            "visual_emotion": v_em,
            "aural_emotion": a_em,
            "same_quadrant": match,
            "label": "real" if match else "fake",
        })
        if not match:
            fake_indices.append(seg_idx)
    return results, fake_indices
