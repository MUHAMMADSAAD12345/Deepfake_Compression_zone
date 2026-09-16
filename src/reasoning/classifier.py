import numpy as np

FAKE_PROB_THRESHOLD = 0.15

def compute_fake_probabilities(vt_results, at_results, im_results):
    vt_fake = sum(1 for r in vt_results if r["label"] == "fake")
    at_fake = sum(1 for r in at_results if r["label"] == "fake")
    im_fake = sum(1 for r in im_results if r["label"] == "fake")
    vt_prob = vt_fake / len(vt_results) if vt_results else 0.0
    at_prob = at_fake / len(at_results) if at_results else 0.0
    im_prob = im_fake / len(im_results) if im_results else 0.0
    return {
        "VT": {"fake_count": vt_fake, "total": len(vt_results), "probability": vt_prob},
        "AT": {"fake_count": at_fake, "total": len(at_results), "probability": at_prob},
        "IM": {"fake_count": im_fake, "total": len(im_results), "probability": im_prob},
    }

def classify_modality(prob_dict, threshold=FAKE_PROB_THRESHOLD):
    results = {}
    for key in ["VT", "AT", "IM"]:
        prob = prob_dict[key]["probability"]
        results[key] = "fake" if prob > threshold else "real"
    return results

def voting_classifier(modality_results):
    votes = {"real": 0, "fake": 0}
    for key, verdict in modality_results.items():
        votes[verdict] += 1
    return "fake" if votes["fake"] >= 2 else "real"

def generate_explanation(prob_dict, modality_results, vt_fake_indices, at_fake_indices, im_results):
    lines = []
    pred = voting_classifier(modality_results)
    lines.append(f"Prediction: {pred.upper()}")
    for key, label in modality_results.items():
        d = prob_dict[key]
        lines.append(
            f"  {key}: {label} (fake transitions: {d['fake_count']}/{d['total']} = {d['probability']:.2%})"
        )
    if vt_fake_indices:
        lines.append(f"  Visual intra-modality fakes at timestamps: {vt_fake_indices[:10]}{'...' if len(vt_fake_indices) > 10 else ''}")
    if at_fake_indices:
        lines.append(f"  Aural intra-modality fakes at timestamps: {at_fake_indices[:10]}{'...' if len(at_fake_indices) > 10 else ''}")
    im_fake = [(r['timestamp_index'], r['visual_emotion'], r['aural_emotion'])
               for r in im_results if r['label'] == 'fake']
    if im_fake:
        samples = [(t, v, a) for t, v, a in im_fake[:5]]
        lines.append(f"  Inter-modality mismatches (timestamp, vis, aural): {samples}")
    return "\n".join(lines)
