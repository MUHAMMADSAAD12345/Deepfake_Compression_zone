import sys, os, json, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

from src.models.fer_model import FERModel, VGG19FER, preprocess_face, EMOTIONS
from src.models.ser_model import SERModel, SERMLP, extract_audio_features, RAVDESS_EMOTIONS
from src.reasoning.intra_modality import intra_modality_reasoning
from src.reasoning.inter_modality import inter_modality_reasoning
from src.reasoning.classifier import (
    compute_fake_probabilities, classify_modality, voting_classifier, generate_explanation,
)
from src.knowledge_base import (
    get_transition_prob, is_normal_transition, get_quadrant, same_quadrant, EMOTIONS as KB_EMOTIONS,
)

print("=" * 60)
print("COMPONENT 1: Knowledge Base")
print("=" * 60)
# Test transition prob between known emotions
p = get_transition_prob("happy", "sad")
print(f"  happy->sad transition prob: {p:.3f}")
assert 0 <= p <= 1, "transition prob out of range"

# Test normal transition
normal = is_normal_transition("happy", "sad")
print(f"  happy->sad is_normal: {normal}")

# Test quadrant
q = get_quadrant("happy")
print(f"  happy quadrant: {q}")
assert 1 <= q <= 4

# Test same quadrant
sq = same_quadrant("happy", "surprise")
print(f"  happy & surprise same_quadrant: {sq}")
print("  KB OK")

print("=" * 60)
print("COMPONENT 2: FER Model (GPU inference)")
print("=" * 60)
fer_weights = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints", "fer2013_vgg19.pt")
fer = FERModel(fer_weights if os.path.exists(fer_weights) else None, device=device)
# Create a dummy face image (48x48 grayscale)
dummy_face = np.random.randint(0, 256, (48, 48), dtype=np.uint8)
emotion, conf = fer.predict(dummy_face)
print(f"  Single predict: {emotion} ({conf:.3f})")
assert emotion in EMOTIONS

batch_faces = [np.random.randint(0, 256, (48, 48), dtype=np.uint8) for _ in range(4)]
emotions, probs = fer.predict_batch(batch_faces)
print(f"  Batch predict ({len(emotions)}): {emotions}")
print("  FER OK")

print("=" * 60)
print("COMPONENT 3: SER Model (GPU inference)")
print("=" * 60)
ser_weights = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints", "ser_weights.pt")
ser = SERModel(ser_weights if os.path.exists(ser_weights) else None, device=device)
# Create dummy audio (1 second at 16kHz)
dummy_audio = np.random.randn(16000).astype(np.float32)
emotion, conf = ser.predict(dummy_audio)
print(f"  Single predict: {emotion} ({conf:.3f})")
assert emotion in RAVDESS_EMOTIONS

batch_audio = [np.random.randn(16000).astype(np.float32) for _ in range(4)]
emotions, probs = ser.predict_batch(batch_audio)
print(f"  Batch predict ({len(emotions)}): {emotions}")
print("  SER OK")

print("=" * 60)
print("COMPONENT 4: Reasoning Modules")
print("=" * 60)
vt_results, vt_fake = intra_modality_reasoning(["happy", "sad", "angry", "happy"])
print(f"  Intra-visual: {len(vt_results)} transitions, {len(vt_fake)} abnormal")

at_results, at_fake = intra_modality_reasoning(["neutral", "happy", "sad", "angry"])
print(f"  Intra-aural: {len(at_results)} transitions, {len(at_fake)} abnormal")

sync_map = [0, 1, 2, 3]
im_results, im_fake = inter_modality_reasoning(
    ["happy", "sad", "angry", "happy"],
    ["neutral", "happy", "sad", "angry"],
    sync_map,
)
print(f"  Inter-modal: {len(im_results)} pairs, {len(im_fake)} mismatches")

print("=" * 60)
print("COMPONENT 5: Classifier & Voting")
print("=" * 60)
prob_dict = compute_fake_probabilities(vt_results, at_results, im_results)
print(f"  Probabilities: VT={prob_dict['VT']['probability']:.2%}, AT={prob_dict['AT']['probability']:.2%}, IM={prob_dict['IM']['probability']:.2%}")

modality_results = classify_modality(prob_dict)
print(f"  Modality results: {modality_results}")

final_pred = voting_classifier(modality_results)
print(f"  Final prediction: {final_pred}")

explanation = generate_explanation(prob_dict, modality_results, vt_fake, at_fake, im_results)
print(f"  Explanation:\n{explanation}")

print("=" * 60)
print("ALL COMPONENTS OK - Pipeline ready")
print("=" * 60)
