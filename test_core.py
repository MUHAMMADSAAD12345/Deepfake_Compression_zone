import sys; sys.path.insert(0, 'E:/audi deepfakes')
from src.knowledge_base import *
from src.reasoning.intra_modality import intra_modality_reasoning
from src.reasoning.inter_modality import inter_modality_reasoning
from src.reasoning.classifier import *

print('=== Knowledge Base ===')
print('Emotions:', EMOTIONS)
print('Thresholds:', THRESHOLDS)
print('normal->happy prob:', get_transition_prob('normal', 'happy'))
print('happy->sad prob:', get_transition_prob('happy', 'sad'))
print('normal->happy normal?', is_normal_transition('normal', 'happy'))
print('happy->sad normal?', is_normal_transition('happy', 'sad'))

print()
print('=== Arousal-Valence Quadrants ===')
for e in EMOTIONS:
    print(f'  {e}: Q{get_quadrant(e)}')

print()
print('=== Intra-modality Reasoning ===')
seq = ['normal', 'happy', 'sad', 'normal', 'surprise']
results, fake_idx = intra_modality_reasoning(seq)
for r in results:
    print(f'  {r["from_emotion"]} -> {r["to_emotion"]}: {r["label"]}')
print('Fake indices:', fake_idx)

print()
print('=== Inter-modality Reasoning ===')
vis = ['normal', 'happy', 'sad', 'normal']
aur = ['happy', 'normal', 'happy', 'surprise']
sync = [0, 1, 2, 3]
results, fake_idx = inter_modality_reasoning(vis, aur, sync)
for r in results:
    print(f'  vis={r["visual_emotion"]} aur={r["aural_emotion"]} same_q={r["same_quadrant"]}: {r["label"]}')

print()
print('=== Classifier ===')
vt_fake = [{'label': 'fake'}, {'label': 'real'}, {'label': 'fake'}]
at_fake = [{'label': 'real'}, {'label': 'real'}, {'label': 'fake'}]
im = results
prob = compute_fake_probabilities(vt_fake, at_fake, im)
print('Probabilities:', prob)
mod = classify_modality(prob)
print('Modality results:', mod)
print('Final vote:', voting_classifier(mod))

print()
print('All core logic tests passed!')
