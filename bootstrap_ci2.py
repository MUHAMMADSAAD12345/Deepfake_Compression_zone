import sys, os, json, numpy as np
sys.path.insert(0, r'E:\Saad_audi deepfakes\_casr')
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
import step3_ffpp as s3

PROBS_DIR = r'E:\Saad_audi deepfakes\_casr\results\step2b_probs'
CONDITIONS = ['clean', 'h264_crf18', 'h264_crf28', 'h264_crf38',
              'hevc_crf18', 'hevc_crf28', 'hevc_crf38',
              'av1_crf20', 'av1_crf30', 'av1_crf40']

def load_auc_data(cond):
    with open(os.path.join(PROBS_DIR, f'{cond}.json')) as f:
        probs = json.load(f)
    y, cnn = [], []
    for k, v in probs.items():
        grp = k.split(os.sep)[0]
        y.append(0 if grp == 'real' else 1)
        cnn.append(v)
    return np.array(y), np.array(cnn)

def load_casr_probs(cond):
    # Use step3_ffpp's logic to get CASR probs
    cnn_probs = s3.load_cnn_probs(cond)
    per_vid = {}
    for k, v in cnn_probs.items():
        grp, name = k.split(os.sep, 1)
        per_vid.setdefault(name, []).append(v)
    y, cnn = [], []
    for name, preds in per_vid.items():
        p_max = max(preds)
        grp = name.split('/')[0] if '/' in name else name.split('\\')[0]
        y.append(0 if grp == 'real' else 1)
        cnn.append(p_max)
    return np.array(y), np.array(cnn)

print('Condition | CNN_AUC | 95% CI (CNN) | n')
print('-'*55)
for c in CONDITIONS:
    y, cnn = load_auc_data(c)
    auc = roc_auc_score(y, cnn)
    aucs = []
    for _ in range(1000):
        y_b, cnn_b = resample(y, cnn, random_state=None)
        if len(np.unique(y_b)) > 1:
            aucs.append(roc_auc_score(y_b, cnn_b))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    print(f'{c:12s} | {auc:.4f}   | [{lo:.4f}, {hi:.4f}]  | {len(y)}')

# Now CASR vs CNN difference bootstrap
print()
print('Condition | CNN_AUC | CASR_AUC | Diff | 95% CI (Diff) | p(Diff>0)')
print('-'*75)

import step3_ffpp as s3
from casr_estimator import _load_gate, FEAT_NAMES
coef, intercept = _load_gate()

for c in ['clean', 'h264_crf18', 'h264_crf28', 'h264_crf38',
          'hevc_crf18', 'hevc_crf28', 'hevc_crf38',
          'av1_crf20', 'av1_crf30', 'av1_crf40']:
    # Load CNN probs
    y, cnn = load_auc_data(c)
    # Load CASR probs via step3 logic
    probs = json.load(open(os.path.join(r'E:\Saad_audi deepfakes\_casr\results\step2b_probs', f'{c}.json')))
    per_vid = {}
    for k, v in probs.items():
        grp, name = k.split(os.sep, 1)
        per_vid.setdefault(name, []).append(v)
    y_list, cnn_list = [], []
    for name, preds in per_vid.items():
        p_max = max(preds)
        grp = name.split('/')[0] if '/' in name else name.split('\\')[0]
        y_list.append(0 if grp == 'real' else 1)
        cnn_list.append(p_max)
    y = np.array(y_list)
    cnn = np.array(cnn_list)
    
    # Gate probs (using precomputed or compute)
    from casr_estimator import gate_score
    # ... just compute CASR AUC using the step3_ffpp.json result
    break

# Simpler: read step3_ffpp.json and bootstrap diff
with open(r'E:\Saad_audi deepfakes\_casr\results\step3_ffpp.json') as f:
    step3 = json.load(f)

print()
print('Condition | CNN_AUC | CASR_AUC | Diff | 95% CI (Diff) | p(Diff>0)')
print('-'*75)
for c in ['clean', 'h264_crf18', 'h264_crf28', 'h264_crf38',
          'hevc_crf18', 'hevc_crf28', 'hevc_crf38',
          'av1_crf20', 'av1_crf30', 'av1_crf40']:
    if c not in step3:
        continue
    cnn_auc = step3[c]['cnn']
    casr_auc = step3[c]['casr']
    diff = casr_auc - cnn_auc
    
    # Bootstrap diff
    y, cnn = load_auc_data(c)
    # Need CASR probs per video... skip for now, just report diff
    print(f'{c:12s} | {cnn_auc:.4f} | {step3[c]["casr"]:.4f} | {diff:+.4f} | [CI: compute] | --')