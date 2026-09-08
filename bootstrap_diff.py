import sys, os, json, numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample

PROBS_DIR = r'E:\Saad_audi deepfakes\_casr\results\step2b_probs'
CONDITIONS = ['clean', 'h264_crf18', 'h264_crf28', 'h264_crf38',
              'hevc_crf18', 'hevc_crf28', 'hevc_crf38',
              'av1_crf20', 'av1_crf30', 'av1_crf40']

def load_per_video(cond):
    with open(os.path.join(r'E:\Saad_audi deepfakes\_casr\results\step2b_probs', f'{cond}.json')) as f:
        probs = json.load(f)
    per_vid = {}
    for k, v in probs.items():
        grp, name = k.split(os.sep, 1)
        per_vid.setdefault((grp, name), []).append(v)
    return per_vid

def load_gate_feats(cond):
    with open(r'E:\Saad_audi deepfakes\_casr\results\step6_gate_feats.json') as f:
        return json.load(f).get(cond, {})

from casr_estimator import _load_gate, FEAT_NAMES
coef, intercept = _load_gate()

results = []
print('Condition | CNN_AUC | CASR_AUC | Diff | 95% CI (Diff) | p(Diff>0)')
print('-'*75)

for c in ['clean', 'h264_crf18', 'h264_crf28', 'h264_crf38',
          'hevc_crf18', 'hevc_crf28', 'hevc_crf38',
          'av1_crf20', 'av1_crf30', 'av1_crf40']:
    per_vid = load_per_video(c)
    gate_feats = load_gate_feats(c)
    
    y_list, cnn_list, casr_list = [], [], []
    for (grp, name), preds in per_vid.items():
        p_max = max(preds)
        label = 0 if grp == 'real' else 1
        # Gate prob
        f = gate_feats.get(f"{grp}/{name}")
        if f is None:
            continue
        vec = [float(f[n]) for n in ['blockiness', 'zero_dct_ratio', 'grid_std', 'brisque']]
        from casr_estimator import _load_gate
        coef, intercept = _load_gate()
        z = float(np.dot(np.log1p(np.abs(vec)), coef) + intercept)
        hp = 1.0 / (1.0 + np.exp(-z))
        # Clean semantic prob = clean CNN prob for same video
        # We'll need to get it from clean condition
        # For now, approximate: at heavy compression, CASR ≈ CNN (hp high)
        # Actually need the clean semantic prob per video
        # Simplify: use the step3_ffpp.json CASR results for point estimate, bootstrap diff
        pass
    # Skip for now - we'll use the step3 results directly
    break

# Simpler: bootstrap the diff using the step3_ffpp.json point estimates
# by simulating the per-video scores
# Actually we need per-video scores for proper bootstrap
# Let's extract per-video CNN and CASR scores from step3_ffpp logic

import step3_ffpp as s3
from casr_estimator import _load_gate, FEAT_NAMES

coef, intercept = _load_gate()
with open(r'E:\Saad_audi deepfakes\_casr\results\step6_gate_feats.json') as f:
    all_gate = json.load(f)

print('Condition | CNN_AUC | CASR_AUC | Diff | 95% CI (Diff) | p(Diff>0)')
print('-'*75)

for c in ['clean', 'h264_crf18', 'h264_crf28', 'h264_crf38',
          'hevc_crf18', 'hevc_crf28', 'hevc_crf38',
          'av1_crf20', 'av1_crf30', 'av1_crf40']:
    # Get per-video CNN and CASR scores from step3_ffpp logic
    cnn_probs = s3.load_cnn_probs(c)
    per_vid = {}
    for k, v in cnn_probs.items():
        grp, name = k.split(os.sep, 1)
        per_vid.setdefault(name, []).append(v)
    
    # Gate features
    gate_feats = all_gate.get(c, {})
    
    y_list, cnn_list, casr_list = [], [], []
    for name, preds in per_vid.items():
        p_max = max(preds)
        grp = name.split(os.sep)[0] if '/' in name else name.split('\\')[0]
        label = 0 if grp == 'real' else 1
        
        # Get clean prob for semantic (same video in clean condition)
        clean_probs = json.load(open(os.path.join(r'E:\Saad_audi deepfakes\_casr\results\step2b_probs', 'clean.json')))
        clean_key = None
        for k in clean_probs:
            if k.endswith(os.sep + name) or k.endswith('\\' + name) or k == name:
                clean_key = k
                break
        if clean_key is None:
            continue
        clean_per_vid = {}
        for k, v in clean_probs.items():
            grp2, n2 = k.split(os.sep, 1)
            clean_per_vid.setdefault(n2, []).append(v)
        if name not in clean_per_vid:
            continue
        p_clean = max(clean_per_vid[name])
        
        # Gate
        f = None
        # Try both key formats
        for key in [f"{grp}/{name}", f"{grp}\\{name}", name]:
            if key in all_gate.get(c, {}):
                f = all_gate[c][key]
                break
        if f is None:
            continue
        vec = [float(f[n]) for n in ['blockiness', 'zero_dct_ratio', 'grid_std', 'brisque']]
        z = float(np.dot(np.log1p(np.abs(vec)), coef) + intercept)
        hp = 1.0 / (1.0 + np.exp(-z))
        w_sem = 1.0 - hp
        p_casr = w_sem * p_max + (1 - w_sem) * p_clean
        
        y_list.append(0 if grp == 'real' else 1)
        cnn_list.append(p_max)
        casr_list.append(p_casr)
    
    if len(y_list) < 100:
        print(f'{c:12s} | insufficient data')
        continue
    
    y = np.array(y_list)
    cnn = np.array(cnn_list)
    casr = np.array(casr_list)
    
    cnn_auc = roc_auc_score(y, cnn)
    casr_auc = roc_auc_score(y, casr)
    diff = casr_auc - cnn_auc
    
    # Bootstrap
    diffs = []
    for _ in range(1000):
        y_b, cnn_b, casr_b = resample(y, cnn, casr, random_state=None)
        if len(np.unique(y_b)) < 2:
            continue
        diffs.append(roc_auc_score(y_b, casr_b) - roc_auc_score(y_b, cnn_b))
    
    if diffs:
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        p_val = np.mean(np.array(diffs) > 0)
    else:
        lo, hi, p_val = 0, 0, 0
    
    cnn_auc = roc_auc_score(y, cnn)
    casr_auc = roc_auc_score(y, casr)
    diff = casr_auc - cnn_auc
    
    print(f'{c:12s} | {cnn_auc:.4f} | {casr_auc:.4f} | {diff:+.4f} | [{lo:.4f}, {hi:.4f}] | {p_val:.4f}')