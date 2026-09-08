import sys, os, json, numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample

PROBS_DIR = r'E:\Saad_audi deepfakes\_casr\results\step2b_probs'

def load_per_video(cond):
    with open(os.path.join(r'E:\Saad_audi deepfakes\_casr\results\step2b_probs', f'{cond}.json')) as f:
        probs = json.load(f)
    per_vid = {}
    for k, v in probs.items():
        grp, name = k.split(os.sep, 1)
        # name includes .npz, remove it for matching
        name_noext = os.path.splitext(name)[0]
        per_vid.setdefault((grp, name_noext), []).append(v)
    return per_vid

def load_clean():
    with open(os.path.join(r'E:\Saad_audi deepfakes\_casr\results\step2b_probs', 'clean.json')) as f:
        probs = json.load(f)
    clean_vid = {}
    for k, v in probs.items():
        grp, name = k.split(os.sep, 1)
        name_noext = os.path.splitext(name)[0]
        clean_vid.setdefault((grp, name_noext), []).append(v)
    return {k: max(v) for k, v in clean_vid.items()}

from casr_estimator import _load_gate
coef, intercept = _load_gate()
clean_probs = load_clean()

with open(r'E:\Saad_audi deepfakes\_casr\results\step6_gate_feats.json') as f:
    all_gate = json.load(f)

print('Condition | CNN_AUC | CASR_AUC | Diff | 95% CI (Diff) | p(Diff>0)')
print('-'*75)

for c in ['clean', 'h264_crf18', 'h264_crf28', 'h264_crf38',
          'hevc_crf18', 'hevc_crf28', 'hevc_crf38',
          'av1_crf20', 'av1_crf30', 'av1_crf40']:
    per_vid = load_per_video(c)
    gate_feats = all_gate.get(c, {})
    
    y_list, cnn_list, casr_list = [], [], []
    for (grp, name_noext), preds in per_vid.items():
        p_max = max(preds)
        label = 0 if grp == 'real' else 1
        # Clean semantic prob
        p_clean = clean_probs.get((grp, name_noext))
        if p_clean is None:
            continue
        # Gate features (gate cache uses forward slash, no ext)
        gate_key = f"{grp}/{name_noext}"
        f = gate_feats.get(gate_key)
        if f is None:
            continue
        vec = [float(f[n]) for n in ['blockiness', 'zero_dct_ratio', 'grid_std', 'brisque']]
        z = float((np.dot(np.log1p(np.abs(vec)), coef) + intercept).item())
        hp = 1.0 / (1.0 + np.exp(-z))
        w_sem = 1.0 - hp
        p_casr = w_sem * p_max + (1 - w_sem) * p_clean
        
        y_list.append(0 if grp == 'real' else 1)
        cnn_list.append(p_max)
        casr_list.append(p_casr)
    
    if len(y_list) < 100:
        print(f'{c:12s} | insufficient data (n={len(y_list)})')
        continue
    
    y = np.array(y_list)
    cnn = np.array(cnn_list)
    casr = np.array(casr_list)
    
    cnn_auc = roc_auc_score(y, cnn)
    casr_auc = roc_auc_score(y, casr)
    diff = casr_auc - roc_auc_score(y, cnn)
    
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