import sys, os, json, numpy as np
sys.path.insert(0, r'E:\Saad_audi deepfakes\_casr')
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample

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
print('Condition | CNN | CASR | Diff | 95% CI (Diff) | p-value (Diff>0)')
print('-'*75)
import step3_ffpp as s3
from casr_estimator import _load_gate, FEAT_NAMES
coef, intercept = _load_gate()

for c in CONDITIONS:
    y, cnn = s3.load_cnn_probs(c)  # need to adapt
    break