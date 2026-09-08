"""Train + save the CASR reliability gate (heavy-compression classifier).

Trains on step1_estimates.csv: heavy (H.264 CRF>=33) vs light (CRF<=23),
logs features, logistic regression. Saves models/gate_logreg.npz consumed by
casr_estimator.gate_score().
"""
import csv
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from casr_estimator import FEAT_NAMES

ROOT = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(ROOT, "results", "step1", "step1_estimates.csv")
OUT = os.path.join(ROOT, "models", "gate_logreg.npz")


def main():
    rows = list(csv.DictReader(open(CSV)))
    X, y = [], []
    for r in rows:
        if not r["condition"].startswith("h264"):
            continue
        crf = int(r["true_val"])
        if crf >= 33 or crf <= 23:
            X.append([np.log1p(abs(float(r[k]))) for k in FEAT_NAMES])
            y.append(1 if crf >= 33 else 0)
    X, y = np.array(X), np.array(y)
    print(f"n={len(y)} heavy={y.sum()} light={(y == 0).sum()}")

    lg = LogisticRegression(max_iter=2000, C=1.0)
    cv = cross_val_score(lg, X, y, cv=5, scoring="roc_auc")
    print(f"CV AUC: {cv.mean():.4f} ± {cv.std():.4f}")

    lg.fit(X, y)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, coef=lg.coef_[0], intercept=lg.intercept_)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()