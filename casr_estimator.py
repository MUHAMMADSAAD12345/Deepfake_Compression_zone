"""CASR reliability branch: compression-state estimation.

Reusable module extracted from step1_estimator.py.
Estimates compression severity from a single frame so the CASR fusion can
gate the semantic (emotion) branch by predicted reliability.

API:
    compression_features(img_bgr) -> dict of 4 features
    gate_score(img_bgr, gate_lr=GATE_MODEL) -> fake-prob of heavy-compression
"""
import os

import cv2
import numpy as np

_DCT8 = None


def _dct8_matrix():
    global _DCT8
    if _DCT8 is None:
        n = 8
        k = np.arange(n)
        f = np.pi * (k[:, None] * (2 * k[None, :] + 1)) / (2 * n)
        M = np.cos(f)
        M[0] *= np.sqrt(1.0 / n)
        M[1:] *= np.sqrt(2.0 / n)
        _DCT8 = M
    return _DCT8


def central_crop_gray(img, size=512):
    h, w = img.shape[:2]
    x0, y0 = max((w - size) // 2, 0), max((h - size) // 2, 0)
    cs = img[y0:y0 + size, x0:x0 + size]
    return cv2.cvtColor(cs, cv2.COLOR_BGR2GRAY)


def blockiness(gray):
    """Wang-style blockiness: boundary-adjacent diffs / in-block diffs."""
    g = gray.astype(np.float64)
    h, w = g.shape
    bc = np.arange(8, w, 8)
    db = np.abs(g[:, bc - 1] - g[:, bc]).sum()
    inner = np.abs(g[:, 1:] - g[:, :-1]).sum(axis=0)
    mask = np.ones(w - 1, dtype=bool)
    mask[bc - 1] = False
    bh = db / max(inner[mask].sum(), 1e-9)
    br = np.arange(8, h, 8)
    dbv = np.abs(g[br - 1, :] - g[br, :]).sum()
    innerv = np.abs(g[1:, :] - g[:-1, :]).sum(axis=1)
    maskv = np.ones(h - 1, dtype=bool)
    maskv[br - 1] = False
    bv = dbv / max(innerv[maskv].sum(), 1e-9)
    return float((bh + bv) / 2.0)


def zero_dct_ratio(gray, eps=1.0):
    """Fraction of ~zero AC DCT coefficients over 8x8 blocks (DCT-II via matmul)."""
    h, w = gray.shape
    hb, wb = h // 8, w // 8
    blocks = (gray[:hb * 8, :wb * 8].astype(np.float64) - 128.0)\
        .reshape(hb, 8, wb, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)
    M = _dct8_matrix()
    d = np.einsum("ij,njk,lk->nil", M, blocks, M)
    return float((np.abs(d) < eps).mean())


def grid_std(gray):
    g = gray.astype(np.float64)
    col_edges = np.abs(np.diff(g, axis=1)).mean(axis=0)
    row_edges = np.abs(np.diff(g, axis=0)).mean(axis=1)
    if len(col_edges) < 16 or len(row_edges) < 16:
        return 0.0
    return float(np.std(col_edges[::8]) + np.std(row_edges[::8]))


def brisque_score(img_bgr):
    import torch
    from piq import brisque
    t = torch.from_numpy(img_bgr.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    with torch.no_grad():
        return float(brisque(t).item())


FEAT_NAMES = ["blockiness", "zero_dct_ratio", "grid_std", "brisque"]


def compression_features(img_bgr):
    """All 4 features for one BGR frame."""
    gray = central_crop_gray(img_bgr)
    return {
        "blockiness": blockiness(gray),
        "zero_dct_ratio": zero_dct_ratio(gray),
        "grid_std": grid_std(gray),
        "brisque": brisque_score(img_bgr),
    }


GATE_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "gate_logreg.npz")

_GATE = None


def _load_gate():
    global _GATE
    if _GATE is None:
        d = np.load(GATE_MODEL_PATH)
        _GATE = (d["coef"], d["intercept"])
    return _GATE


def gate_score(features_or_img, gate=None):
    """P(heavy compression) from features (dict) or a BGR frame.

    Trained: heavy = H.264 CRF>=33 vs light = CRF<=23 (AUC 0.93, Step 1).
    """
    f = features_or_img if isinstance(features_or_img, dict) else compression_features(features_or_img)
    if gate is None:
        coef, intercept = _load_gate()
        z = np.dot(coef, [np.log1p(abs(f[k])) for k in FEAT_NAMES]) + intercept
    else:
        coef, intercept = gate
        z = np.dot(coef, [np.log1p(abs(f[k])) for k in FEAT_NAMES]) + intercept
    return float(1.0 / (1.0 + np.exp(-z.item() if hasattr(z, 'item') else z)))