"""Calibration and confidence metrics."""

from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    tot = 0.0
    n = len(y)
    for i in range(n_bins):
        m = (conf >= bins[i]) & (conf < bins[i + 1] if i < n_bins - 1 else conf <= bins[i + 1])
        if not np.any(m):
            continue
        tot += abs(correct[m].mean() - conf[m].mean()) * m.mean()
    return float(tot)


def brier(probs: np.ndarray, y: np.ndarray) -> float:
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def nll(probs: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(probs[np.arange(len(y)), y], eps, 1.0)
    return float(-np.mean(np.log(p)))


def entropy(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(probs, eps, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def summarize(probs: np.ndarray, y: np.ndarray) -> dict[str, float]:
    pred = probs.argmax(axis=1)
    acc = float((pred == y).mean())
    conf = float(probs.max(axis=1).mean())
    H = float(entropy(probs).mean())
    return {
        "accuracy": acc,
        "confidence": conf,
        "ece": ece(probs, y),
        "brier": brier(probs, y),
        "nll": nll(probs, y),
        "entropy": H,
        "conf_acc_gap": conf - acc,
        "n": float(len(y)),
    }
