"""Clean k=1 HAPT metrics. No multiplicity."""

from __future__ import annotations

import csv

import numpy as np
import torch

from metrics import summarize
from models import late_mean, late_product, predict_proba
from phase4.contextual_psa import ContextualPSA

from .paths import assert_hapt_write

N_CLASSES = 12


def fuse_cpsa_attn(model: ContextualPSA, p1: np.ndarray, p2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    tokens = np.stack([p1, p2], axis=1).astype(np.float32)
    t = torch.from_numpy(tokens)
    with torch.no_grad():
        fused, diag = model(t, return_diagnostics=True)
    return fused.cpu().numpy(), diag["attention"].cpu().numpy()


def per_class_scores(probs: np.ndarray, y: np.ndarray) -> list[dict]:
    pred = probs.argmax(axis=1)
    rows = []
    f1s = []
    recalls = []
    for c in range(N_CLASSES):
        support = int((y == c).sum())
        tp = int(((pred == c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if support else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if support and (prec + rec) else (0.0 if support else float("nan"))
        rows.append(
            {
                "class": c,
                "support": support,
                "recall": rec,
                "precision": prec,
                "f1": f1,
            }
        )
        if support:
            f1s.append(f1)
            recalls.append(rec)
    return rows, (float(np.mean(f1s)) if f1s else float("nan")), (float(np.mean(recalls)) if recalls else float("nan"))


def score_block(probs: np.ndarray, y: np.ndarray) -> dict:
    rec = summarize(probs, y)
    _, macro_f1, bal_acc = per_class_scores(probs, y)
    rec["macro_f1"] = macro_f1
    rec["balanced_accuracy"] = bal_acc
    return rec


def evaluate_clean(enc1, enc2, cpsa, X1te, X2te, yte, subj) -> dict:
    p1 = predict_proba(enc1, X1te)
    p2 = predict_proba(enc2, X2te)
    prod = late_product([p1, p2])
    mean = late_mean([p1, p2])
    cpsa_p, attn = fuse_cpsa_attn(cpsa, p1, p2)
    objects = {
        "M1": p1,
        "M2": p2,
        "late_product": prod,
        "late_mean": mean,
        "cpsa": cpsa_p,
    }
    metric_rows = []
    per_class_rows = []
    subject_rows = []
    for name, probs in objects.items():
        rec = score_block(probs, yte)
        rec.update({"object": name})
        metric_rows.append(rec)
        class_rows, _, _ = per_class_scores(probs, yte)
        for cr in class_rows:
            per_class_rows.append({"object": name, **cr})
        for s in sorted(set(int(v) for v in subj.tolist())):
            m = subj == s
            sr = score_block(probs[m], yte[m])
            sr.update({"object": name, "subject": int(s), "n": int(m.sum())})
            subject_rows.append(sr)
    attn_rows = []
    for j, src in enumerate(("M1", "M2")):
        mass = attn[:, j]
        attn_rows.append(
            {
                "source": src,
                "attn_mean": float(np.mean(mass)),
                "attn_median": float(np.median(mass)),
                "attn_p10": float(np.quantile(mass, 0.10)),
                "attn_p90": float(np.quantile(mass, 0.90)),
                "attn_min": float(np.min(mass)),
                "attn_max": float(np.max(mass)),
            }
        )
    if not np.allclose(attn.sum(axis=1), 1.0, atol=1e-6):
        raise RuntimeError("CPSA attention masses do not sum to 1")
    return {
        "p1": p1,
        "p2": p2,
        "late_product": prod,
        "late_mean": mean,
        "cpsa": cpsa_p,
        "attention": attn,
        "metric_rows": metric_rows,
        "per_class_rows": per_class_rows,
        "subject_rows": subject_rows,
        "attn_rows": attn_rows,
    }


def write_csv(path, rows: list[dict], fields: list[str]) -> None:
    dest = assert_hapt_write(path)
    with dest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            out = {}
            for k in fields:
                v = row.get(k, "")
                if isinstance(v, (float, np.floating)):
                    out[k] = format(float(v), ".17g")
                else:
                    out[k] = v
            w.writerow(out)
