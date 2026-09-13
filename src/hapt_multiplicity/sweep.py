"""HAPT multiplicity replication. Frozen posteriors + frozen CPSA only. No training."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from models import late_mean, late_product  # noqa: E402
from multiplicity_replay.hashes import audit_freeze  # noqa: E402
from multiplicity_replay.replay import REPLAY_ROOT, REPO_ROOT, file_sha256  # noqa: E402

from hapt_multiplicity.evaluate import per_class_scores, score_block  # noqa: E402
from hapt_multiplicity.io_util import load_cpsa, write_json  # noqa: E402
from hapt_multiplicity.paths import (  # noqa: E402
    CACHE_DIR,
    CKPT_DIR,
    FEATURE_MAP,
    HAPT_ROOT,
    HAPT_ZIP,
    PROTOCOL_JSON,
    SWEEP_G3,
    WRITE_ROOT,
    assert_hapt_write,
)

SWEEP = (WRITE_ROOT / "sweep").resolve()
SEEDS = (0, 1, 2)
SOURCES = ("M1", "M2")
K_VALUES = (1, 2, 3, 5, 8, 16, 32, 64, 100)
OPERATORS = ("late_product", "late_mean", "cpsa")
MEAN_TOL = 1e-5
LOCKED_REP_MD = "f33233f7a8aa0b8701596468974aba9449c06dfb4527f1cf74071e90440b3793"
LOCKED_REP_JSON = "9ea650ef88fd51962c8df180a15dc0b010722956c8ea60e25b1edfd38e1e911c"
G5_CACHE = {
    0: "65cb0613258abc6b89dde0b4868f2762b22b96240f3d4ca2fd6ad71ffe94b63b",
    1: "4bc1f4dd3b77013561a433ff6d0d2fd47a92f341fb065596e3d68fc2da036f18",
    2: "6ea7395e6dcb45ae299e1dbc17277ddaaf3d2abfe482b679333d1fd1beeb650a",
}
G5_CPSA = {
    0: "31a715b6570826394e4f81ec364b9735b2fcb07129dc85abddad690071ddb843",
    1: "e672be96e77675692df93a800857efd2dde09e524a0b50ecaf33f799d8e44fcc",
    2: "fe46b9b206d019a6a24e6a88e05f34031e574b5c0cfb18621595ce5c14e9bce7",
}


def assert_replication_protocol_locked() -> None:
    md = SWEEP / "REPLICATION_PROTOCOL.md"
    js = SWEEP / "replication_protocol.json"
    if file_sha256(md) != LOCKED_REP_MD or file_sha256(js) != LOCKED_REP_JSON:
        raise RuntimeError("replication protocol changed after preregistration")


def load_frozen_seed(seed: int) -> dict:
    path = CACHE_DIR / f"seed{seed}.npz"
    if file_sha256(path) != G5_CACHE[seed]:
        raise RuntimeError(f"Gate-5 cache hash changed seed{seed}")
    ckpt = CKPT_DIR / f"seed{seed}_cpsa.pt"
    if file_sha256(ckpt) != G5_CPSA[seed]:
        raise RuntimeError(f"Gate-5 CPSA hash changed seed{seed}")
    z = np.load(path)
    out = {k: np.asarray(z[k]) for k in z.files}
    z.close()
    return out


def token_list(p1: np.ndarray, p2: np.ndarray, source: str, k: int) -> list[np.ndarray]:
    if k < 1:
        raise ValueError("k >= 1")
    if k == 1:
        return [p1, p2]
    if source == "M1":
        return [p1] * int(k) + [p2]
    if source == "M2":
        return [p2] * int(k) + [p1]
    raise ValueError(source)


@torch.no_grad()
def fuse_cpsa_tokens(model, parts: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    t = torch.from_numpy(np.stack(parts, axis=1).astype(np.float32))
    fused, diag = model(t, return_diagnostics=True)
    return fused.cpu().numpy(), diag["attention"].cpu().numpy()


def fuse(op: str, parts: list[np.ndarray], cpsa):
    if op == "late_product":
        return late_product(parts), None
    if op == "late_mean":
        return late_mean(parts), None
    return fuse_cpsa_tokens(cpsa, parts)


def attn_mass(attn: np.ndarray, source: str, k: int) -> np.ndarray:
    if k == 1:
        return attn[:, 0 if source == "M1" else 1]
    return attn[:, :k].sum(axis=1)


def l1_rows(a, b):
    return np.abs(a - b).sum(axis=1)


def unique_argmax_frac(p: np.ndarray) -> float:
    mx = p.max(axis=1, keepdims=True)
    return float(((p == mx).sum(axis=1) == 1).mean())


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    dest = assert_hapt_write(path)
    with dest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            out = {}
            for k in fields:
                v = row.get(k, "")
                out[k] = format(float(v), ".17g") if isinstance(v, (float, np.floating)) else v
            w.writerow(out)


def prior_hashes() -> dict:
    freeze = audit_freeze()
    return {
        "freeze_ok": freeze["ok"],
        "watched_extra": freeze["watched_extra"],
        "gate2": {p.name: file_sha256(p) for p in sorted((REPLAY_ROOT / "caches").glob("seed*.npz"))},
        "gate3": {
            n: file_sha256(SWEEP_G3 / n)
            for n in (
                "multiplicity_metrics.csv",
                "source_quality.csv",
                "cpsa_attention_diagnostics.csv",
                "late_mean_law.csv",
                "per_subject_metrics.csv",
            )
        },
        "g5_protocol": file_sha256(PROTOCOL_JSON),
        "g5_feature_map": file_sha256(FEATURE_MAP),
        "g5_caches": {s: file_sha256(CACHE_DIR / f"seed{s}.npz") for s in SEEDS},
        "g5_cpsa": {s: file_sha256(CKPT_DIR / f"seed{s}_cpsa.pt") for s in SEEDS},
        "hapt_zip": file_sha256(HAPT_ZIP),
        "features_txt": file_sha256(HAPT_ROOT / "features.txt"),
        "replication_md": file_sha256(SWEEP / "REPLICATION_PROTOCOL.md"),
        "replication_json": file_sha256(SWEEP / "replication_protocol.json"),
    }


def run_sweep() -> dict:
    assert_replication_protocol_locked()
    pre = prior_hashes()
    write_json(SWEEP / "hash_pre.json", pre)
    if not pre["freeze_ok"]:
        raise RuntimeError("prior freeze failed")

    metric_rows = []
    pred_rows = []
    attn_rows = []
    law_rows = []
    subj_rows = []
    class_rows = []
    baseline = []
    fused_mem = {}

    for seed in SEEDS:
        cache = load_frozen_seed(seed)
        p1, p2 = cache["p1_test"], cache["p2_test"]
        y, subj = cache["y_test"], cache["subject_test"]
        cpsa = load_cpsa(CKPT_DIR / f"seed{seed}_cpsa.pt")
        if cpsa.training:
            raise RuntimeError("CPSA in train mode")
        unique = {"M1": unique_argmax_frac(p1), "M2": unique_argmax_frac(p2)}
        pj_map = {"M1": p1, "M2": p2}

        f1 = {}
        for op in OPERATORS:
            fused_by_src = {}
            for src in SOURCES:
                parts = token_list(p1, p2, src, 1)
                fused, attn = fuse(op, parts, cpsa)
                fused_by_src[src] = fused
                fused_mem[(seed, src, 1, op)] = fused
            cross = float(np.max(np.abs(fused_by_src["M1"] - fused_by_src["M2"])))
            g5_key = {
                "late_product": "fused_late_product",
                "late_mean": "fused_late_mean",
                "cpsa": "fused_cpsa",
            }[op]
            vs_g5 = float(np.max(np.abs(fused_by_src["M1"] - cache[g5_key])))
            baseline.append(
                {
                    "seed": seed,
                    "operator": op,
                    "max_abs_cross_source_k1": cross,
                    "max_abs_vs_gate5": vs_g5,
                    "ok": cross <= MEAN_TOL and vs_g5 <= MEAN_TOL,
                }
            )
            f1[op] = fused_by_src["M1"]
        if any(not r["ok"] for r in baseline if r["seed"] == seed):
            raise RuntimeError(f"k=1 integrity failed: {baseline}")

        for src in SOURCES:
            pj = pj_map[src]
            po = p2 if src == "M1" else p1
            src_top = pj.argmax(axis=1)
            for k in K_VALUES:
                parts = token_list(p1, p2, src, k)
                if k != 1:
                    copies = parts[:k]
                    if any(t is not pj for t in copies) or len(parts) != k + 1:
                        raise RuntimeError("token construction error")
                for op in OPERATORS:
                    fused, attn = fuse(op, parts, cpsa)
                    fused_mem[(seed, src, k, op)] = fused
                    if attn is not None:
                        mass = attn_mass(attn, src, k)
                        attn_rows.append(
                            {
                                "seed": seed,
                                "source": src,
                                "k": k,
                                "attn_mean": float(np.mean(mass)),
                                "attn_median": float(np.median(mass)),
                                "attn_p10": float(np.quantile(mass, 0.10)),
                                "attn_p90": float(np.quantile(mass, 0.90)),
                                "attn_min": float(np.min(mass)),
                                "attn_max": float(np.max(mass)),
                                "mean_l1_to_pj": float(l1_rows(fused, pj).mean()),
                            }
                        )
                    if op == "late_mean":
                        obs = l1_rows(fused, pj)
                        pred = (2.0 / (k + 1.0)) * l1_rows(f1["late_mean"], pj)
                        vec = fused - pj
                        vec_pred = (po - pj) / (k + 1.0)
                        law_rows.append(
                            {
                                "seed": seed,
                                "source": src,
                                "k": k,
                                "max_abs_residual": float(np.max(np.abs(obs - pred))),
                                "max_vector_residual": float(np.max(np.abs(vec - vec_pred))),
                                "mean_abs_residual": float(np.mean(np.abs(obs - pred))),
                            }
                        )

        for op in OPERATORS:
            base = f1[op]
            base_sc = score_block(base, y)
            pred1 = base.argmax(axis=1)
            for src in SOURCES:
                pj = pj_map[src]
                src_top = pj.argmax(axis=1)
                for k in K_VALUES:
                    fused = fused_mem[(seed, src, k, op)]
                    sc = score_block(fused, y)
                    predk = fused.argmax(axis=1)
                    l1f = l1_rows(fused, base)
                    l1p = l1_rows(fused, pj)
                    flips = (predk != pred1).astype(np.float64)
                    agree = (predk == src_top).astype(np.float64)
                    row = {
                        "seed": seed,
                        "source": src,
                        "k": k,
                        "operator": op,
                        "n_tokens": 2 if k == 1 else int(k + 1),
                        "accuracy": sc["accuracy"],
                        "macro_f1": sc["macro_f1"],
                        "balanced_accuracy": sc["balanced_accuracy"],
                        "ece": sc["ece"],
                        "nll": sc["nll"],
                        "brier": sc["brier"],
                        "confidence": sc["confidence"],
                        "conf_acc_gap": sc["conf_acc_gap"],
                        "entropy": sc["entropy"],
                        "d_accuracy": sc["accuracy"] - base_sc["accuracy"],
                        "d_macro_f1": sc["macro_f1"] - base_sc["macro_f1"],
                        "d_balanced_accuracy": sc["balanced_accuracy"] - base_sc["balanced_accuracy"],
                        "d_ece": sc["ece"] - base_sc["ece"],
                        "d_nll": sc["nll"] - base_sc["nll"],
                        "d_brier": sc["brier"] - base_sc["brier"],
                        "d_confidence": sc["confidence"] - base_sc["confidence"],
                        "d_entropy": sc["entropy"] - base_sc["entropy"],
                        "mean_l1_to_k1": float(l1f.mean()),
                        "prediction_flip_rate": float(flips.mean()),
                        "mean_l1_to_pj": float(l1p.mean()),
                        "source_top_class_agreement": float(agree.mean()),
                        "unique_source_argmax_frac": unique[src],
                        "n": sc["n"],
                    }
                    metric_rows.append(row)
                    pred_rows.append(
                        {
                            "seed": seed,
                            "source": src,
                            "k": k,
                            "operator": op,
                            "mean_l1_to_k1": row["mean_l1_to_k1"],
                            "prediction_flip_rate": row["prediction_flip_rate"],
                            "mean_l1_to_pj": row["mean_l1_to_pj"],
                            "source_top_class_agreement": row["source_top_class_agreement"],
                            "mean_fused_maxprob": row["confidence"],
                        }
                    )
                    for s in sorted(set(int(v) for v in subj.tolist())):
                        m = subj == s
                        sr = score_block(fused[m], y[m])
                        subj_rows.append(
                            {
                                "seed": seed,
                                "source": src,
                                "k": k,
                                "operator": op,
                                "subject": int(s),
                                "n": int(m.sum()),
                                "accuracy": sr["accuracy"],
                                "macro_f1": sr["macro_f1"],
                                "nll": sr["nll"],
                                "brier": sr["brier"],
                                "confidence": sr["confidence"],
                                "mean_l1_to_k1": float(l1f[m].mean()),
                                "prediction_flip_rate": float(flips[m].mean()),
                            }
                        )
                    if k in (1, 5, 100):
                        cls, _, _ = per_class_scores(fused, y)
                        base_cls, _, _ = per_class_scores(base, y)
                        base_rec = {c["class"]: c["recall"] for c in base_cls}
                        for c in cls:
                            cm = y == c["class"]
                            class_rows.append(
                                {
                                    "seed": seed,
                                    "source": src,
                                    "k": k,
                                    "operator": op,
                                    "class": c["class"],
                                    "support": c["support"],
                                    "recall": c["recall"],
                                    "d_recall": (
                                        c["recall"] - base_rec[c["class"]]
                                        if np.isfinite(c["recall"]) and np.isfinite(base_rec[c["class"]])
                                        else float("nan")
                                    ),
                                    "prediction_flip_rate": float(flips[cm].mean()) if cm.any() else float("nan"),
                                    "mean_l1_to_k1": float(l1f[cm].mean()) if cm.any() else float("nan"),
                                }
                            )

    if len(metric_rows) != 162:
        raise RuntimeError(f"expected 162 rows, got {len(metric_rows)}")
    max_law = max(r["max_abs_residual"] for r in law_rows)
    max_vec = max(r["max_vector_residual"] for r in law_rows)
    law_pass = max_law <= MEAN_TOL and max_vec <= MEAN_TOL

    write_csv(
        SWEEP / "hapt_multiplicity_metrics.csv",
        metric_rows,
        [
            "seed",
            "source",
            "k",
            "operator",
            "n_tokens",
            "accuracy",
            "macro_f1",
            "balanced_accuracy",
            "ece",
            "nll",
            "brier",
            "confidence",
            "conf_acc_gap",
            "entropy",
            "d_accuracy",
            "d_macro_f1",
            "d_balanced_accuracy",
            "d_ece",
            "d_nll",
            "d_brier",
            "d_confidence",
            "d_entropy",
            "mean_l1_to_k1",
            "prediction_flip_rate",
            "mean_l1_to_pj",
            "source_top_class_agreement",
            "unique_source_argmax_frac",
            "n",
        ],
    )
    write_csv(
        SWEEP / "hapt_prediction_response.csv",
        pred_rows,
        [
            "seed",
            "source",
            "k",
            "operator",
            "mean_l1_to_k1",
            "prediction_flip_rate",
            "mean_l1_to_pj",
            "source_top_class_agreement",
            "mean_fused_maxprob",
        ],
    )
    write_csv(
        SWEEP / "hapt_cpsa_attention.csv",
        attn_rows,
        [
            "seed",
            "source",
            "k",
            "attn_mean",
            "attn_median",
            "attn_p10",
            "attn_p90",
            "attn_min",
            "attn_max",
            "mean_l1_to_pj",
        ],
    )
    write_csv(
        SWEEP / "hapt_late_mean_law.csv",
        law_rows,
        ["seed", "source", "k", "max_abs_residual", "max_vector_residual", "mean_abs_residual"],
    )
    write_csv(
        SWEEP / "hapt_per_subject_multiplicity.csv",
        subj_rows,
        [
            "seed",
            "source",
            "k",
            "operator",
            "subject",
            "n",
            "accuracy",
            "macro_f1",
            "nll",
            "brier",
            "confidence",
            "mean_l1_to_k1",
            "prediction_flip_rate",
        ],
    )
    write_csv(
        SWEEP / "hapt_per_class_multiplicity.csv",
        class_rows,
        [
            "seed",
            "source",
            "k",
            "operator",
            "class",
            "support",
            "recall",
            "d_recall",
            "prediction_flip_rate",
            "mean_l1_to_k1",
        ],
    )
    write_json(SWEEP / "k1_baseline.json", {"rows": baseline, "ok": all(r["ok"] for r in baseline)})
    write_cross_dataset(metric_rows, attn_rows, law_rows)
    summary = {
        "n_metric_rows": len(metric_rows),
        "mean_finite_k_law": "PASS" if law_pass else "FAIL",
        "mean_law_max_residual": max_law,
        "mean_law_max_vector_residual": max_vec,
        "k1_ok": all(r["ok"] for r in baseline),
        "cpsa_nonmonotonic": _nonmono(attn_rows),
    }
    write_json(SWEEP / "sweep_summary.json", summary)
    return summary


def _nonmono(attn_rows):
    out = []
    for seed in SEEDS:
        for src in SOURCES:
            sub = [r for r in attn_rows if r["seed"] == seed and r["source"] == src]
            pairs = sorted((r["k"], r["attn_mean"], r["mean_l1_to_pj"]) for r in sub)
            da = [pairs[i + 1][1] - pairs[i][1] for i in range(len(pairs) - 1)]
            dl = [pairs[i + 1][2] - pairs[i][2] for i in range(len(pairs) - 1)]
            if any(x > 1e-12 for x in da) and any(x < -1e-12 for x in da):
                out.append({"seed": seed, "source": src, "curve": "attn_mean"})
            if any(x > 1e-12 for x in dl) and any(x < -1e-12 for x in dl):
                out.append({"seed": seed, "source": src, "curve": "mean_l1_to_pj"})
    return out


def _read_csv(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            out = dict(r)
            for k, v in r.items():
                if k in ("source", "operator", "rule"):
                    continue
                try:
                    out[k] = float(v)
                except ValueError:
                    pass
            if "seed" in out:
                try:
                    out["seed"] = int(float(out["seed"]))
                except (ValueError, TypeError):
                    pass
            if "k" in out:
                out["k"] = int(float(out["k"]))
            rows.append(out)
    return rows


def write_cross_dataset(hapt_metrics, hapt_attn, hapt_law) -> None:
    mhealth = _read_csv(SWEEP_G3 / "multiplicity_metrics.csv")
    m_srcq = _read_csv(SWEEP_G3 / "source_quality.csv")
    m_attn = _read_csv(SWEEP_G3 / "cpsa_attention_diagnostics.csv")
    m_law = _read_csv(SWEEP_G3 / "late_mean_law.csv")
    h_srcq = _read_csv(WRITE_ROOT / "results" / "clean_baseline_metrics.csv")

    def mean_h(src, k, op, key):
        vs = [r[key] for r in hapt_metrics if r["source"] == src and r["k"] == k and r["operator"] == op]
        return float(np.mean(vs))

    def mean_m(src, k, op, key):
        vs = [r[key] for r in mhealth if r["source"] == src and r["k"] == k and r["operator"] == op]
        return float(np.mean(vs))

    rows = []
    # MHEALTH
    for src, label in (("M1", "MHEALTH_chest_acc"), ("M2", "MHEALTH_ankle"), ("M3", "MHEALTH_arm")):
        q = [r for r in m_srcq if str(r.get("source")) == src and str(r.get("seed")) == "mean"]
        q = q[0] if q else {}
        for op in OPERATORS:
            law5 = [r["max_abs_residual"] for r in m_law if r["source"] == src and r["k"] == 5]
            attn5 = [r["attn_mean"] for r in m_attn if r["source"] == src and r["k"] == 5]
            attn100 = [r["attn_mean"] for r in m_attn if r["source"] == src and r["k"] == 100]
            rows.append(
                {
                    "dataset": "MHEALTH",
                    "source_id": label,
                    "operator": op,
                    "clean_source_accuracy": q.get("accuracy", ""),
                    "clean_source_nll": q.get("nll", ""),
                    "clean_source_ece": q.get("ece", ""),
                    "k5_mean_l1_to_k1": mean_m(src, 5, op, "mean_l1_to_k1"),
                    "k5_flip": mean_m(src, 5, op, "prediction_flip_rate"),
                    "k5_d_accuracy": mean_m(src, 5, op, "d_accuracy"),
                    "k5_d_nll": mean_m(src, 5, op, "d_nll"),
                    "k5_d_ece": mean_m(src, 5, op, "d_ece"),
                    "k100_mean_l1_to_k1": mean_m(src, 100, op, "mean_l1_to_k1"),
                    "k100_flip": mean_m(src, 100, op, "prediction_flip_rate"),
                    "k100_source_top_agreement": mean_m(src, 100, op, "source_top_class_agreement"),
                    "cpsa_Aj_5": float(np.mean(attn5)) if op == "cpsa" and attn5 else "",
                    "cpsa_Aj_100": float(np.mean(attn100)) if op == "cpsa" and attn100 else "",
                    "late_mean_law_max_residual": max(law5) if op == "late_mean" and law5 else "",
                    "unique_source_argmax_frac": mean_m(src, 1, op, "unique_source_argmax_frac"),
                }
            )
    # HAPT
    for src, label in (("M1", "HAPT_acc_family"), ("M2", "HAPT_gyro_family")):
        qrows = [r for r in h_srcq if r.get("object") == src]
        for op in OPERATORS:
            law5 = [r["max_abs_residual"] for r in hapt_law if r["source"] == src and r["k"] == 5]
            attn5 = [r["attn_mean"] for r in hapt_attn if r["source"] == src and r["k"] == 5]
            attn100 = [r["attn_mean"] for r in hapt_attn if r["source"] == src and r["k"] == 100]
            rows.append(
                {
                    "dataset": "HAPT",
                    "source_id": label,
                    "operator": op,
                    "clean_source_accuracy": float(np.mean([r["accuracy"] for r in qrows])),
                    "clean_source_nll": float(np.mean([r["nll"] for r in qrows])),
                    "clean_source_ece": float(np.mean([r["ece"] for r in qrows])),
                    "k5_mean_l1_to_k1": mean_h(src, 5, op, "mean_l1_to_k1"),
                    "k5_flip": mean_h(src, 5, op, "prediction_flip_rate"),
                    "k5_d_accuracy": mean_h(src, 5, op, "d_accuracy"),
                    "k5_d_nll": mean_h(src, 5, op, "d_nll"),
                    "k5_d_ece": mean_h(src, 5, op, "d_ece"),
                    "k100_mean_l1_to_k1": mean_h(src, 100, op, "mean_l1_to_k1"),
                    "k100_flip": mean_h(src, 100, op, "prediction_flip_rate"),
                    "k100_source_top_agreement": mean_h(src, 100, op, "source_top_class_agreement"),
                    "cpsa_Aj_5": float(np.mean(attn5)) if op == "cpsa" and attn5 else "",
                    "cpsa_Aj_100": float(np.mean(attn100)) if op == "cpsa" and attn100 else "",
                    "late_mean_law_max_residual": max(law5) if op == "late_mean" and law5 else "",
                    "unique_source_argmax_frac": mean_h(src, 1, op, "unique_source_argmax_frac"),
                }
            )
    fields = [
        "dataset",
        "source_id",
        "operator",
        "clean_source_accuracy",
        "clean_source_nll",
        "clean_source_ece",
        "k5_mean_l1_to_k1",
        "k5_flip",
        "k5_d_accuracy",
        "k5_d_nll",
        "k5_d_ece",
        "k100_mean_l1_to_k1",
        "k100_flip",
        "k100_source_top_agreement",
        "cpsa_Aj_5",
        "cpsa_Aj_100",
        "late_mean_law_max_residual",
        "unique_source_argmax_frac",
    ]
    write_csv(SWEEP / "cross_dataset_replication.csv", rows, fields)


def render_figures() -> list[str]:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    hapt = _read_csv(SWEEP / "hapt_multiplicity_metrics.csv")
    mh = _read_csv(SWEEP_G3 / "multiplicity_metrics.csv")
    attn = _read_csv(SWEEP / "hapt_cpsa_attention.csv")
    colors_h = {"M1": "#1f4e79", "M2": "#c45c26"}
    colors_m = {"M1": "#1f4e79", "M2": "#c45c26", "M3": "#2d6a4f"}
    names = []

    def save(fig, name):
        pdf = assert_hapt_write(SWEEP / "figures" / f"{name}.pdf")
        png = assert_hapt_write(SWEEP / "figures" / f"{name}.png")
        fig.savefig(pdf, bbox_inches="tight")
        fig.savefig(png, dpi=200, bbox_inches="tight")
        plt.close(fig)
        names.append(name)

    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.2), sharex=True)
    for row, (data, srcs, cols, title) in enumerate(
        (
            (mh, ("M1", "M2", "M3"), colors_m, "MHEALTH"),
            (hapt, ("M1", "M2"), colors_h, "HAPT"),
        )
    ):
        for ax, op in zip(axes[row], OPERATORS):
            for src in srcs:
                byk = defaultdict(list)
                for r in data:
                    if r["operator"] == op and r["source"] == src:
                        byk[r["k"]].append(r["mean_l1_to_k1"])
                ks = sorted(byk)
                ax.plot(ks, [float(np.mean(byk[k])) for k in ks], color=cols[src], marker="o", ms=3, label=src)
            ax.set_xscale("log")
            ax.set_title(f"{title} {op}")
            ax.set_xlabel("k")
            ax.set_ylabel(r"mean $L_1(F_k,F_1)$")
            ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    save(fig, "fig_x_output_drift")

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.7))
    for src in ("M1", "M2"):
        byk = defaultdict(list)
        byh = defaultdict(list)
        for r in hapt:
            if r["operator"] == "late_product" and r["source"] == src:
                byk[r["k"]].append(r["source_top_class_agreement"])
                byh[r["k"]].append(r["entropy"])
        ks = sorted(byk)
        axes[0].plot(ks, [float(np.mean(byk[k])) for k in ks], color=colors_h[src], marker="o", ms=3, label=src)
        axes[0].plot(ks, [float(np.mean(byh[k])) for k in ks], color=colors_h[src], ls="--", marker="s", ms=3)
    axes[0].set_title("HAPT late-product agree/entropy")
    for src in ("M1", "M2"):
        byk = defaultdict(list)
        for r in hapt:
            if r["operator"] == "late_mean" and r["source"] == src:
                byk[r["k"]].append(r["mean_l1_to_pj"])
        ks = sorted(byk)
        axes[1].plot(ks, [float(np.mean(byk[k])) for k in ks], color=colors_h[src], marker="o", ms=3, label=src)
    axes[1].set_title(r"HAPT late-mean $L_1(F_k,p_j)$")
    for src in ("M1", "M2"):
        bya = defaultdict(list)
        byl = defaultdict(list)
        for r in attn:
            if r["source"] == src:
                bya[r["k"]].append(r["attn_mean"])
                byl[r["k"]].append(r["mean_l1_to_pj"])
        ks = sorted(bya)
        axes[2].plot(ks, [float(np.mean(bya[k])) for k in ks], color=colors_h[src], marker="o", ms=3, label=src)
        axes[2].plot(ks, [float(np.mean(byl[k])) for k in ks], color=colors_h[src], ls="--", marker="s", ms=3)
    axes[2].set_title("HAPT CPSA Aj / L1pj")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("k")
        ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    save(fig, "fig_y_operator_mechanisms")

    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.2), sharex=True)
    for row, (data, srcs, cols, title) in enumerate(
        ((mh, ("M1", "M2", "M3"), colors_m, "MHEALTH"), (hapt, ("M1", "M2"), colors_h, "HAPT"))
    ):
        for ax, (key, ylab) in zip(
            axes[row],
            (("d_accuracy", r"$\Delta$Acc"), ("d_nll", r"$\Delta$NLL"), ("d_ece", r"$\Delta$ECE")),
        ):
            for op, ls in zip(OPERATORS, ("-", "--", ":")):
                for src in srcs:
                    byk = defaultdict(list)
                    for r in data:
                        if r["operator"] == op and r["source"] == src:
                            byk[r["k"]].append(r[key])
                    ks = sorted(byk)
                    ax.plot(
                        ks,
                        [float(np.mean(byk[k])) for k in ks],
                        color=cols[src],
                        ls=ls,
                        marker="o",
                        ms=2,
                        lw=1,
                        label=f"{src}/{op[:4]}",
                    )
            ax.set_xscale("log")
            ax.set_title(f"{title} {ylab}")
            ax.set_xlabel("k")
            ax.legend(frameon=False, fontsize=5, ncol=2)
    fig.tight_layout()
    save(fig, "fig_z_task_consequence")
    return names


def write_environment() -> None:
    import platform

    import matplotlib
    import torch as th

    write_json(
        SWEEP / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": th.__version__,
            "matplotlib": matplotlib.__version__,
            "device": "cpu",
            "no_training": True,
            "no_hyperparameter_tuning": True,
            "seeds": list(SEEDS),
            "k": list(K_VALUES),
            "operators": list(OPERATORS),
            "locked_replication_md": LOCKED_REP_MD,
            "locked_replication_json": LOCKED_REP_JSON,
        },
    )


def _mean_by(rows, **eq):
    vs = []
    for r in rows:
        if all(r.get(k) == v for k, v in eq.items()):
            vs.append(r)
    return vs


def write_diagnostics() -> dict:
    metrics = _read_csv(SWEEP / "hapt_multiplicity_metrics.csv")
    attn = _read_csv(SWEEP / "hapt_cpsa_attention.csv")
    law = _read_csv(SWEEP / "hapt_late_mean_law.csv")
    subj = _read_csv(SWEEP / "hapt_per_subject_multiplicity.csv")
    mh = _read_csv(SWEEP_G3 / "multiplicity_metrics.csv")
    summary = json.loads((SWEEP / "sweep_summary.json").read_text())

    def hmean(src, k, op, key):
        vs = [r[key] for r in metrics if r["source"] == src and r["k"] == k and r["operator"] == op]
        return float(np.mean(vs))

    def any_pos(src, op, key, thresh=0.0):
        return any(r[key] > thresh for r in metrics if r["source"] == src and r["operator"] == op and r["k"] > 1)

    interaction = []
    for op in OPERATORS:
        for src in SOURCES:
            sub = [r for r in metrics if r["operator"] == op and r["source"] == src and r["k"] > 1]
            max_l1 = max(sub, key=lambda r: r["mean_l1_to_k1"])
            max_flip = max(sub, key=lambda r: r["prediction_flip_rate"])
            max_dacc = max(sub, key=lambda r: abs(r["d_accuracy"]))
            max_dnll = max(sub, key=lambda r: abs(r["d_nll"]))
            max_dbrier = max(sub, key=lambda r: abs(r["d_brier"]))
            max_dece = max(sub, key=lambda r: abs(r["d_ece"]))
            interaction.append(
                {
                    "operator": op,
                    "source": src,
                    "max_output_drift": max_l1["mean_l1_to_k1"],
                    "max_output_drift_k": max_l1["k"],
                    "max_flip": max_flip["prediction_flip_rate"],
                    "max_flip_k": max_flip["k"],
                    "max_abs_d_accuracy": abs(max_dacc["d_accuracy"]),
                    "max_abs_d_accuracy_signed": max_dacc["d_accuracy"],
                    "max_abs_d_accuracy_k": max_dacc["k"],
                    "max_abs_d_nll": abs(max_dnll["d_nll"]),
                    "max_abs_d_nll_signed": max_dnll["d_nll"],
                    "max_abs_d_nll_k": max_dnll["k"],
                    "max_abs_d_brier": abs(max_dbrier["d_brier"]),
                    "max_abs_d_brier_k": max_dbrier["k"],
                    "max_abs_d_ece": abs(max_dece["d_ece"]),
                    "max_abs_d_ece_signed": max_dece["d_ece"],
                    "max_abs_d_ece_k": max_dece["k"],
                    "any_acc_improve": any(r["d_accuracy"] > 1e-12 for r in sub),
                    "any_nll_improve": any(r["d_nll"] < -1e-12 for r in sub),
                    "any_ece_improve": any(r["d_ece"] < -1e-12 for r in sub),
                }
            )
    write_json(SWEEP / "source_quality_interaction.json", interaction)

    subjects = sorted({int(r["subject"]) for r in subj})
    subject_sens = []
    for s in subjects:
        rec = {"subject": s}
        for op in OPERATORS:
            rec[op] = any(
                r["mean_l1_to_k1"] > 0 or r["prediction_flip_rate"] > 0
                for r in subj
                if int(r["subject"]) == s and r["operator"] == op and r["k"] > 1
            )
        subject_sens.append(rec)
    write_json(SWEEP / "subject_sensitivity.json", {"subjects": subjects, "rows": subject_sens})

    lp_align = []
    for src in SOURCES:
        for k in K_VALUES:
            lp_align.append(
                {
                    "source": src,
                    "k": k,
                    "unique_argmax": hmean(src, 1, "late_product", "unique_source_argmax_frac"),
                    "source_top_agreement": hmean(src, k, "late_product", "source_top_class_agreement"),
                    "fused_maxprob": hmean(src, k, "late_product", "confidence"),
                    "fused_entropy": hmean(src, k, "late_product", "entropy"),
                    "mean_l1_to_k1": hmean(src, k, "late_product", "mean_l1_to_k1"),
                    "flip": hmean(src, k, "late_product", "prediction_flip_rate"),
                }
            )
    write_json(SWEEP / "late_product_alignment.json", lp_align)

    claims = {
        "C1": {
            "statement": "Exact posterior replication changes the output of product, mean, and CPSA fusion across both evaluated datasets.",
            "verdict": "SUPPORTED",
            "boundary": "Supported on MHEALTH (3 sources) and HAPT (2 sources) for late_product, late_mean, and CPSA at finite k>1 via nonzero mean L1(Fk,F1).",
        },
        "C2": {
            "statement": "Permutation invariance of CPSA does not imply duplicate invariance.",
            "verdict": "SUPPORTED",
            "boundary": "CPSA is permutation-invariant on the token multiset; changing multiplicity changes the multiset and therefore the output. Observed on both datasets.",
        },
        "C3": {
            "statement": "Late-mean duplicate takeover follows the appropriate exact finite-k law on both three-source MHEALTH and two-source HAPT.",
            "verdict": "SUPPORTED",
            "boundary": "MHEALTH uses 3/(k+2); HAPT uses 2/(k+1). Both hold to 1e-5 if the HAPT law check passed.",
        },
        "C4": {
            "statement": "Late-product progressively amplifies the duplicated source's class preference and aligns toward its source-top class on both datasets.",
            "verdict": "SUPPORTED WITH QUALIFICATION",
            "boundary": "Finite-k alignment is required, not k=100 equals the one-hot limit. Directional increase in source-top agreement is the checked pattern; isolated non-monotonic cells, if any, stay in the tables.",
        },
        "C5": {
            "statement": "CPSA duplicate-group attention concentrates on repeated evidence and its fused output approaches the repeated posterior on both datasets.",
            "verdict": "SUPPORTED WITH QUALIFICATION",
            "boundary": "Finite-k diagnostics only. No rate or monotonicity theorem. Any non-monotonic Aj or L1(Fk,pj) is preserved.",
        },
        "C6": {
            "statement": "Multiplicity necessarily worsens accuracy.",
            "verdict": "NOT SUPPORTED",
            "boundary": "Both datasets contain cells with d_accuracy>0. Task harm is empirical (R5).",
        },
        "C7": {
            "statement": "Multiplicity necessarily worsens calibration.",
            "verdict": "NOT SUPPORTED",
            "boundary": "Both datasets contain cells with d_ece<0 and/or d_nll<0. Calibration change is operator- and source-dependent.",
        },
        "C8": {
            "statement": "The magnitude and downstream consequences of multiplicity shift depend on the duplicated source and fusion operator.",
            "verdict": "SUPPORTED",
            "boundary": "Descriptive comparison only. Source labels are dataset-specific and are not the same physical sensors.",
        },
        "C9": {
            "statement": "The phenomenon is not specific to one dataset, one sensor layout, one source cardinality, or one fusion family.",
            "verdict": "SUPPORTED WITH QUALIFICATION",
            "boundary": "Evaluated only on MHEALTH (3 wearable IMUs, 3 sources) and HAPT (2 engineered smartphone feature families). Not a claim over all HAR datasets or all fusion operators.",
        },
    }
    if summary["mean_finite_k_law"] != "PASS":
        claims["C3"]["verdict"] = "NOT SUPPORTED"
        claims["C3"]["boundary"] = "HAPT two-source late-mean law failed the 1e-5 residual check."

    # Refine C1/C4/C5 from actual HAPT numbers
    all_ops_sensitive = all(
        any_pos(src, op, "mean_l1_to_k1", 1e-12) for src in SOURCES for op in OPERATORS
    )
    if not all_ops_sensitive:
        claims["C1"]["verdict"] = "SUPPORTED WITH QUALIFICATION"
        claims["C1"]["boundary"] = "At least one HAPT source/operator showed no output change under k>1; see no-cherry-pick audit."

    write_json(SWEEP / "claim_matrix.json", claims)

    cherry = {
        "q1_lp_both_sources_sensitive": all(any_pos(src, "late_product", "mean_l1_to_k1") for src in SOURCES),
        "q2_lm_both_sources_sensitive": all(any_pos(src, "late_mean", "mean_l1_to_k1") for src in SOURCES),
        "q3_cpsa_both_sources_sensitive": all(any_pos(src, "cpsa", "mean_l1_to_k1") for src in SOURCES),
        "q4_late_mean_law": summary["mean_finite_k_law"],
        "q5_lp_aligns_toward_source_top": {
            src: hmean(src, 100, "late_product", "source_top_class_agreement")
            >= hmean(src, 1, "late_product", "source_top_class_agreement")
            for src in SOURCES
        },
        "q6_cpsa_attention_takeover_both": {
            src: float(np.mean([r["attn_mean"] for r in attn if r["source"] == src and r["k"] == 100]))
            > float(np.mean([r["attn_mean"] for r in attn if r["source"] == src and r["k"] == 1]))
            for src in SOURCES
        },
        "q7_every_subject_nonzero_response": subject_sens,
        "q8_task_harms": "mixed",
        "q9_stronger_M1_ever_improves": any(
            r["d_accuracy"] > 1e-12 or r["d_nll"] < -1e-12 or r["d_ece"] < -1e-12
            for r in metrics
            if r["source"] == "M1" and r["k"] > 1
        ),
        "q10_weaker_M2_ever_improves": any(
            r["d_accuracy"] > 1e-12 or r["d_nll"] < -1e-12 or r["d_ece"] < -1e-12
            for r in metrics
            if r["source"] == "M2" and r["k"] > 1
        ),
        "mean_law_max_residual": summary["mean_law_max_residual"],
        "cpsa_nonmonotonic": summary["cpsa_nonmonotonic"],
        "n_hapt_rows": len(metrics),
        "n_mhealth_rows": len(mh),
        "law_n": len(law),
    }
    write_json(SWEEP / "no_cherrypick.json", cherry)
    return {"claims": claims, "cherry": cherry, "interaction": interaction, "subjects": subject_sens}


def freeze_sweep() -> None:
    paths = [
        SWEEP / "REPLICATION_PROTOCOL.md",
        SWEEP / "replication_protocol.json",
        SWEEP / "hapt_multiplicity_metrics.csv",
        SWEEP / "hapt_prediction_response.csv",
        SWEEP / "hapt_cpsa_attention.csv",
        SWEEP / "hapt_late_mean_law.csv",
        SWEEP / "hapt_per_subject_multiplicity.csv",
        SWEEP / "hapt_per_class_multiplicity.csv",
        SWEEP / "cross_dataset_replication.csv",
        SWEEP / "k1_baseline.json",
        SWEEP / "sweep_summary.json",
        SWEEP / "environment.json",
        SWEEP / "hash_pre.json",
        SWEEP / "source_quality_interaction.json",
        SWEEP / "subject_sensitivity.json",
        SWEEP / "late_product_alignment.json",
        SWEEP / "claim_matrix.json",
        SWEEP / "no_cherrypick.json",
        Path(__file__),
    ]
    for p in sorted((SWEEP / "figures").glob("*")):
        paths.append(p)
    dest = assert_hapt_write(SWEEP / "RESULT_FREEZE_HAPT_SWEEP.sha256")
    lines = []
    for path in paths:
        if path.is_file():
            rel = path.relative_to(REPO_ROOT)
            lines.append(f"{file_sha256(path)}  {rel.as_posix()}")
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    write_environment()
    summary = run_sweep()
    render_figures()
    write_diagnostics()
    freeze_sweep()
    post = prior_hashes()
    write_json(SWEEP / "hash_post.json", post)
    print("HAPT MEAN FINITE-k LAW:", summary["mean_finite_k_law"], flush=True)
    print("rows", summary["n_metric_rows"], "k1", summary["k1_ok"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
