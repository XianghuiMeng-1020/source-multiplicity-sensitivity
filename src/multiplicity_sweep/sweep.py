"""Clean-source × multiplicity × operator response map.

Reads Gate-2 posterior caches and frozen CPSA checkpoints. Writes only under
research/multiplicity_sweep/. Never trains. Never writes Phase-4 or Gate-2 files.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from datasets import load_mhealth  # noqa: E402
from metrics import summarize  # noqa: E402
from models import late_mean, late_product  # noqa: E402

from multiplicity_replay.hashes import audit_freeze  # noqa: E402
from multiplicity_replay.replay import (  # noqa: E402
    CKPT_DIR,
    FROZEN_RESULTS,
    REPLAY_ROOT,
    REPO_ROOT,
    file_sha256,
    load_cpsa,
)

SWEEP_ROOT = (REPO_ROOT / "research" / "multiplicity_sweep").resolve()
SEEDS = (0, 1, 2)
SOURCES = ("M1", "M2", "M3")
SOURCE_KEY = {"M1": "p1_test", "M2": "p2_test", "M3": "p3_test"}
K_VALUES = (1, 2, 3, 5, 8, 16, 32, 64, 100)
OPERATORS = ("late_product", "late_mean", "cpsa")
METRIC_KEYS = ("accuracy", "ece", "nll", "brier", "confidence")
MEAN_LAW_TOL = 1e-5
BASELINE_TOL = 1e-5
GATE2_CACHE_HASHES = {
    0: "ee66750753d8ddd5caae55310f713003ce91abb852dbe2a0e334e8c6e2abbacd",
    1: "a3e3b7a0c7d1076057afe9117987a105acbd573329c6a38166bbede383c33780",
    2: "a23a4ca5cba5d015709944c4e095b8f79c22e541dbe942bd05a509de3b042d15",
}


def assert_sweep_write(path: Path) -> Path:
    dest = Path(path).resolve()
    if FROZEN_RESULTS == dest or FROZEN_RESULTS in dest.parents:
        raise PermissionError(f"sweep write refused inside frozen results: {dest}")
    if REPLAY_ROOT == dest or REPLAY_ROOT in dest.parents:
        raise PermissionError(f"sweep write refused inside Gate-2 replay: {dest}")
    paper = (REPO_ROOT / "paper").resolve()
    if paper == dest or paper in dest.parents:
        raise PermissionError(f"sweep write refused inside paper/: {dest}")
    if SWEEP_ROOT != dest and SWEEP_ROOT not in dest.parents:
        raise PermissionError(f"sweep write refused outside {SWEEP_ROOT}: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    return dest


def write_json(path: Path, payload) -> Path:
    dest = assert_sweep_write(path)
    dest.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    return dest


def ffmt(x) -> str:
    return format(float(x), ".17g")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> Path:
    dest = assert_sweep_write(path)
    with dest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            out = {}
            for k in fieldnames:
                v = row[k]
                out[k] = ffmt(v) if isinstance(v, (float, np.floating)) else v
            w.writerow(out)
    return dest


def load_seed_cache(seed: int) -> dict[str, np.ndarray]:
    path = REPLAY_ROOT / "caches" / f"seed{seed}.npz"
    digest = file_sha256(path)
    if digest != GATE2_CACHE_HASHES[seed]:
        raise RuntimeError(f"Gate-2 cache hash changed for seed{seed}: {digest}")
    z = np.load(path)
    out = {k: np.asarray(z[k]) for k in z.files}
    z.close()
    return out


def token_list(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, source: str, k: int) -> list[np.ndarray]:
    if k < 1:
        raise ValueError("k is total source multiplicity and must be >= 1")
    parts = {"M1": p1, "M2": p2, "M3": p3}
    if source not in parts:
        raise ValueError(source)
    if k == 1:
        return [p1, p2, p3]
    others = [parts[s] for s in SOURCES if s != source]
    return [parts[source]] * int(k) + others


def stack_tokens(parts: list[np.ndarray]) -> np.ndarray:
    return np.stack(parts, axis=1)


@torch.no_grad()
def fuse_cpsa_with_attention(model, tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reuse ContextualPSA.forward(return_diagnostics=True). Inference only."""
    model.eval()
    t = torch.from_numpy(np.asarray(tokens, dtype=np.float32))
    fused, diag = model(t, return_diagnostics=True)
    return fused.cpu().numpy(), diag["attention"].cpu().numpy()


def fuse(operator: str, parts: list[np.ndarray], cpsa_model) -> tuple[np.ndarray, np.ndarray | None]:
    if operator == "late_product":
        return late_product(parts), None
    if operator == "late_mean":
        return late_mean(parts), None
    if operator == "cpsa":
        fused, attn = fuse_cpsa_with_attention(cpsa_model, stack_tokens(parts))
        return fused, attn
    raise ValueError(operator)


def duplicate_attention_mass(attn: np.ndarray, source: str, k: int) -> np.ndarray:
    if k == 1:
        return attn[:, SOURCES.index(source)]
    return attn[:, :k].sum(axis=1)


def l1_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a - b).sum(axis=1)


def unique_argmax_mask(p: np.ndarray) -> np.ndarray:
    mx = p.max(axis=1, keepdims=True)
    return (p == mx).sum(axis=1) == 1


def per_example_nll(probs: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return -np.log(np.clip(probs[np.arange(len(y)), y], eps, 1.0))


def per_example_brier(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1.0
    return np.sum((probs - onehot) ** 2, axis=1)


def load_subjects_aligned(y_test: np.ndarray) -> np.ndarray:
    ds = load_mhealth()
    y_loader = np.asarray(ds.test.y)
    if y_loader.shape != y_test.shape or not np.array_equal(y_loader, y_test):
        raise RuntimeError("cached y_test is not aligned with load_mhealth().test.y")
    subj = np.asarray(ds.test.subject)
    if subj.shape[0] != y_test.shape[0]:
        raise RuntimeError("subject vector length mismatch")
    return subj


def source_quality_rows(caches: dict[int, dict[str, np.ndarray]]) -> list[dict]:
    rows = []
    for seed, cache in caches.items():
        y = cache["y_test"]
        for source in SOURCES:
            p = cache[SOURCE_KEY[source]]
            rec = summarize(p, y)
            pred = p.argmax(axis=1)
            rec.update(
                {
                    "seed": int(seed),
                    "source": source,
                    "argmax_correct_frac": float((pred == y).mean()),
                }
            )
            rows.append(rec)
    for source in SOURCES:
        src_rows = [r for r in rows if r["source"] == source]
        mean = {"seed": "mean", "source": source}
        for key in (
            "accuracy",
            "ece",
            "nll",
            "brier",
            "confidence",
            "conf_acc_gap",
            "entropy",
            "argmax_correct_frac",
            "n",
        ):
            mean[key] = float(np.mean([r[key] for r in src_rows]))
        rows.append(mean)
    return rows


def run_sweep() -> dict:
    caches = {int(s): load_seed_cache(int(s)) for s in SEEDS}
    y_ref = caches[0]["y_test"]
    for seed, cache in caches.items():
        if not np.array_equal(cache["y_test"], y_ref):
            raise RuntimeError(f"y_test differs at seed {seed}")
        for key in ("p1_test", "p2_test", "p3_test"):
            if cache[key].shape != (1987, 12):
                raise RuntimeError(f"{key} seed{seed} shape {cache[key].shape}")
    subjects = load_subjects_aligned(y_ref)
    if sorted(set(int(s) for s in subjects.tolist())) != [8, 9, 10]:
        raise RuntimeError(f"unexpected test subjects {sorted(set(subjects.tolist()))}")

    source_rows = source_quality_rows(caches)
    write_csv(
        SWEEP_ROOT / "source_quality.csv",
        source_rows,
        [
            "seed",
            "source",
            "accuracy",
            "ece",
            "nll",
            "brier",
            "confidence",
            "conf_acc_gap",
            "entropy",
            "argmax_correct_frac",
            "n",
        ],
    )

    cpsa_models = {int(s): load_cpsa(CKPT_DIR / f"seed{s}_cpsa.pt") for s in SEEDS}

    metric_rows: list[dict] = []
    attn_rows: list[dict] = []
    law_rows: list[dict] = []
    subject_rows: list[dict] = []
    baseline = {}
    f1 = {}
    fused_mem: dict[tuple, np.ndarray] = {}

    for seed in SEEDS:
        cache = caches[int(seed)]
        p1, p2, p3 = cache["p1_test"], cache["p2_test"], cache["p3_test"]
        y = cache["y_test"]
        parts0 = {"M1": p1, "M2": p2, "M3": p3}
        cpsa = cpsa_models[int(seed)]

        for operator in OPERATORS:
            fused_k1 = {}
            for source in SOURCES:
                toks = token_list(p1, p2, p3, source, 1)
                fused, attn = fuse(operator, toks, cpsa)
                fused_k1[source] = fused
                fused_mem[(seed, source, 1, operator)] = fused
            max_cross = max(
                float(np.max(np.abs(fused_k1["M1"] - fused_k1["M2"]))),
                float(np.max(np.abs(fused_k1["M1"] - fused_k1["M3"]))),
                float(np.max(np.abs(fused_k1["M2"] - fused_k1["M3"]))),
            )
            rec = {
                "seed": int(seed),
                "operator": operator,
                "max_abs_cross_source_k1": max_cross,
                "cross_source_ok": max_cross <= BASELINE_TOL,
            }
            if operator in ("late_product", "cpsa"):
                gate2 = cache[f"fused_{operator}_A"]
                max_g2 = float(np.max(np.abs(fused_k1["M1"] - gate2)))
                rec["max_abs_vs_gate2_arm_A"] = max_g2
                rec["gate2_arm_A_ok"] = max_g2 <= BASELINE_TOL
            else:
                rec["max_abs_vs_gate2_arm_A"] = None
                rec["gate2_arm_A_ok"] = None
            baseline[(seed, operator)] = rec
            f1[(seed, operator)] = fused_k1["M1"]

        for source in SOURCES:
            pj = parts0[source]
            for k in K_VALUES:
                toks = token_list(p1, p2, p3, source, k)
                if k != 1 and sum(t is pj for t in toks) != k:
                    raise RuntimeError(f"{source} k={k} does not contain k exact copies")
                if k != 1 and len(toks) != k + 2:
                    raise RuntimeError(f"token count {len(toks)} != k+2")
                for operator in OPERATORS:
                    fused, attn = fuse(operator, toks, cpsa)
                    fused_mem[(seed, source, k, operator)] = fused
                    if attn is not None:
                        mass = duplicate_attention_mass(attn, source, k)
                        attn_rows.append(_attn_row(seed, source, k, mass, fused, pj))
                    if operator == "late_mean":
                        law_rows.append(_mean_law_row(seed, source, k, fused, f1[(seed, "late_mean")], pj))

    if any(not v["cross_source_ok"] for v in baseline.values()):
        raise RuntimeError(f"k=1 cross-source mismatch: {baseline}")
    if any(v["gate2_arm_A_ok"] is False for v in baseline.values()):
        raise RuntimeError(f"k=1 Gate-2 arm A mismatch: {baseline}")

    for seed in SEEDS:
        cache = caches[int(seed)]
        y = cache["y_test"]
        parts0 = {s: cache[SOURCE_KEY[s]] for s in SOURCES}
        unique_frac = {
            s: float(unique_argmax_mask(parts0[s]).mean()) for s in SOURCES
        }
        for operator in OPERATORS:
            base_fused = f1[(seed, operator)]
            base_metrics = summarize(base_fused, y)
            pred1 = base_fused.argmax(axis=1)
            for source in SOURCES:
                pj = parts0[source]
                src_top = pj.argmax(axis=1)
                for k in K_VALUES:
                    fused = fused_mem[(seed, source, k, operator)]
                    rec = summarize(fused, y)
                    predk = fused.argmax(axis=1)
                    l1_f1 = l1_rows(fused, base_fused)
                    l1_pj = l1_rows(fused, pj)
                    flips = (predk != pred1).astype(np.float64)
                    agree = (predk == src_top).astype(np.float64)
                    row = {
                        "seed": int(seed),
                        "source": source,
                        "k": int(k),
                        "operator": operator,
                        "n_tokens": 3 if k == 1 else int(k + 2),
                        "accuracy": rec["accuracy"],
                        "ece": rec["ece"],
                        "nll": rec["nll"],
                        "brier": rec["brier"],
                        "confidence": rec["confidence"],
                        "conf_acc_gap": rec["conf_acc_gap"],
                        "entropy": rec["entropy"],
                        "d_accuracy": rec["accuracy"] - base_metrics["accuracy"],
                        "d_ece": rec["ece"] - base_metrics["ece"],
                        "d_nll": rec["nll"] - base_metrics["nll"],
                        "d_brier": rec["brier"] - base_metrics["brier"],
                        "d_confidence": rec["confidence"] - base_metrics["confidence"],
                        "mean_l1_to_k1": float(l1_f1.mean()),
                        "prediction_flip_rate": float(flips.mean()),
                        "mean_l1_to_pj": float(l1_pj.mean()),
                        "source_top_class_agreement": float(agree.mean()),
                        "unique_source_argmax_frac": unique_frac[source],
                        "n": rec["n"],
                    }
                    metric_rows.append(row)
                    for subj in (8, 9, 10):
                        m = subjects == subj
                        subject_rows.append(
                            {
                                "seed": int(seed),
                                "source": source,
                                "k": int(k),
                                "operator": operator,
                                "subject": int(subj),
                                "n": int(m.sum()),
                                "accuracy": float((predk[m] == y[m]).mean()),
                                "nll": float(per_example_nll(fused[m], y[m]).mean()),
                                "brier": float(per_example_brier(fused[m], y[m]).mean()),
                                "confidence": float(fused[m].max(axis=1).mean()),
                                "prediction_flip_rate": float(flips[m].mean()),
                                "mean_l1_to_k1": float(l1_f1[m].mean()),
                            }
                        )

    write_csv(
        SWEEP_ROOT / "multiplicity_metrics.csv",
        metric_rows,
        [
            "seed",
            "source",
            "k",
            "operator",
            "n_tokens",
            "accuracy",
            "ece",
            "nll",
            "brier",
            "confidence",
            "conf_acc_gap",
            "entropy",
            "d_accuracy",
            "d_ece",
            "d_nll",
            "d_brier",
            "d_confidence",
            "mean_l1_to_k1",
            "prediction_flip_rate",
            "mean_l1_to_pj",
            "source_top_class_agreement",
            "unique_source_argmax_frac",
            "n",
        ],
    )
    write_csv(
        SWEEP_ROOT / "cpsa_attention_diagnostics.csv",
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
        SWEEP_ROOT / "late_mean_law.csv",
        law_rows,
        [
            "seed",
            "source",
            "k",
            "max_abs_residual",
            "mean_abs_residual",
            "max_obs_l1",
            "max_pred_l1",
        ],
    )
    write_csv(
        SWEEP_ROOT / "per_subject_metrics.csv",
        subject_rows,
        [
            "seed",
            "source",
            "k",
            "operator",
            "subject",
            "n",
            "accuracy",
            "nll",
            "brier",
            "confidence",
            "prediction_flip_rate",
            "mean_l1_to_k1",
        ],
    )

    max_law = max(r["max_abs_residual"] for r in law_rows)
    law_pass = max_law <= MEAN_LAW_TOL
    summaries = build_summaries(
        metric_rows,
        attn_rows,
        baseline,
        source_rows,
        subject_rows,
        law_pass,
        max_law,
    )
    write_json(SWEEP_ROOT / "baseline_reproduction.json", {"rows": [baseline[k] | {"seed": k[0], "operator": k[1]} for k in baseline], "ok": True})
    write_json(SWEEP_ROOT / "summary.json", summaries)
    return {
        "metric_rows": metric_rows,
        "attn_rows": attn_rows,
        "law_rows": law_rows,
        "subject_rows": subject_rows,
        "source_rows": source_rows,
        "baseline": baseline,
        "summaries": summaries,
        "mean_law_pass": law_pass,
        "mean_law_max_residual": max_law,
    }


def _attn_row(seed, source, k, mass, fused, pj) -> dict:
    return {
        "seed": int(seed),
        "source": source,
        "k": int(k),
        "attn_mean": float(np.mean(mass)),
        "attn_median": float(np.median(mass)),
        "attn_p10": float(np.quantile(mass, 0.10)),
        "attn_p90": float(np.quantile(mass, 0.90)),
        "attn_min": float(np.min(mass)),
        "attn_max": float(np.max(mass)),
        "mean_l1_to_pj": float(l1_rows(fused, pj).mean()),
    }


def _mean_law_row(seed, source, k, fused, f1_arr, pj) -> dict:
    obs = l1_rows(fused, pj)
    pred = (3.0 / (k + 2.0)) * l1_rows(f1_arr, pj)
    resid = np.abs(obs - pred)
    return {
        "seed": int(seed),
        "source": source,
        "k": int(k),
        "max_abs_residual": float(resid.max()),
        "mean_abs_residual": float(resid.mean()),
        "max_obs_l1": float(obs.max()),
        "max_pred_l1": float(pred.max()),
    }


def _nonmonotonic(xs: list[int], ys: list[float]) -> bool:
    pairs = sorted(zip(xs, ys))
    diffs = [pairs[i + 1][1] - pairs[i][1] for i in range(len(pairs) - 1)]
    return any(d > 1e-12 for d in diffs) and any(d < -1e-12 for d in diffs)


def build_summaries(metric_rows, attn_rows, baseline, source_rows, subject_rows, law_pass, max_law) -> dict:
    def rows_op(op):
        return [r for r in metric_rows if r["operator"] == op and r["k"] != 1]

    cherry = {}
    for op in OPERATORS:
        sub = rows_op(op)
        cherry[op] = {
            "all_sources_change_outputs": all(
                any(r["source"] == s and r["mean_l1_to_k1"] > 1e-8 for r in sub) for s in SOURCES
            ),
            "max_mean_l1_to_k1": max(r["mean_l1_to_k1"] for r in sub),
            "max_flip": max(r["prediction_flip_rate"] for r in sub),
        }
    mixed = {
        "acc_up_quality_down": [
            {
                "seed": r["seed"],
                "source": r["source"],
                "k": r["k"],
                "operator": r["operator"],
                "d_accuracy": r["d_accuracy"],
                "d_nll": r["d_nll"],
                "d_brier": r["d_brier"],
                "d_ece": r["d_ece"],
            }
            for r in metric_rows
            if r["k"] != 1
            and r["d_accuracy"] > 1e-12
            and (r["d_nll"] > 1e-12 or r["d_brier"] > 1e-12 or r["d_ece"] > 1e-12)
        ],
        "ece_down_output_shift": [
            {
                "seed": r["seed"],
                "source": r["source"],
                "k": r["k"],
                "operator": r["operator"],
                "d_ece": r["d_ece"],
                "mean_l1_to_k1": r["mean_l1_to_k1"],
                "prediction_flip_rate": r["prediction_flip_rate"],
            }
            for r in metric_rows
            if r["k"] != 1 and r["d_ece"] < -1e-12 and r["mean_l1_to_k1"] > 0.05
        ],
    }
    source_cons = {}
    for op in OPERATORS:
        source_cons[op] = {}
        for source in SOURCES:
            sub = [r for r in metric_rows if r["operator"] == op and r["source"] == source]
            k5 = next(r for r in sub if r["k"] == 5)
            k100 = next(r for r in sub if r["k"] == 100)
            # max over seeds of mean metrics: take seed-wise then max abs
            by_k = {}
            for r in sub:
                by_k.setdefault(r["k"], []).append(r)
            max_flip_row = max(sub, key=lambda r: r["prediction_flip_rate"])
            max_l1_row = max(sub, key=lambda r: r["mean_l1_to_k1"])
            extremes = {}
            for name, signed in (
                ("d_accuracy", True),
                ("d_ece", True),
                ("d_nll", True),
                ("d_brier", True),
            ):
                row = max(sub, key=lambda r: abs(r[name]))
                extremes[name] = {
                    "value": row[name],
                    "abs": abs(row[name]),
                    "k": row["k"],
                    "seed": row["seed"],
                }
            mean_l1_k = {k: float(np.mean([r["mean_l1_to_k1"] for r in vs])) for k, vs in by_k.items()}
            mean_flip_k = {k: float(np.mean([r["prediction_flip_rate"] for r in vs])) for k, vs in by_k.items()}
            k5_rep = {
                "k5_mean_l1": float(np.mean([r["mean_l1_to_k1"] for r in sub if r["k"] == 5])),
                "k100_mean_l1": float(np.mean([r["mean_l1_to_k1"] for r in sub if r["k"] == 100])),
                "k5_mean_flip": float(np.mean([r["prediction_flip_rate"] for r in sub if r["k"] == 5])),
                "k100_mean_flip": float(np.mean([r["prediction_flip_rate"] for r in sub if r["k"] == 100])),
                "l1_continues_after_k5": mean_l1_k[100] > mean_l1_k[5] + 1e-6,
                "sign_d_ece_k5_vs_k100_same": (
                    np.sign(np.mean([r["d_ece"] for r in sub if r["k"] == 5]))
                    == np.sign(np.mean([r["d_ece"] for r in sub if r["k"] == 100]))
                ),
            }
            source_cons[op][source] = {
                "max_prediction_flip": {
                    "value": max_flip_row["prediction_flip_rate"],
                    "k": max_flip_row["k"],
                    "seed": max_flip_row["seed"],
                },
                "max_mean_l1_to_k1": {
                    "value": max_l1_row["mean_l1_to_k1"],
                    "k": max_l1_row["k"],
                    "seed": max_l1_row["seed"],
                },
                "extremes": extremes,
                "k5_row_seed0_if_present": k5,
                "k5_representativeness": k5_rep,
                "mean_l1_by_k": mean_l1_k,
                "mean_flip_by_k": mean_flip_k,
            }

    attn_nonmono = []
    for seed in SEEDS:
        for source in SOURCES:
            sub = [r for r in attn_rows if r["seed"] == seed and r["source"] == source]
            ks = [r["k"] for r in sub]
            if _nonmonotonic(ks, [r["attn_mean"] for r in sub]):
                attn_nonmono.append({"seed": seed, "source": source, "curve": "attn_mean"})
            if _nonmonotonic(ks, [r["mean_l1_to_pj"] for r in sub]):
                attn_nonmono.append({"seed": seed, "source": source, "curve": "mean_l1_to_pj"})

    subj_shift = {}
    for op in OPERATORS:
        subj_shift[op] = {}
        for subj in (8, 9, 10):
            sub = [r for r in subject_rows if r["operator"] == op and r["subject"] == subj and r["k"] != 1]
            subj_shift[op][str(subj)] = {
                "max_mean_l1_to_k1": max(r["mean_l1_to_k1"] for r in sub),
                "max_flip": max(r["prediction_flip_rate"] for r in sub),
                "any_output_shift": max(r["mean_l1_to_k1"] for r in sub) > 1e-8,
            }

    return {
        "mean_finite_k_law": "PASS" if law_pass else "FAIL",
        "mean_finite_k_law_max_residual": max_law,
        "lp_limit_proposition": "VALID",
        "baseline": {f"seed{s}_{op}": baseline[(s, op)] for s, op in baseline},
        "source_quality_means": [r for r in source_rows if r["seed"] == "mean"],
        "no_cherry_picking": cherry,
        "mixed_effects": {
            "n_acc_up_quality_down": len(mixed["acc_up_quality_down"]),
            "n_ece_down_output_shift": len(mixed["ece_down_output_shift"]),
            "acc_up_quality_down_examples": mixed["acc_up_quality_down"][:20],
            "ece_down_output_shift_examples": mixed["ece_down_output_shift"][:20],
        },
        "source_consistency": source_cons,
        "cpsa_nonmonotonic_curves": attn_nonmono,
        "subject_output_shift": subj_shift,
        "n_metric_rows": len(metric_rows),
    }
