"""Gate-5 mean-preserving near-duplicate robustness probe.

Constructs k-token groups whose mean equals the original source posterior,
fuses them with the same frozen operators used by the multiplicity sweep,
and compares against exact-copy trajectories and a matched single-copy
content-perturbation control.

Not a new fusion method. No training. Does not write frozen sweep artifacts.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hapt_multiplicity.io_util import load_cpsa as load_hapt_cpsa  # noqa: E402
from hapt_multiplicity.paths import CACHE_DIR as HAPT_CACHE  # noqa: E402
from hapt_multiplicity.paths import CKPT_DIR as HAPT_CKPT  # noqa: E402
from hapt_multiplicity.sweep import G5_CACHE, G5_CPSA  # noqa: E402
from hapt_multiplicity.sweep import fuse as hp_fuse  # noqa: E402
from hapt_multiplicity.sweep import token_list as hp_token_list  # noqa: E402
from multiplicity_replay.hashes import file_sha256  # noqa: E402
from multiplicity_replay.replay import CKPT_DIR as MH_CKPT  # noqa: E402
from multiplicity_replay.replay import load_cpsa as load_mh_cpsa  # noqa: E402
from multiplicity_sweep.sweep import fuse as mh_fuse  # noqa: E402
from multiplicity_sweep.sweep import load_seed_cache  # noqa: E402
from multiplicity_sweep.sweep import token_list as mh_token_list  # noqa: E402

OUT = (REPO / "research" / "gate5_near_duplicate").resolve()
SEEDS = (0, 1, 2)
K_VALUES = (1, 2, 3, 5, 8, 16, 32, 64, 100)
K_NEAR = (2, 3, 5, 8, 16, 32, 64, 100)
ETAS = (0.01, 0.05)
REGRESSION_K = (2, 5, 16, 100)
OPS = ("late_product", "late_mean", "cpsa")
MH_SOURCES = ("M1", "M2", "M3")
HP_SOURCES = ("M1", "M2")
CONSTRUCT_TOL = 1e-12
RATIO_DENOM_MIN = 1e-6
# Existing sweep mean-law residual is 1e-5; float32 operator-path mean fusion sits here.
MEAN_FUSION_TOL = 1e-5

TABLE1 = {
    ("MHEALTH", "M1", "late_product"): {2: 0.109, 5: 0.178, 16: 0.250, 100: 0.316},
    ("MHEALTH", "M1", "late_mean"): {2: 0.118, 5: 0.270, 16: 0.394, 100: 0.459},
    ("MHEALTH", "M1", "cpsa"): {2: 0.186, 5: 0.362, 16: 0.430, 100: 0.448},
    ("MHEALTH", "M2", "late_product"): {2: 0.193, 5: 0.424, 16: 0.550, 100: 0.595},
    ("MHEALTH", "M2", "late_mean"): {2: 0.139, 5: 0.318, 16: 0.464, 100: 0.540},
    ("MHEALTH", "M2", "cpsa"): {2: 0.223, 5: 0.461, 16: 0.546, 100: 0.565},
    ("MHEALTH", "M3", "late_product"): {2: 0.086, 5: 0.219, 16: 0.370, 100: 0.492},
    ("MHEALTH", "M3", "late_mean"): {2: 0.141, 5: 0.321, 16: 0.469, 100: 0.546},
    ("MHEALTH", "M3", "cpsa"): {2: 0.220, 5: 0.437, 16: 0.516, 100: 0.535},
    ("HAPT", "M1", "late_product"): {2: 0.072, 5: 0.141, 16: 0.180, 100: 0.196},
    ("HAPT", "M1", "late_mean"): {2: 0.109, 5: 0.218, 16: 0.288, 100: 0.320},
    ("HAPT", "M1", "cpsa"): {2: 0.081, 5: 0.157, 16: 0.206, 100: 0.230},
    ("HAPT", "M2", "late_product"): {2: 0.072, 5: 0.201, 16: 0.337, 100: 0.430},
    ("HAPT", "M2", "late_mean"): {2: 0.109, 5: 0.218, 16: 0.288, 100: 0.320},
    ("HAPT", "M2", "cpsa"): {2: 0.094, 5: 0.214, 16: 0.326, 100: 0.399},
}

CONDITION_FIELDS = [
    "dataset",
    "seed",
    "source",
    "operator",
    "k",
    "eta",
    "n_test",
    "n_boundary_degenerate",
    "frac_boundary_degenerate",
    "mean_l1_exact_vs_clean",
    "max_l1_exact_vs_clean",
    "mean_l1_near_vs_clean",
    "max_l1_near_vs_clean",
    "mean_l1_near_vs_exact",
    "max_l1_near_vs_exact",
    "flip_rate_near_vs_clean",
    "flip_count_near_vs_clean",
    "flip_rate_near_vs_exact",
    "flip_count_near_vs_exact",
    "mean_single_copy_l1",
    "mean_maxdir_single_copy_l1",
    "max_maxdir_single_copy_l1",
    "mean_token_l1_to_p",
    "max_token_l1_to_p",
    "frac_token_l1_ge_90pct_eta",
    "movement_ratio",
    "trajectory_error_ratio",
]


def _l1_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a.astype(np.float64) - b.astype(np.float64)).sum(axis=1)


def _linf_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)), axis=1)


def _load_frozen_l1(path: Path) -> dict[tuple, float]:
    out = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            key = (int(row["seed"]), row["source"], row["operator"], int(row["k"]))
            out[key] = float(row["mean_l1_to_k1"])
    return out


def perturbation_basis(p: np.ndarray, eta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (v, d_base, degenerate) for posterior matrix p (n, C) in float64."""
    order = np.argsort(-p, axis=1, kind="stable")
    a = order[:, 0]
    b = order[:, 1]
    n, c = p.shape
    rows = np.arange(n)
    p_a = p[rows, a]
    p_b = p[rows, b]
    v = np.zeros_like(p)
    v[rows, a] = 1.0
    v[rows, b] = -1.0
    d_target = eta / 2.0
    d_base = np.minimum(d_target, 0.9 * np.minimum(p_a, p_b))
    degenerate = p_b == 0.0
    d_base = np.where(degenerate, 0.0, d_base)
    return v, d_base, degenerate


def near_duplicate_group(p: np.ndarray, k: int, eta: float) -> tuple[list[np.ndarray], dict]:
    """Build k mean-preserving near-duplicates of p. Construction is float64."""
    if k < 1:
        raise ValueError("k >= 1")
    p64 = np.asarray(p, dtype=np.float64)
    v, d_base, degenerate = perturbation_basis(p64, eta)
    n_pair = k // 2
    qs: list[np.ndarray] = []
    token_l1 = np.zeros(p64.shape[0], dtype=np.float64)
    if n_pair == 0:
        qs.append(p64.copy())
    else:
        for ell in range(1, n_pair + 1):
            w = 0.5 + 0.5 * (ell / n_pair)
            delta = (w * d_base)[:, None] * v
            qs.append(p64 + delta)
            qs.append(p64 - delta)
            token_l1 = np.maximum(token_l1, np.abs(delta).sum(axis=1))
        if k % 2 == 1:
            qs.append(p64.copy())
    stacked = np.stack(qs, axis=0)
    group_mean = stacked.mean(axis=0)
    row_sums = stacked.sum(axis=2)
    p_row_sums = p64.sum(axis=1)
    finite = bool(np.isfinite(stacked).all())
    nonnegative = bool(np.min(stacked) >= -CONSTRUCT_TOL)
    # Frozen posteriors are float32; they need not sum to 1 at 1e-12.
    # The construction must preserve each token's original row sum and the group mean.
    row_sum_vs_one = float(np.max(np.abs(row_sums - 1.0)))
    row_sum_vs_p = float(np.max(np.abs(row_sums - p_row_sums[None, :])))
    p_row_sum_vs_one = float(np.max(np.abs(p_row_sums - 1.0)))
    mean_err = float(np.max(np.abs(group_mean - p64)))
    min_entry = float(np.min(stacked))
    n_deg = int(degenerate.sum())
    differ = True
    if k > 1 and n_pair >= 1:
        differ_mask = np.max(np.abs(stacked - stacked[:1]), axis=(0, 2)) > 0.0
        differ = bool(np.all(differ_mask | degenerate))
    ok = (
        finite
        and nonnegative
        and row_sum_vs_p <= CONSTRUCT_TOL
        and mean_err <= CONSTRUCT_TOL
        and float(np.max(token_l1)) <= eta + CONSTRUCT_TOL
        and differ
    )
    diag = {
        "finite": finite,
        "nonnegative": nonnegative,
        "row_sum_err": row_sum_vs_one,
        "row_sum_vs_p": row_sum_vs_p,
        "p_row_sum_vs_one": p_row_sum_vs_one,
        "mean_err": mean_err,
        "min_entry": min_entry,
        "max_token_l1": float(np.max(token_l1)),
        "mean_token_l1": float(np.mean(token_l1)),
        "token_l1": token_l1,
        "n_degenerate": n_deg,
        "frac_degenerate": float(n_deg / p64.shape[0]),
        "frac_ge_90": float(np.mean(token_l1 >= 0.9 * eta)),
        "tokens_differ_when_required": differ,
        "ok": ok,
        "d_base": d_base,
        "v": v,
    }
    return qs, diag


def replace_repeated(exact_tokens: list[np.ndarray], qs: list[np.ndarray]) -> list[np.ndarray]:
    k = len(qs)
    if len(exact_tokens) < k:
        raise RuntimeError("exact token list shorter than near-duplicate group")
    dtype = exact_tokens[0].dtype
    replaced = [q.astype(dtype, copy=False) for q in qs]
    return replaced + list(exact_tokens[k:])


def replace_single(clean_tokens: list[np.ndarray], order: tuple[str, ...], source: str, q: np.ndarray) -> list[np.ndarray]:
    out = [t.copy() for t in clean_tokens]
    out[order.index(source)] = q.astype(out[0].dtype, copy=False)
    return out


def _pct(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    return float(np.percentile(np.asarray(xs, dtype=np.float64), q))


def _run_dataset(name, caches, models, sources, order, token_fn, fuse_fn, frozen_l1):
    cond_rows = []
    single_rows = []
    regen_rows = []
    construct_stats = []
    mean_eq = []
    for seed in SEEDS:
        cache = caches[seed]
        model = models[seed]
        p1, p2 = cache["p1_test"], cache["p2_test"]
        p3 = cache.get("p3_test")
        originals = {"M1": p1, "M2": p2}
        if p3 is not None:
            originals["M3"] = p3
        clean = token_fn(p1, p2, p3, sources[0], 1)
        f_clean = {op: fuse_fn(op, clean, model)[0] for op in OPS}
        exact_fused = {}
        for src in sources:
            for k in K_NEAR:
                exact_tokens = token_fn(p1, p2, p3, src, k)
                for op in OPS:
                    fk, _ = fuse_fn(op, exact_tokens, model)
                    exact_fused[(src, k, op)] = (exact_tokens, fk)
                    if k in REGRESSION_K:
                        mean_l1 = float(_l1_rows(fk, f_clean[op]).mean())
                        frozen = frozen_l1[(seed, src, op, k)]
                        regen_rows.append(
                            {
                                "dataset": name,
                                "seed": int(seed),
                                "source": src,
                                "operator": op,
                                "k": int(k),
                                "reproduced_mean_l1": mean_l1,
                                "frozen_mean_l1": frozen,
                                "abs_discrepancy": abs(mean_l1 - frozen),
                            }
                        )
        single_cache = {}
        for src in sources:
            p = originals[src]
            for eta in ETAS:
                qs1, diag1 = near_duplicate_group(p, 2, eta)
                q_plus, q_minus = qs1[0], qs1[1]
                plus_tokens = replace_single(clean, order, src, q_plus)
                minus_tokens = replace_single(clean, order, src, q_minus)
                per_op = {}
                for op in OPS:
                    f_plus, _ = fuse_fn(op, plus_tokens, model)
                    f_minus, _ = fuse_fn(op, minus_tokens, model)
                    l1p = _l1_rows(f_plus, f_clean[op])
                    l1m = _l1_rows(f_minus, f_clean[op])
                    mid = 0.5 * (l1p + l1m)
                    mx = np.maximum(l1p, l1m)
                    rec = {
                        "dataset": name,
                        "seed": int(seed),
                        "source": src,
                        "operator": op,
                        "eta": float(eta),
                        "n_test": int(p.shape[0]),
                        "n_boundary_degenerate": diag1["n_degenerate"],
                        "frac_boundary_degenerate": diag1["frac_degenerate"],
                        "mean_single_copy_l1": float(mid.mean()),
                        "mean_maxdir_single_copy_l1": float(mx.mean()),
                        "max_maxdir_single_copy_l1": float(mx.max()),
                    }
                    single_rows.append(rec)
                    per_op[op] = rec
                single_cache[(src, eta)] = per_op
        for src in sources:
            p = originals[src]
            for k in K_NEAR:
                exact_tokens = exact_fused[(src, k, OPS[0])][0]
                for eta in ETAS:
                    qs, diag = near_duplicate_group(p, k, eta)
                    construct_stats.append(
                        {
                            "dataset": name,
                            "seed": int(seed),
                            "source": src,
                            "k": int(k),
                            "eta": float(eta),
                            **{kk: diag[kk] for kk in (
                                "finite",
                                "nonnegative",
                                "row_sum_err",
                                "row_sum_vs_p",
                                "p_row_sum_vs_one",
                                "mean_err",
                                "min_entry",
                                "max_token_l1",
                                "mean_token_l1",
                                "n_degenerate",
                                "frac_degenerate",
                                "frac_ge_90",
                                "tokens_differ_when_required",
                                "ok",
                            )},
                        }
                    )
                    near_tokens = replace_repeated(exact_tokens, qs)
                    for op in OPS:
                        f_exact = exact_fused[(src, k, op)][1]
                        f_near, _ = fuse_fn(op, near_tokens, model)
                        l1_ec = _l1_rows(f_exact, f_clean[op])
                        l1_nc = _l1_rows(f_near, f_clean[op])
                        l1_ne = _l1_rows(f_near, f_exact)
                        flip_nc = np.argmax(f_near, axis=1) != np.argmax(f_clean[op], axis=1)
                        flip_ne = np.argmax(f_near, axis=1) != np.argmax(f_exact, axis=1)
                        mean_ec = float(l1_ec.mean())
                        mean_nc = float(l1_nc.mean())
                        mean_ne = float(l1_ne.mean())
                        if mean_ec >= RATIO_DENOM_MIN:
                            move_r = mean_nc / mean_ec
                            err_r = mean_ne / mean_ec
                        else:
                            move_r = float("nan")
                            err_r = float("nan")
                        sc = single_cache[(src, eta)][op]
                        if op == "late_mean":
                            others64 = [np.asarray(t, dtype=np.float64) for t in exact_tokens[k:]]
                            f_near64 = np.mean(np.stack(qs + others64, axis=0), axis=0)
                            f_exact64 = np.mean(
                                np.stack([p.astype(np.float64)] * k + others64, axis=0), axis=0
                            )
                            mean_eq.append(
                                {
                                    "dataset": name,
                                    "seed": int(seed),
                                    "source": src,
                                    "k": int(k),
                                    "eta": float(eta),
                                    "max_abs": float(_linf_rows(f_near, f_exact).max()),
                                    "max_l1": float(l1_ne.max()),
                                    "mean_l1": mean_ne,
                                    "max_abs_f64": float(np.max(np.abs(f_near64 - f_exact64))),
                                    "max_l1_f64": float(np.abs(f_near64 - f_exact64).sum(axis=1).max()),
                                }
                            )
                        cond_rows.append(
                            {
                                "dataset": name,
                                "seed": int(seed),
                                "source": src,
                                "operator": op,
                                "k": int(k),
                                "eta": float(eta),
                                "n_test": int(p.shape[0]),
                                "n_boundary_degenerate": diag["n_degenerate"],
                                "frac_boundary_degenerate": diag["frac_degenerate"],
                                "mean_l1_exact_vs_clean": mean_ec,
                                "max_l1_exact_vs_clean": float(l1_ec.max()),
                                "mean_l1_near_vs_clean": mean_nc,
                                "max_l1_near_vs_clean": float(l1_nc.max()),
                                "mean_l1_near_vs_exact": mean_ne,
                                "max_l1_near_vs_exact": float(l1_ne.max()),
                                "flip_rate_near_vs_clean": float(flip_nc.mean()),
                                "flip_count_near_vs_clean": int(flip_nc.sum()),
                                "flip_rate_near_vs_exact": float(flip_ne.mean()),
                                "flip_count_near_vs_exact": int(flip_ne.sum()),
                                "mean_single_copy_l1": sc["mean_single_copy_l1"],
                                "mean_maxdir_single_copy_l1": sc["mean_maxdir_single_copy_l1"],
                                "max_maxdir_single_copy_l1": sc["max_maxdir_single_copy_l1"],
                                "mean_token_l1_to_p": diag["mean_token_l1"],
                                "max_token_l1_to_p": diag["max_token_l1"],
                                "frac_token_l1_ge_90pct_eta": diag["frac_ge_90"],
                                "movement_ratio": move_r,
                                "trajectory_error_ratio": err_r,
                            }
                        )
    return cond_rows, single_rows, regen_rows, construct_stats, mean_eq


def _operator_summary(rows: list[dict]) -> list[dict]:
    out = []
    for op in OPS:
        for eta in ETAS:
            sub = [r for r in rows if r["operator"] == op and r["eta"] == eta]
            move = [r["movement_ratio"] for r in sub if np.isfinite(r["movement_ratio"])]
            terr = [r["trajectory_error_ratio"] for r in sub if np.isfinite(r["trajectory_error_ratio"])]
            out.append(
                {
                    "operator": op,
                    "eta": float(eta),
                    "n_conditions": len(sub),
                    "n_valid_ratio": len(move),
                    "median_movement_ratio": _pct(move, 50),
                    "p5_movement_ratio": _pct(move, 5),
                    "p95_movement_ratio": _pct(move, 95),
                    "min_movement_ratio": float(np.min(move)) if move else float("nan"),
                    "max_movement_ratio": float(np.max(move)) if move else float("nan"),
                    "median_trajectory_error_ratio": _pct(terr, 50),
                    "p95_trajectory_error_ratio": _pct(terr, 95),
                    "max_trajectory_error_ratio": float(np.max(terr)) if terr else float("nan"),
                    "flip_count_near_vs_clean": int(sum(r["flip_count_near_vs_clean"] for r in sub)),
                    "flip_count_near_vs_exact": int(sum(r["flip_count_near_vs_exact"] for r in sub)),
                    "mean_mean_l1_near_vs_clean": float(np.mean([r["mean_l1_near_vs_clean"] for r in sub])),
                    "mean_mean_l1_exact_vs_clean": float(np.mean([r["mean_l1_exact_vs_clean"] for r in sub])),
                    "mean_mean_l1_near_vs_exact": float(np.mean([r["mean_l1_near_vs_exact"] for r in sub])),
                    "mean_single_copy_l1": float(np.mean([r["mean_single_copy_l1"] for r in sub])),
                    "max_maxdir_single_copy_l1": float(np.max([r["max_maxdir_single_copy_l1"] for r in sub])),
                }
            )
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    mh_caches = {s: load_seed_cache(s) for s in SEEDS}
    hp_caches = {}
    for s in SEEDS:
        path = HAPT_CACHE / f"seed{s}.npz"
        if file_sha256(path) != G5_CACHE[s]:
            raise RuntimeError(f"HAPT cache hash changed seed{s}")
        z = np.load(path)
        hp_caches[s] = {k: np.asarray(z[k]) for k in z.files}
        z.close()
        if file_sha256(HAPT_CKPT / f"seed{s}_cpsa.pt") != G5_CPSA[s]:
            raise RuntimeError(f"HAPT CPSA hash changed seed{s}")
    mh_models = {s: load_mh_cpsa(MH_CKPT / f"seed{s}_cpsa.pt") for s in SEEDS}
    hp_models = {s: load_hapt_cpsa(HAPT_CKPT / f"seed{s}_cpsa.pt") for s in SEEDS}

    def mh_tokens(p1, p2, p3, src, k):
        return mh_token_list(p1, p2, p3, src, k)

    def hp_tokens(p1, p2, p3, src, k):
        return hp_token_list(p1, p2, src, k)

    mh_frozen = _load_frozen_l1(REPO / "research" / "multiplicity_sweep" / "multiplicity_metrics.csv")
    hp_frozen = _load_frozen_l1(REPO / "research" / "hapt_multiplicity" / "sweep" / "hapt_multiplicity_metrics.csv")

    cond, single, regen, construct, mean_eq = _run_dataset(
        "MHEALTH", mh_caches, mh_models, MH_SOURCES, MH_SOURCES, mh_tokens, mh_fuse, mh_frozen
    )
    c2, s2, r2, g2, m2 = _run_dataset(
        "HAPT", hp_caches, hp_models, HP_SOURCES, HP_SOURCES, hp_tokens, hp_fuse, hp_frozen
    )
    cond += c2
    single += s2
    regen += r2
    construct += g2
    mean_eq += m2

    with (OUT / "near_duplicate_conditions.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CONDITION_FIELDS)
        w.writeheader()
        w.writerows(cond)

    op_sum = _operator_summary(cond)
    with (OUT / "near_duplicate_operator_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(op_sum[0].keys()))
        w.writeheader()
        w.writerows(op_sum)

    single_fields = [
        "dataset",
        "seed",
        "source",
        "operator",
        "eta",
        "n_test",
        "n_boundary_degenerate",
        "frac_boundary_degenerate",
        "mean_single_copy_l1",
        "mean_maxdir_single_copy_l1",
        "max_maxdir_single_copy_l1",
    ]
    with (OUT / "single_copy_control.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=single_fields)
        w.writeheader()
        w.writerows(single)

    assertion_fail = [c for c in construct if not c["ok"]]
    n_ex_by_eta = defaultdict(int)
    n_deg_by_eta = defaultdict(int)
    n_ge90_num = defaultdict(float)
    n_ge90_den = defaultdict(int)
    # construction rows are per seed/source/k/eta; degenerate counts are example counts
    # Use unique seed/source/eta at k=2 to avoid multiplying k.
    seen_deg = {}
    for c in construct:
        key = (c["dataset"], c["seed"], c["source"], c["eta"])
        if key not in seen_deg:
            seen_deg[key] = (c["n_degenerate"], c["n_degenerate"] / c["frac_degenerate"] if c["frac_degenerate"] else 0.0)
            # n_test recovered below from conditions
        n_ex_by_eta[c["eta"]] += 1
        n_deg_by_eta[c["eta"]] += c["n_degenerate"]
        n_ge90_num[c["eta"]] += c["frac_ge_90"]
        n_ge90_den[c["eta"]] += 1

    n_test_lookup = {}
    for r in cond:
        n_test_lookup[(r["dataset"], r["seed"], r["source"])] = r["n_test"]
    deg_examples = 0
    total_examples = 0
    for (ds, seed, src, eta), (nd, _) in seen_deg.items():
        # each example counted once per eta in seen_deg
        pass
    # Count unique examples × eta
    deg_examples_eta = {0.01: 0, 0.05: 0}
    tot_examples_eta = {0.01: 0, 0.05: 0}
    seen_ex = set()
    for c in construct:
        key = (c["dataset"], c["seed"], c["source"], c["eta"])
        if key in seen_ex:
            continue
        seen_ex.add(key)
        n = n_test_lookup[(c["dataset"], c["seed"], c["source"])]
        deg_examples_eta[c["eta"]] += c["n_degenerate"]
        tot_examples_eta[c["eta"]] += n

    seed_means = defaultdict(list)
    for r in regen:
        seed_means[(r["dataset"], r["source"], r["operator"], r["k"])].append(r["reproduced_mean_l1"])
    table_mismatch = []
    for key, vals in seed_means.items():
        ds, src, op, k = key
        rounded = round(float(np.mean(vals)), 3)
        expected = TABLE1[(ds, src, op)][k]
        if rounded != expected:
            table_mismatch.append({"key": list(key), "rounded": rounded, "table": expected})

    def _worst(rows, key, pred=None):
        pool = [r for r in rows if pred(r)] if pred else rows
        return max(pool, key=lambda r: r[key])

    def _finite_worst(rows, key, pred):
        pool = [r for r in rows if pred(r) and np.isfinite(r[key])]
        return max(pool, key=lambda r: r[key])

    diagnostics = {
        "n_condition_rows": len(cond),
        "n_construction_rows": len(construct),
        "max_row_sum_error": max(c["row_sum_err"] for c in construct),
        "max_row_sum_vs_original_p": max(c["row_sum_vs_p"] for c in construct),
        "max_original_p_row_sum_vs_one": max(c["p_row_sum_vs_one"] for c in construct),
        "max_group_mean_vs_p_elementwise": max(c["mean_err"] for c in construct),
        "min_posterior_entry": min(c["min_entry"] for c in construct),
        "max_achieved_token_l1": {str(eta): max(c["max_token_l1"] for c in construct if c["eta"] == eta) for eta in ETAS},
        "mean_achieved_token_l1": {str(eta): float(np.mean([c["mean_token_l1"] for c in construct if c["eta"] == eta])) for eta in ETAS},
        "frac_reaching_90pct_eta": {
            str(eta): float(np.mean([c["frac_ge_90"] for c in construct if c["eta"] == eta])) for eta in ETAS
        },
        "boundary_degenerate_examples": {
            str(eta): {"count": deg_examples_eta[eta], "n": tot_examples_eta[eta], "fraction": deg_examples_eta[eta] / tot_examples_eta[eta]}
            for eta in ETAS
        },
        "assertion_failures": len(assertion_fail),
        "no_clip_no_renorm": True,
        "construct_tol": CONSTRUCT_TOL,
        "all_finite": all(c["finite"] for c in construct),
        "all_nonnegative": all(c["nonnegative"] for c in construct),
        "all_tokens_differ_when_required": all(c["tokens_differ_when_required"] for c in construct),
        "mean_fusion_max_elementwise": max(m["max_abs"] for m in mean_eq),
        "mean_fusion_max_per_example_l1": max(m["max_l1"] for m in mean_eq),
        "mean_fusion_max_mean_l1": max(m["mean_l1"] for m in mean_eq),
        "mean_fusion_max_elementwise_f64": max(m["max_abs_f64"] for m in mean_eq),
        "mean_fusion_max_per_example_l1_f64": max(m["max_l1_f64"] for m in mean_eq),
        "mean_fusion_operator_path_passed": max(m["max_abs"] for m in mean_eq) <= MEAN_FUSION_TOL,
        "mean_fusion_construction_passed": max(m["max_abs_f64"] for m in mean_eq) <= CONSTRUCT_TOL,
        "mean_fusion_passed": max(m["max_abs"] for m in mean_eq) <= MEAN_FUSION_TOL
        and max(m["max_abs_f64"] for m in mean_eq) <= CONSTRUCT_TOL,
        "regression_max_abs_discrepancy": max(r["abs_discrepancy"] for r in regen),
        "regression_n_compared": len(regen),
        "manuscript_table1_rounding_mismatches": table_mismatch,
    }
    (OUT / "construction_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    with (OUT / "regression_exact_duplicate.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(regen[0].keys()))
        w.writeheader()
        w.writerows(regen)

    summary = {
        **diagnostics,
        "operator_summary": op_sum,
        "worst_product_mean_l1_near_vs_exact": _worst(cond, "mean_l1_near_vs_exact", lambda r: r["operator"] == "late_product"),
        "worst_attention_mean_l1_near_vs_exact": _worst(cond, "mean_l1_near_vs_exact", lambda r: r["operator"] == "cpsa"),
        "worst_product_trajectory_error_ratio": _finite_worst(cond, "trajectory_error_ratio", lambda r: r["operator"] == "late_product"),
        "worst_attention_trajectory_error_ratio": _finite_worst(cond, "trajectory_error_ratio", lambda r: r["operator"] == "cpsa"),
    }
    (OUT / "near_duplicate_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(diagnostics, indent=2))
    print(json.dumps(op_sum, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
