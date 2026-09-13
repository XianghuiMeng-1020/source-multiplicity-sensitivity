"""Gate-3 provenance-collapse negative control.

Construct D_{j,k}(P) with the existing token_list, collapse by source ID,
then fuse with the same frozen operators. No training. Does not write frozen
sweep artifacts.
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

OUT = (REPO / "research" / "gate3_provenance_control").resolve()
SEEDS = (0, 1, 2)
K_VALUES = (1, 2, 3, 5, 8, 16, 32, 64, 100)
REGRESSION_K = (2, 5, 16, 100)
OPS = ("late_product", "late_mean", "cpsa")
MH_SOURCES = ("M1", "M2", "M3")
HP_SOURCES = ("M1", "M2")
COLLAPSE_TOL = 1e-12
NORM_TOL = 1e-6
REGRESSION_TOL = 1e-8

# Manuscript Table 1 seed-mean L1(Fk,F1), 3 d.p. Read-only regression target.
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


def _mh_ids(source: str, k: int) -> list[str]:
    if k == 1:
        return list(MH_SOURCES)
    others = [s for s in MH_SOURCES if s != source]
    return [source] * int(k) + others


def _hp_ids(source: str, k: int) -> list[str]:
    if k == 1:
        return ["M1", "M2"]
    if source == "M1":
        return ["M1"] * int(k) + ["M2"]
    if source == "M2":
        return ["M2"] * int(k) + ["M1"]
    raise ValueError(source)


def collapse(tokens: list[np.ndarray], ids: list[str], order: tuple[str, ...]) -> tuple[list[np.ndarray], dict]:
    if len(tokens) != len(ids):
        raise RuntimeError("token/id length mismatch")
    groups: dict[str, list[np.ndarray]] = {sid: [] for sid in order}
    for tok, sid in zip(tokens, ids):
        if sid not in groups:
            raise RuntimeError(f"unexpected source id {sid}")
        groups[sid].append(tok)
    collapsed = []
    checks = {
        "one_token_per_source": True,
        "finite": True,
        "normalized": True,
        "n_ids": len(order),
        "max_collapse_to_member": 0.0,
    }
    for sid in order:
        g = groups[sid]
        if len(g) < 1:
            raise RuntimeError(f"missing source {sid} after construction")
        stacked = np.stack(g, axis=0).astype(np.float64)
        avg = stacked.mean(axis=0)
        member_dev = float(np.max(np.abs(stacked - stacked[:1])))
        checks["max_collapse_to_member"] = max(checks["max_collapse_to_member"], member_dev)
        if not np.isfinite(avg).all():
            checks["finite"] = False
        row_sum = avg.sum(axis=1)
        if float(np.max(np.abs(row_sum - 1.0))) > NORM_TOL:
            checks["normalized"] = False
        collapsed.append(avg.astype(np.float32, copy=False))
    seen = {sid for sid in ids}
    if seen != set(order) or len(collapsed) != len(order):
        checks["one_token_per_source"] = False
    return collapsed, checks


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


def _run_dataset(name, caches, models, sources, order, token_fn, id_fn, fuse_fn, frozen_l1):
    ctrl_rows = []
    regen_rows = []
    assertions = []
    for seed in SEEDS:
        cache = caches[seed]
        model = models[seed]
        p1, p2 = cache["p1_test"], cache["p2_test"]
        p3 = cache.get("p3_test")
        originals = {"M1": p1, "M2": p2}
        if p3 is not None:
            originals["M3"] = p3
        for op in OPS:
            clean_tokens = token_fn(p1, p2, p3, sources[0], 1)
            f1, _ = fuse_fn(op, clean_tokens, model)
            for src in sources:
                for k in K_VALUES:
                    tokens = token_fn(p1, p2, p3, src, k)
                    ids = id_fn(src, k)
                    if len(tokens) != len(ids):
                        raise RuntimeError("D construction / id tagging mismatch")
                    collapsed, checks = collapse(tokens, ids, order)
                    # exact-copy: collapsed source equals original posterior
                    pj_dev = float(np.max(np.abs(collapsed[order.index(src)].astype(np.float64) - originals[src].astype(np.float64))))
                    checks["collapsed_equals_original"] = pj_dev <= COLLAPSE_TOL
                    checks["collapsed_to_original_max_abs"] = pj_dev
                    assertions.append(
                        {
                            "dataset": name,
                            "seed": int(seed),
                            "source": src,
                            "operator": op,
                            "k": int(k),
                            **checks,
                        }
                    )
                    f_ctrl, _ = fuse_fn(op, collapsed, model)
                    l1 = _l1_rows(f_ctrl, f1)
                    linf = _linf_rows(f_ctrl, f1)
                    flips = np.argmax(f_ctrl, axis=1) != np.argmax(f1, axis=1)
                    ctrl_rows.append(
                        {
                            "dataset": name,
                            "seed": int(seed),
                            "source": src,
                            "operator": op,
                            "k": int(k),
                            "mean_l1_ctrl_vs_clean": float(l1.mean()),
                            "max_l1_ctrl_vs_clean": float(l1.max()),
                            "max_abs_ctrl_vs_clean": float(linf.max()),
                            "flip_rate": float(flips.mean()),
                            "flip_count": int(flips.sum()),
                            "n_test": int(l1.shape[0]),
                        }
                    )
                    if k in REGRESSION_K:
                        fk, _ = fuse_fn(op, tokens, model)
                        mean_l1 = float(_l1_rows(fk, f1).mean())
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
    return ctrl_rows, regen_rows, assertions


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

    ctrl, regen, asserts = _run_dataset(
        "MHEALTH", mh_caches, mh_models, MH_SOURCES, MH_SOURCES, mh_tokens, _mh_ids, mh_fuse, mh_frozen
    )
    c2, r2, a2 = _run_dataset(
        "HAPT", hp_caches, hp_models, HP_SOURCES, HP_SOURCES, hp_tokens, _hp_ids, hp_fuse, hp_frozen
    )
    ctrl += c2
    regen += r2
    asserts += a2

    fields = [
        "dataset",
        "seed",
        "source",
        "operator",
        "k",
        "mean_l1_ctrl_vs_clean",
        "max_l1_ctrl_vs_clean",
        "max_abs_ctrl_vs_clean",
        "flip_rate",
        "flip_count",
        "n_test",
    ]
    with (OUT / "provenance_control.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(ctrl)

    def argmax_row(key):
        return max(ctrl, key=lambda r: r[key])

    by_op = {}
    for op in OPS:
        sub = [r for r in ctrl if r["operator"] == op]
        by_op[op] = {
            "max_mean_l1": max(r["mean_l1_ctrl_vs_clean"] for r in sub),
            "max_per_example_l1": max(r["max_l1_ctrl_vs_clean"] for r in sub),
            "max_elementwise_abs": max(r["max_abs_ctrl_vs_clean"] for r in sub),
            "total_flips": int(sum(r["flip_count"] for r in sub)),
        }

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

    assertion_fail = [
        a
        for a in asserts
        if (not a["one_token_per_source"])
        or (not a["finite"])
        or (not a["normalized"])
        or (not a["collapsed_equals_original"])
    ]

    summary = {
        "n_conditions": len(ctrl),
        "datasets": ["MHEALTH", "HAPT"],
        "seeds": list(SEEDS),
        "sources": {"MHEALTH": list(MH_SOURCES), "HAPT": list(HP_SOURCES)},
        "operators": list(OPS),
        "k": list(K_VALUES),
        "global_max_mean_l1": max(r["mean_l1_ctrl_vs_clean"] for r in ctrl),
        "global_max_per_example_l1": max(r["max_l1_ctrl_vs_clean"] for r in ctrl),
        "global_max_elementwise_abs": max(r["max_abs_ctrl_vs_clean"] for r in ctrl),
        "total_prediction_flips": int(sum(r["flip_count"] for r in ctrl)),
        "max_flip_rate": max(r["flip_rate"] for r in ctrl),
        "worst_mean_l1": {k: argmax_row("mean_l1_ctrl_vs_clean")[k] for k in fields},
        "worst_per_example_l1": {k: argmax_row("max_l1_ctrl_vs_clean")[k] for k in fields},
        "worst_elementwise_abs": {k: argmax_row("max_abs_ctrl_vs_clean")[k] for k in fields},
        "by_operator": by_op,
        "regression_max_abs_discrepancy": max(r["abs_discrepancy"] for r in regen),
        "regression_n_compared": len(regen),
        "manuscript_table1_rounding_mismatches": table_mismatch,
        "assertion_failures": len(assertion_fail),
        "collapse_tol": COLLAPSE_TOL,
        "max_member_identity_dev": max(a["max_collapse_to_member"] for a in asserts),
        "max_collapsed_to_original_abs": max(a["collapsed_to_original_max_abs"] for a in asserts),
    }
    (OUT / "provenance_control_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (OUT / "regression_noncollapsed.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(regen[0].keys()))
        w.writeheader()
        w.writerows(regen)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
