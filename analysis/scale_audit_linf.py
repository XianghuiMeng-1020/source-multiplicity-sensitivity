"""Same-metric scale audit: multiplicity ||Fk-F1||_inf vs Gate-2 order-only max.

Frozen posteriors and CPSA only. No training.
"""

from __future__ import annotations

import csv
import json
import sys
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

OUT = (REPO / "research" / "gate3_scale_audit").resolve()
ORDER_ONLY_MAX = 5.1081180572509766e-5
SEEDS = (0, 1, 2)
K_GT1 = (2, 3, 5, 8, 16, 32, 64, 100)
OPS = ("late_product", "late_mean", "cpsa")


def _linf(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)), axis=1)


def _run(name, caches, models, sources, token_fn, fuse_fn) -> list[dict]:
    rows = []
    for seed in SEEDS:
        cache = caches[seed]
        model = models[seed]
        p1, p2 = cache["p1_test"], cache["p2_test"]
        p3 = cache["p3_test"] if name == "MHEALTH" else None
        for op in OPS:
            parts1 = token_fn(p1, p2, p3, sources[0], 1)
            f1, _ = fuse_fn(op, parts1, model)
            for src in sources:
                for k in K_GT1:
                    parts = token_fn(p1, p2, p3, src, k)
                    fk, _ = fuse_fn(op, parts, model)
                    per = _linf(fk, f1)
                    rows.append(
                        {
                            "dataset": name,
                            "seed": int(seed),
                            "source": src,
                            "operator": op,
                            "k": int(k),
                            "n": int(per.shape[0]),
                            "max_linf": float(per.max()),
                            "mean_linf": float(per.mean()),
                            "min_linf": float(per.min()),
                            "exceeds_order_only_max": bool(per.max() > ORDER_ONLY_MAX),
                            "frac_examples_exceed_order_only": float((per > ORDER_ONLY_MAX).mean()),
                        }
                    )
    return rows


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

    rows = _run("MHEALTH", mh_caches, mh_models, ("M1", "M2", "M3"), mh_tokens, mh_fuse)
    rows += _run("HAPT", hp_caches, hp_models, ("M1", "M2"), hp_tokens, hp_fuse)
    maxes = np.array([r["max_linf"] for r in rows], dtype=np.float64)
    nonzero = maxes[maxes > 0]
    summary = {
        "metric": "max_over_examples ||Fk-F1||_inf",
        "order_only_global_max_linf": ORDER_ONLY_MAX,
        "n_conditions": len(rows),
        "n_nonzero_conditions": int((maxes > 0).sum()),
        "min_nonzero_linf": float(nonzero.min()) if nonzero.size else None,
        "median_linf": float(np.median(maxes)),
        "q25_linf": float(np.quantile(maxes, 0.25)),
        "q75_linf": float(np.quantile(maxes, 0.75)),
        "max_linf": float(maxes.max()),
        "fraction_conditions_exceeding_order_only_max": float((maxes > ORDER_ONLY_MAX).mean()),
        "n_conditions_at_or_below_order_only_max": int((maxes <= ORDER_ONLY_MAX).sum()),
        "by_operator": {},
    }
    for op in OPS:
        sub = np.array([r["max_linf"] for r in rows if r["operator"] == op])
        summary["by_operator"][op] = {
            "min_nonzero": float(sub[sub > 0].min()),
            "median": float(np.median(sub)),
            "max": float(sub.max()),
            "frac_exceed_order_only": float((sub > ORDER_ONLY_MAX).mean()),
        }
    fields = [
        "dataset",
        "seed",
        "source",
        "operator",
        "k",
        "n",
        "max_linf",
        "mean_linf",
        "min_linf",
        "exceeds_order_only_max",
        "frac_examples_exceed_order_only",
    ]
    with (OUT / "multiplicity_linf.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    (OUT / "scale_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
