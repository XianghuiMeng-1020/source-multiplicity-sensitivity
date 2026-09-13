"""Permutation negative control. Frozen posteriors and CPSA only. No training.

Reorders an unchanged posterior multiset and compares fused outputs to the
canonical token order used in the multiplicity sweeps.
"""

from __future__ import annotations

import itertools
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
from multiplicity_sweep.sweep import GATE2_CACHE_HASHES  # noqa: E402
from multiplicity_sweep.sweep import fuse as mh_fuse  # noqa: E402
from multiplicity_sweep.sweep import load_seed_cache  # noqa: E402
from multiplicity_sweep.sweep import token_list as mh_token_list  # noqa: E402

OUT = (REPO / "research" / "gate2_permutation_control").resolve()
CONTROL_SEED = 20270913
SEEDS = (0, 1, 2)
K_VALUES = (1, 2, 3, 5, 8, 16, 32, 64, 100)
OPS = ("late_product", "late_mean", "cpsa")
MH_SOURCES = ("M1", "M2", "M3")
HP_SOURCES = ("M1", "M2")


def _reorder(parts: list[np.ndarray], order: tuple[int, ...] | list[int]) -> list[np.ndarray]:
    return [parts[i] for i in order]


def _multiset_ok(a: list[np.ndarray], b: list[np.ndarray]) -> bool:
    if len(a) != len(b):
        return False
    used = [False] * len(b)
    for x in a:
        hit = False
        for j, y in enumerate(b):
            if used[j]:
                continue
            if x is y or (x.shape == y.shape and np.array_equal(x, y)):
                used[j] = True
                hit = True
                break
        if not hit:
            return False
    return all(used)


def _compare(f0: np.ndarray, f1: np.ndarray) -> dict:
    diff = np.abs(f0.astype(np.float64) - f1.astype(np.float64))
    flips = int((f0.argmax(axis=1) != f1.argmax(axis=1)).sum())
    return {
        "n": int(f0.shape[0]),
        "max_linf": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "pred_changes": flips,
        "allclose_rtol1e6_atol1e8": bool(np.allclose(f0, f1, rtol=1e-6, atol=1e-8)),
    }


def _orders_k1(n_src: int) -> list[tuple[int, ...]]:
    return [p for p in itertools.permutations(range(n_src)) if p != tuple(range(n_src))]


def _orders_expanded(n: int, rng_a: np.random.Generator, rng_b: np.random.Generator) -> dict[str, list[int]]:
    ident = list(range(n))
    rev = list(reversed(ident))
    sh_a = rng_a.permutation(n).tolist()
    sh_b = rng_b.permutation(n).tolist()
    return {"reverse": rev, "shuffle_A": sh_a, "shuffle_B": sh_b}


def _run_dataset(name: str, caches: dict, models: dict, sources: tuple, token_fn, fuse_fn) -> dict:
    k1_rows = []
    grid_rows = []
    n_src = len(sources)
    k1_perms = _orders_k1(n_src)

    for seed in SEEDS:
        cache = caches[seed]
        model = models[seed]
        if name == "MHEALTH":
            p1, p2, p3 = cache["p1_test"], cache["p2_test"], cache["p3_test"]
            n = int(p1.shape[0])
        else:
            p1, p2 = cache["p1_test"], cache["p2_test"]
            p3 = None
            n = int(p1.shape[0])

        canon_k1 = token_fn(p1, p2, p3, sources[0], 1)
        for op in OPS:
            f0, _ = fuse_fn(op, canon_k1, model)
            for perm in k1_perms:
                parts = _reorder(canon_k1, perm)
                if not _multiset_ok(canon_k1, parts):
                    raise RuntimeError(f"{name} k=1 multiset broken seed={seed} perm={perm}")
                f1, _ = fuse_fn(op, parts, model)
                rec = _compare(f0, f1)
                rec.update(
                    {
                        "dataset": name,
                        "seed": int(seed),
                        "operator": op,
                        "k": 1,
                        "perm": list(perm),
                    }
                )
                k1_rows.append(rec)

        rng_a = np.random.default_rng(CONTROL_SEED)
        rng_b = np.random.default_rng(CONTROL_SEED + 1)
        for src in sources:
            for k in K_VALUES:
                parts0 = token_fn(p1, p2, p3, src, k)
                orders = _orders_expanded(len(parts0), rng_a, rng_b)
                for op in OPS:
                    f0, _ = fuse_fn(op, parts0, model)
                    for label, order in orders.items():
                        if order == list(range(len(parts0))):
                            continue
                        parts = _reorder(parts0, order)
                        if not _multiset_ok(parts0, parts):
                            raise RuntimeError(f"{name} grid multiset broken {src} k={k} {label}")
                        f1, _ = fuse_fn(op, parts, model)
                        rec = _compare(f0, f1)
                        rec.update(
                            {
                                "dataset": name,
                                "seed": int(seed),
                                "source": src,
                                "operator": op,
                                "k": int(k),
                                "ordering": label,
                            }
                        )
                        grid_rows.append(rec)
    return {"k1": k1_rows, "grid": grid_rows, "n_examples": n, "k1_perm_count": len(k1_perms)}


def _agg(rows: list[dict], **eq) -> dict:
    sub = [r for r in rows if all(r.get(k) == v for k, v in eq.items())]
    if not sub:
        return {}
    return {
        "n_comparisons": len(sub),
        "n_examples": int(sub[0]["n"]),
        "max_linf": max(r["max_linf"] for r in sub),
        "mean_abs": float(np.mean([r["mean_abs"] for r in sub])),
        "pred_changes": int(sum(r["pred_changes"] for r in sub)),
        "pred_total": int(sum(r["n"] for r in sub)),
    }


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
    for m in list(mh_models.values()) + list(hp_models.values()):
        if m.training:
            raise RuntimeError("CPSA in train mode")

    def mh_tokens(p1, p2, p3, src, k):
        return mh_token_list(p1, p2, p3, src, k)

    def hp_tokens(p1, p2, p3, src, k):
        return hp_token_list(p1, p2, src, k)

    mh = _run_dataset("MHEALTH", mh_caches, mh_models, MH_SOURCES, mh_tokens, mh_fuse)
    hp = _run_dataset("HAPT", hp_caches, hp_models, HP_SOURCES, hp_tokens, hp_fuse)

    all_k1 = mh["k1"] + hp["k1"]
    all_grid = mh["grid"] + hp["grid"]
    all_rows = all_k1 + all_grid

    summary = {
        "control_seed": CONTROL_SEED,
        "no_training": True,
        "k1_permutations": {
            "MHEALTH": "all 3! minus identity (5)",
            "HAPT": "all 2! minus identity (1)",
        },
        "grid_orderings": ["reverse", "shuffle_A", "shuffle_B"],
        "k_values": list(K_VALUES),
        "seeds": list(SEEDS),
        "operators": list(OPS),
        "k1_by_cell": {
            f"{ds}/{op}": _agg(all_k1, dataset=ds, operator=op)
            for ds in ("MHEALTH", "HAPT")
            for op in OPS
        },
        "grid": {
            "datasets": ["MHEALTH", "HAPT"],
            "mhealth_sources": list(MH_SOURCES),
            "hapt_sources": list(HP_SOURCES),
            "operators": list(OPS),
            "seeds": list(SEEDS),
            "k": list(K_VALUES),
            "orderings_per_multiset": ["reverse", "shuffle_A", "shuffle_B"],
            "global_max_linf": max(r["max_linf"] for r in all_grid),
            "global_mean_abs": float(np.mean([r["mean_abs"] for r in all_grid])),
            "ordering_only_pred_flips": int(sum(r["pred_changes"] for r in all_grid)),
            "n_comparisons": len(all_grid),
        },
        "global_max_linf_all_controls": max(r["max_linf"] for r in all_rows),
        "global_ordering_only_pred_flips": int(sum(r["pred_changes"] for r in all_rows)),
        "k1_n_examples": {"MHEALTH": mh["n_examples"], "HAPT": hp["n_examples"]},
        "gate2_cache_hashes_checked": GATE2_CACHE_HASHES,
        "hapt_cache_hashes_checked": G5_CACHE,
    }

    (OUT / "permutation_control_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    fields = [
        "split",
        "dataset",
        "seed",
        "source",
        "operator",
        "k",
        "ordering",
        "n",
        "max_linf",
        "mean_abs",
        "pred_changes",
    ]
    import csv

    with (OUT / "permutation_control.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in all_k1:
            w.writerow({**r, "split": "k1", "source": "", "ordering": str(r.get("perm"))})
        for r in all_grid:
            w.writerow({**r, "split": "grid"})

    print("CONTROL_SEED", CONTROL_SEED)
    print("GLOBAL_MAX_LINF", summary["global_max_linf_all_controls"])
    print("GLOBAL_PRED_FLIPS", summary["global_ordering_only_pred_flips"])
    print("GRID_MAX_LINF", summary["grid"]["global_max_linf"])
    print("GRID_PRED_FLIPS", summary["grid"]["ordering_only_pred_flips"])
    for key, rec in summary["k1_by_cell"].items():
        print(f"K1 {key} max_linf={rec['max_linf']:.8g} flips={rec['pred_changes']}/{rec['pred_total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
