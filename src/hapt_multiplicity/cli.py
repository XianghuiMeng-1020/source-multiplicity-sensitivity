"""HAPT Gate-5 orchestrator: train, freeze, replay. No k>1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import numpy as np

from models import late_mean, late_product, predict_proba
from multiplicity_replay.hashes import audit_freeze
from multiplicity_replay.replay import REPLAY_ROOT, REPO_ROOT, file_sha256

from .evaluate import evaluate_clean, write_csv
from .io_util import (
    assert_protocol_frozen,
    load_cpsa,
    load_encoder,
    save_state,
    write_json,
)
from .loader import integrity_record, load_hapt_two_source
from .paths import (
    CACHE_DIR,
    CKPT_DIR,
    FEATURE_MAP,
    HAPT_ROOT,
    HAPT_ZIP,
    PROTOCOL_JSON,
    RESULTS,
    SWEEP_G3,
    WRITE_ROOT,
    assert_hapt_write,
)
from .train import train_seed

SEEDS = (0, 1, 2)
COMPARE_METRICS = (
    "accuracy",
    "ece",
    "nll",
    "brier",
    "confidence",
    "conf_acc_gap",
    "entropy",
    "macro_f1",
    "balanced_accuracy",
)
G3_TABLES = (
    "multiplicity_metrics.csv",
    "source_quality.csv",
    "cpsa_attention_diagnostics.csv",
    "late_mean_law.csv",
    "per_subject_metrics.csv",
)


def _prior_hashes() -> dict:
    freeze = audit_freeze()
    g2 = {p.name: file_sha256(p) for p in sorted((REPLAY_ROOT / "caches").glob("seed*.npz"))}
    g3 = {name: file_sha256(SWEEP_G3 / name) for name in G3_TABLES}
    hapt_inputs = {
        "hapt.zip": file_sha256(HAPT_ZIP),
        "features.txt": file_sha256(HAPT_ROOT / "features.txt"),
        "X_train.txt": file_sha256(HAPT_ROOT / "Train" / "X_train.txt"),
        "y_train.txt": file_sha256(HAPT_ROOT / "Train" / "y_train.txt"),
        "subject_id_train.txt": file_sha256(HAPT_ROOT / "Train" / "subject_id_train.txt"),
        "X_test.txt": file_sha256(HAPT_ROOT / "Test" / "X_test.txt"),
        "y_test.txt": file_sha256(HAPT_ROOT / "Test" / "y_test.txt"),
        "subject_id_test.txt": file_sha256(HAPT_ROOT / "Test" / "subject_id_test.txt"),
    }
    return {
        "freeze_ok": freeze["ok"],
        "watched_extra": freeze["watched_extra"],
        "checkpoints": freeze["checkpoints"],
        "gate2_caches": g2,
        "gate3_tables": g3,
        "hapt_inputs": hapt_inputs,
        "protocol": file_sha256(PROTOCOL_JSON),
        "feature_map": file_sha256(FEATURE_MAP),
    }


def _sha_file_list(paths: list) -> list[str]:
    lines = []
    for path in paths:
        rel = path.relative_to(REPO_ROOT)
        lines.append(f"{file_sha256(path)}  {rel.as_posix()}")
    return lines


def stage_train() -> dict:
    assert_protocol_frozen()
    pre = _prior_hashes()
    write_json(WRITE_ROOT / "hash_pre.json", pre)
    if not pre["freeze_ok"]:
        raise RuntimeError("prior-gate freeze pre-check failed")

    ds = load_hapt_two_source()
    write_json(RESULTS / "split_integrity.json", integrity_record(ds))

    holdout_rows = []
    all_metrics = []
    all_class = []
    all_subj = []
    all_attn = []
    for seed in SEEDS:
        payload = train_seed(ds, seed)
        save_state(CKPT_DIR / f"seed{seed}_enc_M1.pt", payload["enc1"])
        save_state(CKPT_DIR / f"seed{seed}_enc_M2.pt", payload["enc2"])
        save_state(CKPT_DIR / f"seed{seed}_cpsa.pt", payload["cpsa"])
        np.savez(
            assert_hapt_write(RESULTS / f"holdout_seed{seed}.npz"),
            tr_i=payload["tr_i"],
            ho_i=payload["ho_i"],
        )
        holdout_rows.append({k: payload[k] for k in payload if k not in ("enc1", "enc2", "cpsa", "tr_i", "ho_i")})
        ev = evaluate_clean(
            payload["enc1"],
            payload["enc2"],
            payload["cpsa"],
            ds.test.X1,
            ds.test.X2,
            ds.test.y,
            ds.test.subject,
        )
        _write_seed_cache(seed, ev, ds)
        for row in ev["metric_rows"]:
            all_metrics.append({"seed": seed, **row})
        for row in ev["per_class_rows"]:
            all_class.append({"seed": seed, **row})
        for row in ev["subject_rows"]:
            all_subj.append({"seed": seed, **row})
        for row in ev["attn_rows"]:
            all_attn.append({"seed": seed, **row})
        print(
            f"seed {seed} trained enc={payload['encoder_train_n']} ho={payload['cpsa_holdout_n']} "
            f"sec={payload['wall_clock_sec']:.1f}",
            flush=True,
        )

    write_json(RESULTS / "holdout_audit.json", {"seeds": holdout_rows})
    write_csv(
        RESULTS / "clean_baseline_metrics.csv",
        all_metrics,
        ["seed", "object"] + list(COMPARE_METRICS) + ["n"],
    )
    write_csv(
        RESULTS / "per_class_metrics.csv",
        all_class,
        ["seed", "object", "class", "support", "recall", "precision", "f1"],
    )
    write_csv(
        RESULTS / "clean_per_subject_metrics.csv",
        all_subj,
        ["seed", "object", "subject", "n", "accuracy", "macro_f1", "nll", "brier", "confidence"],
    )
    write_csv(
        RESULTS / "clean_cpsa_attention.csv",
        all_attn,
        ["seed", "source", "attn_mean", "attn_median", "attn_p10", "attn_p90", "attn_min", "attn_max"],
    )
    write_json(
        RESULTS / "environment.json",
        {
            "python": sys.version.replace("\n", " "),
            "executable": sys.executable,
            "numpy": np.__version__,
            "torch": __import__("torch").__version__,
            "device": "cpu",
            "additional_fitted_scaling": False,
            "dataset_provided_feature_scaling": True,
        },
    )
    _write_freeze_list()
    return {"holdout": holdout_rows, "n_metric_rows": len(all_metrics)}


def _write_seed_cache(seed: int, ev: dict, ds) -> None:
    dest = assert_hapt_write(CACHE_DIR / f"seed{seed}.npz")
    arrays = {
        "p1_test": ev["p1"],
        "p2_test": ev["p2"],
        "y_test": ds.test.y,
        "subject_test": ds.test.subject,
        "fused_late_product": ev["late_product"],
        "fused_late_mean": ev["late_mean"],
        "fused_cpsa": ev["cpsa"],
    }
    np.savez(dest, **arrays)
    meta = {
        "seed": seed,
        "shapes": {k: list(v.shape) for k, v in arrays.items()},
        "dtypes": {k: str(v.dtype) for k, v in arrays.items()},
        "test_n": 3162,
        "class_count": 12,
        "row_sum_tol": 1e-5,
        "p1_row_sum_maxdev": float(np.max(np.abs(ev["p1"].sum(1) - 1))),
        "p2_row_sum_maxdev": float(np.max(np.abs(ev["p2"].sum(1) - 1))),
        "finite": bool(np.isfinite(ev["p1"]).all() and np.isfinite(ev["p2"]).all()),
        "checkpoint_sha256": {
            "enc_M1": file_sha256(CKPT_DIR / f"seed{seed}_enc_M1.pt"),
            "enc_M2": file_sha256(CKPT_DIR / f"seed{seed}_enc_M2.pt"),
            "cpsa": file_sha256(CKPT_DIR / f"seed{seed}_cpsa.pt"),
        },
        "feature_map_sha256": file_sha256(FEATURE_MAP),
        "y_test_sha256": hashlib.sha256(ds.test.y.tobytes()).hexdigest(),
        "protocol_sha256": file_sha256(PROTOCOL_JSON),
    }
    write_json(CACHE_DIR / f"seed{seed}_meta.json", meta)


def _write_freeze_list() -> None:
    paths = [
        PROTOCOL_JSON,
        WRITE_ROOT / "PROTOCOL.md",
        FEATURE_MAP,
        RESULTS / "clean_baseline_metrics.csv",
        RESULTS / "per_class_metrics.csv",
        RESULTS / "clean_per_subject_metrics.csv",
        RESULTS / "clean_cpsa_attention.csv",
        RESULTS / "split_integrity.json",
        RESULTS / "holdout_audit.json",
        RESULTS / "environment.json",
    ]
    for seed in SEEDS:
        paths.extend(
            [
                CKPT_DIR / f"seed{seed}_enc_M1.pt",
                CKPT_DIR / f"seed{seed}_enc_M2.pt",
                CKPT_DIR / f"seed{seed}_cpsa.pt",
                CACHE_DIR / f"seed{seed}.npz",
            ]
        )
    dest = assert_hapt_write(RESULTS / "RESULT_FREEZE_HAPT.sha256")
    dest.write_text("\n".join(_sha_file_list(paths)) + "\n", encoding="utf-8")


def _infer_seed(ds, seed: int) -> dict:
    enc1 = load_encoder(CKPT_DIR / f"seed{seed}_enc_M1.pt", 348)
    enc2 = load_encoder(CKPT_DIR / f"seed{seed}_enc_M2.pt", 211)
    cpsa = load_cpsa(CKPT_DIR / f"seed{seed}_cpsa.pt")
    if enc1.training or enc2.training or cpsa.training:
        raise RuntimeError("loaded module in train mode")
    return evaluate_clean(enc1, enc2, cpsa, ds.test.X1, ds.test.X2, ds.test.y, ds.test.subject)


def classify_delta(frozen: float, replayed: float) -> str:
    if frozen == replayed:
        return "EXACT"
    ad = abs(replayed - frozen)
    if ad <= 1e-7:
        return "NUMERICAL_MATCH"
    if ad <= 1e-5:
        return "CLOSE_ONLY"
    return "FAIL"


def stage_replay(tag: str = "run1") -> dict:
    assert_protocol_frozen()
    ds = load_hapt_two_source()
    frozen_rows = list(
        __import__("csv").DictReader((RESULTS / "clean_baseline_metrics.csv").open())
    )
    frozen_idx = {(int(r["seed"]), r["object"]): r for r in frozen_rows}
    comparison = []
    array_diffs = {}
    fail = 0
    for seed in SEEDS:
        ev = _infer_seed(ds, seed)
        z = np.load(CACHE_DIR / f"seed{seed}.npz")
        for key, arr in (
            ("p1_test", ev["p1"]),
            ("p2_test", ev["p2"]),
            ("fused_late_product", ev["late_product"]),
            ("fused_late_mean", ev["late_mean"]),
            ("fused_cpsa", ev["cpsa"]),
        ):
            d = float(np.max(np.abs(arr - z[key])))
            array_diffs[f"seed{seed}_{key}"] = {
                "max_abs": d,
                "bit_identical": bool(arr.shape == z[key].shape and (arr == z[key]).all()),
            }
        z.close()
        for row in ev["metric_rows"]:
            fr = frozen_idx[(seed, row["object"])]
            cell = {"seed": seed, "object": row["object"], "tag": tag}
            for m in COMPARE_METRICS:
                frozen = float(fr[m])
                replayed = float(row[m])
                status = classify_delta(frozen, replayed)
                if status == "FAIL":
                    fail += 1
                cell[m] = {
                    "frozen": frozen,
                    "replayed": replayed,
                    "abs_diff": abs(replayed - frozen),
                    "status": status,
                }
            comparison.append(cell)
    out = {"fail_cells": fail, "comparison": comparison, "array_diffs": array_diffs, "tag": tag}
    write_json(WRITE_ROOT / f"replay_{tag}.json", out)
    return out


def stage_determinism() -> dict:
    a = json.loads((WRITE_ROOT / "replay_run1.json").read_text())
    b = json.loads((WRITE_ROOT / "replay_run2.json").read_text())
    report = {"pairs": {}, "bit_identical_all": True}
    keys = (
        "p1_test",
        "p2_test",
        "fused_late_product",
        "fused_late_mean",
        "fused_cpsa",
    )
    for seed in SEEDS:
        z1 = np.load(WRITE_ROOT / "caches_run1" / f"seed{seed}.npz")
        z2 = np.load(WRITE_ROOT / "caches_run2" / f"seed{seed}.npz")
        for key in keys:
            ident = bool(z1[key].shape == z2[key].shape and (z1[key] == z2[key]).all())
            d = float(np.max(np.abs(z1[key] - z2[key])))
            report["pairs"][f"seed{seed}_{key}"] = {"max_abs": d, "bit_identical": ident}
            if not ident:
                report["bit_identical_all"] = False
        z1.close()
        z2.close()
    write_json(WRITE_ROOT / "determinism.json", report)
    _ = a, b
    return report


def stage_replay_dump(tag: str) -> None:
    """Inference-only dump. Never overwrites frozen caches/."""
    assert_protocol_frozen()
    ds = load_hapt_two_source()
    out_dir = WRITE_ROOT / f"caches_{tag}"
    for seed in SEEDS:
        ev = _infer_seed(ds, seed)
        dest = assert_hapt_write(out_dir / f"seed{seed}.npz")
        np.savez(
            dest,
            p1_test=ev["p1"],
            p2_test=ev["p2"],
            y_test=ds.test.y,
            subject_test=ds.test.subject,
            fused_late_product=ev["late_product"],
            fused_late_mean=ev["late_mean"],
            fused_cpsa=ev["cpsa"],
        )
    stage_replay(tag)


def _spawn(tag: str) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    cmd = [sys.executable, "-m", "hapt_multiplicity", "--stage", "replay-dump", "--tag", tag]
    print("SPAWN", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env)


def stage_all() -> None:
    stage_train()
    _spawn("run1")
    _spawn("run2")
    det = stage_determinism()
    post = _prior_hashes()
    write_json(WRITE_ROOT / "hash_post.json", post)
    run1 = json.loads((WRITE_ROOT / "replay_run1.json").read_text())
    status = "PASS" if run1["fail_cells"] == 0 and det is not None and post["freeze_ok"] else "FAIL"
    write_json(
        WRITE_ROOT / "manifest.json",
        {
            "gate": "GATE 5",
            "protocol_id": "hapt_candidate_a_clean_two_source_v1",
            "training_device": "cpu",
            "additional_fitted_scaling": False,
            "k_greater_than_1": False,
            "replay_fail_cells": run1["fail_cells"],
            "determinism_bit_identical_all": det["bit_identical_all"],
            "status": status,
        },
    )
    print("GATE5", status, "fail_cells", run1["fail_cells"], flush=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="HAPT clean two-source train/freeze/replay")
    p.add_argument("--stage", default="all", choices=("all", "train", "replay-dump", "determinism"))
    p.add_argument("--tag", default="run1")
    args = p.parse_args(argv)
    if args.stage == "train":
        stage_train()
    elif args.stage == "replay-dump":
        stage_replay_dump(args.tag)
    elif args.stage == "determinism":
        stage_determinism()
    else:
        stage_all()
    return 0
