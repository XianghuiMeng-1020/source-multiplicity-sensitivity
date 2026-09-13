"""Gate-2 orchestrator: hash audit, CPU replay, determinism, manifest."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from .hashes import audit_freeze
from .replay import (
    CHECKPOINT_LOAD_METHOD,
    COMPARE_METRICS,
    DEVICE,
    FROZEN_RESULTS,
    REPLAY_ROOT,
    REPO_ROOT,
    SEEDS,
    assert_replay_write,
    compare_to_frozen,
    file_sha256,
    infer_all_seeds,
    source_hashes,
    write_json,
    write_seed_cache,
)


def _env_block() -> dict:
    import numpy
    import torch

    return {
        "python": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "numpy": numpy.__version__,
        "torch": torch.__version__,
        "device": str(DEVICE),
        "mps_available": bool(torch.backends.mps.is_available()),
        "checkpoint_loading": CHECKPOINT_LOAD_METHOD,
        "map_location": "cpu",
    }


def _load_dataset():
    from datasets import load_mhealth

    return load_mhealth()


def _dataset_block(ds) -> dict:
    zip_path = REPO_ROOT / "data" / "raw" / "mhealth.zip"
    return {
        "name": ds.name,
        "train_subjects": sorted(set(int(s) for s in ds.train.subject.tolist())),
        "test_subjects": sorted(set(int(s) for s in ds.test.subject.tolist())),
        "test_n": int(ds.test.n),
        "n_classes": int(ds.n_classes),
        "modality_order": list(ds.modality_order),
        "notes": ds.notes,
        "mhealth_zip_sha256": file_sha256(zip_path) if zip_path.exists() else "ABSENT",
        "X1tr0_subject": int(ds.train.subject[0]),
        "X1tr0_recording_file": "UNVERIFIED",
        "X1tr0_window_start_end": "UNVERIFIED",
        "window_temporal_provenance": "UNVERIFIED",
    }


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def _arrays_bit_identical(a: np.ndarray, b: np.ndarray) -> bool:
    a = np.asarray(a)
    b = np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and bool((a == b).all())


def stage_infer(tag: str) -> dict:
    ds = _load_dataset()
    dataset = _dataset_block(ds)
    if dataset["train_subjects"] != [1, 2, 3, 4, 5, 6, 7]:
        raise RuntimeError(f"train subjects {dataset['train_subjects']}")
    if dataset["test_subjects"] != [8, 9, 10]:
        raise RuntimeError(f"test subjects {dataset['test_subjects']}")
    if dataset["test_n"] != 1987 or dataset["n_classes"] != 12:
        raise RuntimeError("unexpected test n or n_classes")

    hashes = source_hashes()
    cache_dir = REPLAY_ROOT / ("caches" if tag == "run1" else f"caches_{tag}")
    payloads = infer_all_seeds(ds)
    cache_paths = []
    rows = []
    for payload in payloads:
        cache_paths.append(str(write_seed_cache(payload, hashes, cache_dir)))
        rows.extend(payload["rows"])

    comparison = compare_to_frozen(rows) if tag == "run1" else []
    report = {
        "tag": tag,
        "dataset": dataset,
        "environment": _env_block(),
        "source_sha256": hashes,
        "cache_paths": cache_paths,
        "rows": rows,
        "comparison": comparison,
        "fail_cells": sum(
            1
            for cell in comparison
            for m in COMPARE_METRICS
            if cell[m]["status"] == "FAIL"
        ),
    }
    write_json(REPLAY_ROOT / f"infer_{tag}.json", report)
    if tag == "run1":
        write_json(REPLAY_ROOT / "comparison.json", report)
        _write_comparison_csv(comparison)
        _write_max_discrepancy(comparison)
    return report


def _write_comparison_csv(comparison: list[dict]) -> None:
    dest = assert_replay_write(REPLAY_ROOT / "comparison.csv")
    fields = ["rule", "seed", "arm"]
    for m in COMPARE_METRICS:
        fields.extend([f"{m}_frozen", f"{m}_replayed", f"{m}_abs_diff", f"{m}_status"])
    fields.extend(["confidence_gap_frozen", "confidence_gap_replayed"])
    import csv

    with dest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for cell in comparison:
            row = {"rule": cell["rule"], "seed": cell["seed"], "arm": cell["arm"]}
            for m in COMPARE_METRICS:
                row[f"{m}_frozen"] = repr(cell[m]["frozen"])
                row[f"{m}_replayed"] = repr(cell[m]["replayed"])
                row[f"{m}_abs_diff"] = repr(cell[m]["abs_diff"])
                row[f"{m}_status"] = cell[m]["status"]
            row["confidence_gap_frozen"] = repr(cell["confidence_gap_frozen"])
            row["confidence_gap_replayed"] = repr(cell["confidence_gap_replayed"])
            w.writerow(row)


def _write_max_discrepancy(comparison: list[dict]) -> dict:
    out = {"by_system": {}, "by_metric": {}, "global_max": 0.0, "fail_cells": 0}
    for rule in ("late_product", "cpsa"):
        out["by_system"][rule] = {}
        for m in COMPARE_METRICS:
            diffs = [c[m]["abs_diff"] for c in comparison if c["rule"] == rule]
            out["by_system"][rule][m] = max(diffs) if diffs else None
    for m in COMPARE_METRICS:
        diffs = [c[m]["abs_diff"] for c in comparison]
        out["by_metric"][m] = max(diffs) if diffs else None
        out["global_max"] = max(out["global_max"], out["by_metric"][m] or 0.0)
    out["fail_cells"] = sum(
        1 for c in comparison for m in COMPARE_METRICS if c[m]["status"] == "FAIL"
    )
    write_json(REPLAY_ROOT / "max_discrepancy.json", out)
    return out


def _npz_map(cache_dir: Path) -> dict[int, np.lib.npyio.NpzFile]:
    out = {}
    for seed in SEEDS:
        out[int(seed)] = np.load(cache_dir / f"seed{seed}.npz")
    return out


def stage_determinism() -> dict:
    run1 = _npz_map(REPLAY_ROOT / "caches")
    run2 = _npz_map(REPLAY_ROOT / "caches_run2")
    keys_mod = ("p1_test", "p2_test", "p3_test", "p1_stuck", "y_test")
    keys_cpsa = tuple(f"fused_cpsa_{arm}" for arm in "ABCDE")
    keys_lp = tuple(f"fused_late_product_{arm}" for arm in "ABCDE")
    report: dict = {"seeds": {}, "bit_identical_all": True}
    for seed in SEEDS:
        a, b = run1[int(seed)], run2[int(seed)]
        seed_rec = {}
        for key in keys_mod + keys_cpsa + keys_lp:
            d = _max_abs(a[key], b[key])
            ident = _arrays_bit_identical(a[key], b[key])
            seed_rec[key] = {"max_abs_diff": d, "bit_identical": ident}
            if not ident:
                report["bit_identical_all"] = False
        report["seeds"][str(seed)] = seed_rec
    for handle in list(run1.values()) + list(run2.values()):
        handle.close()
    write_json(REPLAY_ROOT / "determinism.json", report)
    return report


def _cache_hashes(cache_dir: Path) -> dict[str, str]:
    out = {}
    for path in sorted(cache_dir.glob("*")):
        if path.is_file():
            out[str(path.relative_to(REPO_ROOT))] = file_sha256(path)
    return out


def write_manifest(
    pre: dict,
    post: dict,
    infer: dict,
    determinism: dict,
    commands: list[str],
) -> Path:
    payload = {
        "gate": "GATE 2",
        "phase4_results_role": "immutable reference artifacts",
        "immutable_reference_dir": str(FROZEN_RESULTS),
        "replay_write_root": str(REPLAY_ROOT),
        "training_executed": False,
        "official_write_command_executed": False,
        "commands": commands,
        "environment": infer["environment"],
        "dataset": infer["dataset"],
        "input_hashes": {
            "mhealth_zip_sha256": infer["dataset"]["mhealth_zip_sha256"],
            "frozen_reference": pre,
        },
        "checkpoint_hashes": pre["checkpoints"],
        "source_hashes": infer["source_sha256"],
        "generated_cache_hashes": {
            "run1": _cache_hashes(REPLAY_ROOT / "caches"),
            "run2": _cache_hashes(REPLAY_ROOT / "caches_run2"),
        },
        "frozen_reference_hashes_pre": pre,
        "frozen_reference_hashes_post": post,
        "fail_cells": infer["fail_cells"],
        "determinism_bit_identical_all": determinism.get("bit_identical_all"),
        "replay_status": (
            "PASS"
            if infer["fail_cells"] == 0 and pre["ok"] and post["ok"] and pre == post
            else "FAIL"
        ),
    }
    dest = write_json(REPLAY_ROOT / "manifest.json", payload)
    return dest


def _spawn_infer(tag: str) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    cmd = [sys.executable, "-m", "multiplicity_replay", "--stage", "infer", "--tag", tag]
    print("SPAWN", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env)


def stage_all() -> dict:
    commands = [
        f"PYTHONPATH=src {sys.executable} -m multiplicity_replay --stage infer --tag run1",
        f"PYTHONPATH=src {sys.executable} -m multiplicity_replay --stage infer --tag run2",
        f"PYTHONPATH=src {sys.executable} -m multiplicity_replay --stage determinism",
    ]
    pre = audit_freeze()
    write_json(REPLAY_ROOT / "hash_pre.json", pre)
    if not pre["ok"]:
        raise RuntimeError("frozen hash pre-check failed")

    infer = stage_infer("run1")
    _spawn_infer("run2")
    determinism = stage_determinism()

    post = audit_freeze()
    write_json(REPLAY_ROOT / "hash_post.json", post)
    if not post["ok"] or pre != post:
        raise RuntimeError("frozen hash post-check failed or hashes drifted")

    manifest = write_manifest(pre, post, infer, determinism, commands)
    return {
        "fail_cells": infer["fail_cells"],
        "manifest": str(manifest),
        "frozen_ok": True,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Frozen-checkpoint multiplicity replay")
    p.add_argument("--stage", choices=("all", "infer", "determinism"), default="all")
    p.add_argument("--tag", default="run1")
    args = p.parse_args(argv)
    if args.stage == "infer":
        stage_infer(args.tag)
        return 0
    if args.stage == "determinism":
        stage_determinism()
        return 0
    stage_all()
    return 0
