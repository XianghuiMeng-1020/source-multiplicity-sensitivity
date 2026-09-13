"""Gate-3 orchestrator: hash audit, sweep, figures, manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from multiplicity_replay.hashes import audit_freeze
from multiplicity_replay.replay import REPLAY_ROOT, REPO_ROOT, file_sha256

from .figures import render_figures
from .sweep import SWEEP_ROOT, write_json, run_sweep


def _hash_tree(root: Path) -> dict[str, str]:
    out = {}
    if not root.exists():
        return out
    for path in sorted(root.rglob("*")):
        if path.is_file():
            out[str(path.relative_to(REPO_ROOT))] = file_sha256(path)
    return out


def _gate2_cache_hashes() -> dict[str, str]:
    out = {}
    for path in sorted((REPLAY_ROOT / "caches").glob("seed*.npz")):
        out[str(path.relative_to(REPO_ROOT))] = file_sha256(path)
    return out


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="Clean-source multiplicity sweep").parse_args(argv)
    pre = audit_freeze()
    pre["gate2_caches"] = _gate2_cache_hashes()
    write_json(SWEEP_ROOT / "hash_pre.json", pre)
    if not pre["ok"]:
        raise RuntimeError("frozen hash pre-check failed")

    result = run_sweep()
    figures = render_figures()

    post = audit_freeze()
    post["gate2_caches"] = _gate2_cache_hashes()
    write_json(SWEEP_ROOT / "hash_post.json", post)
    if not post["ok"]:
        raise RuntimeError("frozen hash post-check failed")
    if pre["watched_extra"] != post["watched_extra"]:
        raise RuntimeError("watched extra hashes drifted")
    if pre["checkpoints"] != post["checkpoints"]:
        raise RuntimeError("checkpoint hashes drifted")
    if pre["gate2_caches"] != post["gate2_caches"]:
        raise RuntimeError("Gate-2 cache hashes drifted")

    new_hashes = _hash_tree(SWEEP_ROOT)
    manifest = {
        "gate": "GATE 3",
        "phase4_results_role": "immutable reference artifacts",
        "gate2_caches_role": "immutable replay caches",
        "training_executed": False,
        "command": f"PYTHONPATH=src {sys.executable} -m multiplicity_sweep",
        "mean_finite_k_law": result["summaries"]["mean_finite_k_law"],
        "lp_limit_proposition": result["summaries"]["lp_limit_proposition"],
        "n_metric_rows": result["summaries"]["n_metric_rows"],
        "figures": figures,
        "frozen_hash_pre_ok": pre["ok"],
        "frozen_hash_post_ok": post["ok"],
        "gate2_caches": post["gate2_caches"],
        "new_output_hashes": new_hashes,
        "sweep_status": (
            "PASS"
            if result["mean_law_pass"] and pre["ok"] and post["ok"]
            else "FAIL"
        ),
    }
    write_json(SWEEP_ROOT / "manifest.json", manifest)
    print("MEAN FINITE-k LAW:", result["summaries"]["mean_finite_k_law"], flush=True)
    print("LP-LIMIT:", result["summaries"]["lp_limit_proposition"], flush=True)
    print("metric rows:", result["summaries"]["n_metric_rows"], flush=True)
    return 0
