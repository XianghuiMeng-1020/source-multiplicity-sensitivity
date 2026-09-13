"""CPU replay of official A–E from frozen checkpoints.

Writes only under research/multiplicity_replay/. Never trains. Never writes
Phase-4 results. Reuses existing standardize / stuck / token / fusion / metric
functions without reimplementing them.
"""

from __future__ import annotations

import csv
import hashlib
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
from models import MLP, predict_proba  # noqa: E402

from phase4.config import (  # noqa: E402
    M1,
    M2,
    M3,
    N_CLASSES,
    SEEDS,
    WRITE_ROOT,
)
from phase4.contextual_psa import ContextualPSA  # noqa: E402
from phase4.run_joint import (  # noqa: E402
    ARMS,
    evaluate_cpsa_arm,
    evaluate_late_product_arm,
    make_trainref_stuck,
    standardize_pair,
    token_list,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REPLAY_ROOT = (REPO_ROOT / "research" / "multiplicity_replay").resolve()
FROZEN_RESULTS = WRITE_ROOT.resolve()
CKPT_DIR = FROZEN_RESULTS / "checkpoints"
JOINT_RAW = FROZEN_RESULTS / "joint_raw.csv"
DEVICE = torch.device("cpu")
ROW_SUM_TOL = 1e-5
NUMERICAL_MATCH = 1e-7
CLOSE_ONLY = 1e-5
COMPARE_METRICS = ("accuracy", "ece", "nll", "brier", "confidence")
CHECKPOINT_LOAD_METHOD = "torch.load(path, map_location='cpu', weights_only=True)"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_replay_write(path: Path) -> Path:
    dest = Path(path).resolve()
    if FROZEN_RESULTS == dest or FROZEN_RESULTS in dest.parents:
        raise PermissionError(f"replay write refused inside frozen results: {dest}")
    if REPLAY_ROOT != dest and REPLAY_ROOT not in dest.parents:
        raise PermissionError(f"replay write refused outside {REPLAY_ROOT}: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    return dest


def load_state_dict(path: Path) -> dict:
    try:
        state = torch.load(path, map_location=DEVICE, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=DEVICE)
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint is not a state_dict: {path}")
    return state


def load_encoder(path: Path, d_in: int) -> MLP:
    model = MLP(d_in, N_CLASSES, hidden=128)
    model.load_state_dict(load_state_dict(path))
    model.to(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_cpsa(path: Path) -> ContextualPSA:
    model = ContextualPSA()
    model.load_state_dict(load_state_dict(path))
    model.to(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def classify_delta(delta: float, frozen: float, replayed: float) -> str:
    if frozen == replayed:
        return "EXACT"
    ad = abs(delta)
    if ad <= NUMERICAL_MATCH:
        return "NUMERICAL_MATCH"
    if ad <= CLOSE_ONLY:
        return "CLOSE_ONLY"
    return "FAIL"


def assert_official_tokens(
    arm: str,
    toks: list[np.ndarray],
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    p1_stuck: np.ndarray,
) -> None:
    if arm == "A" and toks != [p1, p2, p3]:
        raise RuntimeError("arm A token list is not [p1, p2, p3]")
    if arm == "B":
        if len(toks) != 7 or sum(t is p1 for t in toks) != 5:
            raise RuntimeError("arm B does not contain exactly five M1 copies")
        if toks != [p1, p1, p1, p1, p1, p2, p3]:
            raise RuntimeError("arm B is not [p1 x5, p2, p3]")
    if arm == "C" and toks != [p1_stuck, p2, p3]:
        raise RuntimeError("arm C token list is not [p1_stuck, p2, p3]")
    if arm == "D":
        if len(toks) != 7 or sum(t is p1_stuck for t in toks) != 5:
            raise RuntimeError("arm D does not contain exactly five stuck-M1 copies")
        if toks != [p1_stuck, p1_stuck, p1_stuck, p1_stuck, p1_stuck, p2, p3]:
            raise RuntimeError("arm D is not [p1_stuck x5, p2, p3]")
    if arm == "E":
        if toks != [p2, p3]:
            raise RuntimeError("arm E token list is not [p2, p3]")
        if any(t is p1 or t is p1_stuck for t in toks):
            raise RuntimeError("arm E contains an M1 token")


def replay_seed(ds, seed: int) -> dict:
    m1, m2, m3 = ds.modality_order
    if (m1, m2, m3) != (M1, M2, M3):
        raise RuntimeError(f"unexpected modality_order {ds.modality_order}")

    X1tr, X1te, mu1, sd1 = standardize_pair(ds.train.modalities[m1], ds.test.modalities[m1])
    _, X2te, _, _ = standardize_pair(ds.train.modalities[m2], ds.test.modalities[m2])
    _, X3te, _, _ = standardize_pair(ds.train.modalities[m3], ds.test.modalities[m3])
    yte = np.asarray(ds.test.y)

    enc_paths = {
        "enc1": CKPT_DIR / f"seed{seed}_enc1.pt",
        "enc2": CKPT_DIR / f"seed{seed}_enc2.pt",
        "enc3": CKPT_DIR / f"seed{seed}_enc3.pt",
        "cpsa": CKPT_DIR / f"seed{seed}_cpsa.pt",
    }
    enc1 = load_encoder(enc_paths["enc1"], X1te.shape[1])
    enc2 = load_encoder(enc_paths["enc2"], X2te.shape[1])
    enc3 = load_encoder(enc_paths["enc3"], X3te.shape[1])
    cpsa = load_cpsa(enc_paths["cpsa"])
    if enc1.training or enc2.training or enc3.training or cpsa.training:
        raise RuntimeError("a loaded module is in train mode")

    p1 = predict_proba(enc1, X1te)
    p2 = predict_proba(enc2, X2te)
    p3 = predict_proba(enc3, X3te)
    X1_bad = make_trainref_stuck(X1tr, n_rows=len(X1te))
    p1_stuck = predict_proba(enc1, X1_bad)

    fused: dict[str, dict[str, np.ndarray]] = {"late_product": {}, "cpsa": {}}
    rows: list[dict] = []
    for arm in ARMS:
        toks = token_list(arm, p1, p2, p3, p1_stuck)
        assert_official_tokens(arm, toks, p1, p2, p3, p1_stuck)
        lp = evaluate_late_product_arm(arm, p1, p2, p3, p1_stuck)
        cp = evaluate_cpsa_arm(cpsa, arm, p1, p2, p3, p1_stuck)
        fused["late_product"][arm] = lp
        fused["cpsa"][arm] = cp
        for rule, probs in (("late_product", lp), ("cpsa", cp)):
            rec = summarize(probs, yte)
            rec.update({"arm": arm, "seed": int(seed), "rule": rule})
            rows.append(rec)

    return {
        "seed": int(seed),
        "p1": p1,
        "p2": p2,
        "p3": p3,
        "p1_stuck": p1_stuck,
        "y_test": yte,
        "fused": fused,
        "rows": rows,
        "enc_paths": enc_paths,
        "mu1": mu1,
        "sd1": sd1,
        "X1tr0_subject": int(ds.train.subject[0]),
    }


def assert_posteriors(arr: np.ndarray, name: str) -> None:
    if arr.shape != (1987, 12):
        raise RuntimeError(f"{name} shape {arr.shape} != (1987, 12)")
    if not np.isfinite(arr).all():
        raise RuntimeError(f"{name} has non-finite values")
    if (arr < 0).any():
        raise RuntimeError(f"{name} has negative probabilities")
    max_dev = float(np.max(np.abs(arr.sum(axis=1) - 1.0)))
    if max_dev > ROW_SUM_TOL:
        raise RuntimeError(f"{name} row sums deviate by {max_dev}")


def compare_to_frozen(rows: list[dict]) -> list[dict]:
    frozen_rows = list(csv.DictReader(JOINT_RAW.open()))
    index = {(r["rule"], int(r["seed"]), r["arm"]): r for r in frozen_rows}
    if len(index) != 30:
        raise RuntimeError(f"frozen joint_raw has {len(index)} keys, expected 30")
    out = []
    for rec in rows:
        key = (rec["rule"], int(rec["seed"]), rec["arm"])
        if key not in index:
            raise RuntimeError(f"missing frozen row {key}")
        fr = index[key]
        cell = {
            "rule": rec["rule"],
            "seed": int(rec["seed"]),
            "arm": rec["arm"],
            "confidence_gap_replayed": float(rec["confidence"]) - float(rec["accuracy"]),
            "confidence_gap_frozen": float(fr["confidence"]) - float(fr["accuracy"]),
        }
        for m in COMPARE_METRICS:
            frozen = float(fr[m])
            replayed = float(rec[m])
            delta = replayed - frozen
            cell[m] = {
                "frozen": frozen,
                "replayed": replayed,
                "abs_diff": abs(delta),
                "status": classify_delta(delta, frozen, replayed),
            }
        out.append(cell)
    if len(out) != 30:
        raise RuntimeError(f"replay comparison has {len(out)} rows, expected 30")
    return out


def source_hashes() -> dict[str, str]:
    rels = [
        "src/models.py",
        "src/metrics.py",
        "src/datasets.py",
        "src/phase4/run_joint.py",
        "src/phase4/contextual_psa.py",
        "src/phase4/config.py",
        "src/multiplicity_replay/replay.py",
        "src/multiplicity_replay/cli.py",
        "src/multiplicity_replay/hashes.py",
        "src/multiplicity_replay/__main__.py",
    ]
    return {r: file_sha256(REPO_ROOT / r) for r in rels}


def write_seed_cache(payload: dict, source: dict[str, str], cache_dir: Path) -> Path:
    seed = payload["seed"]
    dest = assert_replay_write(cache_dir / f"seed{seed}.npz")
    arrays = {
        "p1_test": np.asarray(payload["p1"]),
        "p2_test": np.asarray(payload["p2"]),
        "p3_test": np.asarray(payload["p3"]),
        "y_test": np.asarray(payload["y_test"]),
        "p1_stuck": np.asarray(payload["p1_stuck"]),
    }
    for rule in ("late_product", "cpsa"):
        for arm, arr in payload["fused"][rule].items():
            arrays[f"fused_{rule}_{arm}"] = np.asarray(arr)
    np.savez(dest, **arrays)
    meta = {
        "dataset": "MHEALTH",
        "seed": seed,
        "class_count": 12,
        "test_n": 1987,
        "shapes": {k: list(v.shape) for k, v in arrays.items()},
        "dtypes": {k: str(v.dtype) for k, v in arrays.items()},
        "checkpoint_paths": {k: str(p) for k, p in payload["enc_paths"].items()},
        "checkpoint_sha256": {k: file_sha256(p) for k, p in payload["enc_paths"].items()},
        "source_sha256": source,
        "standardization": (
            "train-only column mean/std via phase4.run_joint.standardize_pair; "
            "sd floor 1e-6; applied independently per modality"
        ),
        "stuck_rule": (
            "phase4.run_joint.make_trainref_stuck(X1tr, n_test) = tile X1tr[0] "
            "after train-only standardize; encode with frozen enc1"
        ),
        "X1tr0_subject": payload["X1tr0_subject"],
        "X1tr0_recording_file": "UNVERIFIED",
        "X1tr0_window_start_end": "UNVERIFIED",
        "window_temporal_provenance": "UNVERIFIED",
        "device": str(DEVICE),
        "checkpoint_loading": CHECKPOINT_LOAD_METHOD,
        "creation_command": (
            "PYTHONPATH=src /opt/homebrew/opt/python@3.13/bin/python3.13 "
            "-m multiplicity_replay"
        ),
        "phase4_results_role": "immutable reference artifacts",
        "immutable_reference": str(FROZEN_RESULTS),
    }
    meta_path = assert_replay_write(cache_dir / f"seed{seed}_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return dest


def infer_all_seeds(ds) -> list[dict]:
    payloads = []
    y_ref = None
    for seed in SEEDS:
        payload = replay_seed(ds, int(seed))
        for name in ("p1", "p2", "p3", "p1_stuck"):
            assert_posteriors(payload[name], f"seed{seed}_{name}")
        if y_ref is None:
            y_ref = payload["y_test"]
        elif not np.array_equal(y_ref, payload["y_test"]):
            raise RuntimeError("y_test differs across seeds")
        payloads.append(payload)
    return payloads


def write_json(path: Path, payload) -> Path:
    dest = assert_replay_write(path)
    dest.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    return dest
