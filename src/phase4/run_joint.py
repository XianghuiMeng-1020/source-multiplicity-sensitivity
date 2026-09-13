"""Phase-4 joint runner. Official MHEALTH execution is locked by default."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from metrics import summarize  # noqa: E402
from models import late_product, predict_proba, train_mlp  # noqa: E402

from .config import (
    CPSA_EPOCHS,
    DATASET,
    ENCODER_EPOCHS,
    HOLDOUT_FRAC,
    K_DUP,
    LOCK_MESSAGE,
    M1,
    M2,
    M3,
    PROTOCOL_ABORT_MESSAGE,
    PROTOCOL_VERSION,
    SEEDS,
    WRITE_ROOT,
    assert_write_allowed,
)
from .contextual_psa import ContextualPSA, fuse_cpsa, train_cpsa

ARMS = ("A", "B", "C", "D", "E")
ARM_K = {"A": 1, "B": K_DUP, "C": 1, "D": K_DUP, "E": 0}
ARM_RELIABILITY = {
    "A": "clean",
    "B": "clean",
    "C": "hard_stuck_trainref",
    "D": "hard_stuck_trainref",
    "E": "dropped",
}
ARM_N_TOKENS = {"A": 3, "B": 7, "C": 3, "D": 7, "E": 2}
RAW_FIELDS = [
    "arm",
    "seed",
    "rule",
    "k",
    "reliability",
    "n_tokens",
    "accuracy",
    "confidence",
    "ece",
    "nll",
    "brier",
    "entropy",
    "conf_acc_gap",
    "n",
]


def holdout_idx(n: int, frac: float = HOLDOUT_FRAC, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_h = max(1, int(frac * n))
    return idx[n_h:], idx[:n_h]


def standardize_pair(tr: np.ndarray, te: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = tr.mean(axis=0)
    sd = tr.std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return (tr - mu) / sd, (te - mu) / sd, mu, sd


def make_trainref_stuck(X1_train_std: np.ndarray, n_rows: int) -> np.ndarray:
    """Phase-4 stuck: repeat standardized train row 0. Does not accept test arrays."""
    if n_rows < 1:
        raise ValueError("n_rows must be >= 1")
    train = np.asarray(X1_train_std)
    if train.ndim != 2 or train.shape[0] < 1:
        raise ValueError("X1_train_std must be a non-empty 2-d train array")
    stuck_ref = train[0]
    return np.repeat(stuck_ref[None, :], int(n_rows), axis=0)


def token_list(
    arm: str,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    p1_bad: np.ndarray,
) -> list[np.ndarray]:
    if arm == "A":
        return [p1, p2, p3]
    if arm == "B":
        return [p1] * K_DUP + [p2, p3]
    if arm == "C":
        return [p1_bad, p2, p3]
    if arm == "D":
        return [p1_bad] * K_DUP + [p2, p3]
    if arm == "E":
        return [p2, p3]
    raise ValueError(f"unknown arm {arm}")


def stack_tokens(parts: list[np.ndarray]) -> np.ndarray:
    return np.stack(parts, axis=1)


def evaluate_late_product_arm(
    arm: str,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    p1_bad: np.ndarray,
) -> np.ndarray:
    return late_product(token_list(arm, p1, p2, p3, p1_bad))


def evaluate_cpsa_arm(
    model: ContextualPSA,
    arm: str,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    p1_bad: np.ndarray,
) -> np.ndarray:
    tokens = stack_tokens(token_list(arm, p1, p2, p3, p1_bad))
    return fuse_cpsa(model, tokens)


def encoder_slice(X: np.ndarray, y: np.ndarray, tr_i: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return X[tr_i], y[tr_i]


def cpsa_holdout_sets(
    p1_train: np.ndarray,
    p2_train: np.ndarray,
    p3_train: np.ndarray,
    ho_i: np.ndarray,
) -> np.ndarray:
    return np.stack([p1_train[ho_i], p2_train[ho_i], p3_train[ho_i]], axis=1)


def make_result_row(arm: str, seed: int, rule: str, probs: np.ndarray, y: np.ndarray) -> dict:
    row = summarize(probs, y)
    row.update(
        {
            "arm": arm,
            "seed": int(seed),
            "rule": rule,
            "k": ARM_K[arm],
            "reliability": ARM_RELIABILITY[arm],
            "n_tokens": ARM_N_TOKENS[arm],
        }
    )
    return {k: row[k] for k in RAW_FIELDS}


def _load_mhealth():
    from datasets import load_mhealth

    return load_mhealth()


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    dest = assert_write_allowed(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _write_json(path: Path, payload: dict) -> None:
    dest = assert_write_allowed(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _save_checkpoint(path: Path, model) -> None:
    import torch

    dest = assert_write_allowed(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), dest)


def run_official_seed(seed: int, ds) -> list[dict]:
    """Official per-seed path. Must not be invoked in Step 4."""
    m1, m2, m3 = ds.modality_order
    if (m1, m2, m3) != (M1, M2, M3):
        raise RuntimeError(f"unexpected modality_order {ds.modality_order}")
    ytr = ds.train.y
    n_classes = ds.n_classes
    X1tr, X1te, _, _ = standardize_pair(ds.train.modalities[m1], ds.test.modalities[m1])
    X2tr, X2te, _, _ = standardize_pair(ds.train.modalities[m2], ds.test.modalities[m2])
    X3tr, X3te, _, _ = standardize_pair(ds.train.modalities[m3], ds.test.modalities[m3])
    tr_i, ho_i = holdout_idx(len(ytr), HOLDOUT_FRAC, seed)
    if set(tr_i) & set(ho_i):
        raise RuntimeError("tr_i and ho_i must be disjoint")

    enc1 = train_mlp(*encoder_slice(X1tr, ytr, tr_i), n_classes, epochs=ENCODER_EPOCHS, seed=seed)
    enc2 = train_mlp(*encoder_slice(X2tr, ytr, tr_i), n_classes, epochs=ENCODER_EPOCHS, seed=seed + 1)
    enc3 = train_mlp(*encoder_slice(X3tr, ytr, tr_i), n_classes, epochs=ENCODER_EPOCHS, seed=seed + 2)
    enc1.eval()
    enc2.eval()
    enc3.eval()

    p1_ho = predict_proba(enc1, X1tr[ho_i])
    p2_ho = predict_proba(enc2, X2tr[ho_i])
    p3_ho = predict_proba(enc3, X3tr[ho_i])
    cpsa_sets = cpsa_holdout_sets(
        predict_proba(enc1, X1tr),
        predict_proba(enc2, X2tr),
        predict_proba(enc3, X3tr),
        ho_i,
    )
    if not np.allclose(cpsa_sets[:, 0], p1_ho) or cpsa_sets.shape[1] != 3:
        raise RuntimeError("CPSA holdout set construction failed")
    cpsa = train_cpsa(cpsa_sets, ytr[ho_i], seed=seed + 10, epochs=CPSA_EPOCHS)
    cpsa.eval()

    ckpt_dir = WRITE_ROOT / "checkpoints"
    _save_checkpoint(ckpt_dir / f"seed{seed}_enc1.pt", enc1)
    _save_checkpoint(ckpt_dir / f"seed{seed}_enc2.pt", enc2)
    _save_checkpoint(ckpt_dir / f"seed{seed}_enc3.pt", enc3)
    _save_checkpoint(ckpt_dir / f"seed{seed}_cpsa.pt", cpsa)

    X1_bad = make_trainref_stuck(X1tr, n_rows=len(X1te))
    p1 = predict_proba(enc1, X1te)
    p2 = predict_proba(enc2, X2te)
    p3 = predict_proba(enc3, X3te)
    p1_bad = predict_proba(enc1, X1_bad)

    yte = ds.test.y
    rows: list[dict] = []
    for arm in ARMS:
        lp = evaluate_late_product_arm(arm, p1, p2, p3, p1_bad)
        rows.append(make_result_row(arm, seed, "late_product", lp, yte))
        cp = evaluate_cpsa_arm(cpsa, arm, p1, p2, p3, p1_bad)
        rows.append(make_result_row(arm, seed, "cpsa", cp, yte))
    return rows


def run_official() -> int:
    """Full official pipeline. Locked unless both flags are supplied."""
    ds = _load_mhealth()
    if ds.name != DATASET:
        raise RuntimeError(f"expected {DATASET}, got {ds.name}")
    all_rows: list[dict] = []
    for seed in SEEDS:
        all_rows.extend(run_official_seed(int(seed), ds))
    _write_csv(WRITE_ROOT / "joint_raw.csv", all_rows, RAW_FIELDS)
    _write_json(
        WRITE_ROOT / "config.json",
        {
            "protocol": PROTOCOL_VERSION,
            "dataset": DATASET,
            "seeds": list(SEEDS),
            "k_dup": K_DUP,
        },
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase-4 joint factorial (locked by default)")
    p.add_argument("--execute-official", action="store_true")
    p.add_argument("--protocol", default="", type=str)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.execute_official:
        print(LOCK_MESSAGE)
        return 2
    if args.protocol != PROTOCOL_VERSION:
        print(PROTOCOL_ABORT_MESSAGE)
        return 2
    return run_official()


if __name__ == "__main__":
    sys.exit(main())
