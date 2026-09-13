"""Frozen Phase-4 constants and write-guard. Protocol: 03b / v3."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WRITE_ROOT = (REPO_ROOT / "research" / "phase4_9of10" / "results").resolve()
PROTOCOL_VERSION = "phase4_joint_k_x_degrade_v3_approved"

DATASET = "MHEALTH"
M1 = "chest_acc"
M2 = "ankle"
M3 = "arm"
SEEDS = (0, 1, 2)
K_DUP = 5
N_CLASSES = 12
PHI_H = 16
G_HIDDEN = 16
ENCODER_EPOCHS = 25
CPSA_EPOCHS = 25
CPSA_LR = 1e-3
CPSA_BATCH = 128
HOLDOUT_FRAC = 0.2
STUCK_RULE = "X1tr[0]_after_train_standardize"
MODALITY_ID = False
ACC_TOL = 0.005
ECE_BINS = 15
EXPECTED_CPSA_PARAMS = 1009

HISTORICAL_RESULT_DIRS = (
    REPO_ROOT / "research" / "results",
    REPO_ROOT / "research" / "phase2" / "results",
    REPO_ROOT / "research" / "silent_failure_pilot" / "results",
    REPO_ROOT / "research" / "silent_failure_phase2" / "results",
    REPO_ROOT / "research" / "silent_failure_frozen_experiments",
    REPO_ROOT / "research" / "silent_failure_phase3_5" / "results",
)

LOCK_MESSAGE = (
    "Official Phase-4 execution is locked.\n"
    "Run semantic tests first and use the explicit execution flag only "
    "after external authorization."
)

PROTOCOL_ABORT_MESSAGE = (
    "Protocol token missing or wrong. Abort before data loading."
)


def assert_write_allowed(path: str | Path) -> Path:
    """Allow writes only under WRITE_ROOT after absolute resolve.

    ``../`` escapes, historical result dirs, ``paper/``, and ``src/`` fail.
    """
    resolved = Path(path).resolve()
    root = WRITE_ROOT.resolve()
    allowed = resolved == root or root in resolved.parents
    if not allowed:
        raise PermissionError(
            f"Phase-4 write refused: {resolved} is not under {root}"
        )
    return resolved
