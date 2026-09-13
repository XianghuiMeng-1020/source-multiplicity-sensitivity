"""Structural tests for the multiplicity sweep. Do not write frozen artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from multiplicity_replay.replay import FROZEN_RESULTS, REPLAY_ROOT
from multiplicity_sweep.sweep import (
    K_VALUES,
    SWEEP_ROOT,
    assert_sweep_write,
    token_list,
)


def test_k5_is_five_total_copies_not_five_added():
    p1, p2, p3 = [np.full((3, 12), i + 1, dtype=np.float32) for i in range(3)]
    toks = token_list(p1, p2, p3, "M1", 5)
    assert len(toks) == 7
    assert sum(t is p1 for t in toks) == 5
    toks2 = token_list(p1, p2, p3, "M2", 5)
    assert sum(t is p2 for t in toks2) == 5
    assert sum(t is p1 for t in toks2) == 1


def test_k1_ignores_source_identity():
    p1, p2, p3 = [np.full((3, 12), i + 1, dtype=np.float32) for i in range(3)]
    a = token_list(p1, p2, p3, "M1", 1)
    b = token_list(p1, p2, p3, "M2", 1)
    c = token_list(p1, p2, p3, "M3", 1)
    assert a == b == c == [p1, p2, p3]


def test_k_grid_and_token_cardinality():
    p1, p2, p3 = [np.ones((2, 12), dtype=np.float32) for _ in range(3)]
    assert K_VALUES == (1, 2, 3, 5, 8, 16, 32, 64, 100)
    for k in K_VALUES:
        toks = token_list(p1, p2, p3, "M3", k)
        assert len(toks) == (3 if k == 1 else k + 2)


def test_sweep_never_writes_frozen_or_replay():
    with pytest.raises(PermissionError):
        assert_sweep_write(FROZEN_RESULTS / "joint_raw.csv")
    with pytest.raises(PermissionError):
        assert_sweep_write(REPLAY_ROOT / "caches" / "seed0.npz")


def test_full_matrix_present_if_sweep_ran():
    path = SWEEP_ROOT / "multiplicity_metrics.csv"
    if not path.exists():
        pytest.skip("sweep not run yet")
    import csv

    rows = list(csv.DictReader(path.open()))
    keys = {(int(r["seed"]), r["source"], int(r["k"]), r["operator"]) for r in rows}
    assert len(keys) == 3 * 3 * 9 * 3 == 243
    assert len(rows) == 243


def test_mean_law_pass_if_sweep_ran():
    path = SWEEP_ROOT / "summary.json"
    if not path.exists():
        pytest.skip("sweep not run yet")
    s = json.loads(path.read_text())
    assert s["mean_finite_k_law"] == "PASS"
    assert s["n_metric_rows"] == 243
