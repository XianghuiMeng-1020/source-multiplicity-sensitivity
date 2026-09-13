"""Replay-namespace tests. Do not write into frozen Phase-4 results."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_SRC = Path(__file__).resolve().parents[1]
_REPO = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from models import MLP
from phase4.config import K_DUP, N_CLASSES, WRITE_ROOT
from phase4.contextual_psa import ContextualPSA
from phase4.run_joint import token_list

from multiplicity_replay.replay import (
    CKPT_DIR,
    COMPARE_METRICS,
    FROZEN_RESULTS,
    REPLAY_ROOT,
    assert_replay_write,
    load_cpsa,
    load_encoder,
)


def test_frozen_checkpoints_load_without_training(monkeypatch):
    import models
    import phase4.contextual_psa as cpsa_mod

    def boom(*_a, **_k):
        raise RuntimeError("training must not run during replay tests")

    monkeypatch.setattr(models, "train_mlp", boom)
    monkeypatch.setattr(cpsa_mod, "train_cpsa", boom)

    enc = load_encoder(CKPT_DIR / "seed0_enc1.pt", d_in=15)
    cpsa = load_cpsa(CKPT_DIR / "seed0_cpsa.pt")
    assert isinstance(enc, MLP)
    assert isinstance(cpsa, ContextualPSA)
    assert enc.training is False
    assert cpsa.training is False
    assert next(enc.parameters()).device.type == "cpu"
    assert next(cpsa.parameters()).device.type == "cpu"
    for p in list(enc.parameters()) + list(cpsa.parameters()):
        assert p.requires_grad is False


def test_posterior_shapes_and_row_sums():
    cache = REPLAY_ROOT / "caches" / "seed0.npz"
    if not cache.exists():
        pytest.skip("replay caches not written yet")
    z = np.load(cache)
    for key in ("p1_test", "p2_test", "p3_test", "p1_stuck"):
        arr = z[key]
        assert arr.shape == (1987, 12)
        assert np.isfinite(arr).all()
        assert not (arr < 0).any()
        assert float(np.max(np.abs(arr.sum(axis=1) - 1.0))) <= 1e-5
    assert z["y_test"].shape == (1987,)


def test_official_b_has_exactly_five_m1_copies():
    p1, p2, p3, stuck = [np.full((4, N_CLASSES), i + 1, dtype=np.float32) for i in range(4)]
    toks = token_list("B", p1, p2, p3, stuck)
    assert len(toks) == 7
    assert sum(t is p1 for t in toks) == K_DUP == 5
    assert toks == [p1, p1, p1, p1, p1, p2, p3]


def test_official_d_has_exactly_five_stuck_m1_copies():
    p1, p2, p3, stuck = [np.full((4, N_CLASSES), i + 1, dtype=np.float32) for i in range(4)]
    toks = token_list("D", p1, p2, p3, stuck)
    assert len(toks) == 7
    assert sum(t is stuck for t in toks) == K_DUP == 5
    assert toks == [stuck, stuck, stuck, stuck, stuck, p2, p3]


def test_official_e_contains_no_m1():
    p1, p2, p3, stuck = [np.full((4, N_CLASSES), i + 1, dtype=np.float32) for i in range(4)]
    toks = token_list("E", p1, p2, p3, stuck)
    assert toks == [p2, p3]
    assert all(t is not p1 and t is not stuck for t in toks)


def test_replay_never_writes_frozen_results():
    frozen = Path(WRITE_ROOT).resolve()
    assert frozen == FROZEN_RESULTS
    with pytest.raises(PermissionError):
        assert_replay_write(frozen / "joint_raw.csv")
    with pytest.raises(PermissionError):
        assert_replay_write(frozen / "replay_probe.txt")
    dest = assert_replay_write(REPLAY_ROOT / "test_write_probe.txt")
    dest.write_text("ok", encoding="utf-8")
    dest.unlink()


def test_metric_comparison_has_zero_fail_cells():
    path = REPLAY_ROOT / "comparison.json"
    if not path.exists():
        pytest.skip("comparison.json not written yet")
    report = json.loads(path.read_text(encoding="utf-8"))
    fails = [
        (c["rule"], c["seed"], c["arm"], m)
        for c in report["comparison"]
        for m in COMPARE_METRICS
        if c[m]["status"] == "FAIL"
    ]
    assert report["fail_cells"] == 0
    assert fails == []
    assert len(report["comparison"]) == 30


def test_torch_is_cpu_only_for_loaded_weights():
    state = torch.load(CKPT_DIR / "seed0_cpsa.pt", map_location="cpu", weights_only=True)
    for v in state.values():
        if torch.is_tensor(v):
            assert v.device.type == "cpu"
