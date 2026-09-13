"""Contamination and construction assertions. Do not write prior-gate artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hapt_multiplicity.features import assert_partition, load_feature_map
from hapt_multiplicity.loader import load_hapt_two_source
from hapt_multiplicity.paths import (
    CACHE_DIR,
    FEATURE_MAP,
    FROZEN_MHEALTH,
    PROTOCOL_JSON,
    RESULTS,
    WRITE_ROOT,
    assert_hapt_write,
)
from hapt_multiplicity.io_util import assert_protocol_frozen
from multiplicity_replay.replay import file_sha256


def test_protocol_still_preregistered():
    assert_protocol_frozen()
    assert file_sha256(PROTOCOL_JSON) == "ff5ebd78ba53283abf04de177db1178476e01fe4cebec3dbe2eddb85df36f6c9"
    assert file_sha256(FEATURE_MAP) == "f1deccb0aa31346c40a9483f15c502b07dd1957099a909a0357e864e2818a4c2"


def test_feature_partition_and_exclusions():
    fmap = load_feature_map()
    assert_partition(fmap)
    assert len(fmap["M1_indices"]) + len(fmap["M2_indices"]) + 2 == 561
    for name in ("tBodyGyro-AngleWRTGravity-1", "tBodyGyroJerk-AngleWRTGravity-1"):
        assert name not in fmap["M1_names"]
        assert name not in fmap["M2_names"]


def test_official_test_subjects_never_in_train_pool():
    ds = load_hapt_two_source()
    assert set(ds.train.subject.tolist()).isdisjoint(set(ds.test.subject.tolist()))


def test_no_test_rows_in_cpsa_holdout():
    if not (RESULTS / "holdout_seed0.npz").exists():
        pytest.skip("training not finished")
    ds = load_hapt_two_source()
    n_train = len(ds.train.y)
    for seed in (0, 1, 2):
        z = np.load(RESULTS / f"holdout_seed{seed}.npz")
        tr_i, ho_i = z["tr_i"], z["ho_i"]
        z.close()
        assert set(tr_i).isdisjoint(set(ho_i))
        assert tr_i.max() < n_train and ho_i.max() < n_train
        assert min(tr_i.min(), ho_i.min()) >= 0


def test_no_additional_scaling_in_loader_source():
    src = Path(__file__).resolve().with_name("loader.py").read_text(encoding="utf-8")
    assert "standardize" not in src
    assert "mean(axis" not in src


def test_write_guard_blocks_prior_artifacts():
    with pytest.raises(PermissionError):
        assert_hapt_write(FROZEN_MHEALTH / "joint_raw.csv")
    dest = assert_hapt_write(WRITE_ROOT / "contamination_probe.txt")
    dest.write_text("ok", encoding="utf-8")
    dest.unlink()


def test_caches_have_expected_shape_if_present():
    path = CACHE_DIR / "seed0.npz"
    if not path.exists():
        pytest.skip("caches not written")
    z = np.load(path)
    for key in ("p1_test", "p2_test", "fused_late_product", "fused_late_mean", "fused_cpsa"):
        assert z[key].shape == (3162, 12)
        assert np.isfinite(z[key]).all()
        assert float(np.max(np.abs(z[key].sum(1) - 1))) <= 1e-5
        assert not (z[key] < 0).any()
    z.close()
