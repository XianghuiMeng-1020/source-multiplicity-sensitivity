"""Frozen Candidate A feature map. Does not use _group_hapt_features."""

from __future__ import annotations

import json

from multiplicity_replay.replay import file_sha256

from .paths import FEATURE_MAP, LOCKED_FEATURE_MAP_HASH


def load_feature_map() -> dict:
    digest = file_sha256(FEATURE_MAP)
    if digest != LOCKED_FEATURE_MAP_HASH:
        raise RuntimeError(
            f"feature_map.json hash {digest} != locked preregistration {LOCKED_FEATURE_MAP_HASH}"
        )
    return json.loads(FEATURE_MAP.read_text(encoding="utf-8"))


def assert_partition(fmap: dict) -> None:
    m1 = list(fmap["M1_indices"])
    m2 = list(fmap["M2_indices"])
    ex = list(fmap["excluded_indices"])
    if len(m1) != 348 or len(m2) != 211 or len(ex) != 2:
        raise RuntimeError("feature counts are not 348/211/2")
    if len(m1) + len(m2) + len(ex) != 561:
        raise RuntimeError("348+211+2 != 561")
    if set(m1) & set(m2):
        raise RuntimeError("M1 and M2 indices overlap")
    if sorted(set(m1) | set(m2) | set(ex)) != list(range(561)):
        raise RuntimeError("union plus exclusions is not all 561 columns")
    if fmap["excluded_names"] != [
        "tBodyGyro-AngleWRTGravity-1",
        "tBodyGyroJerk-AngleWRTGravity-1",
    ]:
        raise RuntimeError("excluded names are not the Gate-4 pair")
