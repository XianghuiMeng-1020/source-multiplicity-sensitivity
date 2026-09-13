"""Official HAPT Train/Test loader for Candidate A. No extra scaling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import assert_partition, load_feature_map
from .paths import HAPT_ROOT

EXCLUDED = (
    "tBodyGyro-AngleWRTGravity-1",
    "tBodyGyroJerk-AngleWRTGravity-1",
)


@dataclass
class HaptSplit:
    X1: np.ndarray
    X2: np.ndarray
    y: np.ndarray
    subject: np.ndarray


@dataclass
class HaptTwoSource:
    train: HaptSplit
    test: HaptSplit
    feature_map: dict


def _load_split(split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cap = split.capitalize()
    X = np.loadtxt(HAPT_ROOT / cap / f"X_{split}.txt")
    y_raw = np.loadtxt(HAPT_ROOT / cap / f"y_{split}.txt").astype(int).ravel()
    subj = np.loadtxt(HAPT_ROOT / cap / f"subject_id_{split}.txt").astype(int).ravel()
    if y_raw.min() != 1 or y_raw.max() != 12:
        raise RuntimeError(f"{split} official labels are not 1..12")
    y = y_raw - 1
    if X.shape[0] != y.shape[0] or X.shape[0] != subj.shape[0]:
        raise RuntimeError(f"{split} row alignment failed")
    if X.shape[1] != 561:
        raise RuntimeError(f"{split} does not have 561 columns")
    return X, y, subj


def load_hapt_two_source() -> HaptTwoSource:
    fmap = load_feature_map()
    assert_partition(fmap)
    m1 = np.asarray(fmap["M1_indices"], dtype=int)
    m2 = np.asarray(fmap["M2_indices"], dtype=int)
    names = list(fmap["M1_names"]) + list(fmap["M2_names"]) + list(fmap["excluded_names"])
    if any(n in fmap["M1_names"] or n in fmap["M2_names"] for n in EXCLUDED):
        raise RuntimeError("excluded cross-sensor names leaked into M1/M2")

    Xtr, ytr, str_ = _load_split("train")
    Xte, yte, ste = _load_split("test")
    train = HaptSplit(Xtr[:, m1], Xtr[:, m2], ytr, str_)
    test = HaptSplit(Xte[:, m1], Xte[:, m2], yte, ste)

    if train.X1.shape != (7767, 348) or train.X2.shape != (7767, 211):
        raise RuntimeError(f"unexpected train shapes {train.X1.shape} {train.X2.shape}")
    if test.X1.shape != (3162, 348) or test.X2.shape != (3162, 211):
        raise RuntimeError(f"unexpected test shapes {test.X1.shape} {test.X2.shape}")
    tr_s = sorted(set(int(s) for s in train.subject.tolist()))
    te_s = sorted(set(int(s) for s in test.subject.tolist()))
    if len(tr_s) != 21 or len(te_s) != 9:
        raise RuntimeError(f"subject counts {len(tr_s)}/{len(te_s)}")
    if set(tr_s) & set(te_s):
        raise RuntimeError(f"subject overlap {set(tr_s) & set(te_s)}")
    if sorted(set(train.y.tolist())) != list(range(12)):
        raise RuntimeError("train missing a class")
    if sorted(set(test.y.tolist())) != list(range(12)):
        raise RuntimeError("test missing a class")
    if not (np.array_equal(train.y, ytr) and np.array_equal(test.y, yte)):
        raise RuntimeError("label alignment broken")
    _ = names
    return HaptTwoSource(train=train, test=test, feature_map=fmap)


def integrity_record(ds: HaptTwoSource) -> dict:
    def counts(y, subj):
        return {
            "n": int(len(y)),
            "n_subjects": int(len(set(subj.tolist()))),
            "subjects": sorted(int(s) for s in set(subj.tolist())),
            "per_subject": {str(int(s)): int((subj == s).sum()) for s in sorted(set(subj.tolist()))},
            "per_class": {str(int(c)): int((y == c).sum()) for c in range(12)},
        }

    return {
        "train": counts(ds.train.y, ds.train.subject),
        "test": counts(ds.test.y, ds.test.subject),
        "subject_overlap": [],
        "m1_shape_train": list(ds.train.X1.shape),
        "m2_shape_train": list(ds.train.X2.shape),
        "m1_shape_test": list(ds.test.X1.shape),
        "m2_shape_test": list(ds.test.X2.shape),
        "dataset_provided_feature_scaling": True,
        "additional_fitted_scaling": False,
    }
