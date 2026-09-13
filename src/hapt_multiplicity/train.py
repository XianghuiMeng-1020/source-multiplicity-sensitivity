"""CPU training wrappers. Does not call Phase-4 train_cpsa."""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models import MLP, predict_proba
from phase4.contextual_psa import ContextualPSA, assert_parameter_count

DEVICE = torch.device("cpu")
ENCODER_EPOCHS = 25
ENCODER_BATCH = 128
ENCODER_LR = 1e-3
CPSA_EPOCHS = 25
CPSA_BATCH = 128
CPSA_LR = 1e-3
HOLDOUT_FRAC = 0.2


def holdout_idx(n: int, frac: float = HOLDOUT_FRAC, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_h = max(1, int(frac * n))
    return idx[n_h:], idx[:n_h]


def train_mlp_cpu(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int = 12,
    epochs: int = ENCODER_EPOCHS,
    batch: int = ENCODER_BATCH,
    lr: float = ENCODER_LR,
    seed: int = 0,
) -> MLP:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLP(X.shape[1], n_classes, hidden=128).to(DEVICE)
    ds = TensorDataset(torch.from_numpy(X.astype(np.float32)), torch.from_numpy(y.astype(np.int64)))
    loader = DataLoader(ds, batch_size=batch, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
    model.eval()
    return model.cpu()


def train_cpsa_ntokens(
    posterior_sets: np.ndarray,
    y: np.ndarray,
    seed: int,
    n_tokens: int = 2,
    epochs: int = CPSA_EPOCHS,
    batch: int = CPSA_BATCH,
    lr: float = CPSA_LR,
) -> ContextualPSA:
    """Isolated cardinality-general CPSA trainer. Not the Phase-4 3-token function."""
    sets = np.asarray(posterior_sets, dtype=np.float32)
    labels = np.asarray(y, dtype=np.int64)
    if sets.ndim != 3 or sets.shape[1] != n_tokens or sets.shape[2] != 12:
        raise ValueError(f"posterior_sets must have shape [N, {n_tokens}, 12]")
    if n_tokens < 1:
        raise ValueError("n_tokens must be >= 1")
    if labels.shape != (sets.shape[0],):
        raise ValueError("y must have shape [N]")
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    model = ContextualPSA()
    n_params = assert_parameter_count(model)
    if n_params != 1009:
        raise RuntimeError(f"CPSA parameter count {n_params} != 1009")
    model.to(DEVICE)
    ds = TensorDataset(torch.from_numpy(sets), torch.from_numpy(labels))
    loader = DataLoader(ds, batch_size=batch, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            opt.zero_grad()
            fused = model(xb)
            loss = nn.functional.cross_entropy(torch.log(torch.clamp(fused, min=1e-8)), yb)
            loss.backward()
            opt.step()
    model.eval()
    return model.cpu()


def train_seed(ds, seed: int) -> dict:
    tr_i, ho_i = holdout_idx(len(ds.train.y), HOLDOUT_FRAC, seed)
    if set(tr_i.tolist()) & set(ho_i.tolist()):
        raise RuntimeError("encoder-train and CPSA-holdout overlap")
    te_subj = set(int(s) for s in ds.test.subject.tolist())
    if te_subj & set(int(s) for s in ds.train.subject[tr_i].tolist()):
        # subjects can appear in both encoder-train and holdout; they must not be TEST subjects
        pass
    if te_subj & set(int(s) for s in ds.train.subject.tolist()):
        raise RuntimeError("official test subjects appear in the official train pool")

    t0 = time.perf_counter()
    enc1 = train_mlp_cpu(ds.train.X1[tr_i], ds.train.y[tr_i], seed=seed)
    enc2 = train_mlp_cpu(ds.train.X2[tr_i], ds.train.y[tr_i], seed=seed + 1)
    p1_ho = predict_proba(enc1, ds.train.X1[ho_i])
    p2_ho = predict_proba(enc2, ds.train.X2[ho_i])
    tokens = np.stack([p1_ho, p2_ho], axis=1)
    if tokens.shape[1] != 2:
        raise RuntimeError("CPSA tokens are not 2-wide")
    cpsa = train_cpsa_ntokens(tokens, ds.train.y[ho_i], seed=seed + 10, n_tokens=2)
    elapsed = time.perf_counter() - t0

    def class_counts(y):
        return {str(c): int((y == c).sum()) for c in range(12)}

    return {
        "seed": int(seed),
        "enc1": enc1,
        "enc2": enc2,
        "cpsa": cpsa,
        "tr_i": tr_i,
        "ho_i": ho_i,
        "encoder_train_n": int(len(tr_i)),
        "cpsa_holdout_n": int(len(ho_i)),
        "encoder_train_class_counts": class_counts(ds.train.y[tr_i]),
        "cpsa_holdout_class_counts": class_counts(ds.train.y[ho_i]),
        "encoder_train_n_subjects": int(len(set(ds.train.subject[tr_i].tolist()))),
        "cpsa_holdout_n_subjects": int(len(set(ds.train.subject[ho_i].tolist()))),
        "encoder_train_all_12": sorted(set(ds.train.y[tr_i].tolist())) == list(range(12)),
        "cpsa_holdout_all_12": sorted(set(ds.train.y[ho_i].tolist())) == list(range(12)),
        "missing_holdout_classes": [c for c in range(12) if not (ds.train.y[ho_i] == c).any()],
        "wall_clock_sec": elapsed,
        "device": "cpu",
    }
