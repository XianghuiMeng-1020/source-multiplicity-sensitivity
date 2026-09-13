"""Small MLP encoders and fusion rules."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    def __init__(self, d_in: int, n_classes: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_mlp(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    epochs: int = 25,
    batch: int = 128,
    lr: float = 1e-3,
    seed: int = 0,
) -> MLP:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = _device()
    model = MLP(X.shape[1], n_classes).to(dev)
    ds = TensorDataset(torch.from_numpy(X.astype(np.float32)), torch.from_numpy(y.astype(np.int64)))
    loader = DataLoader(ds, batch_size=batch, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
    return model.cpu()


@torch.no_grad()
def predict_proba(model: MLP, X: np.ndarray, batch: int = 512) -> np.ndarray:
    model.eval()
    outs = []
    for i in range(0, len(X), batch):
        logits = model(torch.from_numpy(X[i : i + batch].astype(np.float32)))
        outs.append(torch.softmax(logits, dim=1).numpy())
    return np.concatenate(outs, axis=0)


def late_mean(parts: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    if weights is None:
        weights = [1.0] * len(parts)
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    acc = np.zeros_like(parts[0])
    for p, wi in zip(parts, w):
        acc += wi * p
    return acc


def late_product(parts: list[np.ndarray], weights: list[float] | None = None, eps: float = 1e-12) -> np.ndarray:
    if weights is None:
        weights = [1.0] * len(parts)
    logp = np.zeros_like(parts[0])
    for p, w in zip(parts, weights):
        logp += w * np.log(np.clip(p, eps, 1.0))
    logp -= logp.max(axis=1, keepdims=True)
    out = np.exp(logp)
    return out / out.sum(axis=1, keepdims=True)


def fit_temperature(probs: np.ndarray, y: np.ndarray) -> float:
    """Fit a scalar T on logits recovered from probs (holdout)."""
    eps = 1e-12
    p = np.clip(probs, eps, 1.0)
    logits = np.log(p)
    T = torch.tensor(1.0, requires_grad=True)
    y_t = torch.from_numpy(y.astype(np.int64))
    z = torch.from_numpy(logits.astype(np.float32))
    opt = torch.optim.LBFGS([T], lr=0.25, max_iter=50)

    def closure():
        opt.zero_grad()
        t = torch.clamp(T, min=0.05)
        loss = nn.functional.cross_entropy(z / t, y_t)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.clamp(T.detach(), min=0.05).item())


def apply_temperature(probs: np.ndarray, T: float, eps: float = 1e-12) -> np.ndarray:
    logits = np.log(np.clip(probs, eps, 1.0))
    z = logits / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)
