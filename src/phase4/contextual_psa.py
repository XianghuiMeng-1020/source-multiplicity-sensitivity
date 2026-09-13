"""Frozen contextual posterior-set attention (CPSA). Evaluation instrument only."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .config import (
    CPSA_BATCH,
    CPSA_EPOCHS,
    CPSA_LR,
    EXPECTED_CPSA_PARAMS,
    G_HIDDEN,
    MODALITY_ID,
    N_CLASSES,
    PHI_H,
)


class ContextualPSA(nn.Module):
    """Shared phi + set-mean context + g([h, c, |h-c|]).

    tokens: [batch, n_tokens, 12], n_tokens >= 1.
    """

    def __init__(
        self,
        n_classes: int = N_CLASSES,
        phi_h: int = PHI_H,
        g_hidden: int = G_HIDDEN,
    ) -> None:
        super().__init__()
        if MODALITY_ID:
            raise RuntimeError("MODALITY_ID is frozen False")
        if n_classes != N_CLASSES or phi_h != PHI_H or g_hidden != G_HIDDEN:
            raise ValueError("CPSA widths are frozen at C=12, PHI_H=16, G_HIDDEN=16")
        self.phi = nn.Sequential(nn.Linear(n_classes, phi_h), nn.ReLU())
        self.g = nn.Sequential(
            nn.Linear(3 * phi_h, g_hidden),
            nn.ReLU(),
            nn.Linear(g_hidden, 1),
        )

    def score_inputs(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return h, c, |h-c|, and the 48-d score input."""
        if tokens.ndim != 3 or tokens.size(-1) != N_CLASSES:
            raise ValueError(f"tokens must be [batch, n_tokens, {N_CLASSES}], got {tuple(tokens.shape)}")
        if tokens.size(1) < 1:
            raise ValueError("n_tokens must be >= 1")
        h = self.phi(tokens)
        c = h.mean(dim=1)
        c_exp = c.unsqueeze(1).expand_as(h)
        dist = (h - c_exp).abs()
        score_in = torch.cat([h, c_exp, dist], dim=-1)
        if score_in.size(-1) != 3 * PHI_H:
            raise RuntimeError("score input is not 48-d; contextual terms missing")
        return h, c, dist, score_in

    def forward(
        self,
        tokens: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        h, c, dist, score_in = self.score_inputs(tokens)
        logits = self.g(score_in).squeeze(-1)
        attention = torch.softmax(logits, dim=-1)
        fused = (attention.unsqueeze(-1) * tokens).sum(dim=1)
        if return_diagnostics:
            return fused, {
                "attention": attention,
                "context": c,
                "distance_to_context": dist,
                "score_input": score_in,
            }
        return fused


def count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def assert_parameter_count(model: ContextualPSA | None = None) -> int:
    model = model if model is not None else ContextualPSA()
    n = count_trainable_parameters(model)
    if n != EXPECTED_CPSA_PARAMS:
        raise RuntimeError(f"CPSA parameter count is {n}, expected {EXPECTED_CPSA_PARAMS}")
    return n


@torch.no_grad()
def fuse_cpsa(model: ContextualPSA, tokens: np.ndarray) -> np.ndarray:
    """Evaluate frozen CPSA. tokens: [n_tokens, 12] or [batch, n_tokens, 12]."""
    model.eval()
    t = torch.from_numpy(np.asarray(tokens, dtype=np.float32))
    if t.ndim == 2:
        t = t.unsqueeze(0)
        squeeze = True
    elif t.ndim == 3:
        squeeze = False
    else:
        raise ValueError("tokens must be 2-d or 3-d")
    fused = model(t)
    out = fused.cpu().numpy()
    return out[0] if squeeze else out


def train_cpsa(
    posterior_sets: np.ndarray,
    y: np.ndarray,
    seed: int,
    epochs: int | None = None,
    batch: int | None = None,
    lr: float | None = None,
) -> ContextualPSA:
    """Fit CPSA on clean unique 3-token posterior sets only.

    posterior_sets: [N, 3, 12]
    y: [N]
    No encoders, no raw features, no test path, no augmentation.
    """
    sets = np.asarray(posterior_sets, dtype=np.float32)
    labels = np.asarray(y, dtype=np.int64)
    if sets.ndim != 3 or sets.shape[1] != 3 or sets.shape[2] != N_CLASSES:
        raise ValueError("posterior_sets must have shape [N, 3, 12]")
    if labels.shape != (sets.shape[0],):
        raise ValueError("y must have shape [N]")
    n_epochs = CPSA_EPOCHS if epochs is None else int(epochs)
    n_batch = CPSA_BATCH if batch is None else int(batch)
    n_lr = CPSA_LR if lr is None else float(lr)

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    model = ContextualPSA()
    assert_parameter_count(model)
    device = torch.device("cpu")
    model.to(device)
    ds = TensorDataset(torch.from_numpy(sets), torch.from_numpy(labels))
    loader = DataLoader(ds, batch_size=n_batch, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=n_lr)
    model.train()
    for _ in range(n_epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            fused = model(xb)
            loss = nn.functional.cross_entropy(torch.log(torch.clamp(fused, min=1e-8)), yb)
            loss.backward()
            opt.step()
    model.eval()
    return model.cpu()
