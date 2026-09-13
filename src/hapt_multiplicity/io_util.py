"""JSON/checkpoint IO under the HAPT write root."""

from __future__ import annotations

import json

import torch

from models import MLP
from multiplicity_replay.replay import file_sha256
from phase4.contextual_psa import ContextualPSA

from .paths import PROTOCOL_JSON, assert_hapt_write
from .train import DEVICE

LOCKED_PROTOCOL = "ff5ebd78ba53283abf04de177db1178476e01fe4cebec3dbe2eddb85df36f6c9"


def assert_protocol_frozen() -> None:
    digest = file_sha256(PROTOCOL_JSON)
    if digest != LOCKED_PROTOCOL:
        raise RuntimeError(
            f"protocol.json changed after preregistration: {digest} != {LOCKED_PROTOCOL}"
        )


def write_json(path, payload) -> None:
    dest = assert_hapt_write(path)
    dest.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")


def save_state(path, model) -> None:
    dest = assert_hapt_write(path)
    torch.save(model.state_dict(), dest)


def load_state(path):
    try:
        return torch.load(path, map_location=DEVICE, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=DEVICE)


def load_encoder(path, d_in: int) -> MLP:
    model = MLP(d_in, 12, hidden=128)
    model.load_state_dict(load_state(path))
    model.to(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_cpsa(path) -> ContextualPSA:
    model = ContextualPSA()
    model.load_state_dict(load_state(path))
    model.to(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
