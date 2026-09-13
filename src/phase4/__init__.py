"""Phase-4 joint factorial package. Official MHEALTH execution is locked."""

from .config import PROTOCOL_VERSION, WRITE_ROOT, assert_write_allowed
from .contextual_psa import ContextualPSA, count_trainable_parameters, fuse_cpsa, train_cpsa

__all__ = [
    "ContextualPSA",
    "PROTOCOL_VERSION",
    "WRITE_ROOT",
    "assert_write_allowed",
    "count_trainable_parameters",
    "fuse_cpsa",
    "train_cpsa",
]
