"""Write-guarded HAPT experiment paths."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HAPT_ROOT = (REPO_ROOT / "data" / "raw" / "hapt_extracted").resolve()
HAPT_ZIP = (REPO_ROOT / "data" / "raw" / "hapt.zip").resolve()
WRITE_ROOT = (REPO_ROOT / "research" / "hapt_multiplicity").resolve()
RESULTS = WRITE_ROOT / "results"
CKPT_DIR = RESULTS / "checkpoints"
CACHE_DIR = WRITE_ROOT / "caches"
PROTOCOL_JSON = WRITE_ROOT / "protocol.json"
FEATURE_MAP = WRITE_ROOT / "feature_map.json"
LOCKED_PROTOCOL_HASH = "ff5ebd78ba53283abf04de177db1178476e01fe4cebec3dbe2eddb85df36f6c9"
LOCKED_FEATURE_MAP_HASH = "f1deccb0aa31346c40a9483f15c502b07dd1957099a909a0357e864e2818a4c2"
FROZEN_MHEALTH = (REPO_ROOT / "research" / "phase4_9of10" / "results").resolve()
REPLAY_G2 = (REPO_ROOT / "research" / "multiplicity_replay").resolve()
SWEEP_G3 = (REPO_ROOT / "research" / "multiplicity_sweep").resolve()
PAPER = (REPO_ROOT / "paper").resolve()


def assert_hapt_write(path: Path) -> Path:
    dest = Path(path).resolve()
    if FROZEN_MHEALTH == dest or FROZEN_MHEALTH in dest.parents:
        raise PermissionError(f"HAPT write refused in MHEALTH freeze: {dest}")
    if REPLAY_G2 == dest or REPLAY_G2 in dest.parents:
        raise PermissionError(f"HAPT write refused in Gate-2 replay: {dest}")
    if SWEEP_G3 == dest or SWEEP_G3 in dest.parents:
        raise PermissionError(f"HAPT write refused in Gate-3 sweep: {dest}")
    if PAPER == dest or PAPER in dest.parents:
        raise PermissionError(f"HAPT write refused in paper/: {dest}")
    raw = (REPO_ROOT / "data" / "raw").resolve()
    if raw == dest or raw in dest.parents:
        raise PermissionError(f"HAPT write refused in original data: {dest}")
    if WRITE_ROOT != dest and WRITE_ROOT not in dest.parents:
        raise PermissionError(f"HAPT write refused outside {WRITE_ROOT}: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    return dest
