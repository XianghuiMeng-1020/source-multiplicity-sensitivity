"""Hash audit of immutable Phase-4 artifacts. Read-only on the freeze tree."""

from __future__ import annotations

from pathlib import Path

from .replay import FROZEN_RESULTS, REPO_ROOT, file_sha256

FREEZE_LIST = REPO_ROOT / "research" / "phase4_9of10" / "results" / "RESULT_FREEZE.sha256"
WATCHED_EXTRA = (
    REPO_ROOT / "paper" / "main.tex",
    REPO_ROOT / "paper" / "main.pdf",
    FROZEN_RESULTS / "joint_raw.csv",
    FROZEN_RESULTS / "joint_summary.csv",
)


def parse_freeze_list() -> list[tuple[str, Path]]:
    rows = []
    for line in FREEZE_LIST.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        digest, rel = line.split(None, 1)
        rows.append((digest, (REPO_ROOT / rel).resolve()))
    return rows


def audit_freeze() -> dict:
    entries = []
    mismatches = []
    for expected, path in parse_freeze_list():
        if not path.exists():
            rec = {
                "path": str(path.relative_to(REPO_ROOT)),
                "expected": expected,
                "actual": None,
                "status": "MISSING",
            }
            mismatches.append(rec)
            entries.append(rec)
            continue
        actual = file_sha256(path)
        status = "OK" if actual == expected else "CHANGED"
        rec = {
            "path": str(path.relative_to(REPO_ROOT)),
            "expected": expected,
            "actual": actual,
            "status": status,
        }
        if status != "OK":
            mismatches.append(rec)
        entries.append(rec)
    extra = {}
    for path in WATCHED_EXTRA:
        rel = str(path.relative_to(REPO_ROOT))
        extra[rel] = file_sha256(path) if path.exists() else "ABSENT"
    ckpts = {}
    for path in sorted((FROZEN_RESULTS / "checkpoints").glob("seed*_*.pt")):
        ckpts[str(path.relative_to(REPO_ROOT))] = file_sha256(path)
    return {
        "freeze_entries": entries,
        "mismatches": mismatches,
        "watched_extra": extra,
        "checkpoints": ckpts,
        "ok": not mismatches,
    }
