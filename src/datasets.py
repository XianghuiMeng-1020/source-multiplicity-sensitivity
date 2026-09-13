"""Load real multimodal HAR datasets and split them into sensor groups."""

from __future__ import annotations

import hashlib
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


@dataclass
class MultiModalSplit:
    name: str
    modalities: dict[str, np.ndarray]
    y: np.ndarray
    subject: np.ndarray | None
    extra: dict[str, np.ndarray]

    @property
    def n(self) -> int:
        return int(self.y.shape[0])


@dataclass
class MultiModalDataset:
    name: str
    modality_order: list[str]
    extra_names: list[str]
    train: MultiModalSplit
    test: MultiModalSplit
    n_classes: int
    notes: str


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_extract(zf: zipfile.ZipFile, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    zf.extractall(dest)
    return dest


def _load_txt(path: Path) -> np.ndarray:
    return np.loadtxt(path)


def _group_hapt_features(names: list[str]) -> dict[str, list[int]]:
    acc, gyro, other = [], [], []
    for i, raw in enumerate(names):
        n = raw.lower()
        if "gyro" in n:
            gyro.append(i)
        elif "acc" in n:
            acc.append(i)
        else:
            other.append(i)
    if not other:
        # Mag / angle leftovers sometimes share Acc in the name; peel gravity+angle.
        other = [i for i, n in enumerate(names) if "gravity" in n.lower() or n.lower().startswith("angle")]
        acc = [i for i in acc if i not in other]
    return {"acc": acc, "gyro": gyro, "other": other}


def load_hapt(zip_path: Path | None = None) -> MultiModalDataset:
    zip_path = zip_path or RAW / "hapt.zip"
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)
    extract = RAW / "hapt_extracted"
    if not (extract / "features.txt").exists() and not any(extract.rglob("features.txt")):
        with zipfile.ZipFile(zip_path) as zf:
            _safe_extract(zf, extract)
    feat_file = next(extract.rglob("features.txt"))
    root = feat_file.parent
    names: list[str] = []
    for line in feat_file.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        names.append(parts[-1] if parts[0].isdigit() else parts[0])
    groups = _group_hapt_features(names)

    def _xy(split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # HAPT uses Train/Test; some mirrors use train/test.
        cands = [
            root / split.capitalize() / f"X_{split}.txt",
            root / split / f"X_{split}.txt",
            root / f"X_{split}.txt",
        ]
        x_path = next(p for p in cands if p.exists())
        y_path = x_path.with_name(f"y_{split}.txt")
        subj_cands = [
            x_path.with_name(f"subject_id_{split}.txt"),
            x_path.with_name(f"subject_{split}.txt"),
        ]
        subj_path = next((p for p in subj_cands if p.exists()), None)
        X = _load_txt(x_path)
        y = _load_txt(y_path).astype(int).ravel()
        # labels are 1..12
        y = y - y.min()
        subj = _load_txt(subj_path).astype(int).ravel() if subj_path else np.zeros(len(y), dtype=int)
        return X, y, subj

    Xtr, ytr, str_ = _xy("train")
    Xte, yte, ste = _xy("test")

    def split_mods(X: np.ndarray, y: np.ndarray, subj: np.ndarray) -> MultiModalSplit:
        mods = {k: X[:, idx] for k, idx in groups.items() if idx}
        extra: dict[str, np.ndarray] = {}
        return MultiModalSplit("hapt", mods, y, subj, extra)

    notes = (
        f"zip_sha256={sha256(zip_path)} n_feat={len(names)} "
        f"acc={len(groups['acc'])} gyro={len(groups['gyro'])} other={len(groups['other'])}"
    )
    return MultiModalDataset(
        name="HAPT",
        modality_order=["acc", "gyro", "other"],
        extra_names=[],
        train=split_mods(Xtr, ytr, str_),
        test=split_mods(Xte, yte, ste),
        n_classes=int(max(ytr.max(), yte.max()) + 1),
        notes=notes,
    )


def _window_stats(x: np.ndarray) -> np.ndarray:
    # x: (T, C)
    return np.concatenate(
        [
            x.mean(axis=0),
            x.std(axis=0),
            x.min(axis=0),
            x.max(axis=0),
            np.median(x, axis=0),
        ]
    )


def load_mhealth(zip_path: Path | None = None, fs: int = 50, win_s: float = 2.0, hop_s: float = 1.0) -> MultiModalDataset:
    zip_path = zip_path or RAW / "mhealth.zip"
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)
    extract = RAW / "mhealth_extracted"
    logs = list(extract.rglob("mHealth_subject*.log"))
    if not logs:
        with zipfile.ZipFile(zip_path) as zf:
            _safe_extract(zf, extract)
        logs = list(extract.rglob("mHealth_subject*.log"))
    if not logs:
        raise FileNotFoundError("no mHealth_subject*.log after extract")

    win = int(fs * win_s)
    hop = int(fs * hop_s)
    # columns: chest acc 0-2, ECG 3-4, ankle acc 5-7, ankle gyro 8-10, ankle mag 11-13,
    # arm acc 14-16, arm gyro 17-19, arm mag 20-22, label 23
    groups = {
        "chest_acc": [0, 1, 2],
        "ankle": list(range(5, 14)),
        "arm": list(range(14, 23)),
    }
    extra_cols = {"ecg": [3, 4]}

    recs: list[tuple[dict[str, np.ndarray], int, int]] = []
    for log in sorted(logs):
        subj = int("".join(ch for ch in log.stem if ch.isdigit()) or 0)
        arr = np.loadtxt(log)
        if arr.ndim != 2 or arr.shape[1] < 24:
            continue
        y = arr[:, 23].astype(int)
        for start in range(0, len(arr) - win + 1, hop):
            sl = slice(start, start + win)
            lab = y[sl]
            # majority non-null label; skip null-only windows
            nonzero = lab[lab > 0]
            if nonzero.size < win * 0.8:
                continue
            label = int(np.bincount(nonzero).argmax())
            feats = {}
            for name, cols in groups.items():
                feats[name] = _window_stats(arr[sl][:, cols])
            extras = {name: _window_stats(arr[sl][:, cols]) for name, cols in extra_cols.items()}
            recs.append((feats | extras, label, subj))

    # subject-wise split: subjects 1-7 train, 8-10 test (standard-ish holdout)
    train_subj = {1, 2, 3, 4, 5, 6, 7}
    tr_idx = [i for i, r in enumerate(recs) if r[2] in train_subj]
    te_idx = [i for i, r in enumerate(recs) if r[2] not in train_subj]
    if not te_idx:
        n = len(recs)
        tr_idx = list(range(int(0.7 * n)))
        te_idx = list(range(int(0.7 * n), n))

    def pack(idxs: list[int]) -> MultiModalSplit:
        mods = {k: np.stack([recs[i][0][k] for i in idxs]) for k in groups}
        extra = {k: np.stack([recs[i][0][k] for i in idxs]) for k in extra_cols}
        y = np.array([recs[i][1] for i in idxs], dtype=int)
        y = y - 1  # 1..12 -> 0..11
        subj = np.array([recs[i][2] for i in idxs], dtype=int)
        return MultiModalSplit("mhealth", mods, y, subj, extra)

    train = pack(tr_idx)
    test = pack(te_idx)
    notes = f"zip_sha256={sha256(zip_path)} windows={len(recs)} win={win} hop={hop}"
    return MultiModalDataset(
        name="MHEALTH",
        modality_order=["chest_acc", "ankle", "arm"],
        extra_names=["ecg"],
        train=train,
        test=test,
        n_classes=12,
        notes=notes,
    )
