"""Synthetic semantic tests for Phase-4. Does not load official MHEALTH."""

from __future__ import annotations

import hashlib
import inspect
import itertools
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_SRC = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import pytest
import torch

from models import late_product
from src.phase4.config import (
    EXPECTED_CPSA_PARAMS,
    HISTORICAL_RESULT_DIRS,
    LOCK_MESSAGE,
    N_CLASSES,
    PROTOCOL_ABORT_MESSAGE,
    PROTOCOL_VERSION,
    REPO_ROOT,
    WRITE_ROOT,
    assert_write_allowed,
)
from src.phase4.contextual_psa import (
    ContextualPSA,
    assert_parameter_count,
    count_trainable_parameters,
    fuse_cpsa,
    train_cpsa,
)
from src.phase4.run_joint import (
    ARM_N_TOKENS,
    ARMS,
    RAW_FIELDS,
    cpsa_holdout_sets,
    encoder_slice,
    evaluate_cpsa_arm,
    evaluate_late_product_arm,
    holdout_idx,
    main as run_joint_main,
    make_result_row,
    make_trainref_stuck,
    stack_tokens,
    standardize_pair,
    token_list,
)

def _simplex(seed: int, n: int = 3) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        v = rng.random(N_CLASSES)
        v = v / v.sum()
        out.append(v.astype(np.float32))
    return out


def _batch_tokens(parts: list[np.ndarray]) -> np.ndarray:
    return np.stack([np.stack(parts, axis=0)], axis=0)


def _iter_integrity_files() -> list[Path]:
    files: list[Path] = []
    for d in HISTORICAL_RESULT_DIRS:
        if d.exists():
            files.extend(sorted(p for p in d.rglob("*.csv") if p.is_file()))
    for extra in (REPO_ROOT / "paper" / "main.tex", REPO_ROOT / "paper" / "refs.bib"):
        if extra.exists():
            files.append(extra)
    return files


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _snapshot() -> dict[str, str]:
    return {str(p): _file_sha256(p) for p in _iter_integrity_files()}


@pytest.fixture(scope="session", autouse=True)
def historical_integrity():
    before = _snapshot()
    yield before
    after = _snapshot()
    changed = [p for p, h in before.items() if after.get(p) != h]
    extra = [p for p in after if p not in before]
    assert changed == []
    assert extra == []


def test_h1_parameter_count_exactly_1009():
    model = ContextualPSA()
    assert count_trainable_parameters(model) == EXPECTED_CPSA_PARAMS
    assert assert_parameter_count(model) == 1009
    n_phi = sum(p.numel() for p in model.phi.parameters())
    n_g = sum(p.numel() for p in model.g.parameters())
    assert n_phi == 208
    assert n_g == 801
    assert n_phi + n_g == 1009


def test_h2_k1_permutation_invariance_all_six():
    torch.manual_seed(0)
    model = ContextualPSA().eval()
    p1, p2, p3 = _simplex(1)
    base = None
    for perm in itertools.permutations([p1, p2, p3]):
        fused = fuse_cpsa(model, np.stack(perm, axis=0))
        if base is None:
            base = fused
        else:
            assert np.max(np.abs(fused - base)) <= 1e-5


def test_h2_k5_permutation_invariance_sample():
    torch.manual_seed(1)
    model = ContextualPSA().eval()
    p1, p2, p3 = _simplex(2)
    tokens = [p1, p1, p1, p1, p1, p2, p3]
    identity = fuse_cpsa(model, np.stack(tokens, axis=0))
    reversed_t = fuse_cpsa(model, np.stack(list(reversed(tokens)), axis=0))
    assert np.max(np.abs(identity - reversed_t)) <= 1e-5
    rng = np.random.default_rng(0)
    for _ in range(20):
        order = rng.permutation(7)
        perm = [tokens[i] for i in order]
        fused = fuse_cpsa(model, np.stack(perm, axis=0))
        assert np.max(np.abs(fused - identity)) <= 1e-5


def test_h3_contextual_capacity_dummy_tokens():
    torch.manual_seed(3)
    model = ContextualPSA().eval()
    p1, p2, p3 = _simplex(4)
    s3 = torch.from_numpy(_batch_tokens([p1, p2, p3]))
    s7 = torch.from_numpy(_batch_tokens([p1, p1, p1, p1, p1, p2, p3]))
    h3, c3, d3, _ = model.score_inputs(s3)
    h7, c7, d7, _ = model.score_inputs(s7)
    assert torch.max(torch.abs(c3 - c7)).item() > 1e-6
    assert torch.max(torch.abs(d3[:, 0] - d7[:, 0])).item() > 1e-6
    assert torch.max(torch.abs(h3[:, 0] - h7[:, 0])).item() < 1e-6


def test_h4_score_input_is_48d_not_token_independent():
    model = ContextualPSA()
    assert model.g[0].in_features == 48
    assert model.phi[0].in_features == 12
    p1, p2, p3 = _simplex(5)
    tokens = torch.from_numpy(_batch_tokens([p1, p2, p3]))
    captured = {}

    def hook(_mod, inputs, _out):
        captured["x"] = inputs[0]

    handle = model.g.register_forward_hook(hook)
    model.eval()
    with torch.no_grad():
        model(tokens)
    handle.remove()
    assert captured["x"].shape[-1] == 48
    src = inspect.getsource(ContextualPSA.score_inputs)
    assert "torch.cat" in src
    assert "dist" in src
    assert "mean" in src


def test_h5_train_holdout_separation_mock_ids():
    n = 20
    ids = np.arange(n)
    tr_i, ho_i = holdout_idx(n, frac=0.2, seed=0)
    assert len(set(tr_i) & set(ho_i)) == 0
    assert set(tr_i).isdisjoint(set(ho_i))
    assert set(tr_i).union(set(ho_i)) == set(ids)
    X = ids.reshape(-1, 1).astype(np.float32)
    y = ids.copy()
    X_enc, y_enc = encoder_slice(X, y, tr_i)
    assert set(y_enc.tolist()) == set(tr_i.tolist())
    assert set(X_enc.ravel().astype(int).tolist()) == set(tr_i.tolist())
    p1 = np.repeat((ids[:, None] + 1) / 100.0, 12, axis=1).astype(np.float32)
    p1 = p1 / p1.sum(axis=1, keepdims=True)
    sets = cpsa_holdout_sets(p1, p1, p1, ho_i)
    assert sets.shape == (len(ho_i), 3, 12)
    sig = inspect.signature(train_cpsa)
    assert set(sig.parameters) == {"posterior_sets", "y", "seed", "epochs", "batch", "lr"}
    src = inspect.getsource(train_cpsa)
    assert "load_mhealth" not in src
    assert "tile" not in src
    assert "hard_stuck" not in src
    with pytest.raises(ValueError):
        train_cpsa(np.zeros((4, 5, 12), dtype=np.float32), np.zeros(4, dtype=np.int64), seed=0, epochs=1)


def test_h5_k9_standardize_ignores_test_statistics():
    rng = np.random.default_rng(0)
    tr = rng.normal(size=(8, 4)).astype(np.float32)
    te_a = rng.normal(size=(5, 4)).astype(np.float32)
    te_b = te_a + 50.0
    _, _, mu_a, sd_a = standardize_pair(tr, te_a)
    _, _, mu_b, sd_b = standardize_pair(tr, te_b)
    assert np.allclose(mu_a, mu_b)
    assert np.allclose(sd_a, sd_b)


def test_h6_train_derived_stuck_ignores_any_test_matrix():
    X_train = np.arange(30, dtype=np.float32).reshape(6, 5)
    bad = make_trainref_stuck(X_train, n_rows=4)
    assert bad.shape == (4, 5)
    assert np.allclose(bad, np.repeat(X_train[0][None, :], 4, axis=0))
    sig = inspect.signature(make_trainref_stuck)
    assert set(sig.parameters) == {"X1_train_std", "n_rows"}
    src = inspect.getsource(make_trainref_stuck)
    assert "X1_test" not in src
    assert "X_test" not in src
    a = make_trainref_stuck(X_train, 3)
    b = make_trainref_stuck(X_train, 3)
    assert np.array_equal(a, b)
    for seed in (0, 1, 2):
        _ = seed
        assert np.allclose(make_trainref_stuck(X_train, 5), make_trainref_stuck(X_train, 5))


def test_h7_late_product_list_vs_weight_identity():
    p1, p2, p3 = [np.stack([v, v], axis=0) for v in _simplex(6)]
    left = late_product([p1] * 5 + [p2, p3])
    right = late_product([p1, p2, p3], weights=[5, 1, 1])
    assert np.max(np.abs(left - right)) < 1e-6


def test_h8_arm_cardinalities():
    p1, p2, p3 = [np.stack([v], axis=0) for v in _simplex(7)]
    p1_bad = p1[:, ::-1].copy()
    p1_bad = p1_bad / p1_bad.sum(axis=1, keepdims=True)
    for arm in ARMS:
        parts = token_list(arm, p1, p2, p3, p1_bad)
        assert len(parts) == ARM_N_TOKENS[arm]
        stacked = stack_tokens(parts)
        assert stacked.shape[1] == ARM_N_TOKENS[arm]
    e = token_list("E", p1, p2, p3, p1_bad)
    assert len(e) == 2
    assert np.array_equal(e[0], p2)
    assert np.array_equal(e[1], p3)


def test_h9_same_checkpoint_no_optimizer_on_eval():
    p1, p2, p3 = [np.repeat(v[None, :], 5, axis=0) for v in _simplex(8)]
    p1_bad = np.roll(p1, 1, axis=1)
    p1_bad = p1_bad / p1_bad.sum(axis=1, keepdims=True)
    sets = np.stack([p1[:4], p2[:4], p3[:4]], axis=1)
    y = np.array([0, 1, 2, 3], dtype=np.int64)
    model = train_cpsa(sets, y, seed=11, epochs=2)
    snapshot = [p.detach().clone() for p in model.parameters()]
    assert model.training is False
    for arm in ARMS:
        evaluate_cpsa_arm(model, arm, p1, p2, p3, p1_bad)
        evaluate_late_product_arm(arm, p1, p2, p3, p1_bad)
    assert model.training is False
    for a, b in zip(snapshot, model.parameters()):
        assert torch.equal(a, b)
    for fn in (evaluate_cpsa_arm, evaluate_late_product_arm, fuse_cpsa):
        src = inspect.getsource(fn)
        assert "Adam" not in src
        assert "train_mlp" not in src
        assert "train_cpsa" not in src
        assert "fit_temperature" not in src
        assert "train_gate" not in src
        assert "optim" not in src


def test_h10_two_token_legality_not_quality():
    model = ContextualPSA().eval()
    _p1, p2, p3 = _simplex(9)
    fused = fuse_cpsa(model, np.stack([p2, p3], axis=0))
    assert fused.shape == (12,)
    assert np.all(fused >= 0)
    assert abs(float(fused.sum()) - 1.0) < 1e-5


def test_h11_no_flags_input_dim_12():
    model = ContextualPSA()
    assert model.phi[0].in_features == 12
    srcs = [
        inspect.getsource(ContextualPSA),
        inspect.getsource(train_cpsa),
        Path(__file__).resolve().with_name("contextual_psa.py").read_text(encoding="utf-8"),
        Path(__file__).resolve().with_name("run_joint.py").read_text(encoding="utf-8"),
    ]
    joined = "\n".join(srcs)
    assert "from attention" not in joined
    assert "import attention" not in joined
    assert "train_gate" not in joined
    assert "hard_stuck(" not in joined


def test_h12_write_guard_allows_and_rejects():
    ok1 = REPO_ROOT / "research" / "phase4_9of10" / "results" / "foo.csv"
    ok2 = REPO_ROOT / "research" / "phase4_9of10" / "results" / "checkpoints" / "x.pt"
    assert assert_write_allowed(ok1) == ok1.resolve()
    assert assert_write_allowed(ok2) == ok2.resolve()
    rejected = [
        REPO_ROOT / "research" / "phase2" / "results" / "x.csv",
        REPO_ROOT / "research" / "results" / "x.csv",
        REPO_ROOT / "paper" / "main.tex",
        REPO_ROOT / "src" / "models.py",
        REPO_ROOT / "research" / "phase4_9of10" / "results" / ".." / ".." / "phase2" / "results" / "x.csv",
    ]
    for path in rejected:
        with pytest.raises(PermissionError):
            assert_write_allowed(path)


def test_h13_default_cli_locked(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("load_mhealth must not be called")

    monkeypatch.setattr("src.phase4.run_joint._load_mhealth", boom)
    assert run_joint_main([]) == 2
    assert run_joint_main(["--protocol", PROTOCOL_VERSION]) == 2
    assert run_joint_main(["--execute-official"]) == 2
    assert run_joint_main(["--execute-official", "--protocol", "wrong"]) == 2


def test_h13_subprocess_default_does_not_train():
    proc = subprocess.run(
        [sys.executable, "-m", "src.phase4.run_joint"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "locked" in (proc.stdout + proc.stderr).lower()
    assert LOCK_MESSAGE.splitlines()[0] in proc.stdout
    proc2 = subprocess.run(
        [sys.executable, "-m", "src.phase4.run_joint", "--execute-official", "--protocol", "nope"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc2.returncode != 0
    assert PROTOCOL_ABORT_MESSAGE in proc2.stdout


def test_h14_cpsa_determinism_cpu_two_epochs():
    sets = np.stack([_simplex(10) for _ in range(6)], axis=0).astype(np.float32)
    sets = sets / sets.sum(axis=2, keepdims=True)
    y = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    a = train_cpsa(sets, y, seed=4, epochs=2)
    b = train_cpsa(sets, y, seed=4, epochs=2)
    pa = fuse_cpsa(a, sets)
    pb = fuse_cpsa(b, sets)
    assert np.max(np.abs(pa - pb)) <= 1e-6


def test_h15_historical_csv_unchanged(historical_integrity):
    now = _snapshot()
    changed = [p for p, h in historical_integrity.items() if now.get(p) != h]
    assert changed == []
    assert len(now) == len(historical_integrity)


def test_k13_output_schema_dummy_row():
    p = np.tile(np.array(_simplex(11)[0])[None, :], (3, 1))
    y = np.array([0, 1, 2], dtype=np.int64)
    row = make_result_row("A", 0, "cpsa", p, y)
    for key in (
        "arm",
        "seed",
        "rule",
        "k",
        "reliability",
        "accuracy",
        "confidence",
        "ece",
        "nll",
        "brier",
        "entropy",
    ):
        assert key in row
    assert list(row) == RAW_FIELDS


def test_k16_train_cpsa_cannot_receive_encoders():
    dummy = torch.nn.Linear(4, 4)
    before = [p.detach().clone() for p in dummy.parameters()]
    sets = np.stack([_simplex(12) for _ in range(4)], axis=0).astype(np.float32)
    sets = sets / sets.sum(axis=2, keepdims=True)
    y = np.array([0, 1, 2, 3], dtype=np.int64)
    train_cpsa(sets, y, seed=0, epochs=1)
    for a, b in zip(before, dummy.parameters()):
        assert torch.equal(a, b)
    assert "encoder" not in inspect.signature(train_cpsa).parameters


def test_k18_stuck_does_not_take_labels():
    assert "y" not in inspect.signature(make_trainref_stuck).parameters
    y = np.array([9, 8, 7])
    X = np.ones((3, 2), dtype=np.float32)
    X[0] = [3.0, 4.0]
    _ = make_trainref_stuck(X, 2)
    assert np.array_equal(y, np.array([9, 8, 7]))


def test_no_official_result_files_exist():
    assert not (WRITE_ROOT / "joint_raw.csv").exists()
    assert not (WRITE_ROOT / "joint_summary.csv").exists()
    ckpt = WRITE_ROOT / "checkpoints"
    if ckpt.exists():
        official = list(ckpt.glob("seed*_enc*.pt")) + list(ckpt.glob("seed*_cpsa.pt"))
        assert official == []


def test_k17_stuck_uses_train_row_zero_not_a_test_row():
    X_train = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    bad = make_trainref_stuck(X_train, 3)
    assert np.allclose(bad[0], [1.0, 2.0])
    assert not np.allclose(bad[0], [9.0, 9.0])
