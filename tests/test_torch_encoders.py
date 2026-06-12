"""Tests for the learned (PyTorch) encoders.

Skipped when torch is not installed — except the import-guard test, which
must work everywhere.
"""

import sys

import numpy as np
import pytest

from repleafgbm import RepLeafRegressor
from repleafgbm.encoders import TorchPeriodicEncoder, TorchPLREncoder, make_encoder


def _linear_probe_mse(Z: np.ndarray, target: np.ndarray, l2: float = 1e-3) -> float:
    """Ridge-fit a linear readout on Z; return its MSE (encoder quality probe)."""
    Zc = Z - Z.mean(axis=0)
    tc = target - target.mean()
    w = np.linalg.solve(Zc.T @ Zc + l2 * np.eye(Z.shape[1]), Zc.T @ tc)
    return float(np.mean((Zc @ w - tc) ** 2))


def test_missing_torch_message(monkeypatch):
    """Without torch the fit-time error must say how to install it."""
    monkeypatch.setitem(sys.modules, "torch", None)
    X = np.random.default_rng(0).normal(size=(50, 2))
    with pytest.raises(ImportError, match="pip install"):
        TorchPeriodicEncoder().fit(X, y=X[:, 0])
    # Unsupervised fit (no pretraining) works without torch at all.
    enc = TorchPeriodicEncoder().fit(X)
    assert enc.transform(X).shape == (50, 2 * 5)


torch = pytest.importorskip("torch", reason="torch not installed")


@pytest.fixture
def periodic_problem():
    """Target with a frequency far from the random init's typical draw —
    learned frequencies must adapt to explain it."""
    rng = np.random.default_rng(0)
    X = rng.uniform(-2.0, 2.0, size=(2000, 2))
    y = np.sin(3.0 * X[:, 0]) + rng.normal(0.0, 0.05, 2000)
    return X, y


def test_learned_frequencies_beat_frozen_init(periodic_problem):
    X, y = periodic_problem
    frozen = make_encoder("periodic", n_frequencies=4, random_state=0).fit(X)
    learned = TorchPeriodicEncoder(n_frequencies=4, n_epochs=40, random_state=0)
    learned.fit(X, y=y - y.mean())

    mse_frozen = _linear_probe_mse(frozen.transform(X), y)
    mse_learned = _linear_probe_mse(learned.transform(X), y)
    assert mse_learned < 0.5 * mse_frozen  # training must matter, a lot

    # And the parameters actually moved from the shared initialization.
    assert not np.allclose(learned.frequencies_, frozen.frequencies_)


def test_torch_plr_learns(periodic_problem):
    X, y = periodic_problem
    enc = TorchPLREncoder(n_bins=4, n_outputs=4, n_epochs=40, random_state=0)
    enc.fit(X, y=y - y.mean())
    untrained = TorchPLREncoder(n_bins=4, n_outputs=4, random_state=0).fit(X)
    assert _linear_probe_mse(enc.transform(X), y) < _linear_probe_mse(
        untrained.transform(X), y
    )
    assert enc.output_dim == 2 * 4


def test_determinism(periodic_problem):
    X, y = periodic_problem
    e1 = TorchPeriodicEncoder(n_epochs=5, random_state=7).fit(X, y=y)
    e2 = TorchPeriodicEncoder(n_epochs=5, random_state=7).fit(X, y=y)
    np.testing.assert_array_equal(e1.frequencies_, e2.frequencies_)
    np.testing.assert_array_equal(e1.phases_, e2.phases_)


def test_transform_is_numpy_only(periodic_problem, monkeypatch):
    """After fit, transform/state must not touch torch (saved models predict
    without it)."""
    X, y = periodic_problem
    enc = TorchPeriodicEncoder(n_epochs=3, random_state=0).fit(X, y=y)
    state, config = enc.get_state(), enc.get_config()
    assert all(isinstance(v, np.ndarray) for v in state.values())

    monkeypatch.setitem(sys.modules, "torch", None)  # torch now "missing"
    fresh = make_encoder("torch_periodic", **config)
    fresh.set_state(state)
    np.testing.assert_allclose(fresh.transform(X), enc.transform(X))


def test_end_to_end_and_roundtrip(tmp_path, periodic_problem):
    X, y = periodic_problem
    model = RepLeafRegressor(
        n_estimators=20, num_leaves=8, min_samples_leaf=10,
        leaf_model="embedded_linear", encoder="torch_periodic",
        encoder_params={"n_epochs": 10}, random_state=42,
    )
    model.fit(X, y)
    pred = model.predict(X)
    baseline = float(np.std(y))
    assert float(np.sqrt(np.mean((pred - y) ** 2))) < 0.5 * baseline

    model.save_model(tmp_path / "m")
    loaded = RepLeafRegressor.load_model(tmp_path / "m")
    np.testing.assert_allclose(loaded.predict(X), pred)


def test_torch_plr_state_roundtrip(periodic_problem):
    X, y = periodic_problem
    enc = TorchPLREncoder(n_bins=4, n_outputs=3, n_epochs=3, random_state=1)
    enc.fit(X, y=y)
    fresh = make_encoder(enc.name, **enc.get_config())
    fresh.set_state(enc.get_state())
    np.testing.assert_allclose(enc.transform(X), fresh.transform(X))


def test_pretraining_early_stopping_on_noise(periodic_problem):
    """A pure-noise target must trigger validation early stopping well before
    the epoch budget; a strong-signal target should train longer."""
    X, y = periodic_problem
    rng = np.random.default_rng(3)
    noise = rng.normal(size=len(y))

    enc_noise = TorchPeriodicEncoder(n_epochs=30, patience=3, random_state=0)
    enc_noise.fit(X, y=noise)
    assert enc_noise.pretrain_epochs_used_ < 30

    enc_signal = TorchPeriodicEncoder(n_epochs=30, patience=3, random_state=0)
    enc_signal.fit(X, y=y)
    assert enc_signal.pretrain_epochs_used_ >= enc_noise.pretrain_epochs_used_


def test_regularization_params_serialized(periodic_problem):
    X, y = periodic_problem
    enc = TorchPeriodicEncoder(n_epochs=3, weight_decay=0.5, val_fraction=0.2,
                               patience=2, random_state=0).fit(X, y=y)
    cfg = enc.get_config()
    assert cfg["weight_decay"] == 0.5 and cfg["patience"] == 2
    fresh = make_encoder(enc.name, **cfg)
    fresh.set_state(enc.get_state())
    np.testing.assert_allclose(enc.transform(X), fresh.transform(X))
