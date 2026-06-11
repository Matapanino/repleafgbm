"""Tests for the leaf-linear extrapolation guard (Phase 7).

Beyond a leaf's training support, embedded-linear leaves must extrapolate as
constants (clipped Z), preventing the blow-ups observed on real data
(experiments/results/real_data_validation.md).
"""

import numpy as np
import pytest

from repleafgbm import RepLeafRegressor
from repleafgbm.core.leaf_models import EmbeddedLinearLeafModel, LeafValues


def test_leaf_extrapolates_as_constant_beyond_support():
    rng = np.random.default_rng(0)
    Z = rng.uniform(-1.0, 1.0, size=(200, 2))
    residual = 5.0 * Z[:, 0]  # steep slope: dangerous to extrapolate
    g, h = -residual, np.ones(200)
    model = EmbeddedLinearLeafModel(l2=1e-6, min_samples_linear=5)
    lv = model.fit_leaves([np.arange(200)], g, h, Z)

    inside = lv.predict(np.zeros(1, dtype=np.int64), np.array([[1.0, 0.0]]))
    far_out = lv.predict(np.zeros(1, dtype=np.int64), np.array([[100.0, 0.0]]))
    boundary = lv.predict(np.zeros(1, dtype=np.int64), np.array([[Z[:, 0].max(), 0.0]]))
    # Outside the support the prediction saturates at the boundary value
    # instead of following the slope to ~500.
    np.testing.assert_allclose(far_out, boundary)
    assert far_out[0] < 6.0
    assert inside[0] == pytest.approx(5.0, abs=0.1)


def test_constant_fallback_leaves_have_inactive_bounds():
    rng = np.random.default_rng(1)
    Z = rng.normal(size=(30, 4))
    g, h = -rng.normal(size=30), np.ones(30)
    model = EmbeddedLinearLeafModel(l2=1.0, min_samples_linear=10)
    lv = model.fit_leaves([np.arange(4)], g, h, Z)  # tiny leaf -> constant
    assert np.isinf(lv.z_min[0]).all() and np.isinf(lv.z_max[0]).all()


def test_predictions_bounded_on_outlier_rows():
    """End-to-end reproduction of the Phase 6 failure shape: heavy-tailed
    feature at test time must not blow up predictions."""
    rng = np.random.default_rng(2)
    n = 1500
    X = rng.uniform(0.0, 1.0, size=(n, 3))
    y = 3.0 * X[:, 0] + X[:, 1] + rng.normal(0.0, 0.05, n)
    model = RepLeafRegressor(
        n_estimators=80, learning_rate=0.1, num_leaves=16, min_samples_leaf=20,
        leaf_model="embedded_linear", encoder="identity", random_state=42,
    )
    model.fit(X, y)

    X_out = X.copy()[:100]
    X_out[:, 0] = rng.uniform(50.0, 100.0, 100)  # far outside training range
    pred = model.predict(X_out)
    # Bounded by (roughly) the training target range, not the linear trend.
    assert pred.max() < y.max() + 1.0
    assert np.isfinite(pred).all()


def test_guard_survives_save_load(tmp_path):
    rng = np.random.default_rng(3)
    X = rng.uniform(0.0, 1.0, size=(600, 3))
    y = 4.0 * X[:, 0] + rng.normal(0.0, 0.05, 600)
    model = RepLeafRegressor(
        n_estimators=20, num_leaves=8, min_samples_leaf=10,
        leaf_model="embedded_linear", random_state=42,
    )
    model.fit(X, y)
    X_out = np.column_stack([np.full(50, 30.0), rng.uniform(0, 1, (50, 2))])
    pred = model.predict(X_out)

    model.save_model(tmp_path / "m")
    loaded = RepLeafRegressor.load_model(tmp_path / "m")
    np.testing.assert_allclose(loaded.predict(X_out), pred)
    assert loaded.booster_.leaf_values_[0].z_min is not None


def test_pre_guard_models_load_unclipped(tmp_path):
    """Old leaf_params.npz without zmin/zmax keys load with clipping off."""
    rng = np.random.default_rng(4)
    X = rng.uniform(0.0, 1.0, size=(600, 3))
    y = 4.0 * X[:, 0] + rng.normal(0.0, 0.05, 600)
    model = RepLeafRegressor(
        n_estimators=10, num_leaves=8, min_samples_leaf=10,
        leaf_model="embedded_linear", random_state=42,
    )
    model.fit(X, y)
    model.save_model(tmp_path / "m")

    # Strip the guard arrays, emulating a pre-Phase-7 model directory.
    npz_path = tmp_path / "m" / "leaf_params.npz"
    with np.load(npz_path) as data:
        kept = {k: data[k] for k in data.files if "zmin" not in k and "zmax" not in k}
    np.savez(npz_path, **kept)

    loaded = RepLeafRegressor.load_model(tmp_path / "m")
    assert loaded.booster_.leaf_values_[0].z_min is None
    X_out = np.column_stack([np.full(5, 30.0), rng.uniform(0, 1, (5, 2))])
    # Unclipped: follows the linear trend beyond the data (old behavior),
    # exceeding anything the clipped model would produce.
    assert loaded.predict(X_out).max() > model.predict(X_out).max()


def test_training_rows_unaffected_by_guard():
    """Clipping is a no-op on rows inside each leaf's support, so in-sample
    predictions equal the explicitly unclipped computation."""
    rng = np.random.default_rng(5)
    Z = rng.normal(size=(300, 3))
    residual = Z @ [1.0, -2.0, 0.5]
    g, h = -residual, np.ones(300)
    model = EmbeddedLinearLeafModel(l2=1e-6, min_samples_linear=5)
    rows = [np.arange(150), np.arange(150, 300)]
    lv = model.fit_leaves(rows, g, h, Z)
    leaf_idx = np.repeat([0, 1], 150)

    unclipped = LeafValues(bias=lv.bias, weights=lv.weights)
    np.testing.assert_allclose(
        lv.predict(leaf_idx, Z), unclipped.predict(leaf_idx, Z)
    )
