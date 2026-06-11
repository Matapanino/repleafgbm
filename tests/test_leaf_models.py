"""Tests for leaf model fitting: ridge recovery and constant fallback."""

import numpy as np

from repleafgbm.core.leaf_models import (
    ConstantLeafModel,
    EmbeddedLinearLeafModel,
    LeafValues,
)


def test_constant_leaf_is_newton_step():
    rng = np.random.default_rng(0)
    residual = rng.normal(2.0, 0.1, 100)
    # Squared error: g = -residual, h = 1.
    g, h = -residual, np.ones(100)
    lv = ConstantLeafModel(l2=0.0).fit_leaves([np.arange(100)], g, h, None)
    assert abs(lv.bias[0] - residual.mean()) < 1e-9
    assert lv.weights.shape == (1, 0)


def test_embedded_linear_recovers_linear_target():
    rng = np.random.default_rng(1)
    Z = rng.normal(size=(200, 3))
    residual = 1.0 + 2.0 * Z[:, 0] - 0.5 * Z[:, 2]
    g, h = -residual, np.ones(200)
    model = EmbeddedLinearLeafModel(l2=1e-6, min_samples_linear=5)
    lv = model.fit_leaves([np.arange(200)], g, h, Z)
    np.testing.assert_allclose(lv.bias[0], 1.0, atol=1e-4)
    np.testing.assert_allclose(lv.weights[0], [2.0, 0.0, -0.5], atol=1e-4)

    pred = lv.predict(np.zeros(200, dtype=np.int64), Z)
    np.testing.assert_allclose(pred, residual, atol=1e-4)


def test_small_leaf_falls_back_to_constant():
    rng = np.random.default_rng(2)
    Z = rng.normal(size=(50, 8))
    g, h = -rng.normal(size=50), np.ones(50)
    model = EmbeddedLinearLeafModel(l2=1.0, min_samples_linear=10)
    tiny_leaf = np.arange(6)  # < emb_dim + 2 and < min_samples_linear
    lv = model.fit_leaves([tiny_leaf], g, h, Z)
    assert np.all(lv.weights[0] == 0.0)  # constant fallback: no linear part


def test_mixed_leaves_fallback_per_leaf():
    rng = np.random.default_rng(3)
    Z = rng.normal(size=(120, 2))
    residual = 3.0 * Z[:, 0]
    g, h = -residual, np.ones(120)
    model = EmbeddedLinearLeafModel(l2=1e-6, min_samples_linear=10)
    big, small = np.arange(100), np.arange(100, 120)[:3]
    lv = model.fit_leaves([big, small], g, h, Z)
    assert np.any(lv.weights[0] != 0.0)  # linear fit
    assert np.all(lv.weights[1] == 0.0)  # fallback


def test_weighted_fit_uses_hessian_weights():
    # Two clusters with different Hessian weight; the heavier cluster should
    # dominate the constant fit.
    g = np.array([-1.0, -1.0, -5.0])
    h = np.array([1.0, 1.0, 100.0])
    lv = ConstantLeafModel(l2=0.0).fit_leaves([np.arange(3)], g, h, None)
    # Newton step: sum(-g)/sum(h) = 7/102
    np.testing.assert_allclose(lv.bias[0], 7.0 / 102.0)


def test_leaf_values_predict_indexing():
    lv = LeafValues(bias=np.array([1.0, 2.0]), weights=np.array([[1.0], [0.0]]))
    Z = np.array([[10.0], [10.0]])
    pred = lv.predict(np.array([0, 1]), Z)
    np.testing.assert_allclose(pred, [11.0, 2.0])
