"""Tests for leaf model fitting: ridge recovery and constant fallback."""

import numpy as np
import pytest

from repleafgbm.core.leaf_models import (
    ConstantLeafModel,
    EmbeddedLinearLeafModel,
    LeafValues,
    _fit_weighted_ridge,
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


def test_batched_fit_matches_reference_implementation():
    """The batched normal equations (Phase 11) must agree with the centered
    per-leaf reference (`_fit_weighted_ridge`) on every leaf: weights, bias,
    guard bounds, and fallback decisions."""
    rng = np.random.default_rng(8)
    n, d = 1000, 6
    Z = np.column_stack([
        rng.normal(size=(n, d - 2)),          # standardized-like block
        rng.uniform(0, 1, size=(n, 2)),       # PLR-like bounded block
    ])
    residual = Z @ rng.normal(size=d) + rng.normal(0, 0.3, n)
    g = -residual
    h = np.abs(rng.normal(1.0, 0.2, n)) + 0.1  # non-uniform Hessian weights

    # Mixed leaf sizes, including one below the linear threshold.
    bounds = [0, 300, 640, 652, 1000]
    leaf_rows = [np.arange(bounds[i], bounds[i + 1]) for i in range(4)]

    model = EmbeddedLinearLeafModel(l2=1.0, min_samples_linear=20)
    lv = model.fit_leaves(leaf_rows, g, h, Z)

    min_n = max(20, d + 2)
    for i, rows in enumerate(leaf_rows):
        if rows.shape[0] < min_n:
            assert np.all(lv.weights[i] == 0.0)
            assert np.isinf(lv.z_min[i]).all()
            continue
        b_ref, w_ref = _fit_weighted_ridge(Z[rows], g[rows], h[rows], 1.0)
        np.testing.assert_allclose(lv.weights[i], w_ref, rtol=1e-9, atol=1e-12)
        assert lv.bias[i] == pytest.approx(b_ref, rel=1e-9)
        np.testing.assert_array_equal(lv.z_min[i], Z[rows].min(axis=0))
        np.testing.assert_array_equal(lv.z_max[i], Z[rows].max(axis=0))


@pytest.mark.parametrize("d", [8, 33, 48, 64])
def test_native_and_numpy_stat_paths_agree(monkeypatch, d):
    """The Rust fused-stats path and the NumPy/BLAS path must produce the same
    leaf models (to float noise) across the embedding widths the native gate
    now covers (``_NATIVE_STATS_MAX_DIM`` == 64). With n=1500 rows, d=8/33 take
    the native serial branch while d=48/64 exceed ``LEAF_PARALLEL_MIN_CELLS``
    and exercise the rayon leaf-parallel branch; d>32 also guards the raised
    gate (native vs BLAS up to d=64)."""
    pytest.importorskip("repleafgbm_native", reason="Rust extension not built")
    import repleafgbm.core.leaf_models as lm

    assert d <= lm._NATIVE_STATS_MAX_DIM  # every case must reach the native path
    rng = np.random.default_rng(10)
    n = 1500
    Z = rng.normal(size=(n, d))
    residual = Z @ rng.normal(size=d) + rng.normal(0, 0.2, n)
    g, h = -residual, np.abs(rng.normal(1.0, 0.3, n)) + 0.1
    cuts = [0, 400, 900, 1500]
    leaf_rows = [np.arange(cuts[i], cuts[i + 1]) for i in range(3)]
    model = EmbeddedLinearLeafModel(l2=1.0, min_samples_linear=20)

    lv_native = model.fit_leaves(leaf_rows, g, h, Z)
    monkeypatch.setattr(lm, "_native", None)
    lv_numpy = model.fit_leaves(leaf_rows, g, h, Z)

    # Native (scalar/rayon Gram) and BLAS accumulate in different orders — the
    # native per-element three-way product (h*za*row[b]) vs BLAS's Zl.T @ hZ — so
    # the divergence grows with d; use the project's documented leaf-fit allclose
    # tolerance (as in tests/test_rust_backend.py) rather than the d=8-era 1e-9.
    np.testing.assert_allclose(lv_native.bias, lv_numpy.bias, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(lv_native.weights, lv_numpy.weights,
                               rtol=1e-6, atol=1e-8)
    np.testing.assert_array_equal(lv_native.z_min, lv_numpy.z_min)
    np.testing.assert_array_equal(lv_native.z_max, lv_numpy.z_max)


def test_batched_fit_singular_leaf_falls_back_individually():
    """One leaf with a constant Z block at l2=0 must not poison the batch:
    it falls back to a constant while other leaves keep their linear fits."""
    rng = np.random.default_rng(9)
    n, d = 400, 3
    Z = rng.normal(size=(n, d))
    Z[200:, :] = 1.0  # second leaf: constant embedding -> singular at l2=0
    residual = Z[:, 0] * 2.0 + rng.normal(0, 0.1, n)
    g, h = -residual, np.ones(n)
    model = EmbeddedLinearLeafModel(l2=0.0, min_samples_linear=5)
    lv = model.fit_leaves([np.arange(200), np.arange(200, 400)], g, h, Z)
    assert np.any(lv.weights[0] != 0.0)          # healthy leaf fitted
    assert np.all(lv.weights[1] == 0.0)          # singular leaf fell back
    assert np.isinf(lv.z_min[1]).all()
