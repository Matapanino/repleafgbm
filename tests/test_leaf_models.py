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


def test_multiclass_pooled_matches_per_class(monkeypatch):
    """Session 4: fitting all K class trees' leaves in one pooled native pass
    (``leaf_linear_stats_mc``) must match independent per-class ``fit_leaves``
    *bitwise* — it is a scheduling change only (each pooled leaf still sums its
    own rows in order, reading its class's grad/hess column, and each leaf's
    ridge system is solved independently). Exercises uneven per-class leaf counts
    and a sub-threshold constant-fallback leaf; also checks the NumPy fallback."""
    pytest.importorskip("repleafgbm_native", reason="Rust extension not built")
    import repleafgbm.core.leaf_models as lm

    rng = np.random.default_rng(11)
    n, K, d = 1500, 4, 16
    Z = np.ascontiguousarray(rng.normal(size=(n, d)))
    grad = np.ascontiguousarray(rng.normal(size=(n, K)))
    hess = np.ascontiguousarray(np.abs(rng.normal(1.0, 0.3, (n, K))) + 0.1)

    def leaves(seed, nl):
        parts = np.array_split(np.random.default_rng(seed).permutation(n), nl)
        return [np.sort(p).astype(np.int64) for p in parts]

    # Uneven leaf counts per class; class 3 also gets a tiny sub-threshold leaf.
    rows_per_class = [leaves(20 + k, nl) for k, nl in enumerate([5, 7, 4, 6])]
    rows_per_class[3].append(np.sort(rng.choice(n, 4, replace=False)).astype(np.int64))

    model = EmbeddedLinearLeafModel(l2=1.0, min_samples_linear=20)
    pooled = model.fit_leaves_multiclass(rows_per_class, grad, hess, Z)
    per_class = [
        model.fit_leaves(rows_per_class[k], grad[:, k], hess[:, k], Z)
        for k in range(K)
    ]
    assert len(pooled) == K
    for k in range(K):
        np.testing.assert_array_equal(pooled[k].bias, per_class[k].bias)
        np.testing.assert_array_equal(pooled[k].weights, per_class[k].weights)
        np.testing.assert_array_equal(pooled[k].z_min, per_class[k].z_min)
        np.testing.assert_array_equal(pooled[k].z_max, per_class[k].z_max)

    # Without the native helper, fit_leaves_multiclass routes through per-class
    # BLAS fits, which stay allclose to the native pooled result.
    monkeypatch.setattr(lm, "_native", None)
    fallback = model.fit_leaves_multiclass(rows_per_class, grad, hess, Z)
    for k in range(K):
        np.testing.assert_allclose(fallback[k].bias, pooled[k].bias, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(
            fallback[k].weights, pooled[k].weights, rtol=1e-6, atol=1e-8
        )


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


@pytest.mark.parametrize("clip", [False, True])
@pytest.mark.parametrize("n", [2000, 6000])  # n*d below/above the parallel gate
def test_predict_linear_native_matches_numpy(monkeypatch, clip, n):
    """Session 4: the fused native ``predict_linear`` must match the NumPy
    bias-gather + einsum to float noise, for both the training-eval path
    (clip=False) and the clipped prediction path (clip=True), across the serial
    (small n*d) and rayon (large n*d) branches. Rows are independent, so the two
    native branches are bitwise-identical; both are checked against NumPy here."""
    pytest.importorskip("repleafgbm_native", reason="Rust extension not built")
    import repleafgbm.core.leaf_models as lm

    rng = np.random.default_rng(12)
    L, d = 10, 24
    Z = rng.normal(size=(n, d))
    z_min = rng.normal(size=(L, d)) - 0.5
    z_max = z_min + 1.0 + rng.random((L, d))
    z_min[0] = -np.inf  # constant-fallback leaf: clip must be the identity
    z_max[0] = np.inf
    lv = LeafValues(
        bias=rng.normal(size=L),
        weights=rng.normal(size=(L, d)),
        z_min=z_min,
        z_max=z_max,
    )
    leaf_idx = rng.integers(0, L, n).astype(np.int64)

    assert lm._native is not None and hasattr(lm._native, "predict_linear")
    out_native = lv.predict(leaf_idx, Z, clip=clip)
    monkeypatch.setattr(lm, "_native", None)
    out_numpy = lv.predict(leaf_idx, Z, clip=clip)
    np.testing.assert_allclose(out_native, out_numpy, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("clip", [False, True])
def test_predict_linear_serial_parallel_bitwise(clip):
    """The native ``predict_linear`` serial and rayon branches must be
    *bitwise*-identical: rows are computed independently, so the result must not
    depend on the branch (size gate) or thread count. Computing one large array
    (n*d above the gate -> parallel) and the same rows in small slices (each
    below the gate -> serial) must give exactly equal outputs."""
    pytest.importorskip("repleafgbm_native", reason="Rust extension not built")
    rng = np.random.default_rng(13)
    L, d = 8, 24
    n = 8000  # n*d = 192000 > PREDICT_PARALLEL_MIN -> parallel branch
    Z = rng.normal(size=(n, d))
    z_min = rng.normal(size=(L, d)) - 0.5
    z_max = z_min + 1.0 + rng.random((L, d))
    z_min[0] = -np.inf
    z_max[0] = np.inf
    lv = LeafValues(
        bias=rng.normal(size=L),
        weights=rng.normal(size=(L, d)),
        z_min=z_min,
        z_max=z_max,
    )
    leaf_idx = rng.integers(0, L, n).astype(np.int64)

    full = lv.predict(leaf_idx, Z, clip=clip)  # parallel branch
    step = 1000  # 1000*24 = 24000 < gate -> each slice takes the serial branch
    serial = np.concatenate(
        [lv.predict(leaf_idx[i:i + step], Z[i:i + step], clip=clip)
         for i in range(0, n, step)]
    )
    np.testing.assert_array_equal(full, serial)
