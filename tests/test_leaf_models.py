"""Tests for leaf model fitting: ridge recovery and constant fallback."""

import json

import numpy as np
import pytest

from repleafgbm import RepLeafRegressor
from repleafgbm.core.leaf_models import (
    AdaptiveLeafModel,
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


# --------------------------------------------------------------------------- #
# AdaptiveLeafModel: per-leaf weighted-LOO constant<->embedded_linear gate
# --------------------------------------------------------------------------- #
def test_adaptive_keeps_linear_on_clean_target():
    """A noiseless linear leaf generalizes perfectly, so the gate keeps the
    linear fit and produces exactly the embedded-linear weights."""
    rng = np.random.default_rng(20)
    Z = rng.normal(size=(300, 4))
    residual = 1.0 + 2.0 * Z[:, 0] - 0.5 * Z[:, 2]  # noiseless linear in Z
    g, h = -residual, np.ones(300)
    rows = [np.arange(300)]
    lv_a = AdaptiveLeafModel(
        l2=1e-6, min_samples_linear=10, leaf_gate_margin=0.01
    ).fit_leaves(rows, g, h, Z)
    lv_e = EmbeddedLinearLeafModel(l2=1e-6, min_samples_linear=10).fit_leaves(
        rows, g, h, Z
    )
    assert np.any(lv_a.weights[0] != 0.0)  # linear fit kept
    np.testing.assert_allclose(lv_a.weights[0], lv_e.weights[0], rtol=1e-9, atol=1e-12)


def test_adaptive_demotes_noise_to_constant():
    """A pure-noise leaf above the size threshold: the linear fit does not beat
    the constant in weighted LOO, so the gate demotes it to a constant whose
    bias is the Newton value -g_sum/(h_sum+l2)."""
    rng = np.random.default_rng(21)
    Z = rng.normal(size=(120, 8))
    g, h = -rng.normal(size=120), np.ones(120)  # no linear structure in Z
    lv = AdaptiveLeafModel(
        l2=1.0, min_samples_linear=10, leaf_gate_margin=0.01
    ).fit_leaves([np.arange(120)], g, h, Z)
    assert np.all(lv.weights[0] == 0.0)
    assert np.isinf(lv.z_min[0]).all() and np.isinf(lv.z_max[0]).all()
    assert lv.bias[0] == pytest.approx(-g.sum() / (h.sum() + 1.0))


def test_adaptive_demoted_leaf_matches_embedded_constant(monkeypatch):
    """A gated-to-constant leaf must equal the embedded-linear constant fallback
    bitwise (same Newton bias, zero weights, infinite guard). Forcing the NumPy
    path (both compute the bias by the same bincount) makes it exact; a large
    margin (>=1) demotes every leaf."""
    import repleafgbm.core.leaf_models as lm

    monkeypatch.setattr(lm, "_native", None)
    rng = np.random.default_rng(22)
    Z = rng.normal(size=(120, 8))
    g, h = -rng.normal(size=120), np.abs(rng.normal(1.0, 0.2, 120)) + 0.1
    rows = [np.arange(120)]
    # margin >= 1 => E_lin < (1-margin)*E_const < 0 is never true => always demote.
    lv_a = AdaptiveLeafModel(
        l2=1.0, min_samples_linear=10, leaf_gate_margin=5.0
    ).fit_leaves(rows, g, h, Z)
    # An unreachable min_samples_linear forces the embedded model constant.
    lv_c = EmbeddedLinearLeafModel(l2=1.0, min_samples_linear=10_000).fit_leaves(
        rows, g, h, Z
    )
    np.testing.assert_array_equal(lv_a.bias, lv_c.bias)
    np.testing.assert_array_equal(lv_a.weights, lv_c.weights)
    np.testing.assert_array_equal(lv_a.z_min, lv_c.z_min)
    np.testing.assert_array_equal(lv_a.z_max, lv_c.z_max)


def test_adaptive_margin_brackets():
    """``leaf_gate_margin`` brackets the verdict: 0 keeps a clearly-helpful
    linear leaf; a margin >= 1 forces constant unconditionally."""
    rng = np.random.default_rng(23)
    Z = rng.normal(size=(300, 3))
    residual = 2.0 * Z[:, 0] - Z[:, 1] + 0.3 * rng.normal(size=300)  # mostly linear
    g, h = -residual, np.ones(300)
    rows = [np.arange(300)]
    keep = AdaptiveLeafModel(
        l2=1e-3, min_samples_linear=10, leaf_gate_margin=0.0
    ).fit_leaves(rows, g, h, Z)
    drop = AdaptiveLeafModel(
        l2=1e-3, min_samples_linear=10, leaf_gate_margin=10.0
    ).fit_leaves(rows, g, h, Z)
    assert np.any(keep.weights[0] != 0.0)
    assert np.all(drop.weights[0] == 0.0)


def test_adaptive_loo_demotes_what_insample_keeps_on_noise():
    """The leverage correction is what earns its keep: on a pure-noise leaf the
    linear fit lowers in-sample error (overfit) but not held-out error, so the
    LOO gate demotes it while the in-sample baseline keeps it."""
    rng = np.random.default_rng(24)
    Z = rng.normal(size=(60, 10))
    g, h = -rng.normal(size=60), np.ones(60)
    rows = [np.arange(60)]
    loo = AdaptiveLeafModel(
        l2=1e-3, min_samples_linear=12, leaf_gate="loo", leaf_gate_margin=0.0
    ).fit_leaves(rows, g, h, Z)
    insample = AdaptiveLeafModel(
        l2=1e-3, min_samples_linear=12, leaf_gate="insample", leaf_gate_margin=0.0
    ).fit_leaves(rows, g, h, Z)
    assert np.all(loo.weights[0] == 0.0)  # LOO: noise -> constant
    assert np.any(insample.weights[0] != 0.0)  # in-sample overfit -> linear


def test_adaptive_mixed_verdicts_in_one_tree():
    """One tree, one clean-linear leaf and one pure-noise leaf: the gate keeps
    the first and demotes the second."""
    rng = np.random.default_rng(25)
    Z = rng.normal(size=(400, 6))
    clean, noise = np.arange(200), np.arange(200, 400)
    residual = np.empty(400)
    residual[clean] = 3.0 * Z[clean, 0] - 1.5 * Z[clean, 2]
    residual[noise] = rng.normal(size=200)
    g, h = -residual, np.ones(400)
    lv = AdaptiveLeafModel(
        l2=1e-3, min_samples_linear=20, leaf_gate_margin=0.01
    ).fit_leaves([clean, noise], g, h, Z)
    assert np.any(lv.weights[0] != 0.0)
    assert np.all(lv.weights[1] == 0.0)


def test_adaptive_gate_multioutput():
    """The multi-output vector-leaf path gates one verdict per leaf (summed over
    outputs): a clean-linear leaf is kept, a noise leaf demoted."""
    from repleafgbm.core.multioutput import fit_vector_leaves

    rng = np.random.default_rng(29)
    Z = rng.normal(size=(400, 5))
    clean, noise = np.arange(200), np.arange(200, 400)
    signal = np.column_stack([2.0 * Z[:, 0] - Z[:, 3], Z[:, 1] + 0.5 * Z[:, 4]])
    grad = np.empty((400, 2))
    grad[clean] = -signal[clean]
    grad[noise] = -rng.normal(size=(200, 2))
    hess = np.ones((400, 2))
    adaptive = AdaptiveLeafModel(l2=1e-3, min_samples_linear=20, leaf_gate_margin=0.01)
    lv = fit_vector_leaves(adaptive, [clean, noise], grad, hess, Z, l2=1e-3)
    assert np.any(lv.weights[0] != 0.0)  # clean leaf kept linear (shape (emb, K))
    assert np.all(lv.weights[1] == 0.0)  # noise leaf demoted


def test_adaptive_gate_multioutput_loo_vs_insample():
    """The vector-leaf gate's leverage correction matters too: on a pure-noise
    multi-output leaf the LOO gate demotes while the in-sample baseline keeps."""
    from repleafgbm.core.multioutput import fit_vector_leaves

    rng = np.random.default_rng(40)
    Z = rng.normal(size=(60, 10))
    grad = -rng.normal(size=(60, 2))  # noise, no Z structure
    hess = np.ones((60, 2))
    rows = [np.arange(60)]
    loo = AdaptiveLeafModel(
        l2=1e-3, min_samples_linear=12, leaf_gate="loo", leaf_gate_margin=0.0
    )
    insample = AdaptiveLeafModel(
        l2=1e-3, min_samples_linear=12, leaf_gate="insample", leaf_gate_margin=0.0
    )
    lv_loo = fit_vector_leaves(loo, rows, grad, hess, Z, l2=1e-3)
    lv_ins = fit_vector_leaves(insample, rows, grad, hess, Z, l2=1e-3)
    assert np.all(lv_loo.weights[0] == 0.0)  # LOO: noise -> constant
    assert np.any(lv_ins.weights[0] != 0.0)  # in-sample overfit -> linear


def test_adaptive_is_deterministic():
    """Same inputs -> identical leaf parameters (the gate adds no randomness)."""
    rng = np.random.default_rng(28)
    Z = rng.normal(size=(300, 5))
    residual = Z @ rng.normal(size=5) + 0.5 * rng.normal(size=300)
    g, h = -residual, np.abs(rng.normal(1.0, 0.2, 300)) + 0.1
    rows = [np.arange(150), np.arange(150, 300)]
    model = AdaptiveLeafModel(l2=1.0, min_samples_linear=20, leaf_gate_margin=0.01)
    a = model.fit_leaves(rows, g, h, Z)
    b = model.fit_leaves(rows, g, h, Z)
    np.testing.assert_array_equal(a.weights, b.weights)
    np.testing.assert_array_equal(a.bias, b.bias)


def test_adaptive_gate_stable_across_backends(monkeypatch):
    """Away from the margin boundary, the native and NumPy paths must agree on
    the *set* of linear leaves (the stats are allclose, the verdict identical).
    Uses well-separated clean-linear / noise / clean-linear leaves."""
    pytest.importorskip("repleafgbm_native", reason="Rust extension not built")
    import repleafgbm.core.leaf_models as lm

    rng = np.random.default_rng(27)
    n, d = 1500, 16
    Z = rng.normal(size=(n, d))
    beta = rng.normal(size=d)
    residual = np.empty(n)
    residual[:500] = Z[:500] @ beta + 0.01 * rng.normal(size=500)
    residual[500:1000] = rng.normal(size=500)
    residual[1000:] = Z[1000:] @ beta + 0.01 * rng.normal(size=500)
    g, h = -residual, np.abs(rng.normal(1.0, 0.2, n)) + 0.1
    leaf_rows = [np.arange(0, 500), np.arange(500, 1000), np.arange(1000, 1500)]
    model = AdaptiveLeafModel(l2=1.0, min_samples_linear=20, leaf_gate_margin=0.05)

    lv_native = model.fit_leaves(leaf_rows, g, h, Z)
    monkeypatch.setattr(lm, "_native", None)
    lv_numpy = model.fit_leaves(leaf_rows, g, h, Z)

    mask_native = np.abs(lv_native.weights).sum(axis=1) > 0
    mask_numpy = np.abs(lv_numpy.weights).sum(axis=1) > 0
    np.testing.assert_array_equal(mask_native, mask_numpy)
    np.testing.assert_array_equal(mask_native, [True, False, True])


def test_adaptive_save_load_round_trip(tmp_path):
    """A fitted ``leaf_model="adaptive"`` model with mixed verdicts round-trips:
    predictions are exact and the gate params reload (no serialization bump)."""
    rng = np.random.default_rng(26)
    X = rng.normal(size=(400, 6))
    y = 2.0 * X[:, 0] + np.sin(2.0 * X[:, 1]) + 0.1 * rng.normal(size=400)
    model = RepLeafRegressor(
        n_estimators=25,
        leaf_model="adaptive",
        leaf_gate_margin=0.02,
        encoder="identity",
        random_state=0,
    ).fit(X, y)
    rownorm = np.concatenate([
        np.abs(lv.weights).reshape(lv.weights.shape[0], -1).sum(axis=1)
        for lv in model.booster_.leaf_values_
    ])
    assert (rownorm > 0).any() and (rownorm == 0).any()  # mixed verdicts present

    path = tmp_path / "adaptive_model"
    model.save_model(path)
    loaded = RepLeafRegressor.load_model(path)
    np.testing.assert_array_equal(model.predict(X), loaded.predict(X))
    assert loaded.get_params()["leaf_gate_margin"] == 0.02
    assert loaded.get_params()["leaf_gate"] == "loo"
    # No serialization format bump: a demoted leaf is the existing leaf-array
    # encoding, so an adaptive model stamps the *same* minimum format version as
    # the equivalent embedded_linear model (RepLeafGBM stamps the lowest version
    # its features require, not a global max).
    from repleafgbm.core.serialization import FORMAT_VERSION

    config = json.loads((path / "model_config.json").read_text())
    embedded = RepLeafRegressor(
        n_estimators=25, leaf_model="embedded_linear", encoder="identity",
        random_state=0,
    ).fit(X, y)
    embedded_path = tmp_path / "embedded_model"
    embedded.save_model(embedded_path)
    embedded_config = json.loads((embedded_path / "model_config.json").read_text())
    assert config["format_version"] == embedded_config["format_version"]
    assert config["format_version"] <= FORMAT_VERSION


def test_adaptive_rejects_invalid_gate_params():
    """Gate params are validated at construction, so an invalid value surfaces
    both directly and at estimator ``fit`` (via ``make_leaf_model``)."""
    with pytest.raises(ValueError, match="leaf_gate_margin"):
        AdaptiveLeafModel(leaf_gate_margin=-0.1)
    with pytest.raises(ValueError, match="leaf_gate"):
        AdaptiveLeafModel(leaf_gate="bogus")
    rng = np.random.default_rng(31)
    X, y = rng.normal(size=(60, 4)), rng.normal(size=60)
    with pytest.raises(ValueError, match="leaf_gate"):
        RepLeafRegressor(leaf_model="adaptive", leaf_gate="nope").fit(X, y)


def test_adaptive_handles_near_zero_hessian():
    """Logistic confident-region rows drive h = p(1-p) -> ~0; the LOO gate must
    stay finite (no NaN/inf) whether some or all rows have near-zero Hessian
    (the leverage clamp keeps the leave-one-out division bounded)."""
    rng = np.random.default_rng(50)
    Z = rng.normal(size=(80, 6))
    g = rng.normal(0, 1e-3, 80)
    mixed = np.where(np.arange(80) < 40, 0.25, 1e-12)
    for h in (mixed, np.full(80, 1e-12)):
        lv = AdaptiveLeafModel(
            l2=1.0, min_samples_linear=20, leaf_gate_margin=0.01
        ).fit_leaves([np.arange(80)], g, h, Z)
        assert np.isfinite(lv.bias).all()
        assert np.isfinite(lv.weights).all()


def test_adaptive_multiclass_per_class_verdict():
    """``fit_leaves_multiclass`` applies the gate per (class, leaf): a
    clean-linear class keeps its linear leaves while noise classes are demoted to
    constant."""
    rng = np.random.default_rng(51)
    n, K, d = 600, 3, 6
    Z = np.ascontiguousarray(rng.normal(size=(n, d)))
    grad = np.empty((n, K))
    grad[:, 0] = -(Z @ rng.normal(size=d))  # class 0: linear signal -> keep
    grad[:, 1] = -rng.normal(size=n)  # class 1: noise -> demote
    grad[:, 2] = -rng.normal(size=n)  # class 2: noise -> demote
    hess = np.ascontiguousarray(np.abs(rng.normal(1.0, 0.2, (n, K))) + 0.1)
    parts = np.array_split(np.arange(n), 5)
    rows_per_class = [[np.sort(p).astype(np.int64) for p in parts] for _ in range(K)]
    out = AdaptiveLeafModel(
        l2=1e-3, min_samples_linear=20, leaf_gate_margin=0.01
    ).fit_leaves_multiclass(rows_per_class, grad, hess, Z)
    linear = [int((np.abs(out[k].weights).sum(axis=1) > 0).sum()) for k in range(K)]
    assert linear[0] > 0  # clean-linear class keeps linear leaves
    assert linear[1] == 0 and linear[2] == 0  # noise classes demoted
