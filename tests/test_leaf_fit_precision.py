"""Tests for the opt-in ``leaf_fit_precision='float32_gram'`` wide-emb leaf-fit.

The default ``float64`` path stays the bitwise NumPy<->Rust parity path (covered
by tests/test_rust_backend.py); here we verify the float32 opt-in is (a) numerically
equivalent at allclose/quality level, (b) confined to the wide-emb BLAS path,
(c) robust to the singular-leaf fallback, and (d) a clean public param.

See docs/proposals/float32-wide-embedding-leaf-fit.md.
"""

import numpy as np
import pytest

from repleafgbm import RepLeafRegressor
from repleafgbm.core import leaf_models
from repleafgbm.core.leaf_models import (
    EmbeddedLinearLeafModel,
    make_leaf_model,
)

WIDE = 80  # > _NATIVE_STATS_MAX_DIM (64) -> BLAS Gram path (where float32 applies)


def _leaf_inputs(n=1500, emb=WIDE, n_leaves=4, seed=0):
    """Synthetic single-tree leaf-fit inputs (leaf_rows, grad, hess, Z)."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, emb))
    w = rng.normal(size=emb)
    resid = Z @ w + rng.normal(scale=0.1, size=n)
    grad, hess = -resid, np.ones(n)
    rows = np.array_split(rng.permutation(n), n_leaves)
    return rows, grad, hess, Z


def test_float32_gram_matches_float64_allclose_and_quality():
    rows, grad, hess, Z = _leaf_inputs()
    lv64 = EmbeddedLinearLeafModel(l2=1.0).fit_leaves(rows, grad, hess, Z)
    lv32 = EmbeddedLinearLeafModel(
        l2=1.0, leaf_fit_precision="float32_gram"
    ).fit_leaves(rows, grad, hess, Z)
    leaf_idx = np.concatenate([np.full(len(r), i) for i, r in enumerate(rows)])
    Zc = Z[np.concatenate(rows)]
    p64, p32 = lv64.predict(leaf_idx, Zc), lv32.predict(leaf_idx, Zc)
    # allclose on predictions (not raw weights — the constant-fallback gate can
    # flip on a near-singular leaf), plus RMSE quality-equivalence.
    assert np.allclose(p32, p64, rtol=1e-5, atol=1e-5)
    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))  # noqa: E731
    t = -grad[np.concatenate(rows)]
    assert abs(rmse(p32, t) - rmse(p64, t)) < 1e-4


def test_default_is_float64_and_bitwise():
    rows, grad, hess, Z = _leaf_inputs(seed=1)
    lv_default = EmbeddedLinearLeafModel(l2=1.0).fit_leaves(rows, grad, hess, Z)
    lv_explicit = EmbeddedLinearLeafModel(
        l2=1.0, leaf_fit_precision="float64"
    ).fit_leaves(rows, grad, hess, Z)
    # Default == explicit float64, exactly (bytes), and float32 actually differs.
    assert np.array_equal(lv_default.weights, lv_explicit.weights)
    assert np.array_equal(lv_default.bias, lv_explicit.bias)
    lv32 = EmbeddedLinearLeafModel(
        l2=1.0, leaf_fit_precision="float32_gram"
    ).fit_leaves(rows, grad, hess, Z)
    assert not np.array_equal(lv32.weights, lv_default.weights)  # branch exercised


@pytest.mark.skipif(
    leaf_models._native is None, reason="narrow path uses native ext when present"
)
def test_narrow_embedding_unaffected_by_float32():
    # emb <= 64 takes the native rayon path; the float32 knob must be inert there.
    rows, grad, hess, Z = _leaf_inputs(emb=16, seed=2)
    lv64 = EmbeddedLinearLeafModel(l2=1.0).fit_leaves(rows, grad, hess, Z)
    lv32 = EmbeddedLinearLeafModel(
        l2=1.0, leaf_fit_precision="float32_gram"
    ).fit_leaves(rows, grad, hess, Z)
    assert np.array_equal(lv64.weights, lv32.weights)
    assert np.array_equal(lv64.bias, lv32.bias)


def test_float32_handles_singular_leaf_fallback():
    # Rank-deficient Z (duplicated column) + l2=0 -> singular Gram -> one-by-one
    # constant fallback. float32 must take it without crashing and stay finite.
    rng = np.random.default_rng(3)
    n, emb = 900, WIDE
    Z = rng.normal(size=(n, emb))
    Z[:, 1] = Z[:, 0]  # exact collinearity
    grad, hess = -(Z @ rng.normal(size=emb)), np.ones(n)
    rows = np.array_split(np.arange(n), 3)
    lv = EmbeddedLinearLeafModel(
        l2=0.0, leaf_fit_precision="float32_gram"
    ).fit_leaves(rows, grad, hess, Z)
    assert np.isfinite(lv.weights).all() and np.isfinite(lv.bias).all()


def test_constant_leaf_ignores_precision():
    # make_leaf_model accepts the knob for constant but it is inert.
    m = make_leaf_model("constant", l2=1.0, min_samples_linear=10,
                        leaf_fit_precision="float32_gram")
    assert m.name == "constant"


def test_invalid_precision_raises():
    with pytest.raises(ValueError, match="leaf_fit_precision"):
        make_leaf_model("embedded_linear", l2=1.0, min_samples_linear=10,
                        leaf_fit_precision="bogus")
    with pytest.raises(ValueError, match="leaf_fit_precision"):
        EmbeddedLinearLeafModel(leaf_fit_precision="bogus")


def test_estimator_param_roundtrip_and_quality():
    rng = np.random.default_rng(4)
    n, f = 2000, WIDE
    X = rng.normal(size=(n, f))
    y = 2 * X[:, 0] + np.sin(X[:, 1]) + rng.normal(scale=0.1, size=n)
    common = dict(n_estimators=15, leaf_model="embedded_linear",
                  max_leaf_emb_dim=128, random_state=0)
    m64 = RepLeafRegressor(**common).fit(X, y)
    m32 = RepLeafRegressor(leaf_fit_precision="float32_gram", **common).fit(X, y)
    assert m64.get_params()["leaf_fit_precision"] == "float64"  # default
    assert m32.get_params()["leaf_fit_precision"] == "float32_gram"
    from sklearn.metrics import r2_score
    assert abs(r2_score(y, m64.predict(X)) - r2_score(y, m32.predict(X))) < 5e-3
