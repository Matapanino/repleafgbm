"""Parity tests: Rust kernels vs the NumPy reference backend.

Skipped when the optional ``repleafgbm_native`` extension is not built.
"""

import numpy as np
import pandas as pd
import pytest

from repleafgbm import RepLeafDataset, RepLeafRegressor
from repleafgbm.backends import (
    NumPySplitBackend,
    RustSplitBackend,
    make_split_backend,
)

pytest.importorskip("repleafgbm_native", reason="Rust extension not built")


@pytest.fixture
def node_data():
    rng = np.random.default_rng(0)
    n, F, n_bins_max = 3000, 8, 33
    binned = rng.integers(0, 32, size=(n, F)).astype(np.uint16)
    binned[rng.random((n, F)) < 0.05] = 32  # missing bin
    rows = np.sort(rng.choice(n, size=2000, replace=False)).astype(np.int64)
    grad = rng.normal(size=n)
    hess = np.abs(rng.normal(size=n)) + 0.1
    n_bins_pf = np.full(F, 32, dtype=np.int64)
    return binned, rows, grad, hess, n_bins_max, n_bins_pf


def test_histogram_parity_exact(node_data):
    binned, rows, grad, hess, n_bins_max, _ = node_data
    h_np = NumPySplitBackend().build_histograms(binned, rows, grad, hess, n_bins_max)
    h_rs = RustSplitBackend().build_histograms(binned, rows, grad, hess, n_bins_max)
    # Same accumulation order -> bitwise-identical sums.
    np.testing.assert_array_equal(h_np, h_rs)


def test_split_parity_numeric_and_categorical(node_data):
    binned, rows, grad, hess, n_bins_max, n_bins_pf = node_data
    np_b, rs_b = NumPySplitBackend(), RustSplitBackend()
    hist = np_b.build_histograms(binned, rows, grad, hess, n_bins_max)
    for cat_mask in (
        np.zeros(8, dtype=bool),
        np.array([True, True, False, False, False, False, False, True]),
    ):
        s_np = np_b.find_best_split(hist, n_bins_pf, 20, 1.0, cat_mask,
                                    min_data_per_group=10)
        s_rs = rs_b.find_best_split(hist, n_bins_pf, 20, 1.0, cat_mask,
                                    min_data_per_group=10)
        assert (s_np is None) == (s_rs is None)
        if s_np is not None:
            assert (s_np.feature, s_np.bin) == (s_rs.feature, s_rs.bin)
            assert s_np.gain == pytest.approx(s_rs.gain, rel=1e-9)
            assert (s_np.n_left, s_np.n_right) == (s_rs.n_left, s_rs.n_right)
            if s_np.left_categories is not None:
                np.testing.assert_array_equal(
                    s_np.left_categories, s_rs.left_categories
                )


def test_end_to_end_backend_agreement(regression_data):
    Xtr, ytr, Xte, _ = regression_data
    preds = {}
    for backend in ("numpy", "rust"):
        model = RepLeafRegressor(
            n_estimators=30, num_leaves=8, min_samples_leaf=10,
            leaf_model="embedded_linear", split_backend=backend, random_state=42,
        )
        model.fit(Xtr, ytr)
        preds[backend] = model.predict(Xte)
    np.testing.assert_allclose(preds["numpy"], preds["rust"], rtol=1e-6, atol=1e-8)


def test_end_to_end_with_categoricals_and_missing():
    rng = np.random.default_rng(3)
    n = 1200
    cat = rng.choice(list("abcde"), size=n).astype(object)
    cat[rng.random(n) < 0.05] = None
    x = rng.normal(size=n)
    high = pd.Series(cat).isin(["a", "d"]).to_numpy()
    y = np.where(high, 3.0, -3.0) + x + rng.normal(0, 0.2, n)
    df = pd.DataFrame({"c": cat, "x": x})

    preds = {}
    for backend in ("numpy", "rust"):
        train = RepLeafDataset(df, y, categorical_features=["c"])
        model = RepLeafRegressor(
            n_estimators=20, num_leaves=6, min_samples_leaf=10,
            min_data_per_group=20, split_backend=backend, random_state=42,
        )
        model.fit(train)
        preds[backend] = model.predict(df)
    np.testing.assert_allclose(preds["numpy"], preds["rust"], rtol=1e-6, atol=1e-8)


def test_end_to_end_weighted_backend_agreement():
    """Sample weights fold into g/h upstream of the split kernels, so the
    NumPy and Rust paths must stay bitwise-identical under weighting too."""
    from repleafgbm import RepLeafClassifier

    rng = np.random.default_rng(7)
    n = 800
    X = rng.normal(size=(n, 6))
    y = rng.choice([0, 1, 2, 3], size=n, p=[0.6, 0.25, 0.1, 0.05])
    w = rng.uniform(0.2, 4.0, size=n)
    for leaf in ("constant", "embedded_linear"):
        preds = {}
        for backend in ("numpy", "rust"):
            model = RepLeafClassifier(
                n_estimators=15, num_leaves=8, min_samples_leaf=10,
                leaf_model=leaf, split_backend=backend,
                class_weight="balanced", random_state=42,
            )
            model.fit(X, y, sample_weight=w)
            preds[backend] = model.predict_proba(X)
        np.testing.assert_allclose(
            preds["numpy"], preds["rust"], rtol=1e-6, atol=1e-8
        )


def test_make_split_backend_auto_prefers_rust():
    assert isinstance(make_split_backend("auto"), RustSplitBackend)
    assert isinstance(make_split_backend("numpy"), NumPySplitBackend)
    with pytest.raises(ValueError, match="split_backend"):
        make_split_backend("cuda")
