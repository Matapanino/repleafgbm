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
from repleafgbm.backends.base import SplitCandidate

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


def test_histogram_parity_parallel_branch():
    """Large node exercises the rayon feature-parallel histogram path (not the
    small-node serial branch). Feature-parallel accumulation keeps each
    (feature, bin) cell in row order, so it must stay bitwise-identical to the
    NumPy reference regardless of thread count."""
    rng = np.random.default_rng(1)
    # 40k-row pool, 30k rows sampled: rows.len() * n_features = 30_000 * 12 =
    # 360_000 exceeds PARALLEL_MIN_CELLS (1 << 17 = 131_072) in native/src/lib.rs,
    # so the rayon parallel branch is taken. CI runs this multi-threaded, so the
    # bitwise assert also guards against a future row-parallel kernel that would
    # reorder the per-cell float sums.
    n, n_features, n_bins_max = 40_000, 12, 65
    binned = rng.integers(0, 64, size=(n, n_features)).astype(np.uint16)
    binned[rng.random((n, n_features)) < 0.05] = 64  # missing bin
    rows = np.sort(rng.choice(n, size=30_000, replace=False)).astype(np.int64)
    grad = rng.normal(size=n)
    hess = np.abs(rng.normal(size=n)) + 0.1
    h_np = NumPySplitBackend().build_histograms(binned, rows, grad, hess, n_bins_max)
    h_rs = RustSplitBackend().build_histograms(binned, rows, grad, hess, n_bins_max)
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


@pytest.mark.parametrize("objective", [None, "huber", "quantile"])
def test_multioutput_backend_agreement(objective):
    """Multi-output robust losses feed new g/h values into the same shared
    histogram/scan kernels, so NumPy and Rust must stay in lock-step."""
    rng = np.random.default_rng(11)
    n = 800
    X = rng.normal(size=(n, 6))
    Y = np.column_stack([X[:, 0] * 2 + X[:, 1], -X[:, 2] + 0.5 * X[:, 3]])
    Y += 0.1 * rng.normal(size=(n, 2))
    preds = {}
    for backend in ("numpy", "rust"):
        model = RepLeafRegressor(
            n_estimators=30, num_leaves=8, min_samples_leaf=10,
            leaf_model="embedded_linear", objective=objective,
            split_backend=backend, random_state=42,
        )
        model.fit(X[:600], Y[:600])
        preds[backend] = model.predict(X[600:])
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
    # "cuda" is a recognized (GPU-only) backend now, so an unknown name must be
    # something else; it still raises ValueError, not the cuda ImportError.
    with pytest.raises(ValueError, match="split_backend"):
        make_split_backend("metal")


# --------------------------------------------------------------------------- #
# partition_rows parity: the Rust kernel must reproduce the NumPy reference's
# left/right rows EXACTLY (integer index routing, so assert_array_equal — not
# allclose), including the input row order.
# --------------------------------------------------------------------------- #


def _np_partition_ref(binned, rows, split, missing_bin):
    """Independent NumPy oracle (the pre-refactor partition logic)."""
    b = binned[rows, split.feature]
    if split.left_categories is not None:
        go_left = np.isin(b, split.left_categories) | (b == missing_bin)
    else:
        go_left = (b <= split.bin) | (b == missing_bin)
    return rows[go_left], rows[~go_left]


def _assert_partition_parity(binned, rows, split, missing_bin):
    exp_l, exp_r = _np_partition_ref(binned, rows, split, missing_bin)
    np_l, np_r = NumPySplitBackend().partition_rows(binned, rows, split, missing_bin)
    rs_l, rs_r = RustSplitBackend().partition_rows(binned, rows, split, missing_bin)
    for got_l, got_r in ((np_l, np_r), (rs_l, rs_r)):
        np.testing.assert_array_equal(got_l, exp_l)  # exact index + order
        np.testing.assert_array_equal(got_r, exp_r)
        assert got_l.shape[0] + got_r.shape[0] == rows.shape[0]  # disjoint cover


def test_partition_parity_numeric(node_data):
    binned, rows, *_ = node_data
    missing_bin = 32  # node_data uses n_bins_per_feature == 32 everywhere
    for feature in (0, 3, 7):
        for bin_ in (0, 5, 16, 31):
            split = SplitCandidate(feature, bin_, 0.0, 0, 0)
            _assert_partition_parity(binned, rows, split, missing_bin)


def test_partition_parity_categorical(node_data):
    binned, rows, *_ = node_data
    missing_bin = 32
    for cats in (
        np.array([1, 4, 7, 9, 15], dtype=np.int64),
        np.array([0, 31], dtype=np.int64),          # boundary codes
        np.array([3, 30, 31], dtype=np.int64),       # 30/31 may be absent from node
    ):
        split = SplitCandidate(2, -1, 0.0, 0, 0, left_categories=cats)
        _assert_partition_parity(binned, rows, split, missing_bin)


def test_partition_native_edges():
    """Native kernel over degenerate partitions (empty/all-left/all-right)."""
    missing_bin = 5
    binned = np.array([[0], [1], [2], [3], [4], [5]] * 6, dtype=np.uint16)  # bin 5 == missing
    rows = np.arange(binned.shape[0], dtype=np.int64)

    # all-left: every non-missing bin (0..4) <= 4, missing(5) also left
    _assert_partition_parity(binned, rows, SplitCandidate(0, 4, 0.0, 0, 0), missing_bin)
    # singleton categorical subset: only code 2 (and missing) go left
    _assert_partition_parity(
        binned, rows, SplitCandidate(0, -1, 0.0, 0, 0, left_categories=np.array([2], np.int64)),
        missing_bin,
    )
    # all-right: no missing present, bins 1..4 all > 0
    nm = np.array([[1], [2], [3], [4]] * 9, dtype=np.uint16)
    nm_rows = np.arange(nm.shape[0], dtype=np.int64)
    _assert_partition_parity(nm, nm_rows, SplitCandidate(0, 0, 0.0, 0, 0), missing_bin)
    # empty rows -> two empty arrays
    empty = np.array([], dtype=np.int64)
    gl, gr = RustSplitBackend().partition_rows(
        binned, empty, SplitCandidate(0, 4, 0.0, 0, 0), missing_bin
    )
    assert gl.shape[0] == 0 and gr.shape[0] == 0


def test_partition_node_sizes(node_data):
    """Native kernel stays exact across tiny and large node row counts (the
    kernel handles every size — there is no min-rows fallback gate)."""
    binned, rows, *_ = node_data
    missing_bin = 32
    split = SplitCandidate(1, 12, 0.0, 0, 0)
    for size in (1, 8, 64, 500, rows.shape[0]):
        _assert_partition_parity(binned, rows[:size], split, missing_bin)


def test_partition_rows_dtype_robustness(node_data):
    """Non-int64 / non-contiguous row arrays are normalized before the FFI call."""
    binned, rows, *_ = node_data
    missing_bin = 32
    split = SplitCandidate(4, 20, 0.0, 0, 0)
    rs = RustSplitBackend()
    for variant in (rows.astype(np.int32), rows[::2]):  # both > gate -> native path
        exp_l, exp_r = _np_partition_ref(binned, variant, split, missing_bin)
        got_l, got_r = rs.partition_rows(binned, variant, split, missing_bin)
        np.testing.assert_array_equal(got_l, exp_l)
        np.testing.assert_array_equal(got_r, exp_r)


def test_partition_older_native_fallback(node_data, monkeypatch):
    """A new RustSplitBackend against an older repleafgbm_native (no
    partition_rows symbol) gracefully falls back to the NumPy default."""
    import types

    binned, rows, *_ = node_data
    missing_bin = 32
    split = SplitCandidate(0, 10, 0.0, 0, 0)
    rs = RustSplitBackend()
    stub = types.SimpleNamespace(
        build_histograms=rs._native.build_histograms,
        find_best_split=rs._native.find_best_split,
    )  # no partition_rows attribute
    monkeypatch.setattr(rs, "_native", stub)
    exp_l, exp_r = _np_partition_ref(binned, rows, split, missing_bin)
    got_l, got_r = rs.partition_rows(binned, rows, split, missing_bin)
    np.testing.assert_array_equal(got_l, exp_l)
    np.testing.assert_array_equal(got_r, exp_r)
