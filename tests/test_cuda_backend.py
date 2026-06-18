"""Parity tests: CuPy CUDA histogram kernel vs the NumPy reference backend.

Skipped unless CuPy is installed *and* a usable NVIDIA GPU is present, so the
suite stays green on CPU-only machines (macOS dev box, default CI lane). These
run on a GPU via the Colab dev loop (``scripts/colab_gpu_test.sh``).

Parity here is **allclose, not bitwise**: the GPU histogram uses float64
``atomicAdd`` whose summation order is not fixed, so sums differ from NumPy's
``bincount`` in the low bits. End-to-end predictions still agree to float noise
(``rtol=1e-6``), the same bar the Rust end-to-end tests use.
"""

import numpy as np
import pandas as pd
import pytest

from repleafgbm import RepLeafDataset, RepLeafRegressor
from repleafgbm.backends import (
    CudaSplitBackend,
    NumPySplitBackend,
    make_split_backend,
)

cp = pytest.importorskip("cupy", reason="CuPy not installed")
try:
    if cp.cuda.runtime.getDeviceCount() < 1:  # pragma: no cover - hardware gate
        pytest.skip("no CUDA device available", allow_module_level=True)
except Exception as exc:  # pragma: no cover - driver/runtime missing
    pytest.skip(f"CUDA runtime unavailable: {exc}", allow_module_level=True)


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


def test_histogram_parity_allclose(node_data):
    binned, rows, grad, hess, n_bins_max, _ = node_data
    h_np = NumPySplitBackend().build_histograms(binned, rows, grad, hess, n_bins_max)
    # build_histograms returns a resident device array (Phase B2); copy to host.
    h_cu = cp.asnumpy(
        CudaSplitBackend().build_histograms(binned, rows, grad, hess, n_bins_max)
    )
    # grad/hess sums: float-noise agreement (GPU atomic-add reordering).
    np.testing.assert_allclose(h_np, h_cu, rtol=1e-9, atol=1e-9)
    # Counts are exact integer sums (< 2**53), so they match bitwise.
    np.testing.assert_array_equal(h_np[:, :, 2], h_cu[:, :, 2])


def test_histogram_subtractable(node_data):
    binned, rows, grad, hess, n_bins_max, _ = node_data
    cu = CudaSplitBackend()
    half = rows.size // 2
    left, right = rows[:half], rows[half:]
    # Resident device histograms (Phase B2): subtraction stays on-device, as in
    # the grower; copy to host only for the assertions.
    h_parent = cu.build_histograms(binned, rows, grad, hess, n_bins_max)
    h_left = cu.build_histograms(binned, left, grad, hess, n_bins_max)
    h_right = cu.build_histograms(binned, right, grad, hess, n_bins_max)
    # The tree grower derives a child as parent - sibling; that must hold to
    # float noise so the additive structure is preserved.
    np.testing.assert_allclose(
        cp.asnumpy(h_parent), cp.asnumpy(h_left + h_right), rtol=1e-9, atol=1e-9
    )
    np.testing.assert_allclose(
        cp.asnumpy(h_parent - h_left), cp.asnumpy(h_right), rtol=1e-9, atol=1e-9
    )


def test_split_parity_numeric_and_categorical(node_data):
    # Phase B2: the numeric scan + argmax run on-device (hist_cu is a resident
    # device array); categorical subsets fall back to the host. Both must still
    # agree with the reference on the selected split.
    binned, rows, grad, hess, n_bins_max, n_bins_pf = node_data
    np_b, cu_b = NumPySplitBackend(), CudaSplitBackend()
    hist_np = np_b.build_histograms(binned, rows, grad, hess, n_bins_max)
    hist_cu = cu_b.build_histograms(binned, rows, grad, hess, n_bins_max)
    for cat_mask in (
        np.zeros(8, dtype=bool),
        np.array([True, True, False, False, False, False, False, True]),
    ):
        s_np = np_b.find_best_split(hist_np, n_bins_pf, 20, 1.0, cat_mask,
                                    min_data_per_group=10)
        s_cu = cu_b.find_best_split(hist_cu, n_bins_pf, 20, 1.0, cat_mask,
                                    min_data_per_group=10)
        assert (s_np is None) == (s_cu is None)
        if s_np is not None:
            assert (s_np.feature, s_np.bin) == (s_cu.feature, s_cu.bin)
            assert s_np.gain == pytest.approx(s_cu.gain, rel=1e-6)
            assert (s_np.n_left, s_np.n_right) == (s_cu.n_left, s_cu.n_right)
            if s_np.left_categories is not None:
                np.testing.assert_array_equal(
                    s_np.left_categories, s_cu.left_categories
                )


def test_find_best_split_accepts_host_array(node_data):
    """find_best_split also accepts a host array (cp.asarray uploads it), and on
    the same input agrees with the reference scan."""
    binned, rows, grad, hess, n_bins_max, n_bins_pf = node_data
    np_b, cu_b = NumPySplitBackend(), CudaSplitBackend()
    hist_np = np_b.build_histograms(binned, rows, grad, hess, n_bins_max)
    cat_mask = np.zeros(8, dtype=bool)
    s_np = np_b.find_best_split(hist_np, n_bins_pf, 20, 1.0, cat_mask)
    s_cu = cu_b.find_best_split(hist_np, n_bins_pf, 20, 1.0, cat_mask)
    assert (s_np is None) == (s_cu is None)
    if s_np is not None:
        assert (s_np.feature, s_np.bin) == (s_cu.feature, s_cu.bin)
        assert s_np.gain == pytest.approx(s_cu.gain, rel=1e-6)
        assert (s_np.n_left, s_np.n_right) == (s_cu.n_left, s_cu.n_right)


def test_split_parity_gpu_scan_path():
    """A histogram above the adaptive threshold exercises the on-device numeric
    scan (small ones fall back to the host path tested above). It must still
    agree with the reference on the selected split, numeric and categorical."""
    from repleafgbm.backends.cuda_backend import _GPU_SCAN_MIN_CELLS

    rng = np.random.default_rng(1)
    F = 200
    n_bins_max = (_GPU_SCAN_MIN_CELLS // F) + 2  # ensure F * n_bins_max >= thresh
    assert F * n_bins_max >= _GPU_SCAN_MIN_CELLS
    n_cats = n_bins_max - 1  # missing bin at index n_cats
    n = 6000
    binned = rng.integers(0, n_cats, size=(n, F)).astype(np.uint16)
    binned[rng.random((n, F)) < 0.05] = n_cats  # missing bin
    rows = np.arange(n, dtype=np.int64)
    grad = rng.normal(size=n)
    hess = np.abs(rng.normal(size=n)) + 0.1
    n_bins_pf = np.full(F, n_cats, dtype=np.int64)

    np_b, cu_b = NumPySplitBackend(), CudaSplitBackend()
    hist_np = np_b.build_histograms(binned, rows, grad, hess, n_bins_max)
    hist_cu = cu_b.build_histograms(binned, rows, grad, hess, n_bins_max)
    cat_mask = np.zeros(F, dtype=bool)
    cat_mask[:3] = True  # a few categorical features → host fallback within scan
    for mask in (np.zeros(F, dtype=bool), cat_mask):
        s_np = np_b.find_best_split(hist_np, n_bins_pf, 20, 1.0, mask,
                                    min_data_per_group=10)
        s_cu = cu_b.find_best_split(hist_cu, n_bins_pf, 20, 1.0, mask,
                                    min_data_per_group=10)
        assert (s_np is None) == (s_cu is None)
        if s_np is not None:
            assert (s_np.feature, s_np.bin) == (s_cu.feature, s_cu.bin)
            assert s_np.gain == pytest.approx(s_cu.gain, rel=1e-6)
            assert (s_np.n_left, s_np.n_right) == (s_cu.n_left, s_cu.n_right)
            if s_np.left_categories is not None:
                np.testing.assert_array_equal(
                    s_np.left_categories, s_cu.left_categories
                )


def test_end_to_end_backend_agreement(regression_data):
    Xtr, ytr, Xte, _ = regression_data
    preds = {}
    for backend in ("numpy", "cuda"):
        model = RepLeafRegressor(
            n_estimators=30, num_leaves=8, min_samples_leaf=10,
            leaf_model="embedded_linear", split_backend=backend, random_state=42,
        )
        model.fit(Xtr, ytr)
        preds[backend] = model.predict(Xte)
    np.testing.assert_allclose(preds["numpy"], preds["cuda"], rtol=1e-6, atol=1e-8)


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
    for backend in ("numpy", "cuda"):
        train = RepLeafDataset(df, y, categorical_features=["c"])
        model = RepLeafRegressor(
            n_estimators=20, num_leaves=6, min_samples_leaf=10,
            min_data_per_group=20, split_backend=backend, random_state=42,
        )
        model.fit(train)
        preds[backend] = model.predict(df)
    np.testing.assert_allclose(preds["numpy"], preds["cuda"], rtol=1e-6, atol=1e-8)


def test_end_to_end_weighted_backend_agreement():
    """Sample weights fold into g/h upstream of the split kernels, so the
    NumPy and CUDA paths must still agree to float noise under weighting."""
    from repleafgbm import RepLeafClassifier

    rng = np.random.default_rng(7)
    n = 800
    X = rng.normal(size=(n, 6))
    y = rng.choice([0, 1, 2, 3], size=n, p=[0.6, 0.25, 0.1, 0.05])
    w = rng.uniform(0.2, 4.0, size=n)
    for leaf in ("constant", "embedded_linear"):
        preds = {}
        for backend in ("numpy", "cuda"):
            model = RepLeafClassifier(
                n_estimators=15, num_leaves=8, min_samples_leaf=10,
                leaf_model=leaf, split_backend=backend,
                class_weight="balanced", random_state=42,
            )
            model.fit(X, y, sample_weight=w)
            preds[backend] = model.predict_proba(X)
        np.testing.assert_allclose(
            preds["numpy"], preds["cuda"], rtol=1e-6, atol=1e-8
        )


def test_make_split_backend_cuda_returns_cuda():
    assert isinstance(make_split_backend("cuda"), CudaSplitBackend)
    # "auto" must never pick the GPU backend implicitly.
    assert not isinstance(make_split_backend("auto"), CudaSplitBackend)


# --------------------------------------------------------------------------- #
# Private transfer counters (profiling aid consumed by benchmarks/gpu_profile.py)
# --------------------------------------------------------------------------- #
def test_transfer_counters_build_histograms(node_data):
    """A single build accounts for the binned upload (once) plus the per-node
    rows + gathered grad/hess uploads (the transfer the next optimization cuts)."""
    binned, rows, grad, hess, n_bins_max, _ = node_data
    cu = CudaSplitBackend()
    cu.build_histograms(binned, rows, grad, hess, n_bins_max)
    s = cu.get_transfer_stats()
    n_sel = rows.size
    assert s["rows_h2d_bytes"] == 8 * n_sel
    assert s["gradhess_h2d_bytes"] == 16 * n_sel
    assert s["binned_h2d_bytes"] == 2 * binned.size
    assert s["binned_uploads"] == 1
    assert s["n_hist_builds"] == 1


def test_transfer_counters_binned_cached_across_builds(node_data):
    """The resident binned matrix uploads once (Phase B1); only the small
    per-node rows/grad/hess uploads recur across builds on the same matrix."""
    binned, rows, grad, hess, n_bins_max, _ = node_data
    cu = CudaSplitBackend()
    cu.build_histograms(binned, rows[:1000], grad, hess, n_bins_max)
    cu.build_histograms(binned, rows[1000:], grad, hess, n_bins_max)
    s = cu.get_transfer_stats()
    assert s["binned_uploads"] == 1  # cache hit on the second build
    assert s["binned_h2d_bytes"] == 2 * binned.size
    assert s["n_hist_builds"] == 2
    assert s["rows_h2d_bytes"] == 8 * rows.size  # 1000 + 2000 rows total


def test_reset_transfer_stats(node_data):
    binned, rows, grad, hess, n_bins_max, _ = node_data
    cu = CudaSplitBackend()
    cu.build_histograms(binned, rows, grad, hess, n_bins_max)
    assert any(v > 0 for v in cu.get_transfer_stats().values())
    cu.reset_transfer_stats()
    assert all(v == 0 for v in cu.get_transfer_stats().values())


def test_transfer_counters_small_scan_copies_full_histogram(node_data):
    """Small histograms copy the whole (F, B, 3) array back for the host scan;
    no winning-scalar pack on this path."""
    binned, rows, grad, hess, n_bins_max, n_bins_pf = node_data
    cu = CudaSplitBackend()
    hist = cu.build_histograms(binned, rows, grad, hess, n_bins_max)
    n_features = binned.shape[1]
    assert n_features * n_bins_max < 32_768  # below the adaptive threshold
    cu.reset_transfer_stats()
    cu.find_best_split(hist, n_bins_pf, 20, 1.0, np.zeros(n_features, dtype=bool))
    s = cu.get_transfer_stats()
    assert s["n_small_scans"] == 1
    assert s["hist_d2h_bytes"] == 24 * n_features * n_bins_max
    assert s["n_gpu_scans"] == 0
    assert s["winner_d2h_bytes"] == 0
    assert s["cat_slice_d2h_bytes"] == 0


def test_transfer_counters_large_scan_packs_winner_and_cat_slices():
    """Large histograms keep the array resident: only the 4-scalar winner pack
    (32 bytes) crosses back, plus one slice per categorical feature scanned."""
    from repleafgbm.backends.cuda_backend import _GPU_SCAN_MIN_CELLS

    rng = np.random.default_rng(2)
    F = 200
    n_bins_max = (_GPU_SCAN_MIN_CELLS // F) + 2
    assert F * n_bins_max >= _GPU_SCAN_MIN_CELLS
    n_cats = n_bins_max - 1
    n = 4000
    binned = rng.integers(0, n_cats, size=(n, F)).astype(np.uint16)
    rows = np.arange(n, dtype=np.int64)
    grad = rng.normal(size=n)
    hess = np.abs(rng.normal(size=n)) + 0.1
    n_bins_pf = np.full(F, n_cats, dtype=np.int64)

    cu = CudaSplitBackend()
    hist = cu.build_histograms(binned, rows, grad, hess, n_bins_max)

    # Numeric-only: winner pack crosses back, full histogram stays resident.
    cu.reset_transfer_stats()
    cu.find_best_split(hist, n_bins_pf, 20, 1.0, np.zeros(F, dtype=bool))
    s = cu.get_transfer_stats()
    assert s["n_gpu_scans"] == 1
    assert s["winner_d2h_bytes"] == 32
    assert s["hist_d2h_bytes"] == 0
    assert s["n_small_scans"] == 0
    assert s["cat_slice_d2h_bytes"] == 0

    # With categoricals: one (n_bins_max, 3) slice per categorical feature.
    cat_mask = np.zeros(F, dtype=bool)
    cat_mask[:3] = True
    cu.reset_transfer_stats()
    cu.find_best_split(hist, n_bins_pf, 20, 1.0, cat_mask, min_data_per_group=10)
    s = cu.get_transfer_stats()
    assert s["n_cat_slices"] == 3
    assert s["cat_slice_d2h_bytes"] == 24 * n_bins_max * 3


def test_booster_exposes_split_backend_handle(regression_data):
    """The fitted booster retains its split backend so profilers can read the
    CUDA transfer counters after an end-to-end fit (benchmarks/gpu_profile.py)."""
    Xtr, ytr, _, _ = regression_data
    model = RepLeafRegressor(
        n_estimators=10, num_leaves=8, min_samples_leaf=10,
        leaf_model="constant", split_backend="cuda", random_state=42,
    ).fit(Xtr, ytr)
    backend = model.booster_.split_backend_
    assert isinstance(backend, CudaSplitBackend)
    stats = backend.get_transfer_stats()
    # A real fit performs many node builds, so the per-node gather is non-zero
    # and the binned matrix uploaded exactly once.
    assert stats["n_hist_builds"] > 0
    assert stats["gradhess_h2d_bytes"] > 0
    assert stats["binned_uploads"] == 1
