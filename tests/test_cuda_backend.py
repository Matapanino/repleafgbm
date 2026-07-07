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


def test_device_cuda_macro_matches_explicit_split_backend(regression_data):
    """``device="cuda"`` (ADR 0007) must select the CUDA backend and follow the
    exact same code path as spelling ``split_backend="cuda"`` out — same seed +
    same backend is deterministic, so predictions agree bitwise."""
    Xtr, ytr, Xte, _ = regression_data
    common = dict(
        n_estimators=30, num_leaves=8, min_samples_leaf=10,
        leaf_model="embedded_linear", random_state=42,
    )
    via_macro = RepLeafRegressor(device="cuda", **common).fit(Xtr, ytr)
    explicit = RepLeafRegressor(split_backend="cuda", **common).fit(Xtr, ytr)
    assert isinstance(via_macro.booster_.split_backend_, CudaSplitBackend)
    np.testing.assert_array_equal(
        via_macro.predict(Xte), explicit.predict(Xte)
    )


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
    s = cu.get_transfer_stats()
    assert any(v > 0 for k, v in s.items() if k != "scan_min_cells")
    cu.reset_transfer_stats()
    s = cu.get_transfer_stats()
    # Transfer/work counters zero out; the effective threshold is config (the
    # value that produced the path counts), so reset re-seeds it rather than
    # zeroing it.
    assert all(v == 0 for k, v in s.items() if k != "scan_min_cells")
    assert s["scan_min_cells"] == cu._scan_min_cells


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


# --------------------------------------------------------------------------- #
# Adaptive scan-threshold override (REPLEAFGBM_CUDA_SCAN_MIN_CELLS)
#
# The threshold is resolved once at backend construction, so each test sets the
# env var (monkeypatch auto-restores → no leak) *before* building the backend.
# The CPU-safe resolver unit tests live in tests/test_cuda_scan_threshold.py;
# these verify the GPU-side behaviour: which scan path a node takes and that the
# selected split still matches the NumPy reference on the forced path.
# --------------------------------------------------------------------------- #
def test_scan_min_cells_in_stats_default(monkeypatch):
    monkeypatch.delenv("REPLEAFGBM_CUDA_SCAN_MIN_CELLS", raising=False)
    assert CudaSplitBackend().get_transfer_stats()["scan_min_cells"] == 32_768


def test_scan_min_cells_in_stats_override(monkeypatch):
    monkeypatch.setenv("REPLEAFGBM_CUDA_SCAN_MIN_CELLS", "12345")
    cu = CudaSplitBackend()
    assert cu.get_transfer_stats()["scan_min_cells"] == 12345
    assert cu._scan_min_cells == 12345


def test_env_threshold_forces_host_scan(monkeypatch):
    """A very high threshold pushes even a large histogram onto the host scan
    path (n_small_scans), and the selected split still matches the reference."""
    from repleafgbm.backends.cuda_backend import _GPU_SCAN_MIN_CELLS

    monkeypatch.setenv("REPLEAFGBM_CUDA_SCAN_MIN_CELLS", "1000000000")
    rng = np.random.default_rng(4)
    F = 200
    n_bins_max = (_GPU_SCAN_MIN_CELLS // F) + 2  # large enough to GPU-scan by default
    assert F * n_bins_max >= _GPU_SCAN_MIN_CELLS
    n_cats = n_bins_max - 1
    n = 4000
    binned = rng.integers(0, n_cats, size=(n, F)).astype(np.uint16)
    rows = np.arange(n, dtype=np.int64)
    grad = rng.normal(size=n)
    hess = np.abs(rng.normal(size=n)) + 0.1
    n_bins_pf = np.full(F, n_cats, dtype=np.int64)

    np_b, cu_b = NumPySplitBackend(), CudaSplitBackend()
    assert cu_b.get_transfer_stats()["scan_min_cells"] == 1_000_000_000
    hist_np = np_b.build_histograms(binned, rows, grad, hess, n_bins_max)
    hist_cu = cu_b.build_histograms(binned, rows, grad, hess, n_bins_max)
    cu_b.reset_transfer_stats()
    cat = np.zeros(F, dtype=bool)
    s_cu = cu_b.find_best_split(hist_cu, n_bins_pf, 20, 1.0, cat)
    s_np = np_b.find_best_split(hist_np, n_bins_pf, 20, 1.0, cat)
    st = cu_b.get_transfer_stats()
    assert st["n_small_scans"] == 1 and st["n_gpu_scans"] == 0  # forced to host
    assert (s_np is None) == (s_cu is None)
    if s_np is not None:
        assert (s_np.feature, s_np.bin) == (s_cu.feature, s_cu.bin)
        assert s_np.gain == pytest.approx(s_cu.gain, rel=1e-6)
        assert (s_np.n_left, s_np.n_right) == (s_cu.n_left, s_cu.n_right)


def test_env_threshold_forces_gpu_scan(monkeypatch, node_data):
    """Threshold 0 pushes even a small histogram onto the on-device scan
    (n_gpu_scans); the selected split must still agree with the reference."""
    monkeypatch.setenv("REPLEAFGBM_CUDA_SCAN_MIN_CELLS", "0")
    binned, rows, grad, hess, n_bins_max, n_bins_pf = node_data  # small (F=8)
    np_b, cu_b = NumPySplitBackend(), CudaSplitBackend()
    assert cu_b.get_transfer_stats()["scan_min_cells"] == 0
    hist_np = np_b.build_histograms(binned, rows, grad, hess, n_bins_max)
    hist_cu = cu_b.build_histograms(binned, rows, grad, hess, n_bins_max)
    cu_b.reset_transfer_stats()
    cat = np.zeros(8, dtype=bool)
    s_cu = cu_b.find_best_split(hist_cu, n_bins_pf, 20, 1.0, cat)
    s_np = np_b.find_best_split(hist_np, n_bins_pf, 20, 1.0, cat)
    st = cu_b.get_transfer_stats()
    assert st["n_gpu_scans"] == 1 and st["n_small_scans"] == 0  # forced to GPU
    assert (s_np is None) == (s_cu is None)
    if s_np is not None:
        assert (s_np.feature, s_np.bin) == (s_cu.feature, s_cu.bin)
        assert s_np.gain == pytest.approx(s_cu.gain, rel=1e-6)
        assert (s_np.n_left, s_np.n_right) == (s_cu.n_left, s_cu.n_right)


# --------------------------------------------------------------------------- #
# Multi-output (shared-routing) device path
#
# Multi-output trees stack the K per-output histograms into (F, B, 3, K) and
# scan for a split whose gain is the per-output Newton gain summed over outputs.
# The CUDA backend keeps that stack resident and scans it on-device
# (build_histograms_multioutput / find_best_split_multioutput), mirroring the
# scalar Phase-B2 path; parity is allclose, not bitwise.
# --------------------------------------------------------------------------- #
def _multioutput_xy(n: int, n_outputs: int, seed: int):
    """``n_outputs`` correlated regression targets over shared raw features."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    cols = [
        X[:, 0] + 0.5 * X[:, 1] ** 2 - X[:, 2] + np.sin((k + 2) * X[:, 3])
        + 0.3 * k * X[:, 4]
        for k in range(n_outputs)
    ]
    Y = np.column_stack(cols) + 0.05 * rng.normal(size=(n, n_outputs))
    return X, Y


@pytest.mark.parametrize("leaf_model", ["constant", "embedded_linear"])
def test_multioutput_end_to_end_backend_agreement(leaf_model):
    """End-to-end multi-output fit agrees across numpy/cuda to float noise at the
    adaptive default threshold: these narrow nodes take the host small-scan, so
    the split selection (and thus the tree) is identical — only leaf values carry
    the GPU histogram's float noise. (The on-device scan, which can flip a
    near-tied split, is exercised separately below.)"""
    X, Y = _multioutput_xy(1200, n_outputs=3, seed=11)
    Xtr, Ytr, Xte = X[:900], Y[:900], X[900:]
    preds = {}
    for backend in ("numpy", "cuda"):
        model = RepLeafRegressor(
            n_estimators=30, num_leaves=8, min_samples_leaf=10,
            leaf_model=leaf_model, split_backend=backend, random_state=42,
        ).fit(Xtr, Ytr)
        preds[backend] = model.predict(Xte)
    assert preds["numpy"].shape == (300, 3)
    np.testing.assert_allclose(preds["numpy"], preds["cuda"], rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("leaf_model", ["constant", "embedded_linear"])
def test_multioutput_device_scan_quality_matches(monkeypatch, leaf_model):
    """Forcing the on-device scan for every node (threshold 0), the cuda model
    stays quality-equivalent to numpy. The device scan sums per-output Newton
    gains with CuPy reductions whose low bits differ from NumPy, so on a
    near-tied node the argmax can pick a *different but equally good* split; that
    reroutes a handful of rows. The bulk of predictions still agree to the
    rtol=1e-6 parity bar and the test RMSE matches — the flips are quality-neutral.
    This is the on-device analogue of the scalar Phase-B2 scan (the narrow default
    keeps both on the host); exact-tree parity is not guaranteed once the scan
    runs on the GPU, but model quality is."""
    monkeypatch.setenv("REPLEAFGBM_CUDA_SCAN_MIN_CELLS", "0")
    X, Y = _multioutput_xy(1200, n_outputs=3, seed=11)
    Xtr, Ytr, Xte, Yte = X[:900], Y[:900], X[900:], Y[900:]
    preds = {}
    for backend in ("numpy", "cuda"):
        model = RepLeafRegressor(
            n_estimators=30, num_leaves=8, min_samples_leaf=10,
            leaf_model=leaf_model, split_backend=backend, random_state=42,
        ).fit(Xtr, Ytr)
        preds[backend] = model.predict(Xte)
    np_pred, cu_pred = preds["numpy"], preds["cuda"]
    # Coarse "few rows rerouted" sanity bound: the bulk agree to the rtol=1e-6
    # bar; only the rare near-tied flips exceed it (observed ~1-2% of elements on
    # a T4). The RMSE check below is the real quality gate — a scan that actually
    # drifted (wrong, not near-tied, splits) would reroute far more rows AND
    # degrade RMSE past its 5% guard — so this bound stays loose to avoid flaking
    # on GPU run-to-run flip variation.
    within = np.abs(cu_pred - np_pred) <= 1e-8 + 1e-6 * np.abs(np_pred)
    assert np.mean(~within) < 0.10
    # Quality-equivalent: flipped splits are near-tied, so test RMSE matches.
    rmse_np = float(np.sqrt(np.mean((np_pred - Yte) ** 2)))
    rmse_cu = float(np.sqrt(np.mean((cu_pred - Yte) ** 2)))
    assert abs(rmse_cu - rmse_np) < 0.05 * rmse_np


def test_multioutput_end_to_end_categoricals_and_missing():
    """Multi-output routes categoricals as ordered thresholds (no subset split);
    the device scan must still agree end-to-end with categoricals + missing."""
    rng = np.random.default_rng(13)
    n = 1200
    cat = rng.choice(list("abcde"), size=n).astype(object)
    cat[rng.random(n) < 0.05] = None
    x = rng.normal(size=n)
    high = pd.Series(cat).isin(["a", "d"]).to_numpy()
    y0 = np.where(high, 3.0, -3.0) + x
    y1 = np.where(high, -1.0, 2.0) - 0.5 * x
    Y = np.column_stack([y0, y1]) + rng.normal(0, 0.2, (n, 2))
    df = pd.DataFrame({"c": cat, "x": x})

    preds = {}
    for backend in ("numpy", "cuda"):
        train = RepLeafDataset(df, Y, categorical_features=["c"])
        model = RepLeafRegressor(
            n_estimators=20, num_leaves=6, min_samples_leaf=10,
            min_data_per_group=20, split_backend=backend, random_state=42,
        )
        model.fit(train)
        preds[backend] = model.predict(df)
    assert preds["numpy"].shape == (n, 2)
    np.testing.assert_allclose(preds["numpy"], preds["cuda"], rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("scan_min_cells", ["0", "1000000000"])  # device vs host
@pytest.mark.parametrize("weighted", [False, True])
def test_multioutput_split_scan_device_vs_host_parity(
    monkeypatch, scan_min_cells, weighted
):
    """Unit parity of the multi-output summed-gain scan vs the NumPy reference,
    on both the on-device scan (threshold 0) and the host small-scan fallback
    (huge threshold), with unweighted (h==count) and weighted (h!=count) hess."""
    monkeypatch.setenv("REPLEAFGBM_CUDA_SCAN_MIN_CELLS", scan_min_cells)
    rng = np.random.default_rng(15)
    n, F, n_bins_max, K = 4000, 6, 33, 3
    binned = rng.integers(0, 32, size=(n, F)).astype(np.uint16)
    binned[rng.random((n, F)) < 0.05] = 32  # missing bin
    rows = np.sort(rng.choice(n, size=2500, replace=False)).astype(np.int64)
    n_bins_pf = np.full(F, 32, dtype=np.int64)
    grad = rng.normal(size=(n, K))
    hess = (
        np.abs(rng.normal(size=(n, K))) + 0.1 if weighted else np.ones((n, K))
    )
    np_b, cu_b = NumPySplitBackend(), CudaSplitBackend()
    hist_np = np_b.build_histograms_multioutput(binned, rows, grad, hess, n_bins_max)
    hist_cu = cu_b.build_histograms_multioutput(binned, rows, grad, hess, n_bins_max)
    assert hist_cu.shape == (F, n_bins_max, 3, K)
    s_np = np_b.find_best_split_multioutput(hist_np, n_bins_pf, 20, 1.0)
    s_cu = cu_b.find_best_split_multioutput(hist_cu, n_bins_pf, 20, 1.0)
    assert (s_np is None) == (s_cu is None)
    assert s_np is not None  # the constructed node has a valid split
    # Exact (feature, bin) equality holds here because this fixture's winning
    # split is not near-tied; on a near-tied node the on-device CuPy reduction
    # could flip the argmax to an equally-good split (see
    # test_multioutput_device_scan_quality_matches). The gain is what is
    # guaranteed allclose either way.
    assert (s_np.feature, s_np.bin) == (s_cu.feature, s_cu.bin)
    assert s_np.gain == pytest.approx(s_cu.gain, rel=1e-6)
    assert (s_np.n_left, s_np.n_right) == (s_cu.n_left, s_cu.n_right)


def test_multioutput_histogram_resident_and_subtractable():
    """The multi-output build returns a resident CuPy (F, B, 3, K) array whose
    sibling subtraction (parent - child == sibling) holds to float noise, so the
    grower keeps it on-device as on the scalar path."""
    rng = np.random.default_rng(17)
    n, F, n_bins_max, K = 3000, 5, 33, 3
    binned = rng.integers(0, 32, size=(n, F)).astype(np.uint16)
    binned[rng.random((n, F)) < 0.05] = 32
    rows = np.sort(rng.choice(n, size=2000, replace=False)).astype(np.int64)
    grad = rng.normal(size=(n, K))
    hess = np.abs(rng.normal(size=(n, K))) + 0.1
    cu = CudaSplitBackend()
    half = rows.size // 2
    left, right = rows[:half], rows[half:]
    h_parent = cu.build_histograms_multioutput(binned, rows, grad, hess, n_bins_max)
    h_left = cu.build_histograms_multioutput(binned, left, grad, hess, n_bins_max)
    h_right = cu.build_histograms_multioutput(binned, right, grad, hess, n_bins_max)
    assert isinstance(h_parent, cp.ndarray)
    assert h_parent.shape == (F, n_bins_max, 3, K)
    np.testing.assert_allclose(
        cp.asnumpy(h_parent), cp.asnumpy(h_left + h_right), rtol=1e-9, atol=1e-9
    )
    np.testing.assert_allclose(
        cp.asnumpy(h_parent - h_left), cp.asnumpy(h_right), rtol=1e-9, atol=1e-9
    )


def test_multioutput_device_scan_gate_off_matches_host(monkeypatch):
    """With the kill switch off (REPLEAFGBM_CUDA_MO_DEVICE_SCAN=0) the build
    returns a host stack and the scan delegates to the host reference — the
    pre-device behavior — without touching NumPy."""
    monkeypatch.setenv("REPLEAFGBM_CUDA_MO_DEVICE_SCAN", "0")
    rng = np.random.default_rng(19)
    n, F, n_bins_max, K = 2000, 5, 33, 2
    binned = rng.integers(0, 32, size=(n, F)).astype(np.uint16)
    rows = np.arange(n, dtype=np.int64)
    grad = rng.normal(size=(n, K))
    hess = np.ones((n, K))
    n_bins_pf = np.full(F, 32, dtype=np.int64)

    cu = CudaSplitBackend()
    assert cu._mo_device_scan is False
    hist_cu = cu.build_histograms_multioutput(binned, rows, grad, hess, n_bins_max)
    assert not isinstance(hist_cu, cp.ndarray)  # host stack, not resident
    s_cu = cu.find_best_split_multioutput(hist_cu, n_bins_pf, 20, 1.0)

    np_b = NumPySplitBackend()
    hist_np = np_b.build_histograms_multioutput(binned, rows, grad, hess, n_bins_max)
    s_np = np_b.find_best_split_multioutput(hist_np, n_bins_pf, 20, 1.0)
    assert (s_np.feature, s_np.bin) == (s_cu.feature, s_cu.bin)
    assert s_np.gain == pytest.approx(s_cu.gain, rel=1e-6)


def test_multioutput_scan_counters(monkeypatch):
    """The multi-output scan reuses the transfer counters: a small histogram
    copies the whole (F, B, 3, K) stack back (host fallback); a large one keeps
    it resident and only the 32-byte winner crosses back."""
    rng = np.random.default_rng(21)
    K = 3

    # Small node (F * n_bins_max below the default threshold): host small-scan.
    n, F, n_bins_max = 2000, 8, 33
    assert F * n_bins_max < 32_768
    binned = rng.integers(0, 32, size=(n, F)).astype(np.uint16)
    rows = np.arange(n, dtype=np.int64)
    grad = rng.normal(size=(n, K))
    hess = np.ones((n, K))
    n_bins_pf = np.full(F, 32, dtype=np.int64)
    cu = CudaSplitBackend()
    hist = cu.build_histograms_multioutput(binned, rows, grad, hess, n_bins_max)
    cu.reset_transfer_stats()
    cu.find_best_split_multioutput(hist, n_bins_pf, 20, 1.0)
    s = cu.get_transfer_stats()
    assert s["n_small_scans"] == 1
    assert s["hist_d2h_bytes"] == 24 * F * n_bins_max * K
    assert s["n_gpu_scans"] == 0 and s["winner_d2h_bytes"] == 0

    # Force the on-device scan (threshold 0): only the winner pack crosses back.
    monkeypatch.setenv("REPLEAFGBM_CUDA_SCAN_MIN_CELLS", "0")
    cu = CudaSplitBackend()
    hist = cu.build_histograms_multioutput(binned, rows, grad, hess, n_bins_max)
    cu.reset_transfer_stats()
    cu.find_best_split_multioutput(hist, n_bins_pf, 20, 1.0)
    s = cu.get_transfer_stats()
    assert s["n_gpu_scans"] == 1 and s["winner_d2h_bytes"] == 32
    assert s["hist_d2h_bytes"] == 0 and s["n_small_scans"] == 0


# --------------------------------------------------------------------------- #
# Node-batched depthwise scan (REPLEAFGBM_CUDA_BATCHED_SCAN; Stage 2)
# --------------------------------------------------------------------------- #
def _force_batched_gpu(cu):
    """Force the on-device batched scan even for a small batch (test-only)."""
    cu._batched_scan = True
    cu._scan_min_cells = 0
    return cu


def test_batched_scan_device_matches_reference(node_data):
    """Device batched scan agrees with the NumPy per-node reference (allclose)."""
    binned, rows, grad, hess, n_bins_max, n_bins_pf = node_data
    np_b = NumPySplitBackend()
    cu = _force_batched_gpu(CudaSplitBackend())
    subsets = [rows[:800], rows[400:1400], rows[600:], rows[::2]]
    ref = [
        np_b.find_best_split(
            np_b.build_histograms(binned, r, grad, hess, n_bins_max),
            n_bins_pf, 20, 1.0,
        )
        for r in subsets
    ]
    hists_cu = [cu.build_histograms(binned, r, grad, hess, n_bins_max) for r in subsets]
    got = cu.find_best_split_batched(hists_cu, n_bins_pf, 20, 1.0)
    assert len(got) == len(ref)
    for a, b in zip(got, ref):
        assert (a is None) == (b is None)
        if a is not None:
            assert a.feature == b.feature and a.bin == b.bin
            np.testing.assert_allclose(a.gain, b.gain, rtol=1e-6, atol=1e-9)
            assert a.n_left == b.n_left and a.n_right == b.n_right


def test_batched_scan_with_categoricals_matches_reference(node_data):
    """Batched scan with a categorical feature (host subset scan per node)."""
    binned, rows, grad, hess, n_bins_max, n_bins_pf = node_data
    cat_mask = np.zeros(8, dtype=bool)
    cat_mask[3] = True
    np_b = NumPySplitBackend()
    cu = _force_batched_gpu(CudaSplitBackend())
    subsets = [rows[:900], rows[300:], rows[::2]]
    ref = [
        np_b.find_best_split(
            np_b.build_histograms(binned, r, grad, hess, n_bins_max),
            n_bins_pf, 20, 1.0, cat_mask,
        )
        for r in subsets
    ]
    hists_cu = [cu.build_histograms(binned, r, grad, hess, n_bins_max) for r in subsets]
    got = cu.find_best_split_batched(hists_cu, n_bins_pf, 20, 1.0, cat_mask)
    for a, b in zip(got, ref):
        assert (a is None) == (b is None)
        if a is not None:
            assert a.feature == b.feature and a.bin == b.bin
            np.testing.assert_allclose(a.gain, b.gain, rtol=1e-6, atol=1e-9)


def test_batched_scan_on_by_default(node_data, monkeypatch):
    """Default (env unset) → batched scan on; the grower takes the device path."""
    monkeypatch.delenv("REPLEAFGBM_CUDA_BATCHED_SCAN", raising=False)
    cu = CudaSplitBackend()  # default: _batched_scan True, supports_batched_scan True
    assert cu.supports_batched_scan is True


def test_batched_scan_kill_switch_loops_per_node(node_data, monkeypatch):
    """Kill switch (REPLEAFGBM_CUDA_BATCHED_SCAN=0) → the base per-node loop, exactly."""
    monkeypatch.setenv("REPLEAFGBM_CUDA_BATCHED_SCAN", "0")
    binned, rows, grad, hess, n_bins_max, n_bins_pf = node_data
    cu = CudaSplitBackend()  # env=0 → _batched_scan False, supports_batched_scan False
    assert cu.supports_batched_scan is False
    hists = [cu.build_histograms(binned, rows[:1000], grad, hess, n_bins_max)]
    loop = [cu.find_best_split(hists[0], n_bins_pf, 20, 1.0)]
    batched = cu.find_best_split_batched(hists, n_bins_pf, 20, 1.0)
    assert len(batched) == 1
    a, b = batched[0], loop[0]
    assert (a is None) == (b is None)
    if a is not None:
        assert a.feature == b.feature and a.bin == b.bin


def test_depthwise_batched_e2e_quality_matches(monkeypatch):
    """Depthwise cuda fit: batched device scan vs per-node is quality-equivalent."""
    from sklearn.metrics import r2_score

    from repleafgbm import RepLeafRegressor

    rng = np.random.default_rng(7)
    X = rng.normal(size=(4000, 12))
    y = 2 * X[:, 0] + np.sin(X[:, 1]) - 1.5 * (X[:, 2] > 0) + rng.normal(scale=0.1, size=4000)
    common = dict(
        grow_policy="depthwise", max_depth=5, num_leaves=63, n_estimators=20,
        leaf_model="constant", split_backend="cuda", random_state=0,
    )
    # off = explicit kill switch (per-node loop); on = the new default (env unset).
    monkeypatch.setenv("REPLEAFGBM_CUDA_BATCHED_SCAN", "0")
    off = RepLeafRegressor(**common).fit(X, y)
    monkeypatch.delenv("REPLEAFGBM_CUDA_BATCHED_SCAN", raising=False)
    on = RepLeafRegressor(**common).fit(X, y)
    # Near-tied splits can flip on-device (allclose, not bitwise): assert
    # quality-equivalence, not identical trees.
    assert abs(r2_score(y, off.predict(X)) - r2_score(y, on.predict(X))) < 5e-3
