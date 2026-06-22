"""Feature-parallel binning must be bitwise-identical to the serial loop.

Binning is upstream of the split backend, so any change to its output would
ripple into the NumPy/Rust histogram parity. The thread-pool path runs the
identical per-feature NumPy calls, so it must produce exactly the same
thresholds and bin indices regardless of thread count — these tests pin that.
"""

import numpy as np
import pytest

from repleafgbm.core.histogram import bin_features, compute_bin_thresholds
from repleafgbm.utils.parallel import (
    PARALLEL_MIN_CELLS,
    resolve_n_threads,
)


@pytest.fixture
def mixed_features():
    """A wide-enough matrix to exceed the parallel gate, covering every binning
    branch: high-cardinality continuous (quantile path), low-cardinality
    (midpoint path), a constant column (empty thresholds), a fully-missing
    column, and scattered NaNs (missing bin)."""
    rng = np.random.default_rng(0)
    n, n_features = 50_000, 30
    X = rng.normal(size=(n, n_features))
    X[:, 0] = rng.integers(0, 10, size=n).astype(float)   # midpoint branch
    X[:, 1] = rng.integers(0, 200, size=n).astype(float)  # midpoint branch
    X[:, 2] = 3.14                                         # constant -> empty
    mask = rng.random((n, n_features)) < 0.02
    X[mask] = np.nan                                       # scattered missing
    X[:, 3] = np.nan                                       # fully missing
    assert n * n_features >= PARALLEL_MIN_CELLS, "fixture must exercise parallel"
    return X


def test_thresholds_thread_count_invariant(mixed_features):
    serial = compute_bin_thresholds(mixed_features, max_bins=256, n_threads=1)
    parallel = compute_bin_thresholds(mixed_features, max_bins=256, n_threads=8)
    default = compute_bin_thresholds(mixed_features, max_bins=256)
    assert len(serial) == len(parallel) == len(default) == mixed_features.shape[1]
    for a, b, c in zip(serial, parallel, default):
        np.testing.assert_array_equal(a, b)  # bitwise: identical np calls
        np.testing.assert_array_equal(a, c)


def test_bin_features_thread_count_invariant(mixed_features):
    thresholds = compute_bin_thresholds(mixed_features, max_bins=256, n_threads=1)
    serial = bin_features(mixed_features, thresholds, n_threads=1)
    parallel = bin_features(mixed_features, thresholds, n_threads=8)
    default = bin_features(mixed_features, thresholds)
    assert serial.dtype == np.uint16
    np.testing.assert_array_equal(serial, parallel)
    np.testing.assert_array_equal(serial, default)


def test_small_input_stays_serial_and_correct():
    """Below the parallel gate the serial path runs; output is still correct."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 5))
    assert X.shape[0] * X.shape[1] < PARALLEL_MIN_CELLS
    thresholds = compute_bin_thresholds(X, max_bins=16)
    binned = bin_features(X, thresholds)
    # forcing threads on tiny data must not change anything
    np.testing.assert_array_equal(
        binned, bin_features(X, thresholds, n_threads=8)
    )
    np.testing.assert_array_equal(
        binned, bin_features(X, compute_bin_thresholds(X, max_bins=16, n_threads=8))
    )


def test_resolve_n_threads_env(monkeypatch):
    monkeypatch.setenv("REPLEAFGBM_NUM_THREADS", "3")
    assert resolve_n_threads(10) == 3
    assert resolve_n_threads(2) == 2  # never more workers than features
    monkeypatch.setenv("REPLEAFGBM_NUM_THREADS", "0")  # invalid -> fall back
    assert resolve_n_threads(8) >= 1
    monkeypatch.delenv("REPLEAFGBM_NUM_THREADS", raising=False)
    assert resolve_n_threads(8) >= 1
