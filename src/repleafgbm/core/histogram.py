"""Feature binning for histogram-based split search.

Raw features are quantized once per ``fit`` into small integer bins. Split
search then works on bin indices, which keeps per-node cost linear in the
number of rows and bins. This mirrors LightGBM-style histogram training and
gives the future Rust/C++/CUDA backends a simple data layout to target:
a (n_rows, n_features) uint16 matrix plus per-feature threshold arrays.

Bin semantics for feature j with thresholds t[0] < ... < t[k-1]:

* bin b in [0, k-1]  -> x <= t[b] (and x > t[b-1] for b > 0)
* bin k              -> x > t[k-1]
* bin k + 1          -> missing (NaN). Missing always routes left in v0.
"""

from __future__ import annotations

import numpy as np


def compute_bin_thresholds(X: np.ndarray, max_bins: int = 256) -> list[np.ndarray]:
    """Per-feature sorted candidate thresholds from quantiles of non-NaN values.

    Returns a list of float64 arrays (possibly empty for constant features).
    """
    n_features = X.shape[1]
    thresholds: list[np.ndarray] = []
    qs = np.linspace(0.0, 1.0, max_bins + 1)[1:-1]  # interior quantiles
    for j in range(n_features):
        col = X[:, j]
        valid = col[~np.isnan(col)]
        if valid.size == 0:
            thresholds.append(np.empty(0, dtype=np.float64))
            continue
        uniq = np.unique(valid)
        if uniq.size <= max_bins:
            # Midpoints between consecutive unique values are exact candidates.
            cand = (uniq[:-1] + uniq[1:]) / 2.0
        else:
            cand = np.unique(np.quantile(valid, qs))
        thresholds.append(cand.astype(np.float64))
    return thresholds


def bin_features(X: np.ndarray, thresholds: list[np.ndarray]) -> np.ndarray:
    """Quantize features into bin indices (uint16) following the semantics above."""
    n_rows, n_features = X.shape
    binned = np.empty((n_rows, n_features), dtype=np.uint16)
    for j in range(n_features):
        t = thresholds[j]
        col = X[:, j]
        missing = np.isnan(col)
        # searchsorted with side="left": x <= t[b] -> bin b; x > t[-1] -> len(t).
        b = np.searchsorted(t, col, side="left")
        b[missing] = len(t) + 1
        binned[:, j] = b.astype(np.uint16)
    return binned


def missing_bin(thresholds_j: np.ndarray) -> int:
    """Bin index reserved for missing values of one feature."""
    return len(thresholds_j) + 1
