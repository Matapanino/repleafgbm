"""Splitter: orchestrates split search on binned raw features.

The Splitter owns the binned matrix and per-feature thresholds for one tree
fit, delegates the numeric scan to a backend kernel, and converts winning bin
indices back to real-valued thresholds for the stored tree.
"""

from __future__ import annotations

import numpy as np

from repleafgbm.backends.base import BaseSplitBackend, SplitCandidate
from repleafgbm.backends.numpy_backend import NumPySplitBackend
from repleafgbm.core.histogram import bin_features, compute_bin_thresholds


class Splitter:
    """Finds and applies axis-aligned splits on raw features.

    Args:
        X_raw: Raw feature matrix (float64; categoricals already ordinal-encoded).
        max_bins: Maximum number of histogram bins per feature.
        min_samples_leaf: Minimum rows per child.
        l2: L2 term used in the Newton gain formula.
        backend: Split-search kernel (defaults to NumPy).
    """

    def __init__(
        self,
        X_raw: np.ndarray,
        max_bins: int = 256,
        min_samples_leaf: int = 1,
        l2: float = 1.0,
        backend: BaseSplitBackend | None = None,
    ) -> None:
        self.min_samples_leaf = min_samples_leaf
        self.l2 = l2
        self.backend = backend or NumPySplitBackend()
        self.thresholds = compute_bin_thresholds(X_raw, max_bins=max_bins)
        self.binned = bin_features(X_raw, self.thresholds)
        # Non-missing bin count per feature: len(thresholds) + 1.
        self.n_bins_per_feature = np.array(
            [len(t) + 1 for t in self.thresholds], dtype=np.int64
        )
        # Shared histogram width: widest feature's bins + its missing bin.
        self.n_bins_max = int(self.n_bins_per_feature.max()) + 1

    def build_histograms(
        self, rows: np.ndarray, grad: np.ndarray, hess: np.ndarray
    ) -> np.ndarray:
        """Node histogram (n_features, n_bins_max, 3); see backends.base."""
        return self.backend.build_histograms(
            self.binned, rows, grad, hess, self.n_bins_max
        )

    def find_best_split(self, hist: np.ndarray) -> SplitCandidate | None:
        return self.backend.find_best_split(
            hist, self.n_bins_per_feature, self.min_samples_leaf, self.l2
        )

    def threshold_value(self, split: SplitCandidate) -> float:
        """Real-valued threshold for a winning bin split (x <= t goes left)."""
        return float(self.thresholds[split.feature][split.bin])

    def partition(self, rows: np.ndarray, split: SplitCandidate) -> tuple[np.ndarray, np.ndarray]:
        """Partition rows into (left, right); missing values go left."""
        b = self.binned[rows, split.feature]
        missing_bin = int(self.n_bins_per_feature[split.feature])
        go_left = (b <= split.bin) | (b == missing_bin)
        return rows[go_left], rows[~go_left]
