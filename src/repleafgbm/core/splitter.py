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
        categorical_indices: Columns holding ordinal category codes. They get
            one bin per category and gradient-sorted *subset* splits instead
            of ordered thresholds. Features with more than ``max_bins``
            categories silently fall back to the ordered treatment.
        cat_smooth / min_data_per_group / max_cat_threshold: Categorical
            overfitting guards (LightGBM semantics and defaults); see
            ``BaseSplitBackend.find_best_split``.
    """

    def __init__(
        self,
        X_raw: np.ndarray,
        max_bins: int = 256,
        min_samples_leaf: int = 1,
        l2: float = 1.0,
        backend: BaseSplitBackend | None = None,
        categorical_indices: list[int] | None = None,
        cat_smooth: float = 10.0,
        min_data_per_group: int = 100,
        max_cat_threshold: int = 32,
    ) -> None:
        self.min_samples_leaf = min_samples_leaf
        self.l2 = l2
        self.cat_smooth = cat_smooth
        self.min_data_per_group = min_data_per_group
        self.max_cat_threshold = max_cat_threshold
        self.backend = backend or NumPySplitBackend()
        n_features = X_raw.shape[1]
        self.is_categorical = np.zeros(n_features, dtype=bool)
        for f in categorical_indices or []:
            col = X_raw[:, f]
            valid = col[~np.isnan(col)]
            if valid.size and int(valid.max()) + 1 <= max_bins:
                self.is_categorical[f] = True

        # Numerical features: quantile thresholds. Categorical features:
        # the ordinal code *is* the bin (empty threshold list).
        numeric_X = X_raw.copy()
        numeric_X[:, self.is_categorical] = np.nan  # skip quantile work
        self.thresholds = compute_bin_thresholds(numeric_X, max_bins=max_bins)
        self.binned = bin_features(numeric_X, self.thresholds)
        self.n_bins_per_feature = np.array(
            [len(t) + 1 for t in self.thresholds], dtype=np.int64
        )
        for f in np.flatnonzero(self.is_categorical):
            col = X_raw[:, f]
            n_cats = int(np.nanmax(col)) + 1
            codes = np.where(np.isnan(col), n_cats, col).astype(np.uint16)
            self.binned[:, f] = codes  # missing -> bin n_cats
            self.n_bins_per_feature[f] = n_cats
            self.thresholds[f] = np.empty(0, dtype=np.float64)
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
            hist,
            self.n_bins_per_feature,
            self.min_samples_leaf,
            self.l2,
            categorical_mask=self.is_categorical,
            cat_smooth=self.cat_smooth,
            min_data_per_group=self.min_data_per_group,
            max_cat_threshold=self.max_cat_threshold,
        )

    def threshold_value(self, split: SplitCandidate) -> float:
        """Real-valued threshold for a winning bin split (x <= t goes left)."""
        return float(self.thresholds[split.feature][split.bin])

    def partition(self, rows: np.ndarray, split: SplitCandidate) -> tuple[np.ndarray, np.ndarray]:
        """Partition rows into (left, right); missing values go left."""
        b = self.binned[rows, split.feature]
        missing_bin = int(self.n_bins_per_feature[split.feature])
        if split.left_categories is not None:
            # Categorical bins are the ordinal codes themselves.
            go_left = np.isin(b, split.left_categories) | (b == missing_bin)
        else:
            go_left = (b <= split.bin) | (b == missing_bin)
        return rows[go_left], rows[~go_left]
