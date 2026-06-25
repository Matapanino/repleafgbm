"""Splitter: orchestrates split search on binned raw features.

The Splitter owns the binned matrix and per-feature thresholds for one tree
fit, delegates the numeric scan to a backend kernel, and converts winning bin
indices back to real-valued thresholds for the stored tree.
"""

from __future__ import annotations

import numpy as np

from repleafgbm.backends import numpy_backend as _nb
from repleafgbm.backends.base import BaseSplitBackend, SplitCandidate, _as_host
from repleafgbm.backends.numpy_backend import NumPySplitBackend
from repleafgbm.core.histogram import bin_features, compute_bin_thresholds
from repleafgbm.core.profiling import PhaseProfiler, timed


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
        profiler: PhaseProfiler | None = None,
    ) -> None:
        self.min_samples_leaf = min_samples_leaf
        self.l2 = l2
        self.cat_smooth = cat_smooth
        self.min_data_per_group = min_data_per_group
        self.max_cat_threshold = max_cat_threshold
        self.backend = backend or NumPySplitBackend()
        #: Optional per-phase profiler (None disables timing; see core.profiling).
        self._profiler = profiler
        n_features = X_raw.shape[1]
        self.is_categorical = np.zeros(n_features, dtype=bool)
        for f in categorical_indices or []:
            col = X_raw[:, f]
            valid = col[~np.isnan(col)]
            if valid.size and int(valid.max()) + 1 <= max_bins:
                self.is_categorical[f] = True

        with timed(self._profiler, "binning"):
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
        """Node histogram; see backends.base.

        Scalar grad/hess give the ``(n_features, n_bins_max, 3)`` layout. For
        multi-output regression grad/hess are ``(n_rows, n_outputs)`` and the
        per-output scalar histograms are stacked into
        ``(n_features, n_bins_max, 3, n_outputs)`` — sibling subtraction stays
        valid (it is linear in the stacked array). The backend owns the stack
        (``build_histograms_multioutput``): the host default reuses the scalar
        kernel per output, while a device backend (CUDA) keeps it resident.
        """
        with timed(self._profiler, "histogram"):
            if grad.ndim == 1:
                return self.backend.build_histograms(
                    self.binned, rows, grad, hess, self.n_bins_max
                )
            return self.backend.build_histograms_multioutput(
                self.binned, rows, grad, hess, self.n_bins_max
            )

    def find_best_split(self, hist: np.ndarray) -> SplitCandidate | None:
        with timed(self._profiler, "split_scan"):
            if hist.ndim == 4:  # multi-output: shared-routing numerical scan
                return self.backend.find_best_split_multioutput(
                    hist, self.n_bins_per_feature, self.min_samples_leaf, self.l2
                )
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

    def find_best_split_batched(
        self, hists: list[np.ndarray]
    ) -> list[SplitCandidate | None]:
        """Best split for each node of a depthwise level, in ONE backend call.

        ``hists`` is one scalar histogram per frontier node (numeric + categorical
        subset scan). Mirrors :meth:`find_best_split` but hands the whole batch to
        the backend so a device backend can scan it in a single kernel launch; the
        host default loops the per-node scan (bitwise-identical). Scalar targets
        only — the grower keeps the per-node path for multi-output.
        """
        with timed(self._profiler, "split_scan"):
            return self.backend.find_best_split_batched(
                hists,
                self.n_bins_per_feature,
                self.min_samples_leaf,
                self.l2,
                categorical_mask=self.is_categorical,
                cat_smooth=self.cat_smooth,
                min_data_per_group=self.min_data_per_group,
                max_cat_threshold=self.max_cat_threshold,
            )

    def find_best_level_split(
        self, hists: list[np.ndarray]
    ) -> tuple[int, int] | None:
        """Shared ``(feature, bin)`` for one symmetric-tree level (host-side).

        Maximizes the summed per-node gain across ``hists`` (one histogram per
        node at the level); see
        :func:`~repleafgbm.backends.numpy_backend.find_best_level_split`. Runs on
        the host for every backend (device-resident histograms are pulled with
        ``_as_host``), so NumPy and Rust agree automatically — there is no
        per-backend kernel here.
        """
        with timed(self._profiler, "split_scan"):
            return _nb.find_best_level_split(
                [_as_host(h) for h in hists],
                self.n_bins_per_feature,
                self.min_samples_leaf,
                self.l2,
            )

    def split_at(
        self, hist: np.ndarray, feature: int, bin_: int
    ) -> SplitCandidate:
        """Per-node SplitCandidate at a fixed numeric ``(feature, bin)``.

        Symmetric growth applies a level's shared rule to every node; see
        :func:`~repleafgbm.backends.numpy_backend.split_at`.
        """
        return _nb.split_at(
            _as_host(hist), feature, bin_, self.n_bins_per_feature, self.l2
        )

    def threshold_value(self, split: SplitCandidate) -> float:
        """Real-valued threshold for a winning bin split (x <= t goes left).

        Numerical features map the winning bin to its quantile threshold.
        Categorical features have no threshold array: in single-output trees
        they always produce subset splits (handled via ``left_categories``),
        but multi-output trees route them as ordered thresholds on the ordinal
        code, where the code value itself is the threshold (``code <= bin``).
        """
        if self.is_categorical[split.feature]:
            return float(split.bin)
        return float(self.thresholds[split.feature][split.bin])

    def partition(self, rows: np.ndarray, split: SplitCandidate) -> tuple[np.ndarray, np.ndarray]:
        """Partition rows into (left, right); missing values go left.

        Delegates the row routing to the split backend. The NumPy reference
        lives on :class:`BaseSplitBackend`; the Rust backend overrides it with a
        fused single-pass native kernel. Both preserve the input row order, so
        the children are identical across backends.
        """
        with timed(self._profiler, "partition"):
            missing_bin = int(self.n_bins_per_feature[split.feature])
            return self.backend.partition_rows(self.binned, rows, split, missing_bin)
