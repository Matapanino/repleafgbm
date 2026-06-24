"""Backend interface for split search kernels.

The contract is histogram-based so that tree growth can use the classic
sibling-subtraction trick: a node's histogram equals its parent's minus its
sibling's, halving histogram construction work. Native (Rust/C++/CUDA)
backends implement the same two kernels over the same memory layout.

Histogram layout: float64 array of shape ``(n_features, n_bins_max, 3)``
with channels ``(grad_sum, hess_sum, count)``. Feature ``f`` uses bin
indices ``0 .. n_bins_per_feature[f]`` where the last one holds missing
values; higher indices are zero padding so all features share one array.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class SplitCandidate:
    """Best split found for one node.

    Attributes:
        feature: Column index in the raw/binned feature matrix.
        bin: Bin index b; rows with bin <= b (plus missing) go left.
            -1 for categorical subset splits.
        gain: Newton gain of the split versus keeping the node as a leaf.
        n_left / n_right: Row counts of the resulting children.
        left_categories: For categorical splits, the category codes routed
            left (bin index == ordinal code for natively-binned categorical
            features). Missing values also go left (native convention);
            codes not in the set — including categories absent from the
            node — go right. None for numerical threshold splits.
    """

    feature: int
    bin: int
    gain: float
    n_left: int
    n_right: int
    left_categories: np.ndarray | None = None


class BaseSplitBackend(ABC):
    """Histogram construction and split scanning on binned raw features.

    Implementations see only integer bin matrices and float buffers — no
    pandas, no Python objects — so the same contract can be satisfied by
    native kernels later.
    """

    @abstractmethod
    def build_histograms(
        self,
        binned: np.ndarray,
        rows: np.ndarray,
        grad: np.ndarray,
        hess: np.ndarray,
        n_bins_max: int,
    ) -> np.ndarray:
        """Accumulate (grad, hess, count) per feature/bin for the given rows.

        Returns the ``(n_features, n_bins_max, 3)`` layout described above.
        Histograms must be exactly subtractable: ``parent - child == sibling``.

        The return may be a backend-resident device array (e.g. the CUDA backend
        keeps the histogram on the GPU across a tree's nodes); the grower treats
        it opaquely, requiring only elementwise subtraction and indexing, and
        passes it back to :meth:`find_best_split`. Subtractability holds to float
        noise on the CUDA path (allclose, not bitwise; see ADR 0005).
        """

    @abstractmethod
    def find_best_split(
        self,
        hist: np.ndarray,
        n_bins_per_feature: np.ndarray,
        min_samples_leaf: int,
        l2: float,
        categorical_mask: np.ndarray | None = None,
        cat_smooth: float = 10.0,
        min_data_per_group: int = 100,
        max_cat_threshold: int = 32,
    ) -> SplitCandidate | None:
        """Scan a node histogram for the best split, or None if no valid gain.

        ``hist`` is whatever :meth:`build_histograms` returned (a host array, or
        a device-resident array for the CUDA backend). Missing values always go
        to the left child (v0 convention). Features flagged in
        ``categorical_mask`` are scanned as subset splits (gradient-sorted
        categories) instead of ordered thresholds, governed by the three
        LightGBM-style categorical guards.
        """

    def partition_rows(
        self,
        binned: np.ndarray,
        rows: np.ndarray,
        split: SplitCandidate,
        missing_bin: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Route ``rows`` into ``(left, right)`` children by ``split``.

        Missing values (bin index ``missing_bin``) always go left (v0
        convention). Numeric splits send bins ``<= split.bin`` left; categorical
        subset splits send bins in ``split.left_categories`` left. Both children
        preserve the input row order — required so the downstream per-row
        histogram accumulation stays bitwise-parity-able across backends.

        This is the NumPy reference; native backends may override it with a
        faster kernel that produces the **same** left/right rows in the **same**
        order. Concrete (not abstract) so the NumPy and CUDA backends inherit it
        unchanged.
        """
        b = binned[rows, split.feature]
        if split.left_categories is not None:
            # Categorical bins are the ordinal codes themselves.
            go_left = np.isin(b, split.left_categories) | (b == missing_bin)
        else:
            go_left = (b <= split.bin) | (b == missing_bin)
        return rows[go_left], rows[~go_left]
