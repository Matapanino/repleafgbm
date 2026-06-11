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
        gain: Newton gain of the split versus keeping the node as a leaf.
        n_left / n_right: Row counts of the resulting children.
    """

    feature: int
    bin: int
    gain: float
    n_left: int
    n_right: int


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
        """

    @abstractmethod
    def find_best_split(
        self,
        hist: np.ndarray,
        n_bins_per_feature: np.ndarray,
        min_samples_leaf: int,
        l2: float,
    ) -> SplitCandidate | None:
        """Scan a node histogram for the best split, or None if no valid gain.

        Missing values always go to the left child (v0 convention).
        """
