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


def _as_host(hist):
    """Return a NumPy view of a (possibly device-resident) histogram.

    The CUDA backend returns resident CuPy device arrays; the host multi-output
    stack/scan defaults need a NumPy array. CuPy arrays expose ``.get()`` (a host
    copy); NumPy/Rust arrays have no ``.get`` and pass through unchanged.
    """
    getter = getattr(hist, "get", None)
    return getter() if callable(getter) else hist


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

    # ----------------------------------------------------------------- #
    # Multi-output (shared-routing) split search — optional fast paths.
    #
    # Multi-output regression grows one shared tree per round whose leaves emit
    # an (n_outputs,) vector; at each node the K per-output scalar histograms are
    # stacked into (n_features, n_bins_max, 3, n_outputs) and scanned for a split
    # whose gain is the per-output Newton gain summed over outputs. These two
    # methods carry that stack + scan so a device backend can keep the stack
    # resident and scan it on-device (the CUDA override) instead of round-tripping
    # every output to the host. The base implementations reproduce the historical
    # host behavior exactly, so NumPy/Rust stay byte-for-byte unchanged.
    # ----------------------------------------------------------------- #
    def build_histograms_multioutput(
        self,
        binned: np.ndarray,
        rows: np.ndarray,
        grad: np.ndarray,
        hess: np.ndarray,
        n_bins_max: int,
    ) -> np.ndarray:
        """Stacked per-output histogram for one shared-routing node.

        ``grad``/``hess`` are ``(n_rows, n_outputs)``. Returns the
        ``(n_features, n_bins_max, 3, n_outputs)`` stack of the per-output scalar
        histograms; sibling subtraction stays valid (it is linear in the stacked
        array) and :meth:`find_best_split_multioutput` scans it.

        This default builds one scalar histogram per output via
        :meth:`build_histograms` and stacks them on the host (each pulled to the
        host first — a no-op for NumPy/Rust, a ``.get()`` for a device backend),
        reproducing the splitter's previous behavior exactly. A device backend
        (CUDA) overrides it to keep the stack resident on the GPU.
        """
        return np.stack(
            [
                _as_host(
                    self.build_histograms(
                        binned, rows, grad[:, k], hess[:, k], n_bins_max
                    )
                )
                for k in range(grad.shape[1])
            ],
            axis=-1,
        )

    def find_best_split_multioutput(
        self,
        hist: np.ndarray,
        n_bins_per_feature: np.ndarray,
        min_samples_leaf: int,
        l2: float,
    ) -> SplitCandidate | None:
        """Best shared-routing numeric split for a multi-output node.

        ``hist`` is the ``(n_features, n_bins_max, 3, n_outputs)`` stack from
        :meth:`build_histograms_multioutput`. The gain is the per-output Newton
        gain summed over outputs with a single shared left/right partition;
        missing values go left (v0). Every feature is scanned as an ordered
        threshold — multi-output trees do not produce categorical subset splits.

        This default delegates to the NumPy reference
        :func:`~repleafgbm.backends.numpy_backend.find_best_split_multioutput`
        on a host array, so NumPy/Rust stay byte-for-byte identical. The CUDA
        backend overrides it to scan the resident device array, copying back only
        the winning split's scalars.
        """
        from repleafgbm.backends.numpy_backend import (
            find_best_split_multioutput as _ref,
        )

        return _ref(_as_host(hist), n_bins_per_feature, min_samples_leaf, l2)

    # ----------------------------------------------------------------- #
    # Node-batched split search — optional fast path.
    #
    # Depthwise growth expands a whole level at once; scanning each node's
    # histogram in a separate call is launch-bound on a device backend. A backend
    # that sets ``supports_batched_scan = True`` is handed the level's M node
    # histograms together so it can find all M best splits in one kernel launch
    # (the CUDA override). The default loops :meth:`find_best_split` per node, so
    # the result is byte-for-byte the per-node path — NumPy/Rust are unchanged and
    # the grower's batched path is bitwise-identical to the per-node one.
    # ----------------------------------------------------------------- #
    #: Whether the grower may hand this backend a level's histograms as a batch.
    #: Default off → the grower keeps the per-node path; the CUDA backend flips it
    #: on only when its env gate is set (REPLEAFGBM_CUDA_BATCHED_SCAN).
    supports_batched_scan: bool = False

    def find_best_split_batched(
        self,
        hists,
        n_bins_per_feature: np.ndarray,
        min_samples_leaf: int,
        l2: float,
        categorical_mask: np.ndarray | None = None,
        cat_smooth: float = 10.0,
        min_data_per_group: int = 100,
        max_cat_threshold: int = 32,
    ) -> list[SplitCandidate | None]:
        """Best split for each of M node histograms (one per frontier node).

        ``hists`` is an iterable of M per-node histograms in the
        :meth:`find_best_split` layout. Returns an M-length list of
        ``SplitCandidate | None`` in input order. This default loops
        :meth:`find_best_split`, so it is exactly the per-node scan; a device
        backend overrides it to scan the batch in one kernel launch (the result
        must stay allclose + quality-equivalent, not necessarily bitwise — near-tied
        splits can flip via low-bit device reductions; see ADR 0005).
        """
        return [
            self.find_best_split(
                h, n_bins_per_feature, min_samples_leaf, l2,
                categorical_mask, cat_smooth, min_data_per_group, max_cat_threshold,
            )
            for h in hists
        ]

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
