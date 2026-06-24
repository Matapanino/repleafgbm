"""Rust implementation of the split-search kernels (optional extension).

Wraps the ``repleafgbm_native`` compiled module (see ``native/``), which
implements the same two-kernel contract as the NumPy backend with matching
tie-breaking semantics. Build it with:

    pip install ./native        # requires a Rust toolchain

The two backends agree to floating-point noise; cross-backend predictions
are validated in tests/test_rust_backend.py. Determinism (same seed, same
model) holds within a backend.
"""

from __future__ import annotations

import numpy as np

from repleafgbm.backends.base import BaseSplitBackend, SplitCandidate


class RustSplitBackend(BaseSplitBackend):
    """Compiled split kernels behind the standard backend contract."""

    def __init__(self) -> None:
        try:
            import repleafgbm_native
        except ImportError as exc:
            raise ImportError(
                "The Rust backend requires the repleafgbm_native extension. "
                'Build it with: pip install ./native  (Rust toolchain needed), '
                'or use split_backend="numpy".'
            ) from exc
        self._native = repleafgbm_native
        # Cached feature-major (n_features, n_rows) copy of the binned matrix.
        # Runtime-only handle — the backend is never serialized
        # (Booster.__getstate__ drops split_backend_).
        self._binned_fmajor: np.ndarray | None = None
        self._binned_fmajor_src: np.ndarray | None = None

    def build_histograms(
        self,
        binned: np.ndarray,
        rows: np.ndarray,
        grad: np.ndarray,
        hess: np.ndarray,
        n_bins_max: int,
    ) -> np.ndarray:
        return self._native.build_histograms(
            self._feature_major(binned),
            np.ascontiguousarray(rows, dtype=np.int64),
            np.ascontiguousarray(grad, dtype=np.float64),
            np.ascontiguousarray(hess, dtype=np.float64),
            int(n_bins_max),
        )

    def _feature_major(self, binned: np.ndarray) -> np.ndarray:
        """Cached feature-major ``(n_features, n_rows)`` copy of ``binned``.

        The native kernel parallelizes over features and reads each feature's
        bins as a contiguous slice; a row-major matrix would force a strided
        per-feature gather that is memory-bound and barely scales (~1.3x). The
        transpose is computed once per binned matrix — the Splitter reuses one
        for the whole fit — and reused across every node and class. Values are
        unchanged, so NumPy/Rust histograms stay bitwise-identical.

        Invalidated by object identity; the cache also *holds* a reference to the
        source so its identity cannot be recycled by the GC while cached (a fresh
        ``binned`` is a distinct object → recompute). This keeps it correct even
        if a backend instance were ever reused across fits.
        """
        if self._binned_fmajor_src is not binned:
            self._binned_fmajor = np.ascontiguousarray(
                np.asarray(binned, dtype=np.uint16).T
            )
            self._binned_fmajor_src = binned
        return self._binned_fmajor

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
        if categorical_mask is None:
            categorical_mask = np.zeros(hist.shape[0], dtype=bool)
        result = self._native.find_best_split(
            np.ascontiguousarray(hist, dtype=np.float64),
            np.ascontiguousarray(n_bins_per_feature, dtype=np.int64),
            int(min_samples_leaf),
            float(l2),
            np.ascontiguousarray(categorical_mask, dtype=bool),
            float(cat_smooth),
            int(min_data_per_group),
            int(max_cat_threshold),
        )
        if result is None:
            return None
        feature, bin_, gain, n_left, n_right, cats = result
        return SplitCandidate(
            feature=int(feature),
            bin=int(bin_),
            gain=float(gain),
            n_left=int(n_left),
            n_right=int(n_right),
            left_categories=None if cats is None else np.asarray(cats, dtype=np.int64),
        )

    def partition_rows(
        self,
        binned: np.ndarray,
        rows: np.ndarray,
        split: SplitCandidate,
        missing_bin: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Route rows with the fused native kernel; NumPy default otherwise.

        Falls back to the bitwise-identical NumPy reference only when an older
        ``repleafgbm_native`` build predates the kernel. Inputs are normalized
        exactly as in :meth:`build_histograms` (feature-major bins, contiguous
        int64 rows), and the native kernel preserves row order so the children
        are identical to the NumPy path. The kernel beats NumPy at every node
        size (3-5x numeric, 10-15x categorical; see
        benchmarks/partition_microbench.py), so there is no min-rows gate.
        """
        native = getattr(self._native, "partition_rows", None)
        if native is None:
            return super().partition_rows(binned, rows, split, missing_bin)
        left_categories = (
            None
            if split.left_categories is None
            else np.ascontiguousarray(split.left_categories, dtype=np.int64)
        )
        return native(
            self._feature_major(binned),
            np.ascontiguousarray(rows, dtype=np.int64),
            int(split.feature),
            int(split.bin),
            left_categories,
            int(missing_bin),
        )
