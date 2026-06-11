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

    def build_histograms(
        self,
        binned: np.ndarray,
        rows: np.ndarray,
        grad: np.ndarray,
        hess: np.ndarray,
        n_bins_max: int,
    ) -> np.ndarray:
        return self._native.build_histograms(
            np.ascontiguousarray(binned, dtype=np.uint16),
            np.ascontiguousarray(rows, dtype=np.int64),
            np.ascontiguousarray(grad, dtype=np.float64),
            np.ascontiguousarray(hess, dtype=np.float64),
            int(n_bins_max),
        )

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
