"""Deterministic LightGBM-style row and per-tree feature sampling."""

from __future__ import annotations

import hashlib

import numpy as np

from repleafgbm.utils.random import check_random_state


class TreeSampler:
    """Resolve row and feature subsets for successive boosting trees."""

    def __init__(
        self,
        n_rows: int,
        n_features: int,
        subsample: float,
        subsample_freq: int,
        colsample_bytree: float,
        random_state: int | np.random.Generator | None,
    ) -> None:
        if not 0.0 < subsample <= 1.0:
            raise ValueError(f"subsample must be in (0, 1], got {subsample}")
        if not isinstance(subsample_freq, (int, np.integer)) or subsample_freq < 0:
            raise ValueError(
                f"subsample_freq must be a non-negative integer, got {subsample_freq}"
            )
        if not 0.0 < colsample_bytree <= 1.0:
            raise ValueError(
                f"colsample_bytree must be in (0, 1], got {colsample_bytree}"
            )
        self.n_rows = n_rows
        self.n_features = n_features
        self.subsample = float(subsample)
        self.subsample_freq = int(subsample_freq)
        self.colsample_bytree = float(colsample_bytree)
        self.rng = check_random_state(random_state)
        self._all_rows = np.arange(n_rows, dtype=np.int64)
        self._all_features = np.arange(n_features, dtype=np.int64)
        self._rows = self._all_rows

    def rows(self, iteration: int) -> np.ndarray:
        """Rows for a round; refresh every ``subsample_freq`` rounds."""
        enabled = self.subsample < 1.0 and self.subsample_freq > 0
        if enabled and iteration % self.subsample_freq == 0:
            size = max(1, int(self.subsample * self.n_rows))
            self._rows = np.sort(
                self.rng.choice(self.n_rows, size=size, replace=False)
            ).astype(np.int64, copy=False)
        return self._rows

    def features(self) -> np.ndarray:
        """Fresh feature subset for one tree."""
        if self.colsample_bytree >= 1.0:
            return self._all_features
        size = max(1, int(self.colsample_bytree * self.n_features))
        return np.sort(
            self.rng.choice(self.n_features, size=size, replace=False)
        ).astype(np.int64, copy=False)


def row_fingerprint(rows: np.ndarray) -> str:
    """Compact deterministic diagnostic for a sampled row set."""
    return hashlib.sha256(np.asarray(rows, dtype=np.int64).tobytes()).hexdigest()[:16]
