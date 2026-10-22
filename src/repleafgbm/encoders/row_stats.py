"""Stateless row aggregates over ordered numerical columns."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from repleafgbm.encoders.base import BaseEncoder


class RowStatsEncoder(BaseEncoder):
    """Return mean, std, min, max, zero count, missing count, slope, half gap.

    ``columns=None`` selects the full numerical block; explicit indices select
    and order the block (not the raw matrix, which may include categoricals).
    No standardization is applied: statistics describe each row's own units,
    and the leaf ridge fit absorbs their scales. NaNs are excluded from
    aggregates and counted separately; an empty aggregate is zero. Standard
    deviation uses ddof=0. Slope fits observed values against their positions
    in the selected block, preserving gaps due to missing values; fewer than
    two observations give slope zero. The half gap is first-half mean minus
    second-half mean, split at ``n_columns // 2``; a single column gives zero.
    These conventions are chosen, unmeasured defaults. All-missing rows give
    seven zeros plus the missing count. Fit only validates nonempty input;
    there is no fitted state or fixed input width, and targets/weights are
    ignored. Transform can also be used directly.
    """

    name = "row_stats"

    def __init__(self, columns: Sequence[int] | None = None) -> None:
        if columns is None:
            self.columns = None
            return
        try:
            values = tuple(columns)
        except TypeError as exc:
            raise ValueError("columns must be a non-empty sequence of integer indices") from exc
        if (not values or any(not isinstance(i, (int, np.integer)) or i < 0 for i in values)
                or len(set(values)) != len(values)):
            raise ValueError("columns must contain distinct, nonnegative integer indices")
        self.columns = tuple(int(i) for i in values)

    def _select(self, X_num: np.ndarray) -> np.ndarray:
        X_num = np.asarray(X_num, dtype=np.float64)
        if X_num.ndim != 2:
            raise ValueError("X_num must be 2-D")
        if X_num.shape[1] == 0:
            raise ValueError("X_num must contain at least one numerical column")
        if self.columns is not None:
            if max(self.columns) >= X_num.shape[1]:
                raise ValueError(
                    f"columns index {max(self.columns)} exceeds the {X_num.shape[1]} "
                    "columns in the numerical-feature block"
                )
            X_num = X_num[:, self.columns]
        if np.isinf(X_num).any():
            raise ValueError("Selected X_num columns must contain finite values or NaN")
        return X_num

    def fit(
        self, X_num: np.ndarray, y: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> RowStatsEncoder:
        X_num = self._select(X_num)
        if X_num.shape[0] == 0:
            raise ValueError("RowStatsEncoder requires at least one training row")
        return self

    def transform(self, X_num: np.ndarray) -> np.ndarray:
        X_num = self._select(X_num)
        valid = ~np.isnan(X_num)
        count = valid.sum(axis=1)
        denominator = np.maximum(count, 1)
        values = np.where(valid, X_num, 0.0)
        mean = values.sum(axis=1) / denominator
        centered = np.where(valid, values - mean[:, None], 0.0)
        std = np.sqrt(np.sum(centered**2, axis=1) / denominator)
        minimum = np.where(count > 0, np.where(valid, values, np.inf).min(axis=1), 0.0)
        maximum = np.where(count > 0, np.where(valid, values, -np.inf).max(axis=1), 0.0)
        positions = np.arange(X_num.shape[1], dtype=np.float64)
        position_mean = (valid * positions).sum(axis=1) / denominator
        dx = np.where(valid, positions - position_mean[:, None], 0.0)
        ss = np.sum(dx**2, axis=1)
        slope = np.divide(np.sum(dx * centered, axis=1), ss,
                          out=np.zeros(len(X_num), dtype=np.float64), where=ss > 0)
        split = X_num.shape[1] // 2
        gap = np.zeros(len(X_num), dtype=np.float64)
        if split:
            left = values[:, :split].sum(axis=1) / np.maximum(valid[:, :split].sum(axis=1), 1)
            right = values[:, split:].sum(axis=1) / np.maximum(valid[:, split:].sum(axis=1), 1)
            gap = left - right
        return np.column_stack([mean, std, minimum, maximum, (valid & (values == 0)).sum(axis=1),
                                (~valid).sum(axis=1), slope, gap])

    @property
    def output_dim(self) -> int:
        return 8

    def get_config(self) -> dict:
        return {"columns": None if self.columns is None else list(self.columns)}

    def get_state(self) -> dict[str, np.ndarray]:
        return {}

    def set_state(self, state: dict[str, np.ndarray]) -> None:
        if state:
            raise ValueError("RowStatsEncoder has no fitted state; pass an empty state")
