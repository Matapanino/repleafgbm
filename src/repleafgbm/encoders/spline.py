"""Quantile-knot hinge spaces with an unbounded linear direction."""

from __future__ import annotations

import numpy as np

from repleafgbm.encoders.base import BaseEncoder


class SplineEncoder(BaseEncoder):
    """Encode each column as standardized hinges followed by a linear term.

    ``n_knots=4`` is an unmeasured, chosen default. Knots use equally spaced
    interior quantiles (excluding 0 and 1); repeated knots are retained.
    Each feature block is ``[max(x_std - knot, 0), ..., x_std]``, so its
    linear slot preserves extrapolation in both directions. NaNs produce
    an all-zero block, as in PLR. Entirely missing or constant columns use
    scale 1. Fit requires at least one row and one numerical column.
    Targets and sample weights are ignored.
    """

    name = "spline"

    def __init__(self, n_knots: int = 4) -> None:
        if not isinstance(n_knots, (int, np.integer)) or n_knots < 1:
            raise ValueError("n_knots must be an integer >= 1")
        self.n_knots = int(n_knots)
        self.knots_: np.ndarray | None = None

    @staticmethod
    def _validate(X_num: np.ndarray) -> np.ndarray:
        X_num = np.asarray(X_num, dtype=np.float64)
        if X_num.ndim != 2:
            raise ValueError("X_num must be 2-D")
        if X_num.shape[1] == 0:
            raise ValueError("X_num must contain at least one numerical column")
        if np.isinf(X_num).any():
            raise ValueError("X_num must contain finite values or NaN")
        return X_num

    def fit(
        self, X_num: np.ndarray, y: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> SplineEncoder:
        X_num = self._validate(X_num)
        if X_num.shape[0] == 0:
            raise ValueError("SplineEncoder requires at least one training row")
        count = np.maximum((~np.isnan(X_num)).sum(axis=0), 1)
        self.mean_ = np.nansum(X_num, axis=0) / count
        centered = np.where(np.isnan(X_num), 0.0, X_num - self.mean_)
        std = np.sqrt(np.sum(centered**2, axis=0) / count)
        self.scale_ = np.where(std > 0, std, 1.0)
        self.knots_ = np.zeros((X_num.shape[1], self.n_knots), dtype=np.float64)
        qs = np.linspace(0.0, 1.0, self.n_knots + 2)[1:-1]
        for j in range(X_num.shape[1]):
            valid = X_num[~np.isnan(X_num[:, j]), j]
            if valid.size:
                self.knots_[j] = (np.quantile(valid, qs) - self.mean_[j]) / self.scale_[j]
        return self

    def transform(self, X_num: np.ndarray) -> np.ndarray:
        self._check_fitted("knots_")
        X_num = self._validate(X_num)
        if X_num.shape[1] != self.mean_.size:
            raise ValueError(f"Expected {self.mean_.size} numerical features, got {X_num.shape[1]}")
        missing = np.isnan(X_num)
        standardized = (np.where(missing, self.mean_, X_num) - self.mean_) / self.scale_
        Z = np.empty((len(X_num), self.mean_.size, self.n_knots + 1), dtype=np.float64)
        Z[:, :, :-1] = np.maximum(standardized[:, :, None] - self.knots_, 0.0)
        Z[:, :, -1] = standardized
        Z[missing] = 0.0
        return Z.reshape(len(X_num), self.output_dim)

    @property
    def output_dim(self) -> int:
        self._check_fitted("knots_")
        return int(self.knots_.shape[0] * (self.n_knots + 1))

    def get_config(self) -> dict:
        return {"n_knots": self.n_knots}

    def get_state(self) -> dict[str, np.ndarray]:
        self._check_fitted("knots_")
        return {"knots": self.knots_, "mean": self.mean_, "scale": self.scale_}

    def set_state(self, state: dict[str, np.ndarray]) -> None:
        self.knots_ = np.asarray(state["knots"], dtype=np.float64)
        self.mean_ = np.asarray(state["mean"], dtype=np.float64)
        self.scale_ = np.asarray(state["scale"], dtype=np.float64)
