"""Interpolated empirical-rank spaces with optional normal scores."""

from __future__ import annotations

import numpy as np

from repleafgbm.encoders.base import BaseEncoder


def _normal_ppf(p: np.ndarray) -> np.ndarray:
    """Acklam's rational inverse-normal approximation, evaluated with NumPy.

    The coefficients and 0.02425 branch point belong to the approximation,
    rather than model hyperparameters. Callers supply probabilities in (0, 1).
    """
    a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
         1.383577518672690e2, -3.066479806614716e1, 2.506628277459239]
    b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
         6.680131188771972e1, -1.328068155288572e1, 1.0]
    c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838,
         -2.549732539343734, 4.374664141464968, 2.938163982698783]
    d = [7.784695709041462e-3, 3.224671290700398e-1,
         2.445134137142996, 3.754408661907416, 1.0]
    tail = (p < 0.02425) | (p > 1.0 - 0.02425)
    out = np.empty_like(p)
    q = p[~tail] - 0.5
    r = q * q
    out[~tail] = q * np.polyval(a, r) / np.polyval(b, r)
    q = np.sqrt(-2.0 * np.log(np.minimum(p[tail], 1.0 - p[tail])))
    out[tail] = np.polyval(c, q) / np.polyval(d, q) * np.where(p[tail] < 0.5, 1, -1)
    return out


class QuantileEncoder(BaseEncoder):
    """Map columns to interpolated ECDF ranks using a fitted quantile table.

    ``n_quantiles=100`` and ``output='uniform'`` are unmeasured, chosen
    defaults. Quantiles span [0, 1]; ties get the midpoint of their rank
    interval. Normal scores use an inverse-CDF approximation and a chosen,
    unmeasured endpoint clip of 1e-7 (about +/-5.2 normal units). NaNs produce
    zero. All-missing columns use a zero quantile table. Fit needs at least one
    row and column; targets/weights are ignored. Only the quantile table is
    fitted state.
    """

    name = "quantile"

    def __init__(self, n_quantiles: int = 100, output: str = "uniform") -> None:
        if not isinstance(n_quantiles, (int, np.integer)) or n_quantiles < 2:
            raise ValueError("n_quantiles must be an integer >= 2")
        if output not in ("uniform", "normal"):
            raise ValueError("output must be 'uniform' or 'normal'")
        self.n_quantiles = int(n_quantiles)
        self.output = output
        self.quantiles_: np.ndarray | None = None

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
    ) -> QuantileEncoder:
        X_num = self._validate(X_num)
        if X_num.shape[0] == 0:
            raise ValueError("QuantileEncoder requires at least one training row")
        qs = np.linspace(0.0, 1.0, self.n_quantiles)
        self.quantiles_ = np.zeros((X_num.shape[1], self.n_quantiles), dtype=np.float64)
        for j in range(X_num.shape[1]):
            valid = X_num[~np.isnan(X_num[:, j]), j]
            if valid.size:
                self.quantiles_[j] = np.quantile(valid, qs)
        return self

    def transform(self, X_num: np.ndarray) -> np.ndarray:
        self._check_fitted("quantiles_")
        X_num = self._validate(X_num)
        if X_num.shape[1] != self.quantiles_.shape[0]:
            raise ValueError(
                f"Expected {self.quantiles_.shape[0]} numerical features, got {X_num.shape[1]}"
            )
        Z = np.zeros((len(X_num), self.output_dim), dtype=np.float64)
        qs = np.linspace(0.0, 1.0, self.n_quantiles)
        for j, table in enumerate(self.quantiles_):
            col = X_num[:, j]
            rank = 0.5 * (np.interp(col, table, qs)
                          - np.interp(-col, -table[::-1], -qs[::-1]))
            if self.output == "normal":
                rank = _normal_ppf(np.clip(rank, 1e-7, 1.0 - 1e-7))
            Z[:, j] = rank
            Z[np.isnan(col), j] = 0.0
        return Z

    @property
    def output_dim(self) -> int:
        self._check_fitted("quantiles_")
        return int(self.quantiles_.shape[0])

    def get_config(self) -> dict:
        return {"n_quantiles": self.n_quantiles, "output": self.output}

    def get_state(self) -> dict[str, np.ndarray]:
        self._check_fitted("quantiles_")
        return {"quantiles": self.quantiles_}

    def set_state(self, state: dict[str, np.ndarray]) -> None:
        self.quantiles_ = np.asarray(state["quantiles"], dtype=np.float64)
