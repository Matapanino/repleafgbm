"""Identity encoder: standardized raw numerical features."""

from __future__ import annotations

import numpy as np

from repleafgbm.encoders.base import BaseEncoder


class IdentityEncoder(BaseEncoder):
    """Pass numerical features through, optionally standardized.

    Standardization (default) keeps ridge-regularized leaf models scale-free:
    a single ``l2_leaf`` value then behaves comparably across features.
    Missing values are imputed with the training mean (i.e. 0 after
    standardization).
    """

    name = "identity"

    def __init__(self, standardize: bool = True) -> None:
        self.standardize = standardize
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X_num: np.ndarray) -> IdentityEncoder:
        X_num = np.asarray(X_num, dtype=np.float64)
        self.mean_ = np.nanmean(X_num, axis=0) if X_num.size else np.zeros(X_num.shape[1])
        std = np.nanstd(X_num, axis=0) if X_num.size else np.ones(X_num.shape[1])
        self.scale_ = np.where(std > 0, std, 1.0)
        return self

    def transform(self, X_num: np.ndarray) -> np.ndarray:
        self._check_fitted("mean_")
        X_num = np.asarray(X_num, dtype=np.float64)
        # Mean imputation for missing values (0 in standardized space).
        Z = np.where(np.isnan(X_num), self.mean_, X_num)
        if self.standardize:
            Z = (Z - self.mean_) / self.scale_
        return Z

    @property
    def output_dim(self) -> int:
        self._check_fitted("mean_")
        return int(self.mean_.shape[0])

    def get_config(self) -> dict:
        return {"standardize": self.standardize}

    def get_state(self) -> dict[str, np.ndarray]:
        self._check_fitted("mean_")
        return {"mean": self.mean_, "scale": self.scale_}

    def set_state(self, state: dict[str, np.ndarray]) -> None:
        self.mean_ = np.asarray(state["mean"], dtype=np.float64)
        self.scale_ = np.asarray(state["scale"], dtype=np.float64)
