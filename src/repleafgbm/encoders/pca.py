"""Standardized principal-component spaces computed with NumPy SVD."""

from __future__ import annotations

import numpy as np

from repleafgbm.encoders.base import BaseEncoder


class PCAEncoder(BaseEncoder):
    """Project standardized columns onto their leading right singular vectors.

    ``n_components=8`` is an unmeasured, chosen default, clipped to the number
    of columns. Fit needs at least that many rows and at least two rows.
    NaNs are mean-imputed (zero in standardized space); entirely missing or
    constant columns use scale 1. ``whiten=False`` is the chosen default.
    Whitening gives nondegenerate training components unit sample variance;
    numerically null components are zeroed using the usual SVD rank tolerance
    ``eps * max(n_rows, n_features) * largest_singular_value``.
    The training row count is saved as scalar array metadata for whitening.
    Targets and sample weights are ignored, like the other fixed encoders.
    """

    name = "pca"

    def __init__(self, n_components: int = 8, whiten: bool = False) -> None:
        if not isinstance(n_components, (int, np.integer)) or n_components < 1:
            raise ValueError("n_components must be an integer >= 1")
        self.n_components = int(n_components)
        self.whiten = bool(whiten)
        self.components_: np.ndarray | None = None

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
    ) -> PCAEncoder:
        X_num = self._validate(X_num)
        n_rows, n_features = X_num.shape
        n_components = min(self.n_components, n_features)
        if n_rows < max(2, n_components):
            raise ValueError(f"PCAEncoder requires at least {max(2, n_components)} rows")
        count = np.maximum((~np.isnan(X_num)).sum(axis=0), 1)
        self.mean_ = np.nansum(X_num, axis=0) / count
        centered = np.where(np.isnan(X_num), 0.0, X_num - self.mean_)
        std = np.sqrt(np.sum(centered**2, axis=0) / count)
        self.scale_ = np.where(std > 0, std, 1.0)
        _, singular_values, components = np.linalg.svd(
            centered / self.scale_, full_matrices=False
        )
        self.components_ = components[:n_components].copy()
        # Fix the arbitrary SVD sign by the largest absolute loading.
        pivots = np.argmax(np.abs(self.components_), axis=1)
        self.components_ *= np.sign(self.components_[np.arange(n_components), pivots])[:, None]
        self.singular_values_ = singular_values[:n_components].copy()
        self.n_samples_ = np.asarray(n_rows, dtype=np.int64)
        return self

    def transform(self, X_num: np.ndarray) -> np.ndarray:
        self._check_fitted("components_")
        X_num = self._validate(X_num)
        if X_num.shape[1] != self.mean_.size:
            raise ValueError(f"Expected {self.mean_.size} numerical features, got {X_num.shape[1]}")
        standardized = (np.where(np.isnan(X_num), self.mean_, X_num) - self.mean_) / self.scale_
        Z = standardized @ self.components_.T
        if self.whiten:
            tolerance = (np.finfo(np.float64).eps * max(int(self.n_samples_), self.mean_.size)
                         * self.singular_values_[0])
            factors = np.divide(
                np.sqrt(self.n_samples_ - 1), self.singular_values_,
                out=np.zeros_like(self.singular_values_), where=self.singular_values_ > tolerance,
            )
            Z *= factors
        return Z

    @property
    def output_dim(self) -> int:
        self._check_fitted("components_")
        return int(self.components_.shape[0])

    def get_config(self) -> dict:
        return {"n_components": self.n_components, "whiten": self.whiten}

    def get_state(self) -> dict[str, np.ndarray]:
        self._check_fitted("components_")
        return {"mean": self.mean_, "scale": self.scale_, "components": self.components_,
                "singular_values": self.singular_values_, "n_samples": self.n_samples_}

    def set_state(self, state: dict[str, np.ndarray]) -> None:
        self.mean_ = np.asarray(state["mean"], dtype=np.float64)
        self.scale_ = np.asarray(state["scale"], dtype=np.float64)
        self.components_ = np.asarray(state["components"], dtype=np.float64)
        self.singular_values_ = np.asarray(state["singular_values"], dtype=np.float64)
        self.n_samples_ = np.asarray(state["n_samples"], dtype=np.int64)
