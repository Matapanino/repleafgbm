"""Seeded Gaussian random Fourier spaces for an RBF kernel."""

from __future__ import annotations

import numpy as np

from repleafgbm.encoders.base import BaseEncoder
from repleafgbm.utils.random import check_random_state


class RandomFourierEncoder(BaseEncoder):
    """Return cosines then sines of joint standardized random projections.

    ``n_components=16``, ``gamma='median'``, and seed 0 are unmeasured, chosen
    defaults. Output width is twice n_components; features are divided by
    ``sqrt(n_components)``. Frequencies follow N(0, 2*gamma), with uniform
    phases in [0, 2*pi).
    The median heuristic sets ``gamma = 1 / (2 * median_distance**2)`` over
    unordered distinct row pairs from at most 2,048 seeded sampled rows;
    2,048 is a chosen, unmeasured compute cap. A zero median uses gamma=1,
    a chosen fallback. Explicit gamma must be finite and positive.
    NaNs are mean-imputed in standardized space; entirely missing or constant
    columns use scale 1. Median mode requires two rows, fixed gamma one row.
    Targets/weights are ignored; the model seed is injected by make_encoder.
    """

    name = "rff"

    def __init__(
        self, n_components: int = 16, gamma: str | float = "median",
        random_state: int | None = 0,
    ) -> None:
        if not isinstance(n_components, (int, np.integer)) or n_components < 1:
            raise ValueError("n_components must be an integer >= 1")
        if isinstance(gamma, str):
            if gamma != "median":
                raise ValueError("gamma must be 'median' or a finite positive number")
        elif not np.isscalar(gamma) or not np.isfinite(gamma) or gamma <= 0:
            raise ValueError("gamma must be 'median' or a finite positive number")
        if random_state is not None and not isinstance(random_state, (int, np.integer)):
            raise ValueError("random_state must be an integer or None")
        self.n_components = int(n_components)
        self.gamma = gamma if isinstance(gamma, str) else float(gamma)
        self.random_state = None if random_state is None else int(random_state)
        self.frequencies_: np.ndarray | None = None

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
    ) -> RandomFourierEncoder:
        X_num = self._validate(X_num)
        minimum = 2 if self.gamma == "median" else 1
        if len(X_num) < minimum:
            raise ValueError(f"RandomFourierEncoder requires at least {minimum} training rows")
        count = np.maximum((~np.isnan(X_num)).sum(axis=0), 1)
        self.mean_ = np.nansum(X_num, axis=0) / count
        centered = np.where(np.isnan(X_num), 0.0, X_num - self.mean_)
        std = np.sqrt(np.sum(centered**2, axis=0) / count)
        self.scale_ = np.where(std > 0, std, 1.0)
        standardized = centered / self.scale_
        rng = check_random_state(self.random_state)
        gamma = self.gamma
        if gamma == "median":
            sample = standardized
            if len(sample) > 2048:
                sample = sample[rng.choice(len(sample), 2048, replace=False)]
            norms = np.sum(sample**2, axis=1)
            squared = np.maximum(norms[:, None] + norms[None, :] - 2 * (sample @ sample.T), 0)
            distance = np.median(np.sqrt(squared[np.triu_indices(len(sample), k=1)]))
            gamma = 1.0 / (2.0 * distance**2) if distance > 0 else 1.0
        self.gamma_ = np.asarray(gamma, dtype=np.float64)
        self.frequencies_ = rng.normal(
            0.0, np.sqrt(2.0 * self.gamma_), size=(X_num.shape[1], self.n_components)
        )
        self.phases_ = rng.uniform(0.0, 2.0 * np.pi, size=self.n_components)
        return self

    def transform(self, X_num: np.ndarray) -> np.ndarray:
        self._check_fitted("frequencies_")
        X_num = self._validate(X_num)
        if X_num.shape[1] != self.mean_.size:
            raise ValueError(f"Expected {self.mean_.size} numerical features, got {X_num.shape[1]}")
        standardized = (np.where(np.isnan(X_num), self.mean_, X_num) - self.mean_) / self.scale_
        angles = standardized @ self.frequencies_ + self.phases_
        return np.concatenate([np.cos(angles), np.sin(angles)], axis=1) / np.sqrt(self.n_components)

    @property
    def output_dim(self) -> int:
        self._check_fitted("frequencies_")
        return int(2 * self.frequencies_.shape[1])

    def get_config(self) -> dict:
        return {"n_components": self.n_components, "gamma": self.gamma,
                "random_state": self.random_state}

    def get_state(self) -> dict[str, np.ndarray]:
        self._check_fitted("frequencies_")
        return {"mean": self.mean_, "scale": self.scale_, "frequencies": self.frequencies_,
                "phases": self.phases_, "gamma": self.gamma_}

    def set_state(self, state: dict[str, np.ndarray]) -> None:
        self.mean_ = np.asarray(state["mean"], dtype=np.float64)
        self.scale_ = np.asarray(state["scale"], dtype=np.float64)
        self.frequencies_ = np.asarray(state["frequencies"], dtype=np.float64)
        self.phases_ = np.asarray(state["phases"], dtype=np.float64)
        self.gamma_ = np.asarray(state["gamma"], dtype=np.float64)
