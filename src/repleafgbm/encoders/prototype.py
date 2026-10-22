"""Distances to frozen prototypes learned by NumPy Lloyd iterations."""

from __future__ import annotations

import numpy as np

from repleafgbm.encoders.base import BaseEncoder
from repleafgbm.utils.random import check_random_state


class PrototypeDistanceEncoder(BaseEncoder):
    """Represent standardized rows by Euclidean distances or RBF affinities.

    ``n_prototypes=8``, ``max_iter=20``, ``output='distance'``, and seed 0
    are unmeasured, chosen defaults. Centers start at seeded sampled rows
    without replacement; Lloyd updates retain empty-cluster centers and stop
    when centers are exactly unchanged or max_iter is reached. Fit requires
    at least n_prototypes rows. NaNs are mean-imputed (zero in standardized
    space); entirely missing or constant columns use scale 1.
    RBF output is ``exp(-d**2 / (2*sigma**2))``, with sigma the median of
    unordered distinct center-pair distances. A single center or zero median
    uses sigma=1, a chosen, unmeasured fallback. Targets/weights are ignored.
    """

    name = "prototype"

    def __init__(
        self, n_prototypes: int = 8, max_iter: int = 20, output: str = "distance",
        random_state: int | None = 0,
    ) -> None:
        if not isinstance(n_prototypes, (int, np.integer)) or n_prototypes < 1:
            raise ValueError("n_prototypes must be an integer >= 1")
        if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
            raise ValueError("max_iter must be an integer >= 1")
        if output not in ("distance", "rbf"):
            raise ValueError("output must be 'distance' or 'rbf'")
        if random_state is not None and not isinstance(random_state, (int, np.integer)):
            raise ValueError("random_state must be an integer or None")
        self.n_prototypes = int(n_prototypes)
        self.max_iter = int(max_iter)
        self.output = output
        self.random_state = None if random_state is None else int(random_state)
        self.centers_: np.ndarray | None = None

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

    @staticmethod
    def _squared_distances(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        # Avoid an n_rows * n_prototypes * n_features temporary and cancellation.
        return np.column_stack([np.sum((X - center)**2, axis=1) for center in centers])

    def fit(
        self, X_num: np.ndarray, y: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> PrototypeDistanceEncoder:
        X_num = self._validate(X_num)
        if len(X_num) < self.n_prototypes:
            raise ValueError(
                f"PrototypeDistanceEncoder requires at least {self.n_prototypes} training rows"
            )
        count = np.maximum((~np.isnan(X_num)).sum(axis=0), 1)
        self.mean_ = np.nansum(X_num, axis=0) / count
        centered = np.where(np.isnan(X_num), 0.0, X_num - self.mean_)
        std = np.sqrt(np.sum(centered**2, axis=0) / count)
        self.scale_ = np.where(std > 0, std, 1.0)
        standardized = centered / self.scale_
        rng = check_random_state(self.random_state)
        centers = standardized[rng.choice(len(X_num), self.n_prototypes, replace=False)].copy()
        for _ in range(self.max_iter):
            labels = np.argmin(self._squared_distances(standardized, centers), axis=1)
            updated = centers.copy()
            for j in range(self.n_prototypes):
                members = standardized[labels == j]
                if len(members):
                    updated[j] = members.mean(axis=0)
            if np.array_equal(updated, centers):
                break
            centers = updated
        self.centers_ = centers
        sigma = 1.0
        if self.n_prototypes > 1:
            distances = np.sqrt(self._squared_distances(centers, centers))
            median = np.median(distances[np.triu_indices(self.n_prototypes, k=1)])
            if median > 0:
                sigma = median
        self.sigma_ = np.asarray(sigma, dtype=np.float64)
        return self

    def transform(self, X_num: np.ndarray) -> np.ndarray:
        self._check_fitted("centers_")
        X_num = self._validate(X_num)
        if X_num.shape[1] != self.mean_.size:
            raise ValueError(f"Expected {self.mean_.size} numerical features, got {X_num.shape[1]}")
        standardized = (np.where(np.isnan(X_num), self.mean_, X_num) - self.mean_) / self.scale_
        squared = self._squared_distances(standardized, self.centers_)
        if self.output == "rbf":
            return np.exp(-squared / (2.0 * self.sigma_**2))
        return np.sqrt(squared)

    @property
    def output_dim(self) -> int:
        self._check_fitted("centers_")
        return int(self.centers_.shape[0])

    def get_config(self) -> dict:
        return {"n_prototypes": self.n_prototypes, "max_iter": self.max_iter,
                "output": self.output, "random_state": self.random_state}

    def get_state(self) -> dict[str, np.ndarray]:
        self._check_fitted("centers_")
        return {"centers": self.centers_, "sigma": self.sigma_,
                "mean": self.mean_, "scale": self.scale_}

    def set_state(self, state: dict[str, np.ndarray]) -> None:
        self.centers_ = np.asarray(state["centers"], dtype=np.float64)
        self.sigma_ = np.asarray(state["sigma"], dtype=np.float64)
        self.mean_ = np.asarray(state["mean"], dtype=np.float64)
        self.scale_ = np.asarray(state["scale"], dtype=np.float64)
