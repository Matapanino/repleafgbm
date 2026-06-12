"""Cross-interaction encoder: standardized features + selected pairwise products.

Phase 16 opens the interaction-aware encoder line that Phases 13/14/14b
motivated: per-feature learned encoders found nothing on real data that the
router doesn't already exploit, so the remaining hypothesis is that leaves
benefit from *cross-feature* structure. This encoder is the deterministic,
NumPy-only control for that hypothesis: it appends the ``n_pairs`` pairwise
products ``x_i * x_j`` most correlated with the supervised pretraining
target (the initial Newton residual) to the standardized features. If the
learned MLP variant cannot beat this, interactions per se are not the
missing ingredient.
"""

from __future__ import annotations

import numpy as np

from repleafgbm.encoders.base import BaseEncoder


class CrossInteractionEncoder(BaseEncoder):
    """Standardized numericals plus top-``n_pairs`` pairwise products.

    Selection is supervised and happens once in ``fit`` (then frozen, v0
    rule): all products of standardized feature pairs are scored by absolute
    Pearson correlation with the pretraining target ``y``; without a target
    the first pairs in lexicographic order are used. Products are
    standardized with their training statistics; missing values are mean
    imputed (0 in standardized space) before multiplication.

    Output: ``(n_rows, n_features + min(n_pairs, n_features*(n_features-1)/2))``.
    """

    name = "cross"

    def __init__(self, n_pairs: int = 16) -> None:
        if n_pairs < 1:
            raise ValueError(f"n_pairs must be >= 1, got {n_pairs}")
        self.n_pairs = n_pairs
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.pairs_: np.ndarray | None = None  # (k, 2) int feature indices
        self.prod_mean_: np.ndarray | None = None
        self.prod_scale_: np.ndarray | None = None

    def fit(self, X_num: np.ndarray, y: np.ndarray | None = None) -> CrossInteractionEncoder:
        X_num = np.asarray(X_num, dtype=np.float64)
        self.mean_ = np.nanmean(X_num, axis=0)
        std = np.nanstd(X_num, axis=0)
        self.scale_ = np.where(std > 0, std, 1.0)

        n_features = X_num.shape[1]
        i_idx, j_idx = np.triu_indices(n_features, k=1)
        if i_idx.size == 0:
            self.pairs_ = np.zeros((0, 2), dtype=np.int64)
            self.prod_mean_ = np.zeros(0)
            self.prod_scale_ = np.ones(0)
            return self

        x_std = self._standardize(X_num)
        products = x_std[:, i_idx] * x_std[:, j_idx]  # (n, n_all_pairs)
        if y is not None and i_idx.size > self.n_pairs:
            yc = np.asarray(y, dtype=np.float64) - np.mean(y)
            pc = products - products.mean(axis=0)
            denom = np.sqrt(np.sum(pc**2, axis=0) * np.sum(yc**2)) + 1e-12
            score = np.abs(pc.T @ yc) / denom
            # Stable order among ties so the same seed gives the same model.
            top = np.argsort(-score, kind="stable")[: self.n_pairs]
        else:
            top = np.arange(min(self.n_pairs, i_idx.size))
        top = np.sort(top)
        self.pairs_ = np.stack([i_idx[top], j_idx[top]], axis=1)

        chosen = products[:, top]
        self.prod_mean_ = chosen.mean(axis=0)
        prod_std = chosen.std(axis=0)
        self.prod_scale_ = np.where(prod_std > 0, prod_std, 1.0)
        return self

    def _standardize(self, X_num: np.ndarray) -> np.ndarray:
        Z = np.where(np.isnan(X_num), self.mean_, X_num)
        return (Z - self.mean_) / self.scale_

    def transform(self, X_num: np.ndarray) -> np.ndarray:
        self._check_fitted("pairs_")
        X_num = np.asarray(X_num, dtype=np.float64)
        x_std = self._standardize(X_num)
        if self.pairs_.shape[0] == 0:
            return x_std
        products = x_std[:, self.pairs_[:, 0]] * x_std[:, self.pairs_[:, 1]]
        products = (products - self.prod_mean_) / self.prod_scale_
        return np.concatenate([x_std, products], axis=1)

    @property
    def output_dim(self) -> int:
        self._check_fitted("pairs_")
        return int(self.mean_.shape[0] + self.pairs_.shape[0])

    def get_config(self) -> dict:
        return {"n_pairs": self.n_pairs}

    def get_state(self) -> dict[str, np.ndarray]:
        self._check_fitted("pairs_")
        return {
            "mean": self.mean_,
            "scale": self.scale_,
            "pairs": self.pairs_,
            "prod_mean": self.prod_mean_,
            "prod_scale": self.prod_scale_,
        }

    def set_state(self, state: dict[str, np.ndarray]) -> None:
        self.mean_ = np.asarray(state["mean"], dtype=np.float64)
        self.scale_ = np.asarray(state["scale"], dtype=np.float64)
        self.pairs_ = np.asarray(state["pairs"], dtype=np.int64)
        self.prod_mean_ = np.asarray(state["prod_mean"], dtype=np.float64)
        self.prod_scale_ = np.asarray(state["prod_scale"], dtype=np.float64)
