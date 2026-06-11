"""Simplified piecewise-linear (PLR-style) numerical feature encoder.

This is a deliberately simple variant of the piecewise-linear embeddings from
"On Embeddings for Numerical Features in Tabular Deep Learning" (Gorishniy et
al., 2022). Each numerical feature is mapped to ``n_bins`` components using
quantile bin edges: component ``t`` is 0 below bin ``t``, 1 above it, and
linear inside it. There is no learned linear layer and no periodic component
in v0; those belong to future PyTorch encoders.
"""

from __future__ import annotations

import numpy as np

from repleafgbm.encoders.base import BaseEncoder


class SimplePLREncoder(BaseEncoder):
    """Quantile-based piecewise-linear encoding of numerical features.

    Args:
        n_bins: Number of piecewise-linear components per feature. The output
            dimension is ``n_features * (n_bins + add_linear)``. The default
            of 4 follows experiments/results/plr_projection_gap.md: fewer,
            wider components generalized better in every tested setting,
            because leaf-local ridge fits have few samples per leaf and
            high-dimensional PLR triggers constant fallbacks.
        add_linear: Append the standardized raw value of each feature to its
            PLR block. PLR components saturate outside the training range, so
            leaf models built on them cannot extrapolate; the linear term
            restores an unbounded direction (cf. the linear component of PLR
            in Gorishniy et al. 2022 and PBLD in RealMLP).

    Missing values are encoded as an all-zero block for that feature
    (including the linear term, which is mean-imputed in standardized space).
    """

    name = "plr"

    def __init__(self, n_bins: int = 4, add_linear: bool = True) -> None:
        if n_bins < 2:
            raise ValueError(f"n_bins must be >= 2, got {n_bins}")
        self.n_bins = n_bins
        self.add_linear = add_linear
        # edges_ has shape (n_features, n_bins + 1): quantile bin boundaries.
        self.edges_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X_num: np.ndarray, y: np.ndarray | None = None) -> SimplePLREncoder:
        X_num = np.asarray(X_num, dtype=np.float64)
        n_features = X_num.shape[1]
        qs = np.linspace(0.0, 1.0, self.n_bins + 1)
        edges = np.empty((n_features, self.n_bins + 1), dtype=np.float64)
        for j in range(n_features):
            col = X_num[:, j]
            valid = col[~np.isnan(col)]
            if valid.size == 0:
                edges[j] = np.arange(self.n_bins + 1, dtype=np.float64)
                continue
            e = np.quantile(valid, qs)
            # Guarantee strictly increasing edges even for near-constant
            # features, so the linear interpolation below never divides by 0.
            # nextafter (not "+ epsilon") stays correct at any magnitude:
            # 1e15 + 1e-12 == 1e15 would silently keep a zero-width bin.
            for t in range(1, self.n_bins + 1):
                if e[t] <= e[t - 1]:
                    e[t] = np.nextafter(e[t - 1], np.inf)
            edges[j] = e
        self.edges_ = edges
        self.mean_ = np.nan_to_num(np.nanmean(X_num, axis=0), nan=0.0)
        std = np.nan_to_num(np.nanstd(X_num, axis=0), nan=1.0)
        self.scale_ = np.where(std > 0, std, 1.0)
        return self

    def transform(self, X_num: np.ndarray) -> np.ndarray:
        self._check_fitted("edges_")
        X_num = np.asarray(X_num, dtype=np.float64)
        n_rows, n_features = X_num.shape
        if n_features != self.edges_.shape[0]:
            raise ValueError(
                f"Expected {self.edges_.shape[0]} numerical features, got {n_features}"
            )
        d = self.n_bins + int(self.add_linear)
        Z = np.zeros((n_rows, n_features * d), dtype=np.float64)
        for j in range(n_features):
            col = X_num[:, j]
            missing = np.isnan(col)
            lo = self.edges_[j, :-1]  # (n_bins,)
            width = self.edges_[j, 1:] - lo
            # Broadcast: (n_rows, n_bins); NaN rows stay all-zero. Degenerate
            # bins have nextafter-tiny widths, so the division can overflow to
            # inf; clip maps that to the correct saturated value 1.0.
            with np.errstate(over="ignore"):
                block = (col[:, None] - lo[None, :]) / width[None, :]
            np.clip(block, 0.0, 1.0, out=block)
            block[missing] = 0.0
            Z[:, j * d : j * d + self.n_bins] = block
            if self.add_linear:
                lin = (col - self.mean_[j]) / self.scale_[j]
                lin[missing] = 0.0  # mean imputation in standardized space
                Z[:, j * d + self.n_bins] = lin
        return Z

    @property
    def output_dim(self) -> int:
        self._check_fitted("edges_")
        return int(self.edges_.shape[0] * (self.n_bins + int(self.add_linear)))

    def get_config(self) -> dict:
        return {"n_bins": self.n_bins, "add_linear": self.add_linear}

    def get_state(self) -> dict[str, np.ndarray]:
        self._check_fitted("edges_")
        return {"edges": self.edges_, "mean": self.mean_, "scale": self.scale_}

    def set_state(self, state: dict[str, np.ndarray]) -> None:
        self.edges_ = np.asarray(state["edges"], dtype=np.float64)
        self.mean_ = np.asarray(state["mean"], dtype=np.float64)
        self.scale_ = np.asarray(state["scale"], dtype=np.float64)
