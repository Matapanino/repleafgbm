"""Leaf models: per-leaf predictors fitted on boosting statistics.

This is the core RepLeafGBM idea. A leaf is not a constant; it may be a small
ridge-regularized linear model over the representation Z:

    f_t(x) = b_{leaf} + w_{leaf}^T z_theta(x)

All leaf models are fitted to the Newton targets ``t_i = -g_i / h_i`` with
weights ``h_i``, which makes the same code path exact for squared error
(h = 1, t = residual) and a Newton approximation for other losses.

Overfitting guards (docs/design.md section "Leaf model variants"):

* ridge penalty ``l2`` on weights (never on the intercept),
* fallback to a constant leaf when the leaf has too few samples relative to
  the embedding dimension.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

#: Per-tree leaf parameters: output = bias[leaf] + Z @ weights[leaf].
#: For constant leaves, the weight row is all zeros (weights may have zero
#: columns when the whole tree is constant-leaf).


@dataclass
class LeafValues:
    bias: np.ndarray  # (n_leaves,)
    weights: np.ndarray  # (n_leaves, emb_dim) — emb_dim may be 0
    #: Per-leaf embedding clip bounds, (n_leaves, emb_dim). At prediction
    #: time Z is clipped to the range the leaf was fitted on, so beyond its
    #: training support a leaf extrapolates as a constant — the guard that
    #: prevents linear blow-ups on feature-space outliers (Phase 6/7,
    #: experiments/results/real_data_validation.md). None disables clipping
    #: (models saved before this guard existed).
    z_min: np.ndarray | None = None
    z_max: np.ndarray | None = None

    def predict(self, leaf_idx: np.ndarray, Z: np.ndarray | None) -> np.ndarray:
        out = self.bias[leaf_idx]
        if self.weights.shape[1] > 0:
            assert Z is not None, "embedding matrix required for linear leaves"
            if self.z_min is not None:
                Z = np.clip(Z, self.z_min[leaf_idx], self.z_max[leaf_idx])
            out = out + np.einsum("ij,ij->i", Z, self.weights[leaf_idx])
        return out

    @property
    def emb_dim(self) -> int:
        return int(self.weights.shape[1])


class BaseLeafModel(ABC):
    """Fits per-leaf parameters from gradients/Hessians and embeddings."""

    name: str = "base"

    #: Whether this leaf model consumes the embedding matrix Z.
    uses_embeddings: bool = False

    @abstractmethod
    def fit_leaves(
        self,
        leaf_rows: list[np.ndarray],
        grad: np.ndarray,
        hess: np.ndarray,
        Z: np.ndarray | None,
    ) -> LeafValues:
        """Fit parameters for every leaf of one tree."""


class ConstantLeafModel(BaseLeafModel):
    """Classic GBDT leaf: a single Newton step value per leaf."""

    name = "constant"
    uses_embeddings = False

    def __init__(self, l2: float = 1.0) -> None:
        self.l2 = l2

    def fit_leaves(
        self,
        leaf_rows: list[np.ndarray],
        grad: np.ndarray,
        hess: np.ndarray,
        Z: np.ndarray | None = None,
    ) -> LeafValues:
        bias = np.array(
            [_newton_constant(grad[r], hess[r], self.l2) for r in leaf_rows],
            dtype=np.float64,
        )
        return LeafValues(bias=bias, weights=np.zeros((len(leaf_rows), 0)))


class EmbeddedLinearLeafModel(BaseLeafModel):
    """Ridge-regularized linear model over the representation Z in each leaf.

    Args:
        l2: Ridge penalty on the weight vector (intercept unpenalized).
        min_samples_linear: Minimum leaf size to fit a linear model; smaller
            leaves fall back to a constant. The effective minimum is
            ``max(min_samples_linear, emb_dim + 2)`` so the weighted normal
            equations stay well-posed.
    """

    name = "embedded_linear"
    uses_embeddings = True

    def __init__(self, l2: float = 1.0, min_samples_linear: int = 10) -> None:
        self.l2 = l2
        self.min_samples_linear = min_samples_linear

    def fit_leaves(
        self,
        leaf_rows: list[np.ndarray],
        grad: np.ndarray,
        hess: np.ndarray,
        Z: np.ndarray | None,
    ) -> LeafValues:
        if Z is None:
            raise ValueError("EmbeddedLinearLeafModel requires an embedding matrix Z")
        emb_dim = Z.shape[1]
        n_leaves = len(leaf_rows)
        bias = np.zeros(n_leaves, dtype=np.float64)
        weights = np.zeros((n_leaves, emb_dim), dtype=np.float64)
        # Extrapolation guard: bounds stay infinite (no-op) for constant
        # leaves, whose weights are zero anyway.
        z_min = np.full((n_leaves, emb_dim), -np.inf, dtype=np.float64)
        z_max = np.full((n_leaves, emb_dim), np.inf, dtype=np.float64)
        min_n = max(self.min_samples_linear, emb_dim + 2)
        for i, rows in enumerate(leaf_rows):
            if rows.shape[0] < min_n:
                bias[i] = _newton_constant(grad[rows], hess[rows], self.l2)
                continue
            b, w = _fit_weighted_ridge(Z[rows], grad[rows], hess[rows], self.l2)
            if w is None:  # singular system: constant fallback
                bias[i] = _newton_constant(grad[rows], hess[rows], self.l2)
            else:
                bias[i], weights[i] = b, w
                z_min[i] = Z[rows].min(axis=0)
                z_max[i] = Z[rows].max(axis=0)
        return LeafValues(bias=bias, weights=weights, z_min=z_min, z_max=z_max)


def _newton_constant(g: np.ndarray, h: np.ndarray, l2: float) -> float:
    """Optimal constant leaf value: -sum(g) / (sum(h) + l2)."""
    return float(-g.sum() / (h.sum() + l2))


def _fit_weighted_ridge(
    Z: np.ndarray, g: np.ndarray, h: np.ndarray, l2: float
) -> tuple[float, np.ndarray | None]:
    """Solve the leaf's second-order objective for an affine model.

    Minimizes sum_i h_i * (b + w.z_i - t_i)^2 + l2 * ||w||^2 with t_i = -g_i/h_i,
    which equals the Newton objective up to a constant (docs/math.md).
    Centering Z and t by their h-weighted means decouples the (unpenalized)
    intercept from the weights.
    """
    h_sum = h.sum()
    t = -g / h
    z_mean = (h[:, None] * Z).sum(axis=0) / h_sum
    t_mean = float((h * t).sum() / h_sum)
    Zc = Z - z_mean
    tc = t - t_mean

    d = Z.shape[1]
    A = (Zc * h[:, None]).T @ Zc + l2 * np.eye(d)
    rhs = (Zc * h[:, None]).T @ tc
    try:
        w = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        return 0.0, None
    if not np.all(np.isfinite(w)):
        return 0.0, None
    b = t_mean - float(w @ z_mean)
    return b, w


def make_leaf_model(name: str, l2: float, min_samples_linear: int) -> BaseLeafModel:
    """Factory for the ``leaf_model`` parameter.

    ``raw_linear`` reuses the embedded-linear machinery; the model wrapper is
    responsible for supplying standardized raw numerical features as Z.
    """
    if name == "constant":
        return ConstantLeafModel(l2=l2)
    if name in ("embedded_linear", "raw_linear"):
        return EmbeddedLinearLeafModel(l2=l2, min_samples_linear=min_samples_linear)
    raise ValueError(
        f"Unknown leaf_model {name!r}. Available: 'constant', 'embedded_linear', 'raw_linear'"
    )
