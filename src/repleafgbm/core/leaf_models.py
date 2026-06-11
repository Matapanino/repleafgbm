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

try:  # Optional compiled fast path for fused per-leaf statistics (native/).
    import repleafgbm_native as _native_module

    _native = getattr(_native_module, "leaf_linear_stats", None) and _native_module
except ImportError:  # pragma: no cover - depends on optional extension
    _native = None

#: Above this embedding width the BLAS-based NumPy path beats the scalar
#: fused pass, so the native helper is only used for narrow embeddings.
_NATIVE_STATS_MAX_DIM = 32

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

    def predict(
        self, leaf_idx: np.ndarray, Z: np.ndarray | None, clip: bool = True
    ) -> np.ndarray:
        """Leaf outputs for routed rows.

        ``clip=False`` skips the extrapolation guard; it is only valid for
        the rows the leaves were fitted on (the booster's training-score
        update), where clipping to the leaf's own min/max is exactly the
        identity.
        """
        out = self.bias[leaf_idx]
        if self.weights.shape[1] > 0:
            assert Z is not None, "embedding matrix required for linear leaves"
            if clip and self.z_min is not None:
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
        """Fit all leaves of one tree with batched normal equations.

        Per-leaf work is reduced to two BLAS products (the weighted Gram
        matrix and the gradient projection); all systems are then solved in
        one batched ``np.linalg.solve`` call. Uses the uncentered identities

            sum_i h (z - z_mean)(z - z_mean)^T = M - S_h z_mean z_mean^T
            sum_i h (z - z_mean)(t - t_mean)   = -G_z + S_g z_mean

        with M = sum h z z^T, G_z = sum g z (since h t = -g). This is
        algebraically identical to the centered reference implementation
        (`_fit_weighted_ridge`, kept for parity testing) and numerically
        safe here because encoders produce standardized/bounded embeddings;
        degenerate directions are damped by the ridge term as before.
        """
        if Z is None:
            raise ValueError("EmbeddedLinearLeafModel requires an embedding matrix Z")
        emb_dim = Z.shape[1]
        n_leaves = len(leaf_rows)
        weights = np.zeros((n_leaves, emb_dim), dtype=np.float64)
        # Extrapolation guard: bounds stay infinite (no-op) for constant
        # leaves, whose weights are zero anyway.
        z_min = np.full((n_leaves, emb_dim), -np.inf, dtype=np.float64)
        z_max = np.full((n_leaves, emb_dim), np.inf, dtype=np.float64)

        # Gather everything once in leaf order; per-leaf data is then a
        # contiguous view (no per-leaf fancy indexing).
        sizes = np.array([r.shape[0] for r in leaf_rows], dtype=np.int64)
        order = np.concatenate(leaf_rows) if leaf_rows else np.empty(0, np.int64)
        offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        min_n = max(self.min_samples_linear, emb_dim + 2)
        linear = np.flatnonzero(sizes >= min_n)
        k = linear.size

        if k and _native is not None and emb_dim <= _NATIVE_STATS_MAX_DIM:
            # Fused single-pass statistics in Rust (narrow embeddings only:
            # for wide ones the BLAS Gram below wins).
            g_sum, h_sum, s_hz, A, gz, zmn, zmx = _native.leaf_linear_stats(
                np.ascontiguousarray(Z), grad, hess, order, offsets,
                linear.astype(np.int64),
            )
            bias = -g_sum / (h_sum + self.l2)
            z_mean = s_hz / h_sum[linear][:, None]
            t_mean = -g_sum[linear] / h_sum[linear]
            A -= z_mean[:, :, None] * s_hz[:, None, :]
            rhs = -gz - t_mean[:, None] * s_hz
            z_min[linear] = zmn
            z_max[linear] = zmx
        else:
            seg = np.repeat(np.arange(n_leaves), sizes)
            g_seg = grad[order]
            h_seg = hess[order]
            g_sum = np.bincount(seg, weights=g_seg, minlength=n_leaves)
            h_sum = np.bincount(seg, weights=h_seg, minlength=n_leaves)
            bias = -g_sum / (h_sum + self.l2)
            if k == 0:
                return LeafValues(bias=bias, weights=weights, z_min=z_min, z_max=z_max)

            Z_seg = Z[order]
            hZ_seg = Z_seg * h_seg[:, None]
            A = np.empty((k, emb_dim, emb_dim), dtype=np.float64)
            rhs = np.empty((k, emb_dim), dtype=np.float64)
            z_mean = np.empty((k, emb_dim), dtype=np.float64)
            t_mean = np.empty(k, dtype=np.float64)
            for j, i in enumerate(linear):
                sl = slice(offsets[i], offsets[i + 1])
                Zl = Z_seg[sl]
                hZ = hZ_seg[sl]
                s_hz = hZ.sum(axis=0)
                z_mean[j] = s_hz / h_sum[i]
                t_mean[j] = -g_sum[i] / h_sum[i]
                A[j] = Zl.T @ hZ
                A[j] -= np.outer(z_mean[j], s_hz)
                rhs[j] = -(g_seg[sl] @ Zl) - t_mean[j] * s_hz
                z_min[i] = Zl.min(axis=0)
                z_max[i] = Zl.max(axis=0)
        A[:, np.arange(emb_dim), np.arange(emb_dim)] += self.l2

        try:
            w = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            # Rare: some leaf's system is exactly singular. Solve one by one
            # so only the degenerate leaves fall back to constants.
            w = np.zeros((k, emb_dim), dtype=np.float64)
            for j in range(k):
                try:
                    w[j] = np.linalg.solve(A[j], rhs[j])
                except np.linalg.LinAlgError:
                    w[j] = np.nan  # handled by the finite check below

        ok = np.isfinite(w).all(axis=1)
        for j, i in enumerate(linear):
            if ok[j]:
                weights[i] = w[j]
                bias[i] = t_mean[j] - w[j] @ z_mean[j]
            else:  # constant fallback: Newton bias kept, guard disabled
                z_min[i] = -np.inf
                z_max[i] = np.inf
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
