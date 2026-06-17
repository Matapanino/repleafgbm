"""Training objectives (loss functions) with gradients and Hessians.

Each objective supplies first/second-order statistics of the loss with
respect to the raw score F(x). Trees are grown on these statistics and leaf
models are fitted to the Newton targets ``-g / h`` with weights ``h``
(see docs/math.md).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

import numpy as np


class BaseObjective(ABC):
    """Abstract objective: maps raw scores to gradients/Hessians/predictions."""

    name: str = "base"

    @abstractmethod
    def init_score(self, y: np.ndarray, weight: np.ndarray | None = None) -> float:
        """Optimal constant raw score F_0 for this loss.

        ``weight`` are optional per-row sample weights; when given the optimum
        is the weighted one (e.g. weighted mean / weighted class priors).
        ``weight=None`` reproduces the unweighted result exactly.
        """

    @abstractmethod
    def grad_hess(self, y: np.ndarray, raw_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-row gradient and Hessian of the loss w.r.t. raw_pred."""

    @abstractmethod
    def transform(self, raw_pred: np.ndarray) -> np.ndarray:
        """Map raw scores to output space (identity / probability)."""


class SquaredError(BaseObjective):
    """Mean squared error for regression. g = F - y, h = 1."""

    name = "squared_error"

    def init_score(self, y: np.ndarray, weight: np.ndarray | None = None) -> float:
        return float(_weighted_mean(y, weight))

    def grad_hess(self, y: np.ndarray, raw_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return raw_pred - y, np.ones_like(y)

    def transform(self, raw_pred: np.ndarray) -> np.ndarray:
        return raw_pred


class BinaryLogistic(BaseObjective):
    """Logistic loss for binary classification on labels in {0, 1}.

    g = sigmoid(F) - y, h = sigmoid(F) * (1 - sigmoid(F)).

    Args:
        label_smoothing: If >0, the hard targets are softened to
            ``y * (1 - eps) + eps / 2`` before computing gradients and the
            init score. This regularizes over-confident probabilities;
            ``eps = 0`` reproduces the unsmoothed objective exactly.
    """

    name = "binary_logistic"

    def __init__(self, label_smoothing: float = 0.0) -> None:
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError(
                f"label_smoothing must be in [0, 1), got {label_smoothing}"
            )
        self.label_smoothing = label_smoothing

    def _smooth(self, y: np.ndarray) -> np.ndarray:
        if self.label_smoothing == 0.0:
            return y
        return y * (1.0 - self.label_smoothing) + self.label_smoothing / 2.0

    def init_score(self, y: np.ndarray, weight: np.ndarray | None = None) -> float:
        p = float(np.clip(_weighted_mean(self._smooth(y), weight), 1e-12, 1 - 1e-12))
        return float(np.log(p / (1.0 - p)))

    def grad_hess(self, y: np.ndarray, raw_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = _sigmoid(raw_pred)
        return p - self._smooth(y), np.maximum(p * (1.0 - p), 1e-12)

    def transform(self, raw_pred: np.ndarray) -> np.ndarray:
        return _sigmoid(raw_pred)


class Huber(BaseObjective):
    """Huber loss for outlier-robust regression.

    g = clip(F - y, -delta, delta), h = 1 (the LightGBM convention: the true
    Hessian is 0 beyond delta, which would let outlier-only leaves blow up).
    With h = 1 the Newton targets are clipped residuals, so leaf fits — and
    linear leaves in particular — see outliers with bounded influence.
    """

    name = "huber"

    def __init__(self, delta: float = 1.0) -> None:
        if delta <= 0:
            raise ValueError(f"huber delta must be positive, got {delta}")
        self.delta = delta

    def init_score(self, y: np.ndarray, weight: np.ndarray | None = None) -> float:
        return float(_weighted_quantile(y, 0.5, weight))

    def grad_hess(self, y: np.ndarray, raw_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.clip(raw_pred - y, -self.delta, self.delta), np.ones_like(y)

    def transform(self, raw_pred: np.ndarray) -> np.ndarray:
        return raw_pred


class Quantile(BaseObjective):
    """Pinball (quantile) loss: the model predicts the alpha-quantile.

    g = (1 - alpha) where F >= y and -alpha where F < y, h = 1 — the loss is
    piecewise linear, so boosting takes fixed-size steps whose sign balance
    converges to the alpha-quantile within each leaf.
    """

    name = "quantile"

    def __init__(self, alpha: float = 0.5) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"quantile alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha

    def init_score(self, y: np.ndarray, weight: np.ndarray | None = None) -> float:
        return float(_weighted_quantile(y, self.alpha, weight))

    def grad_hess(self, y: np.ndarray, raw_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        grad = np.where(raw_pred >= y, 1.0 - self.alpha, -self.alpha)
        return grad, np.ones_like(y)

    def transform(self, raw_pred: np.ndarray) -> np.ndarray:
        return raw_pred


class PoissonRegression(BaseObjective):
    """Poisson deviance for non-negative count targets; F is the log-mean.

    L = exp(F) - y*F, g = exp(F) - y, h = exp(F). Raw scores are clipped to
    [-30, 30] inside exp for overflow safety. Output transform: exp.
    """

    name = "poisson"

    def init_score(self, y: np.ndarray, weight: np.ndarray | None = None) -> float:
        if (y < 0).any():
            raise ValueError(
                "objective='poisson' requires non-negative targets; "
                "found negative values in y"
            )
        mean = float(_weighted_mean(y, weight))
        if mean <= 0:
            raise ValueError(
                "objective='poisson' requires a positive target mean "
                "(y is all zeros)"
            )
        return float(np.log(mean))

    def grad_hess(self, y: np.ndarray, raw_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu = np.exp(np.clip(raw_pred, -30.0, 30.0))
        return mu - y, np.maximum(mu, 1e-12)

    def transform(self, raw_pred: np.ndarray) -> np.ndarray:
        return np.exp(np.clip(raw_pred, -30.0, 30.0))


class MulticlassSoftmax:
    """Softmax cross-entropy for K-class classification on labels 0..K-1.

    The vector-valued counterpart of :class:`BaseObjective`: raw scores are
    (n_rows, n_classes) matrices, one column per class. Gradients use the
    diagonal Hessian approximation standard in GBDTs (LightGBM/XGBoost):
    g_k = p_k - 1{y=k}, h_k = p_k * (1 - p_k). Consumed by
    :class:`~repleafgbm.core.multiclass.MulticlassBooster`, which grows one
    tree per class per boosting round on these per-class statistics.
    """

    name = "multiclass_softmax"

    def __init__(self, n_classes: int, label_smoothing: float = 0.0) -> None:
        if n_classes < 3:
            raise ValueError(
                f"MulticlassSoftmax requires n_classes >= 3, got {n_classes}; "
                "use binary_logistic for two classes"
            )
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError(
                f"label_smoothing must be in [0, 1), got {label_smoothing}"
            )
        self.n_classes = n_classes
        self.label_smoothing = label_smoothing

    def init_score(self, y: np.ndarray, weight: np.ndarray | None = None) -> np.ndarray:
        """Log class priors, shape (n_classes,). Softmax-invariant shift aside,
        this is the optimal constant score matrix. With label smoothing the
        priors are the mean smoothed target ``(1 - eps) * empirical + eps / K``.
        Sample weights make the priors the weighted class frequencies."""
        labels = y.astype(np.int64)
        counts = np.bincount(labels, weights=weight, minlength=self.n_classes)
        priors = counts / counts.sum()
        if self.label_smoothing:
            priors = (
                (1.0 - self.label_smoothing) * priors
                + self.label_smoothing / self.n_classes
            )
        return np.log(np.clip(priors, 1e-12, None))

    def grad_hess(self, y: np.ndarray, raw_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-row, per-class gradient and Hessian, both (n_rows, n_classes).

        The hard one-hot target is softened to
        ``(1 - eps) * onehot + eps / K`` when ``label_smoothing`` is set, so
        ``g_k = p_k - target_k``."""
        p = _softmax(raw_pred)
        grad = p.copy()
        eps = self.label_smoothing
        if eps:
            grad -= eps / self.n_classes
        grad[np.arange(y.shape[0]), y.astype(np.int64)] -= 1.0 - eps
        hess = np.maximum(p * (1.0 - p), 1e-12)
        return grad, hess

    def transform(self, raw_pred: np.ndarray) -> np.ndarray:
        return _softmax(raw_pred)


@runtime_checkable
class MultiOutputObjective(Protocol):
    """Structural interface for multi-output (vector-valued) regression losses.

    The vector-valued counterpart of :class:`BaseObjective`, consumed by
    :class:`~repleafgbm.core.multioutput.MultiOutputBooster`. Raw scores and
    targets are ``(n_rows, n_outputs)`` matrices; ``init_score`` returns a
    ``(n_outputs,)`` vector. Every implementation keeps a **constant Hessian**
    (``h = 1``) so the shared-Gram vector-leaf solve in ``core.multioutput``
    applies unchanged (docs/math.md).
    """

    name: str
    n_outputs: int

    def init_score(self, y: np.ndarray, weight: np.ndarray | None = None) -> np.ndarray: ...

    def grad_hess(
        self, y: np.ndarray, raw_pred: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def transform(self, raw_pred: np.ndarray) -> np.ndarray: ...


class MultiOutputSquaredError:
    """Squared error for multi-output (vector-valued) regression.

    The vector-valued counterpart of :class:`SquaredError`: raw scores and
    targets are (n_rows, n_outputs) matrices and every output shares the same
    routing tree (vector leaves, docs/math.md). g = F - Y, h = 1. Consumed by
    :class:`~repleafgbm.core.multioutput.MultiOutputBooster`, which grows one
    tree per round whose leaves emit an (n_outputs,) vector.
    """

    name = "multioutput_squared_error"

    def __init__(self, n_outputs: int) -> None:
        if n_outputs < 2:
            raise ValueError(
                f"MultiOutputSquaredError requires n_outputs >= 2, got {n_outputs}; "
                "use squared_error for a single target"
            )
        self.n_outputs = n_outputs

    def init_score(self, y: np.ndarray, weight: np.ndarray | None = None) -> np.ndarray:
        """Per-output (weighted) means, shape (n_outputs,)."""
        if weight is None:
            return np.mean(y, axis=0)
        w = weight[:, None]
        return (w * y).sum(axis=0) / w.sum()

    def grad_hess(self, y: np.ndarray, raw_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-row, per-output gradient and Hessian, both (n_rows, n_outputs)."""
        return raw_pred - y, np.ones_like(y)

    def transform(self, raw_pred: np.ndarray) -> np.ndarray:
        return raw_pred


class MultiOutputHuber:
    """Huber loss for multi-output (vector-valued), outlier-robust regression.

    The vector-valued counterpart of :class:`Huber`: raw scores and targets are
    (n_rows, n_outputs) matrices sharing one routing tree. Per output,
    g = clip(F - Y, -delta, delta) and h = 1 (the LightGBM convention, as in
    :class:`Huber`). Because the Hessian stays constant across outputs, the
    shared-Gram vector-leaf solve in :mod:`repleafgbm.core.multioutput` applies
    unchanged — only the gradient (clipped residuals) and the init score (the
    per-output median) differ from squared error.
    """

    name = "multioutput_huber"

    def __init__(self, n_outputs: int, delta: float = 1.0) -> None:
        if n_outputs < 2:
            raise ValueError(
                f"MultiOutputHuber requires n_outputs >= 2, got {n_outputs}; "
                "use objective='huber' for a single target"
            )
        if delta <= 0:
            raise ValueError(f"huber delta must be positive, got {delta}")
        self.n_outputs = n_outputs
        self.delta = delta

    def init_score(self, y: np.ndarray, weight: np.ndarray | None = None) -> np.ndarray:
        """Per-output (weighted) medians, shape (n_outputs,)."""
        return np.array(
            [_weighted_quantile(y[:, k], 0.5, weight) for k in range(y.shape[1])],
            dtype=np.float64,
        )

    def grad_hess(self, y: np.ndarray, raw_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-row, per-output gradient and Hessian, both (n_rows, n_outputs)."""
        return np.clip(raw_pred - y, -self.delta, self.delta), np.ones_like(y)

    def transform(self, raw_pred: np.ndarray) -> np.ndarray:
        return raw_pred


class MultiOutputQuantile:
    """Pinball (quantile) loss for multi-output regression: each output is the
    alpha-quantile of its target.

    The vector-valued counterpart of :class:`Quantile`: raw scores and targets
    are (n_rows, n_outputs) matrices sharing one routing tree. Per output,
    g = (1 - alpha) where F >= Y and -alpha where F < Y, and h = 1 — the same
    constant-Hessian, piecewise-linear objective as :class:`Quantile`, so the
    shared-Gram vector-leaf solve applies unchanged.
    """

    name = "multioutput_quantile"

    def __init__(self, n_outputs: int, alpha: float = 0.5) -> None:
        if n_outputs < 2:
            raise ValueError(
                f"MultiOutputQuantile requires n_outputs >= 2, got {n_outputs}; "
                "use objective='quantile' for a single target"
            )
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"quantile alpha must be in (0, 1), got {alpha}")
        self.n_outputs = n_outputs
        self.alpha = alpha

    def init_score(self, y: np.ndarray, weight: np.ndarray | None = None) -> np.ndarray:
        """Per-output (weighted) alpha-quantiles, shape (n_outputs,)."""
        return np.array(
            [_weighted_quantile(y[:, k], self.alpha, weight) for k in range(y.shape[1])],
            dtype=np.float64,
        )

    def grad_hess(self, y: np.ndarray, raw_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-row, per-output gradient and Hessian, both (n_rows, n_outputs)."""
        grad = np.where(raw_pred >= y, 1.0 - self.alpha, -self.alpha)
        return grad, np.ones_like(y)

    def transform(self, raw_pred: np.ndarray) -> np.ndarray:
        return raw_pred


def _weighted_mean(y: np.ndarray, weight: np.ndarray | None) -> float:
    """Mean of ``y`` (weighted by ``weight`` when given)."""
    if weight is None:
        return float(np.mean(y))
    return float(np.dot(weight, y) / weight.sum())


def _weighted_quantile(y: np.ndarray, q: float, weight: np.ndarray | None) -> float:
    """The ``q``-quantile of ``y`` with optional per-row weights.

    With ``weight=None`` this matches ``np.quantile(y, q)`` (linear
    interpolation). The weighted version interpolates the inverse of the
    cumulative weight at the midpoints of each sample's weight interval, which
    reduces to the unweighted definition when all weights are equal.
    """
    if weight is None:
        return float(np.quantile(y, q))
    order = np.argsort(y, kind="stable")
    ys = y[order]
    ws = weight[order]
    cum = np.cumsum(ws)
    total = cum[-1]
    if total <= 0:
        return float(np.quantile(y, q))
    # Midpoint of each sample's cumulative-weight interval, normalized to [0, 1].
    midpoints = (cum - 0.5 * ws) / total
    return float(np.interp(q, midpoints, ys))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _softmax(x: np.ndarray) -> np.ndarray:
    """Row-wise stable softmax for (n_rows, n_classes) score matrices."""
    z = x - x.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)


_OBJECTIVE_REGISTRY: dict[str, type[BaseObjective]] = {
    SquaredError.name: SquaredError,
    BinaryLogistic.name: BinaryLogistic,
    Huber.name: Huber,
    Quantile.name: Quantile,
    PoissonRegression.name: PoissonRegression,
}


def get_objective(name: str) -> BaseObjective:
    if name not in _OBJECTIVE_REGISTRY:
        raise ValueError(
            f"Unknown objective {name!r}. Available: {sorted(_OBJECTIVE_REGISTRY)}"
        )
    return _OBJECTIVE_REGISTRY[name]()


#: Multi-output regression objectives, keyed by ``.name``. Each is constructed
#: with ``n_outputs`` and default loss parameters (delta / alpha) — fidelity of
#: those parameters is a fit-time concern only, since every multi-output loss
#: has an identity output transform, so a *fitted* model predicts identically
#: regardless (used by :mod:`repleafgbm.core.serialization` on load).
_MULTIOUTPUT_OBJECTIVE_REGISTRY: dict[str, type] = {
    MultiOutputSquaredError.name: MultiOutputSquaredError,
    MultiOutputHuber.name: MultiOutputHuber,
    MultiOutputQuantile.name: MultiOutputQuantile,
}


def get_multioutput_objective(name: str, n_outputs: int) -> MultiOutputObjective:
    if name not in _MULTIOUTPUT_OBJECTIVE_REGISTRY:
        raise ValueError(
            f"Unknown multi-output objective {name!r}. "
            f"Available: {sorted(_MULTIOUTPUT_OBJECTIVE_REGISTRY)}"
        )
    return _MULTIOUTPUT_OBJECTIVE_REGISTRY[name](n_outputs)
