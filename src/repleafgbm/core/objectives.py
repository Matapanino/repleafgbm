"""Training objectives (loss functions) with gradients and Hessians.

Each objective supplies first/second-order statistics of the loss with
respect to the raw score F(x). Trees are grown on these statistics and leaf
models are fitted to the Newton targets ``-g / h`` with weights ``h``
(see docs/math.md).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseObjective(ABC):
    """Abstract objective: maps raw scores to gradients/Hessians/predictions."""

    name: str = "base"

    @abstractmethod
    def init_score(self, y: np.ndarray) -> float:
        """Optimal constant raw score F_0 for this loss."""

    @abstractmethod
    def grad_hess(self, y: np.ndarray, raw_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-row gradient and Hessian of the loss w.r.t. raw_pred."""

    @abstractmethod
    def transform(self, raw_pred: np.ndarray) -> np.ndarray:
        """Map raw scores to output space (identity / probability)."""


class SquaredError(BaseObjective):
    """Mean squared error for regression. g = F - y, h = 1."""

    name = "squared_error"

    def init_score(self, y: np.ndarray) -> float:
        return float(np.mean(y))

    def grad_hess(self, y: np.ndarray, raw_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return raw_pred - y, np.ones_like(y)

    def transform(self, raw_pred: np.ndarray) -> np.ndarray:
        return raw_pred


class BinaryLogistic(BaseObjective):
    """Logistic loss for binary classification on labels in {0, 1}.

    g = sigmoid(F) - y, h = sigmoid(F) * (1 - sigmoid(F)).
    """

    name = "binary_logistic"

    def init_score(self, y: np.ndarray) -> float:
        p = float(np.clip(np.mean(y), 1e-12, 1 - 1e-12))
        return float(np.log(p / (1.0 - p)))

    def grad_hess(self, y: np.ndarray, raw_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = _sigmoid(raw_pred)
        return p - y, np.maximum(p * (1.0 - p), 1e-12)

    def transform(self, raw_pred: np.ndarray) -> np.ndarray:
        return _sigmoid(raw_pred)


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

    def __init__(self, n_classes: int) -> None:
        if n_classes < 3:
            raise ValueError(
                f"MulticlassSoftmax requires n_classes >= 3, got {n_classes}; "
                "use binary_logistic for two classes"
            )
        self.n_classes = n_classes

    def init_score(self, y: np.ndarray) -> np.ndarray:
        """Log class priors, shape (n_classes,). Softmax-invariant shift aside,
        this is the optimal constant score matrix."""
        counts = np.bincount(y.astype(np.int64), minlength=self.n_classes)
        priors = np.clip(counts / y.shape[0], 1e-12, None)
        return np.log(priors)

    def grad_hess(self, y: np.ndarray, raw_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-row, per-class gradient and Hessian, both (n_rows, n_classes)."""
        p = _softmax(raw_pred)
        grad = p.copy()
        grad[np.arange(y.shape[0]), y.astype(np.int64)] -= 1.0
        hess = np.maximum(p * (1.0 - p), 1e-12)
        return grad, hess

    def transform(self, raw_pred: np.ndarray) -> np.ndarray:
        return _softmax(raw_pred)


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
}


def get_objective(name: str) -> BaseObjective:
    if name not in _OBJECTIVE_REGISTRY:
        raise ValueError(
            f"Unknown objective {name!r}. Available: {sorted(_OBJECTIVE_REGISTRY)}"
        )
    return _OBJECTIVE_REGISTRY[name]()
