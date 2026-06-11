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


def _sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


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
