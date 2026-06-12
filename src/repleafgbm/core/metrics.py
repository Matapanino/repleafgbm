"""Evaluation metrics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np


class BaseMetric(ABC):
    """Abstract evaluation metric computed on transformed predictions."""

    name: str = "base"
    #: True if smaller values are better.
    minimize: bool = True

    @abstractmethod
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float: ...


class RMSE(BaseMetric):
    name = "rmse"

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


class MAE(BaseMetric):
    name = "mae"

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(np.abs(y_true - y_pred)))


class LogLoss(BaseMetric):
    name = "logloss"

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        p = np.clip(y_pred, 1e-12, 1 - 1e-12)
        return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


class Accuracy(BaseMetric):
    """Accuracy at a 0.5 probability threshold (y_pred are probabilities)."""

    name = "accuracy"
    minimize = False

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean((y_pred >= 0.5) == (y_true == 1)))


class AUC(BaseMetric):
    """ROC AUC via the rank-sum (Mann-Whitney U) formulation with tie handling."""

    name = "auc"
    minimize = False

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pos = y_true == 1
        n_pos = int(pos.sum())
        n_neg = len(y_true) - n_pos
        if n_pos == 0 or n_neg == 0:
            raise ValueError("AUC is undefined when y_true contains a single class")
        # Average (1-based) ranks of the scores, ties sharing their mean rank.
        _, inverse, counts = np.unique(y_pred, return_inverse=True, return_counts=True)
        group_start = np.cumsum(counts) - counts
        avg_rank = group_start + (counts + 1) / 2.0
        ranks = avg_rank[inverse]
        u = ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0
        return float(u / (n_pos * n_neg))


class _CallableMetric(BaseMetric):
    """Adapter wrapping a plain ``(y_true, y_pred) -> float`` callable."""

    def __init__(
        self,
        fn: Callable[[np.ndarray, np.ndarray], float],
        name: str,
        minimize: bool,
    ) -> None:
        self._fn = fn
        self.name = name
        self.minimize = minimize

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(self._fn(y_true, y_pred))


def make_metric(
    fn: Callable[[np.ndarray, np.ndarray], float],
    name: str | None = None,
    minimize: bool = True,
) -> BaseMetric:
    """Wrap a callable as an eval metric usable for monitoring/early stopping.

    Args:
        fn: ``(y_true, y_pred) -> float``. ``y_pred`` is on the prediction
            scale (probabilities for binary classification, values for
            regression).
        name: Key used in ``evals_result_``; defaults to ``fn.__name__``.
        minimize: False for greater-is-better metrics — this drives the early
            stopping direction.
    """
    if not callable(fn):
        raise TypeError(f"make_metric expects a callable, got {type(fn).__name__}")
    return _CallableMetric(fn, name or getattr(fn, "__name__", "custom_metric"), minimize)


_METRIC_REGISTRY: dict[str, type[BaseMetric]] = {
    RMSE.name: RMSE,
    MAE.name: MAE,
    LogLoss.name: LogLoss,
    Accuracy.name: Accuracy,
    AUC.name: AUC,
}


def get_metric(name: str) -> BaseMetric:
    if name not in _METRIC_REGISTRY:
        raise ValueError(f"Unknown metric {name!r}. Available: {sorted(_METRIC_REGISTRY)}")
    return _METRIC_REGISTRY[name]()
