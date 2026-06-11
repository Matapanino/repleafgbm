"""Out-of-fold (OOF) prediction utility.

Generic over estimators (works for external base models *and* RepLeafGBM
itself), with no external dependency: stacking features for training rows
must come from models that never saw those rows, otherwise the second-stage
model learns from leaked targets.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold


def _take(X: Any, idx: np.ndarray) -> Any:
    if hasattr(X, "iloc"):
        return X.iloc[idx]
    return X[idx]


def _default_predict(model: Any, X: Any) -> np.ndarray:
    """Score with the most informative method available."""
    if hasattr(model, "predict_score"):
        return np.asarray(model.predict_score(X), dtype=np.float64)
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1], dtype=np.float64)
    return np.asarray(model.predict(X), dtype=np.float64)


def oof_predictions(
    make_model: Callable[[], Any],
    X: Any,
    y: np.ndarray,
    n_splits: int = 5,
    stratify: bool = False,
    random_state: int = 0,
    predict_fn: Callable[[Any, Any], np.ndarray] | None = None,
) -> tuple[np.ndarray, list[Any]]:
    """K-fold out-of-fold predictions.

    Args:
        make_model: Zero-argument factory returning a fresh unfitted
            estimator with ``fit(X, y)`` (e.g.
            ``lambda: LightGBMExternalModel(task="regression")``).
        X: Feature matrix (ndarray or DataFrame).
        y: Target vector.
        n_splits: Number of folds.
        stratify: Use StratifiedKFold (for classification targets).
        random_state: Fold shuffling seed.
        predict_fn: ``(model, X) -> (n,) scores``. Defaults to
            ``predict_score`` / ``predict_proba[:, 1]`` / ``predict``,
            whichever exists first.

    Returns:
        ``(oof, models)``: scores of shape (n,) where each entry was
        predicted by the fold model that did not train on it, and the
        ``n_splits`` fitted models (e.g. to average their predictions on
        new data).
    """
    y = np.asarray(y)
    predict = predict_fn or _default_predict
    splitter = (
        StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        if stratify
        else KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    )
    oof = np.full(y.shape[0], np.nan, dtype=np.float64)
    models: list[Any] = []
    for train_idx, valid_idx in splitter.split(np.zeros(y.shape[0]), y if stratify else None):
        model = make_model()
        model.fit(_take(X, train_idx), y[train_idx])
        oof[valid_idx] = predict(model, _take(X, valid_idx))
        models.append(model)
    return oof, models
