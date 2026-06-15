"""CatBoost as an external base model (external_model mode).

Same duck-typed contract as the LightGBM and XGBoost base models (``fit`` /
``predict_score`` / ``predict_leaf_indices`` / ``n_trees_``), so it plugs
into ``oof_predictions`` / ``external_feature_frame`` / ``augment_features``
unchanged. The integration is deliberately shallow (docs/roadmap.md v0.3):
CatBoost's native categorical handling is available by passing
``cat_features`` through ``cb_params``, but the recommended path for
categorical data remains :class:`~repleafgbm.data.RepLeafDataset`, where
both models see the same ordinal-encoded matrix. Route extraction
(router_extraction mode) remains LightGBM-only.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

import numpy as np

from repleafgbm.external.lightgbm_model import _resolve_data

_DEFAULT_PARAMS: dict[str, Any] = {
    "iterations": 200,
    "learning_rate": 0.05,
    # CatBoost writes a catboost_info/ log directory to the CWD by default
    # (a tmp/ subdirectory even with writing disabled); a library must not
    # litter the user's filesystem, so both knobs point away from the CWD.
    "allow_writing_files": False,
    "train_dir": os.path.join(tempfile.gettempdir(), "repleafgbm_catboost_info"),
}


def _require_catboost():
    try:
        import catboost
    except ImportError as exc:  # pragma: no cover - exercised via mocked test
        raise ImportError(
            "CatBoost is required for repleafgbm.external.CatBoostExternalModel. "
            'Install it with: pip install catboost  (or pip install "repleafgbm[bench]")'
        ) from exc
    return catboost


class CatBoostExternalModel:
    """CatBoost base model with score and leaf-index extraction.

    Args:
        task: "regression" or "binary".
        random_state: Seed forwarded to CatBoost (``random_seed``).
        **cb_params: Any CatBoostRegressor/CatBoostClassifier parameters;
            merged over defaults (iterations=200, learning_rate=0.05).

    Example:
        >>> base = CatBoostExternalModel(task="regression", random_state=42)
        >>> base.fit(X_train, y_train)
        >>> score = base.predict_score(X)          # (n,)
        >>> leaves = base.predict_leaf_indices(X)  # (n, n_trees) int32
    """

    def __init__(self, task: str = "regression", random_state: int = 0, **cb_params: Any):
        if task not in ("regression", "binary"):
            raise ValueError(f"task must be 'regression' or 'binary', got {task!r}")
        self.task = task
        self.random_state = random_state
        self.cb_params = {**_DEFAULT_PARAMS, **cb_params}
        self.model_: Any = None
        #: Number of trees of the best model under early stopping (the
        #: LightGBM convention: 1-based count), None otherwise.
        self.best_iteration_: int | None = None

    def fit(
        self,
        X: Any,
        y: Any | None = None,
        eval_set: list | None = None,
        early_stopping_rounds: int | None = None,
    ) -> CatBoostExternalModel:
        """Fit the base model, optionally with CatBoost-native early stopping.

        Args:
            eval_set: List of (X, y) tuples or RepLeafDataset objects.
                CatBoost monitors the first entry.
            early_stopping_rounds: CatBoost early-stopping patience; requires
                eval_set. All grown trees are kept (``use_best_model=False``)
                and predictions are pinned to ``best_iteration_``, matching
                the LightGBM/XGBoost wrappers.
        """
        catboost = _require_catboost()
        X, y = _resolve_data(X, y)
        if y is None:
            raise ValueError("Training data must include a target (y)")
        cls = (
            catboost.CatBoostRegressor
            if self.task == "regression"
            else catboost.CatBoostClassifier
        )
        self.model_ = cls(random_seed=self.random_state, verbose=False, **self.cb_params)

        fit_kwargs: dict[str, Any] = {}
        if eval_set is not None:
            resolved = []
            for item in eval_set:
                if isinstance(item, tuple):
                    Xe, ye = _resolve_data(item[0], item[1])
                else:
                    Xe, ye = _resolve_data(item, None)
                if ye is None:
                    raise ValueError("eval_set entries must include a target (y)")
                resolved.append((Xe, np.asarray(ye, dtype=np.float64)))
            fit_kwargs["eval_set"] = resolved
            if early_stopping_rounds is not None:
                fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
                fit_kwargs["use_best_model"] = False
        elif early_stopping_rounds is not None:
            raise ValueError("early_stopping_rounds requires eval_set")

        self.model_.fit(X, y, **fit_kwargs)
        # CatBoost's best iteration is a 0-based index; expose the 1-based
        # tree count to match the other external models.
        self.best_iteration_ = (
            int(self.model_.get_best_iteration()) + 1
            if early_stopping_rounds is not None
            else None
        )
        return self

    def predict_score(self, X: Any) -> np.ndarray:
        """(n,) prediction: target estimate (regression) or P(y=1) (binary)."""
        self._check_fitted()
        X, _ = _resolve_data(X, None)
        end = self.best_iteration_ or 0  # 0 means all trees in CatBoost
        if self.task == "binary":
            proba = self.model_.predict_proba(X, ntree_end=end)
            return np.asarray(proba[:, 1], dtype=np.float64)
        return np.asarray(self.model_.predict(X, ntree_end=end), dtype=np.float64)

    def predict_leaf_indices(self, X: Any) -> np.ndarray:
        """(n, n_trees) int32 leaf assignment per tree (best trees under
        early stopping)."""
        self._check_fitted()
        X, _ = _resolve_data(X, None)
        end = self.best_iteration_ or 0
        leaves = np.asarray(
            self.model_.calc_leaf_indexes(X, ntree_start=0, ntree_end=end),
            dtype=np.int32,
        )
        if leaves.ndim == 1:
            leaves = leaves.reshape(-1, 1)
        return leaves

    @property
    def n_trees_(self) -> int:
        self._check_fitted()
        return int(self.model_.tree_count_)

    def _check_fitted(self) -> None:
        if self.model_ is None:
            raise RuntimeError("CatBoostExternalModel is not fitted yet; call fit() first")
