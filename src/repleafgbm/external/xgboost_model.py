"""XGBoost as an external base model (external_model mode).

Same contract as :class:`~repleafgbm.external.lightgbm_model.LightGBMExternalModel`:
the XGBoost model is trained independently and only its outputs (scores,
leaf indices) are exposed as stacking features. It plugs into
``oof_predictions`` / ``external_feature_frame`` / ``augment_features``
unchanged (they are duck-typed over ``fit`` / ``predict_score`` /
``predict_leaf_indices``). Route extraction (router_extraction mode) remains
LightGBM-only.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from repleafgbm.external.lightgbm_model import _resolve_data

_DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "learning_rate": 0.05,
}


def _require_xgboost():
    try:
        import xgboost
    except ImportError as exc:  # pragma: no cover - exercised via mocked test
        raise ImportError(
            "XGBoost is required for repleafgbm.external.XGBoostExternalModel. "
            "Install it with: pip install xgboost"
        ) from exc
    return xgboost


class XGBoostExternalModel:
    """XGBoost base model with score and leaf-index extraction.

    Args:
        task: "regression" or "binary".
        random_state: Seed forwarded to XGBoost.
        **xgb_params: Any XGBRegressor/XGBClassifier parameters; merged over
            defaults (n_estimators=200, learning_rate=0.05). Custom XGBoost
            objectives go here too (e.g. ``objective="count:poisson"``).

    Example:
        >>> base = XGBoostExternalModel(task="regression", random_state=42)
        >>> base.fit(X_train, y_train)
        >>> score = base.predict_score(X)          # (n,)
        >>> leaves = base.predict_leaf_indices(X)  # (n, n_trees) int32
    """

    def __init__(self, task: str = "regression", random_state: int = 0, **xgb_params: Any):
        if task not in ("regression", "binary"):
            raise ValueError(f"task must be 'regression' or 'binary', got {task!r}")
        self.task = task
        self.random_state = random_state
        self.xgb_params = {**_DEFAULT_PARAMS, **xgb_params}
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
    ) -> XGBoostExternalModel:
        """Fit the base model, optionally with XGBoost-native early stopping.

        Args:
            eval_set: List of (X, y) tuples or RepLeafDataset objects.
            early_stopping_rounds: XGBoost early-stopping patience; requires
                eval_set. ``best_iteration_`` is set and predictions use the
                best iteration.
        """
        xgb = _require_xgboost()
        X, y = _resolve_data(X, y)
        if y is None:
            raise ValueError("Training data must include a target (y)")
        cls = xgb.XGBRegressor if self.task == "regression" else xgb.XGBClassifier
        # XGBoost takes the patience at construction time (fit-time argument
        # was removed in 2.0).
        self.model_ = cls(
            random_state=self.random_state,
            verbosity=0,
            early_stopping_rounds=early_stopping_rounds,
            **self.xgb_params,
        )

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
            fit_kwargs["verbose"] = False
        elif early_stopping_rounds is not None:
            raise ValueError("early_stopping_rounds requires eval_set")

        self.model_.fit(X, y, **fit_kwargs)
        # xgboost's best_iteration is a 0-based index; expose the 1-based
        # tree count to match LightGBMExternalModel.
        self.best_iteration_ = (
            int(self.model_.best_iteration) + 1
            if early_stopping_rounds is not None
            else None
        )
        return self

    def predict_score(self, X: Any) -> np.ndarray:
        """(n,) prediction: target estimate (regression) or P(y=1) (binary)."""
        self._check_fitted()
        X, _ = _resolve_data(X, None)
        kwargs = self._iteration_kwargs()
        if self.task == "binary":
            return np.asarray(self.model_.predict_proba(X, **kwargs)[:, 1], dtype=np.float64)
        return np.asarray(self.model_.predict(X, **kwargs), dtype=np.float64)

    def predict_leaf_indices(self, X: Any) -> np.ndarray:
        """(n, n_trees) int32 leaf assignment per tree (best trees under
        early stopping)."""
        self._check_fitted()
        X, _ = _resolve_data(X, None)
        leaves = np.asarray(
            self.model_.apply(X, **self._iteration_kwargs()), dtype=np.int32
        )
        if leaves.ndim == 1:
            leaves = leaves.reshape(-1, 1)
        return leaves

    def _iteration_kwargs(self) -> dict[str, Any]:
        """Pin predictions to the early-stopping best model explicitly
        (apply() does not honor best_iteration on its own)."""
        if self.best_iteration_ is None:
            return {}
        return {"iteration_range": (0, self.best_iteration_)}

    @property
    def n_trees_(self) -> int:
        self._check_fitted()
        return int(self.model_.get_booster().num_boosted_rounds())

    def _check_fitted(self) -> None:
        if self.model_ is None:
            raise RuntimeError("XGBoostExternalModel is not fitted yet; call fit() first")
