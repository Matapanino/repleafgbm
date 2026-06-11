"""LightGBM as an external base model (external_model mode).

This is deliberately *not* a routing backend: the LightGBM model is trained
independently, and only its outputs (scores, leaf indices) are exposed as
features for RepLeafGBM. Reusing LightGBM's routing inside RepLeafGBM is the
separate router_extraction mode (docs/adr/0002-router-extraction.md, design
only).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from repleafgbm.data import RepLeafDataset

_DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 31,
}


def _require_lightgbm():
    try:
        import lightgbm
    except ImportError as exc:  # pragma: no cover - exercised via mocked test
        raise ImportError(
            "LightGBM is required for repleafgbm.external.LightGBMExternalModel. "
            'Install it with: pip install lightgbm  (or pip install "repleafgbm[external]")'
        ) from exc
    return lightgbm


def _resolve_data(X: Any, y: Any | None) -> tuple[Any, Any | None]:
    """Accept arrays/DataFrames as-is; unwrap RepLeafDataset to encoded floats.

    Going through RepLeafDataset is the recommended path for categorical
    data: both RepLeafGBM and the external model then see the same
    ordinal-encoded matrix (unseen categories as NaN).
    """
    if isinstance(X, RepLeafDataset):
        if y is not None:
            raise ValueError("Pass y inside the RepLeafDataset, not separately")
        return X.get_raw_features(), X.y
    return X, y


class LightGBMExternalModel:
    """LightGBM base model with score and leaf-index extraction.

    Args:
        task: "regression" or "binary".
        random_state: Seed forwarded to LightGBM.
        **lgb_params: Any LGBMRegressor/LGBMClassifier parameters; merged
            over defaults (n_estimators=200, learning_rate=0.05,
            num_leaves=31).

    Example:
        >>> base = LightGBMExternalModel(task="regression", random_state=42)
        >>> base.fit(X_train, y_train)
        >>> score = base.predict_score(X)         # (n,)
        >>> leaves = base.predict_leaf_indices(X)  # (n, n_trees) int32
    """

    def __init__(self, task: str = "regression", random_state: int = 0, **lgb_params: Any):
        if task not in ("regression", "binary"):
            raise ValueError(f"task must be 'regression' or 'binary', got {task!r}")
        self.task = task
        self.random_state = random_state
        self.lgb_params = {**_DEFAULT_PARAMS, **lgb_params}
        self.model_: Any = None
        self.best_iteration_: int | None = None

    def fit(
        self,
        X: Any,
        y: Any | None = None,
        eval_set: list | None = None,
        early_stopping_rounds: int | None = None,
    ) -> LightGBMExternalModel:
        """Fit the base model, optionally with LightGBM-native early stopping.

        Args:
            eval_set: List of (X, y) tuples or RepLeafDataset objects.
            early_stopping_rounds: LightGBM early-stopping patience; requires
                eval_set. ``best_iteration_`` is set, predictions use the
                best iteration, and ``extract_routes`` only extracts trees up
                to it.
        """
        lgb = _require_lightgbm()
        X, y = _resolve_data(X, y)
        if y is None:
            raise ValueError("Training data must include a target (y)")
        cls = lgb.LGBMRegressor if self.task == "regression" else lgb.LGBMClassifier
        self.model_ = cls(random_state=self.random_state, verbose=-1, **self.lgb_params)

        kwargs: dict[str, Any] = {}
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
            kwargs["eval_set"] = resolved
            if early_stopping_rounds is not None:
                kwargs["callbacks"] = [
                    lgb.early_stopping(early_stopping_rounds, verbose=False)
                ]
        elif early_stopping_rounds is not None:
            raise ValueError("early_stopping_rounds requires eval_set")

        self.model_.fit(X, y, **kwargs)
        self.best_iteration_ = (
            self.model_.best_iteration_ if early_stopping_rounds is not None else None
        )
        return self

    def predict_score(self, X: Any) -> np.ndarray:
        """(n,) prediction: target estimate (regression) or P(y=1) (binary)."""
        self._check_fitted()
        X, _ = _resolve_data(X, None)
        if self.task == "binary":
            return np.asarray(self.model_.predict_proba(X)[:, 1], dtype=np.float64)
        return np.asarray(self.model_.predict(X), dtype=np.float64)

    def predict_leaf_indices(self, X: Any) -> np.ndarray:
        """(n, n_trees) int32 leaf assignment per tree."""
        self._check_fitted()
        X, _ = _resolve_data(X, None)
        leaves = np.asarray(self.model_.predict(X, pred_leaf=True), dtype=np.int32)
        if leaves.ndim == 1:
            leaves = leaves.reshape(-1, 1)
        return leaves

    @property
    def n_trees_(self) -> int:
        self._check_fitted()
        return int(self.model_.booster_.num_trees())

    def _check_fitted(self) -> None:
        if self.model_ is None:
            raise RuntimeError("LightGBMExternalModel is not fitted yet; call fit() first")
