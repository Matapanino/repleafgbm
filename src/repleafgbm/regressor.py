"""RepLeafRegressor: sklearn-compatible regression estimator."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import RegressorMixin

from repleafgbm.sklearn import BaseRepLeafModel


class RepLeafRegressor(RegressorMixin, BaseRepLeafModel):
    """Gradient boosting regressor with representation-conditioned leaves.

    Squared-error objective. See :class:`~repleafgbm.sklearn.BaseRepLeafModel`
    for all hyperparameters.

    Example:
        >>> model = RepLeafRegressor(n_estimators=50, leaf_model="embedded_linear",
        ...                          encoder="plr", random_state=42)
        >>> model.fit(X_train, y_train)
        >>> pred = model.predict(X_test)
    """

    _objective_name = "squared_error"
    _eval_metric_name = "rmse"

    def predict(self, X: Any) -> np.ndarray:
        """Predict target values for X (array, DataFrame, or RepLeafDataset)."""
        return self._predict_raw(X)
