"""RepLeafRegressor: sklearn-compatible regression estimator."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import RegressorMixin

from repleafgbm.sklearn import BaseRepLeafModel


class RepLeafRegressor(RegressorMixin, BaseRepLeafModel):
    """Gradient boosting regressor with representation-conditioned leaves.

    Squared-error objective by default; ``objective`` accepts "huber",
    "quantile", and "poisson" (or parameterized instances such as
    ``Quantile(alpha=0.9)``) for robust, quantile, and count regression.
    The default eval metric stays "rmse" — for quantile models consider a
    pinball loss via :func:`repleafgbm.make_metric`. See
    :class:`~repleafgbm.sklearn.BaseRepLeafModel` for all hyperparameters.

    Example:
        >>> model = RepLeafRegressor(n_estimators=50, leaf_model="embedded_linear",
        ...                          encoder="plr", random_state=42)
        >>> model.fit(X_train, y_train)
        >>> pred = model.predict(X_test)
    """

    _objective_name = "squared_error"
    _eval_metric_name = "rmse"

    def predict(self, X: Any) -> np.ndarray:
        """Predict target values for X (array, DataFrame, or RepLeafDataset).

        Predictions are on the target scale: the objective's output transform
        is applied to the raw score (identity for squared error, huber, and
        quantile; exp for poisson, whose raw score is the log-mean).
        """
        raw = self._predict_raw(X)  # checks fitted state first
        return self.booster_.objective.transform(raw)
