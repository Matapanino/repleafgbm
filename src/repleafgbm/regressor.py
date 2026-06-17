"""RepLeafRegressor: sklearn-compatible regression estimator."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import RegressorMixin

from repleafgbm.core.booster import Booster, BoosterParams
from repleafgbm.core.multioutput import MultiOutputBooster
from repleafgbm.core.objectives import MultiOutputSquaredError
from repleafgbm.data import RepLeafDataset
from repleafgbm.sklearn import BaseRepLeafModel


class RepLeafRegressor(RegressorMixin, BaseRepLeafModel):
    """Gradient boosting regressor with representation-conditioned leaves.

    Squared-error objective by default; ``objective`` accepts "huber",
    "quantile", and "poisson" (or parameterized instances such as
    ``Quantile(alpha=0.9)``) for robust, quantile, and count regression.
    The default eval metric stays "rmse" — for quantile models consider a
    pinball loss via :func:`repleafgbm.make_metric`. See
    :class:`~repleafgbm.sklearn.BaseRepLeafModel` for all hyperparameters.

    Multi-output regression: pass a 2-D ``y`` of shape (n_rows, n_outputs) and
    the model grows shared-routing **vector leaves** (one tree per round whose
    leaves emit a vector; routing is shared across outputs). Multi-output is
    squared-error only — the ``objective`` parameter must stay at its default.
    ``predict`` then returns an (n_rows, n_outputs) array.

    Example:
        >>> model = RepLeafRegressor(n_estimators=50, leaf_model="embedded_linear",
        ...                          encoder="plr", random_state=42)
        >>> model.fit(X_train, y_train)
        >>> pred = model.predict(X_test)
    """

    _objective_name = "squared_error"
    _eval_metric_name = "rmse"

    def _prepare_target(self, dataset: RepLeafDataset, is_train: bool) -> RepLeafDataset:
        dataset = super()._prepare_target(dataset, is_train)
        if is_train:
            self.n_outputs_ = dataset.y.shape[1] if dataset.y.ndim == 2 else 1
        return dataset

    def _make_booster(self, params: BoosterParams) -> Booster | MultiOutputBooster:
        if getattr(self, "n_outputs_", 1) > 1:
            if getattr(self, "objective", None) is not None:
                raise ValueError(
                    "multi-output regression (2-D y) supports squared error "
                    "only; leave the objective parameter at its default"
                )
            return MultiOutputBooster(
                params, MultiOutputSquaredError(self.n_outputs_)
            )
        return super()._make_booster(params)

    def _pretrain_target(
        self, dataset: RepLeafDataset, sample_weight: np.ndarray | None = None
    ) -> np.ndarray | None:
        # The multi-output Newton residual is a matrix; learned-encoder
        # pretraining targets are scalar for now, so multi-output encoders fit
        # unsupervised (identity/plr are unaffected). A scalar multi-output
        # pretraining target is a roadmap item (docs/roadmap.md).
        if getattr(self, "n_outputs_", 1) > 1:
            return None
        return super()._pretrain_target(dataset, sample_weight=sample_weight)

    def predict(self, X: Any) -> np.ndarray:
        """Predict target values for X (array, DataFrame, or RepLeafDataset).

        Predictions are on the target scale: the objective's output transform
        is applied to the raw score (identity for squared error, huber, and
        quantile; exp for poisson, whose raw score is the log-mean).
        """
        raw = self._predict_raw(X)  # checks fitted state first
        return self.booster_.objective.transform(raw)
