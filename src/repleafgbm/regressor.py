"""RepLeafRegressor: sklearn-compatible regression estimator."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import RegressorMixin

from repleafgbm.core.booster import Booster, BoosterParams
from repleafgbm.core.multioutput import MultiOutputBooster
from repleafgbm.core.objectives import (
    Huber,
    MultiOutputHuber,
    MultiOutputObjective,
    MultiOutputQuantile,
    MultiOutputSquaredError,
    Quantile,
    SquaredError,
)
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
    leaves emit a vector; routing is shared across outputs). Multi-output
    supports the constant-Hessian losses — squared error (default), "huber",
    and "quantile" (or instances such as ``Quantile(alpha=0.9)``); "poisson" is
    rejected (its Hessian is not constant across outputs). ``predict`` then
    returns an (n_rows, n_outputs) array.

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

    def _build_multioutput_objective(self) -> MultiOutputObjective:
        """Map the ``objective`` parameter onto its multi-output counterpart.

        Multi-output regression shares the constant-Hessian (h=1) family with
        the scalar path: squared error (default), Huber, and quantile. The
        scalar objective is resolved via
        :meth:`~repleafgbm.sklearn.BaseRepLeafModel._build_objective` (honoring
        registered names and parameterized instances such as
        ``Quantile(alpha=0.9)``), then lifted to the vector loss; loss
        parameters (``delta``/``alpha``) carry over. Non-constant-Hessian
        objectives (e.g. poisson) are rejected — they would break the
        shared-Gram vector-leaf solve (docs/math.md).
        """
        n_outputs = self.n_outputs_
        if getattr(self, "objective", None) is None:
            return MultiOutputSquaredError(n_outputs)
        scalar = self._build_objective()
        if isinstance(scalar, SquaredError):
            return MultiOutputSquaredError(n_outputs)
        if isinstance(scalar, Huber):
            return MultiOutputHuber(n_outputs, delta=scalar.delta)
        if isinstance(scalar, Quantile):
            return MultiOutputQuantile(n_outputs, alpha=scalar.alpha)
        raise ValueError(
            "multi-output regression (2-D y) supports the constant-Hessian "
            "losses squared_error, huber, and quantile only; objective "
            f"{scalar.name!r} is not supported for 2-D y"
        )

    def _make_booster(self, params: BoosterParams) -> Booster | MultiOutputBooster:
        if getattr(self, "n_outputs_", 1) > 1:
            return MultiOutputBooster(params, self._build_multioutput_objective())
        return super()._make_booster(params)

    def _pretrain_target(
        self, dataset: RepLeafDataset, sample_weight: np.ndarray | None = None
    ) -> np.ndarray | None:
        # Multi-output: the supervised pretraining target is the (n_rows,
        # n_outputs) negative-gradient residual at the weighted initial score,
        # matching the first round of MultiOutputBooster for the chosen
        # objective (``Y - mean`` for squared error; clipped/quantile residuals
        # at the per-output median/alpha-quantile for huber/quantile). Learned
        # encoders pretrain a K-output head on it (torch_encoders._pretrain);
        # unlearned encoders ignore it. A single output keeps the scalar
        # residual. See docs/math.md.
        if getattr(self, "n_outputs_", 1) <= 1:
            return super()._pretrain_target(dataset, sample_weight=sample_weight)
        objective = self._build_multioutput_objective()
        init = objective.init_score(dataset.y, weight=sample_weight)  # (n_outputs,)
        f0 = np.tile(init, (dataset.n_rows, 1))  # (n, n_outputs)
        grad0, _ = objective.grad_hess(dataset.y, f0)
        return -grad0

    def predict(self, X: Any) -> np.ndarray:
        """Predict target values for X (array, DataFrame, or RepLeafDataset).

        Predictions are on the target scale: the objective's output transform
        is applied to the raw score (identity for squared error, huber, and
        quantile; exp for poisson, whose raw score is the log-mean).
        """
        raw = self._predict_raw(X)  # checks fitted state first
        return self.booster_.objective.transform(raw)
