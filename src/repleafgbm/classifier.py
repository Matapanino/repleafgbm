"""RepLeafClassifier: sklearn-compatible binary and multiclass classifier.

Binary targets use the logistic objective (one tree per round). Targets with
three or more classes automatically use softmax boosting with one tree per
class per round (:mod:`repleafgbm.core.multiclass`); the routing/leaf-model
machinery and the frozen-encoder rule are identical in both modes.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.multiclass import type_of_target

from repleafgbm.core.booster import Booster, BoosterParams
from repleafgbm.core.metrics import BaseMetric, get_metric
from repleafgbm.core.multiclass import MulticlassBooster
from repleafgbm.core.objectives import (
    BaseObjective,
    BinaryLogistic,
    MulticlassSoftmax,
    _sigmoid,
    _softmax,
)
from repleafgbm.data import RepLeafDataset
from repleafgbm.sklearn import BaseRepLeafModel


class RepLeafClassifier(ClassifierMixin, BaseRepLeafModel):
    """Gradient boosting classifier with representation-conditioned leaves.

    Two classes: logistic objective on labels mapped to {0, 1}. Three or
    more classes: softmax objective, one tree per class per boosting round
    (``n_estimators`` counts rounds, so the ensemble holds
    ``n_estimators * n_classes`` trees). The default ``eval_metric`` is
    "logloss" for binary and "multi_logloss" for multiclass. See
    :class:`~repleafgbm.sklearn.BaseRepLeafModel` for hyperparameters.
    """

    _objective_name = "binary_logistic"
    _eval_metric_name = "logloss"
    #: Targets are class labels (not numeric) and never multi-output.
    _y_numeric = False
    _multi_output = False
    #: Subclasses that replay external binary routes (router_extraction)
    #: set this to False to keep rejecting 3+ class targets.
    _supports_multiclass = True

    def _prepare_target(self, dataset: RepLeafDataset, is_train: bool) -> RepLeafDataset:
        if dataset.y is None:
            raise ValueError("Training data must include a target (y)")
        if is_train:
            target_type = type_of_target(dataset.y)
            if target_type not in ("binary", "multiclass"):
                raise ValueError(
                    f"Unknown label type: {target_type!r}. RepLeafClassifier "
                    "requires a classification target (binary or multiclass); "
                    "use RepLeafRegressor for continuous targets."
                )
        if is_train and getattr(self, "objective", None) is not None:
            raise ValueError(
                "RepLeafClassifier selects its objective from the target "
                "(logistic for 2 classes, softmax for 3+); the objective "
                "parameter applies to RepLeafRegressor only"
            )
        classes = np.unique(dataset.y)
        if is_train:
            if classes.shape[0] < 2:
                raise ValueError(
                    f"RepLeafClassifier needs at least 2 classes; got "
                    f"{classes.shape[0]} class in the training target"
                )
            if classes.shape[0] > 2 and not self._supports_multiclass:
                raise ValueError(
                    f"{type(self).__name__} supports binary targets only; "
                    f"got {classes.shape[0]} classes"
                )
            self.classes_ = classes
        unknown = np.setdiff1d(classes, self.classes_)
        if unknown.size:
            raise ValueError(f"eval_set contains labels {unknown} not seen in training")
        if self.classes_.shape[0] == 2:
            y_enc = (dataset.y == self.classes_[1]).astype(np.float64)
        else:
            # classes_ is sorted (np.unique), so searchsorted is the code map.
            y_enc = np.searchsorted(self.classes_, dataset.y).astype(np.float64)
        # Shallow copy: share the encoded feature matrix, replace the target.
        remapped = copy.copy(dataset)
        remapped.y = y_enc
        return remapped

    def _resolve_sample_weight(
        self, dataset: RepLeafDataset, sample_weight: Any | None
    ) -> np.ndarray | None:
        """Fold ``class_weight`` into the per-row sample weight.

        The explicit ``sample_weight`` (validated by the base) is multiplied by
        the per-row weights implied by ``class_weight`` — a ``{label: weight}``
        dict (keyed by the original labels) or "balanced". ``dataset.y`` holds
        the 0..K-1 class codes at this point, so a dict's keys are remapped to
        codes via ``classes_``."""
        base = super()._resolve_sample_weight(dataset, sample_weight)
        class_weight = getattr(self, "class_weight", None)
        if class_weight is None:
            return base
        y_codes = dataset.y.astype(np.int64)
        if isinstance(class_weight, dict):
            remapped: dict[int, float] = {}
            for label, value in class_weight.items():
                idx = int(np.searchsorted(self.classes_, label))
                if idx >= self.classes_.shape[0] or self.classes_[idx] != label:
                    raise ValueError(
                        f"class_weight key {label!r} is not one of the training "
                        f"classes {list(self.classes_)}"
                    )
                remapped[idx] = value
            class_weight = remapped
        cw_weights = compute_sample_weight(class_weight, y_codes)
        return cw_weights if base is None else base * cw_weights

    @property
    def n_classes_(self) -> int:
        return int(self.classes_.shape[0])

    def _make_booster(self, params: BoosterParams) -> Booster | MulticlassBooster:
        if self.n_classes_ > 2:
            return MulticlassBooster(
                params,
                MulticlassSoftmax(
                    self.n_classes_, label_smoothing=self._label_smoothing
                ),
            )
        return super()._make_booster(params)

    def _build_objective(self) -> BaseObjective:
        """Binary logistic objective with the estimator's label smoothing.
        The objective is selected from the target (the ``objective`` parameter
        is rejected in :meth:`_prepare_target`)."""
        return BinaryLogistic(label_smoothing=self._label_smoothing)

    @property
    def _label_smoothing(self) -> float:
        """Label smoothing, defaulting to 0 for subclasses (router extraction)
        whose reduced ``__init__`` omits the parameter."""
        return getattr(self, "label_smoothing", 0.0)

    def _resolve_eval_metric(self) -> BaseMetric:
        if self.eval_metric is None and self.n_classes_ > 2:
            return get_metric("multi_logloss")
        return super()._resolve_eval_metric()

    def _pretrain_target(self, dataset: RepLeafDataset) -> np.ndarray | None:
        # The multiclass Newton residual is an (n_rows, n_classes) matrix;
        # learned-encoder pretraining targets are scalar for now, so
        # multiclass encoders fit unsupervised (identity/plr are unaffected).
        if self.n_classes_ > 2:
            return None
        return super()._pretrain_target(dataset)

    def predict_proba(self, X: Any) -> np.ndarray:
        """Class probabilities of shape (n_rows, n_classes), columns ordered
        by ``classes_``."""
        raw = self._predict_raw(X)
        if raw.ndim == 2:
            return _softmax(raw)
        p1 = _sigmoid(raw)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: Any) -> np.ndarray:
        """Predicted class labels."""
        proba = self.predict_proba(X)
        if self.n_classes_ == 2:  # keep the historical p >= 0.5 tie rule
            return self.classes_[(proba[:, 1] >= 0.5).astype(int)]
        return self.classes_[np.argmax(proba, axis=1)]

    def _extra_config(self) -> dict:
        return {"classes_": self.classes_.tolist()}

    def _restore_extra_config(self, config: dict) -> None:
        self.classes_ = np.asarray(config["classes_"])
