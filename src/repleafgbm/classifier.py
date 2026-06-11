"""RepLeafClassifier: sklearn-compatible binary classifier.

v0 supports binary classification only; multiclass is on the roadmap
(vector-leaf / one-vs-rest designs are discussed in docs/roadmap.md).
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
from sklearn.base import ClassifierMixin

from repleafgbm.core.objectives import _sigmoid
from repleafgbm.data import RepLeafDataset
from repleafgbm.sklearn import BaseRepLeafModel


class RepLeafClassifier(ClassifierMixin, BaseRepLeafModel):
    """Binary gradient boosting classifier with representation-conditioned leaves.

    Logistic objective on labels mapped to {0, 1}. See
    :class:`~repleafgbm.sklearn.BaseRepLeafModel` for hyperparameters.
    """

    _objective_name = "binary_logistic"
    _eval_metric_name = "logloss"

    def _prepare_target(self, dataset: RepLeafDataset, is_train: bool) -> RepLeafDataset:
        if dataset.y is None:
            raise ValueError("Training data must include a target (y)")
        classes = np.unique(dataset.y)
        if is_train:
            if classes.shape[0] != 2:
                raise ValueError(
                    f"RepLeafClassifier supports binary targets in v0; "
                    f"got {classes.shape[0]} classes"
                )
            self.classes_ = classes
        unknown = np.setdiff1d(classes, self.classes_)
        if unknown.size:
            raise ValueError(f"eval_set contains labels {unknown} not seen in training")
        y01 = (dataset.y == self.classes_[1]).astype(np.float64)
        # Shallow copy: share the encoded feature matrix, replace the target.
        remapped = copy.copy(dataset)
        remapped.y = y01
        return remapped

    def predict_proba(self, X: Any) -> np.ndarray:
        """Class probabilities of shape (n_rows, 2), columns ordered by classes_."""
        p1 = self._sigmoid_predict(X)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: Any) -> np.ndarray:
        """Predicted class labels."""
        p1 = self._sigmoid_predict(X)
        return self.classes_[(p1 >= 0.5).astype(int)]

    def _sigmoid_predict(self, X: Any) -> np.ndarray:
        raw = self._predict_raw(X)
        # Numerically stable sigmoid (no overflow for large |raw|).
        return _sigmoid(raw)

    def _extra_config(self) -> dict:
        return {"classes_": self.classes_.tolist()}

    def _restore_extra_config(self, config: dict) -> None:
        self.classes_ = np.asarray(config["classes_"])
