"""sklearn-compatible base estimator shared by regressor and classifier.

This module glues the three core components together (dataset, encoder,
booster) and exposes the familiar fit/predict/save_model/load_model surface.
All heavy lifting lives in :mod:`repleafgbm.core`.
"""

from __future__ import annotations

import copy
import inspect
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator

from repleafgbm.core.booster import Booster, BoosterParams
from repleafgbm.core.leaf_models import make_leaf_model
from repleafgbm.core.metrics import get_metric
from repleafgbm.core.objectives import get_objective
from repleafgbm.core.serialization import load_model_dir, save_model_dir
from repleafgbm.data import RepLeafDataset
from repleafgbm.encoders import RandomProjectionEncoder, make_encoder
from repleafgbm.encoders.base import BaseEncoder


class BaseRepLeafModel(BaseEstimator):
    """Shared implementation for RepLeafRegressor / RepLeafClassifier.

    Args:
        n_estimators: Number of boosting rounds (trees).
        learning_rate: Shrinkage applied to every tree's contribution.
        num_leaves: Maximum leaves per tree (leaf-wise growth).
        max_depth: Maximum tree depth; -1 means unlimited.
        min_samples_leaf: Minimum rows per leaf for a split to be valid.
        leaf_model: "constant", "embedded_linear", or "raw_linear".
        encoder: Encoder name ("identity", "plr") or a BaseEncoder instance.
            Ignored for leaf_model="constant"; for "raw_linear" a standardizing
            identity encoder is always used.
        encoder_params: Keyword arguments for the encoder constructor when
            ``encoder`` is a name (e.g. {"n_bins": 16} for "plr").
        freeze_encoder: Must be True in v0. Encoder parameters are fitted once
            before boosting and never updated during boosting.
        max_leaf_emb_dim: Upper bound on the embedding dimension used by leaf
            models. Wider encoders are reduced with a fixed random projection
            — but note that projection consistently degraded accuracy in
            experiments (experiments/results/plr_projection_gap.md); a
            UserWarning is emitted when it engages. Prefer reducing the
            encoder dimension (e.g. fewer PLR bins) instead.
        l2_leaf: Ridge penalty for leaf models and the split gain denominator.
        max_bins: Histogram bins per feature for split search.
        early_stopping_rounds: Stop training when the first eval_set's metric
            has not improved for this many rounds; ``best_iteration_`` is set
            and ``predict`` uses the best iteration. Requires eval_set.
        eval_metric: Metric name for eval_set monitoring (and early stopping).
            Defaults to "rmse" for regression and "logloss" for
            classification; also available: "mae", "auc", "accuracy".
        random_state: Seed controlling all internal randomness.
    """

    _objective_name: str = "squared_error"
    _eval_metric_name: str = "rmse"

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        max_depth: int = -1,
        min_samples_leaf: int = 20,
        leaf_model: str = "embedded_linear",
        encoder: str | BaseEncoder = "identity",
        encoder_params: dict | None = None,
        freeze_encoder: bool = True,
        max_leaf_emb_dim: int = 64,
        l2_leaf: float = 1.0,
        max_bins: int = 256,
        early_stopping_rounds: int | None = None,
        eval_metric: str | None = None,
        random_state: int | None = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.leaf_model = leaf_model
        self.encoder = encoder
        self.encoder_params = encoder_params
        self.freeze_encoder = freeze_encoder
        self.max_leaf_emb_dim = max_leaf_emb_dim
        self.l2_leaf = l2_leaf
        self.max_bins = max_bins
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric
        self.random_state = random_state

    # ------------------------------------------------------------------ #
    # Fitting
    # ------------------------------------------------------------------ #
    def fit(
        self,
        X: Any,
        y: Any | None = None,
        eval_set: list[RepLeafDataset | tuple] | None = None,
    ) -> BaseRepLeafModel:
        """Fit on (X, y) arrays/DataFrames or a RepLeafDataset.

        Args:
            X: Feature matrix or a RepLeafDataset that already contains y.
            y: Target vector; must be None when X is a RepLeafDataset.
            eval_set: Optional list of RepLeafDataset (or (X, y) tuples)
                evaluated after every boosting round; results are stored in
                ``evals_result_``.
        """
        if not self.freeze_encoder:
            raise NotImplementedError(
                "freeze_encoder=False (encoder updates during boosting) is not "
                "supported in v0; see docs/roadmap.md"
            )

        dataset = self._build_dataset(X, y)
        dataset = self._prepare_target(dataset, is_train=True)
        self.metadata_ = dataset.metadata
        self.n_features_in_ = dataset.n_features
        self.feature_names_in_ = np.asarray(dataset.feature_names, dtype=object)

        leaf_model = make_leaf_model(
            self.leaf_model, l2=self.l2_leaf, min_samples_linear=2 * self.min_samples_leaf
        )
        self.encoder_ = (
            self._build_and_fit_encoder(dataset) if leaf_model.uses_embeddings else None
        )

        eval_sets = self._prepare_eval_sets(eval_set)

        if self.early_stopping_rounds is not None and not eval_sets:
            raise ValueError(
                "early_stopping_rounds requires eval_set; pass eval_set=[...] to fit()"
            )
        params = BoosterParams(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            l2_leaf=self.l2_leaf,
            max_bins=self.max_bins,
            early_stopping_rounds=self.early_stopping_rounds,
        )
        self.booster_ = Booster(params, get_objective(self._objective_name))
        self.booster_.fit(
            dataset,
            self.encoder_,
            leaf_model,
            eval_sets=eval_sets or None,
            eval_metric=get_metric(self.eval_metric or self._eval_metric_name),
        )
        self.evals_result_ = self.booster_.evals_result_
        return self

    def get_feature_importance(self, importance_type: str = "gain") -> np.ndarray:
        """Raw per-feature importance ("gain" or "split"), aligned with
        ``feature_names_in_``. Aggregated over the trees prediction uses
        (i.e. up to ``best_iteration_`` under early stopping)."""
        self._check_is_fitted()
        return self.booster_.feature_importance(
            self.n_features_in_, importance_type=importance_type
        )

    @property
    def feature_importances_(self) -> np.ndarray:
        """Gain-based importances normalized to sum to 1 (sklearn convention)."""
        importance = self.get_feature_importance("gain")
        total = importance.sum()
        return importance / total if total > 0 else importance

    @property
    def best_iteration_(self) -> int | None:
        """Best number of trees found by early stopping (None if unused/unfitted)."""
        booster = getattr(self, "booster_", None)
        return None if booster is None else booster.best_iteration_

    @property
    def best_score_(self) -> float | None:
        """Eval metric value at ``best_iteration_`` (None if unused/unfitted)."""
        booster = getattr(self, "booster_", None)
        return None if booster is None else booster.best_score_

    def _build_dataset(self, X: Any, y: Any | None) -> RepLeafDataset:
        if isinstance(X, RepLeafDataset):
            if y is not None:
                raise ValueError("Pass y inside the RepLeafDataset, not separately")
            return X
        return RepLeafDataset(X, y)

    def _prepare_eval_sets(self, eval_set: Any) -> list[tuple[str, RepLeafDataset]]:
        """Normalize eval_set entries (datasets or (X, y) tuples) for the booster."""
        eval_sets = []
        for i, item in enumerate(eval_set or []):
            if isinstance(item, RepLeafDataset):
                self._check_metadata_compatible(item, context=f"eval_set[{i}]")
                ds = item
            else:
                ds = self._eval_tuple_to_dataset(item)
            ds = self._prepare_target(ds, is_train=False)
            eval_sets.append((f"valid_{i}", ds))
        return eval_sets

    def _eval_tuple_to_dataset(self, item: tuple) -> RepLeafDataset:
        if not (isinstance(item, tuple) and len(item) == 2):
            raise ValueError(
                "eval_set entries must be RepLeafDataset objects or (X, y) tuples"
            )
        # Re-apply the training metadata so preprocessing matches.
        return RepLeafDataset(item[0], item[1], metadata=self.metadata_)

    def _prepare_target(self, dataset: RepLeafDataset, is_train: bool) -> RepLeafDataset:
        """Hook for label handling; classifier overrides to map labels to 0/1."""
        if dataset.y is None:
            raise ValueError("Training data must include a target (y)")
        if not np.issubdtype(dataset.y.dtype, np.number):
            raise ValueError(
                f"Regression target must be numeric, got dtype {dataset.y.dtype}"
            )
        return dataset

    def _build_and_fit_encoder(self, dataset: RepLeafDataset) -> BaseEncoder:
        if self.leaf_model == "raw_linear":
            encoder: BaseEncoder = make_encoder("identity", standardize=True)
        elif isinstance(self.encoder, BaseEncoder):
            encoder = copy.deepcopy(self.encoder)
        else:
            encoder = make_encoder(
                self.encoder,
                _default_random_state=self.random_state,
                **(self.encoder_params or {}),
            )

        X_num = dataset.get_numerical_features()
        if X_num.shape[1] == 0:
            raise ValueError(
                f"leaf_model={self.leaf_model!r} requires at least one numerical "
                "feature; use leaf_model='constant' for all-categorical data"
            )
        encoder.fit(X_num)
        if encoder.output_dim > self.max_leaf_emb_dim:
            warnings.warn(
                f"Encoder output dimension ({encoder.output_dim}) exceeds "
                f"max_leaf_emb_dim ({self.max_leaf_emb_dim}); applying a random "
                "projection. Projection consistently degraded accuracy in our "
                "experiments (experiments/results/plr_projection_gap.md) — "
                "prefer a lower-dimensional encoder (e.g. plr with fewer "
                "n_bins) or a larger max_leaf_emb_dim.",
                UserWarning,
                stacklevel=2,
            )
            encoder = RandomProjectionEncoder(
                encoder, out_dim=self.max_leaf_emb_dim, random_state=self.random_state
            ).fit(X_num)
        return encoder

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #
    def _predict_raw(self, X: Any) -> np.ndarray:
        self._check_is_fitted()
        if isinstance(X, RepLeafDataset):
            self._check_metadata_compatible(X, context="predict input")
            dataset = X
        else:
            dataset = RepLeafDataset(X, metadata=self.metadata_)
        X_raw = dataset.get_raw_features()
        Z = dataset.get_embeddings(self.encoder_) if self.encoder_ is not None else None
        return self.booster_.predict_raw(X_raw, Z)

    def _check_metadata_compatible(self, dataset: RepLeafDataset, context: str) -> None:
        """Reject datasets whose preprocessing differs from training.

        A RepLeafDataset built independently from a different sample can map
        the same category to a different ordinal code (e.g. when the sample is
        missing a category), which would silently corrupt evaluation or
        prediction. Equality of the metadata is required; share it explicitly:

            valid = RepLeafDataset(X, y, metadata=train_data.metadata)
        """
        if dataset.metadata is self.metadata_:
            return
        if dataset.metadata.to_dict() != self.metadata_.to_dict():
            raise ValueError(
                f"{context} was built with feature metadata that differs from "
                "the training data (feature names/types or categorical code "
                "maps do not match), which would silently mis-encode features. "
                "Construct it with the training metadata: "
                "RepLeafDataset(X, y, metadata=train_data.metadata) or "
                "metadata=model.metadata_."
            )

    def _check_is_fitted(self) -> None:
        if getattr(self, "booster_", None) is None:
            raise RuntimeError(
                f"{type(self).__name__} is not fitted yet; call fit() first"
            )

    # ------------------------------------------------------------------ #
    # Save / load
    # ------------------------------------------------------------------ #
    def save_model(self, path: str | Path) -> None:
        """Save the fitted model to a directory (docs/serialization.md)."""
        self._check_is_fitted()
        config = self._serializable_config(self.get_params())
        config.update(self._extra_config())
        save_model_dir(
            path,
            model_class=type(self).__name__,
            config=config,
            booster=self.booster_,
            encoder=self.encoder_,
            metadata=self.metadata_,
        )

    def _serializable_config(self, config: dict) -> dict:
        """Make get_params() JSON-safe; subclasses extend for their params."""
        # Encoder instances are not JSON; persist the fitted encoder separately
        # and record only its registry name here.
        if isinstance(config.get("encoder"), BaseEncoder):
            config["encoder"] = config["encoder"].name
        return config

    def _extra_config(self) -> dict:
        """Subclass hook for extra serialized state (e.g. class labels)."""
        return {}

    @classmethod
    def load_model(cls, path: str | Path) -> BaseRepLeafModel:
        """Load a model saved with :meth:`save_model`."""
        parts = load_model_dir(path)
        if parts["model_class"] != cls.__name__:
            raise ValueError(
                f"Model directory contains a {parts['model_class']}, "
                f"but load_model was called on {cls.__name__}"
            )
        init_params = set(inspect.signature(cls.__init__).parameters) - {"self"}
        kwargs = {k: v for k, v in parts["config"].items() if k in init_params}
        model = cls(**kwargs)
        model.booster_ = parts["booster"]
        model.encoder_ = parts["encoder"]
        model.metadata_ = parts["metadata"]
        model.n_features_in_ = parts["metadata"].n_features
        model.feature_names_in_ = np.asarray(parts["metadata"].feature_names, dtype=object)
        model._restore_extra_config(parts["config"])
        return model

    def _restore_extra_config(self, config: dict) -> None:
        """Subclass hook mirroring :meth:`_extra_config`."""
