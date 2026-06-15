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
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import column_or_1d

from repleafgbm.core.booster import Booster, BoosterParams
from repleafgbm.core.leaf_models import make_leaf_model
from repleafgbm.core.metrics import BaseMetric, get_metric, make_metric
from repleafgbm.core.objectives import BaseObjective, get_objective
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
        cat_smooth / min_data_per_group / max_cat_threshold: Categorical
            subset-split guards with LightGBM semantics and defaults:
            Hessian smoothing of the category sort ratio, minimum node rows
            for a category to be eligible for the left subset, and the cap
            on left-subset size (scanned from both ends of the sorted
            order). See docs/categorical_features.md.
        split_backend: Split kernel implementation: "auto" (compiled Rust
            kernels when the optional ``repleafgbm_native`` extension is
            installed, NumPy otherwise), "numpy", or "rust". Backends agree
            to floating-point noise; same seed + same backend is fully
            deterministic.
        early_stopping_rounds: Stop training when the first eval_set's metric
            has not improved for this many rounds; ``best_iteration_`` is set
            and ``predict`` uses the best iteration. Requires eval_set.
        eval_metric: Metric for eval_set monitoring (and early stopping). A
            registered name ("rmse", "mae", "logloss", "auc", "accuracy"), a
            :class:`~repleafgbm.core.metrics.BaseMetric` instance, or a plain
            ``(y_true, y_pred) -> float`` callable (wrapped via
            :func:`~repleafgbm.core.metrics.make_metric`; assumed
            smaller-is-better — pass a ``make_metric(..., minimize=False)``
            result for greater-is-better metrics). Defaults to "rmse" for
            regression and "logloss" for classification. Custom metrics are
            saved by name only: a reloaded model must be given the metric
            object again before refitting with eval sets.
        objective: Regression loss override: a registered name
            ("squared_error", "huber", "quantile", "poisson") or a
            :class:`~repleafgbm.core.objectives.BaseObjective` instance for
            non-default parameters (e.g. ``Huber(delta=2.0)``,
            ``Quantile(alpha=0.9)``). None uses the estimator's default
            (squared error). The classifier selects its objective from the
            target and rejects this parameter. Like custom metrics, objective
            instances are saved by registry name only: predictions reload
            exactly, but refitting a reloaded model uses the named
            objective's default parameters unless the instance is passed
            again.
        label_smoothing: Classification only (ignored by the regressor). If
            >0, hard targets are softened before computing gradients —
            ``y*(1-eps) + eps/2`` for binary, ``(1-eps)*onehot + eps/K`` for
            multiclass — which regularizes over-confident probabilities. It is
            restored from the saved config on reload (the objective itself
            serializes by name only). Must be in [0, 1).
        random_state: Seed controlling all internal randomness.
    """

    _objective_name: str = "squared_error"
    _eval_metric_name: str = "rmse"
    #: Whether the target must be numeric (regressor) or may be labels
    #: (classifier overrides to False). Drives array validation of ``y``.
    _y_numeric: bool = True

    def _more_tags(self) -> dict:
        # NaN is a supported value (missing routes left), so check_estimator's
        # finiteness checks should test inf-rejection only, not NaN-rejection.
        # Both estimators are supervised and require y.
        return {"allow_nan": True, "requires_y": True}

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
        cat_smooth: float = 10.0,
        min_data_per_group: int = 100,
        max_cat_threshold: int = 32,
        split_backend: str = "auto",
        early_stopping_rounds: int | None = None,
        eval_metric: str | BaseMetric | Any | None = None,
        objective: str | BaseObjective | None = None,
        label_smoothing: float = 0.0,
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
        self.cat_smooth = cat_smooth
        self.min_data_per_group = min_data_per_group
        self.max_cat_threshold = max_cat_threshold
        self.split_backend = split_backend
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric
        self.objective = objective
        self.label_smoothing = label_smoothing
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
            cat_smooth=self.cat_smooth,
            min_data_per_group=self.min_data_per_group,
            max_cat_threshold=self.max_cat_threshold,
            split_backend=self.split_backend,
            early_stopping_rounds=self.early_stopping_rounds,
        )
        self.booster_ = self._make_booster(params)
        self.booster_.fit(
            dataset,
            self.encoder_,
            leaf_model,
            eval_sets=eval_sets or None,
            eval_metric=self._resolve_eval_metric(),
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

    def _make_booster(self, params: BoosterParams) -> Booster:
        """Hook: the classifier overrides this to return a
        :class:`~repleafgbm.core.multiclass.MulticlassBooster` for 3+ classes."""
        return Booster(params, self._build_objective())

    def _build_objective(self) -> BaseObjective:
        """Resolve the ``objective`` parameter (name, instance, or None for
        the estimator default). Subclasses with a reduced __init__ (router
        extraction) have no ``objective`` attribute and get the default."""
        objective = getattr(self, "objective", None)
        if objective is None:
            return get_objective(self._objective_name)
        if isinstance(objective, BaseObjective):
            return copy.deepcopy(objective)
        return get_objective(objective)

    def _resolve_eval_metric(self) -> BaseMetric:
        """Accept a metric name, BaseMetric instance, or plain callable."""
        metric = self.eval_metric
        if metric is None:
            return get_metric(self._eval_metric_name)
        if isinstance(metric, BaseMetric):
            return metric
        if callable(metric):
            return make_metric(metric)
        return get_metric(metric)

    #: Regression supports 2-D multi-output targets; the classifier does not.
    _multi_output: bool = True

    @staticmethod
    def _is_dataframe(X: Any) -> bool:
        """Duck-typed pandas DataFrame check (avoids a hard import here)."""
        return hasattr(X, "iloc") and hasattr(X, "columns")

    def _validate_array_X(self, X: Any) -> Any:
        """sklearn-conformant validation for plain array-likes (rejects sparse,
        complex, infinite, empty, and 1-D inputs with the standard messages).
        DataFrame / RepLeafDataset inputs keep their own preprocessing path,
        which handles categoricals. NaN is allowed (it routes left)."""
        return check_array(
            X, accept_sparse=False, force_all_finite="allow-nan", dtype="numeric"
        )

    def _validate_array_Xy(self, X: Any, y: Any) -> tuple[Any, Any]:
        X, y = check_X_y(
            X,
            y,
            accept_sparse=False,
            force_all_finite="allow-nan",
            dtype="numeric",
            multi_output=self._multi_output,
            y_numeric=self._y_numeric,
        )
        # A single-column 2-D target is raveled with a DataConversionWarning,
        # matching sklearn convention; a genuine multi-output (n, k>1) target
        # is kept (regression vector leaves, Phase 22).
        if self._multi_output and y.ndim == 2 and y.shape[1] == 1:
            y = column_or_1d(y, warn=True)
        return X, y

    def _build_dataset(self, X: Any, y: Any | None) -> RepLeafDataset:
        if isinstance(X, RepLeafDataset):
            if y is not None:
                raise ValueError("Pass y inside the RepLeafDataset, not separately")
            return X
        if not self._is_dataframe(X):
            if y is None:
                raise ValueError(
                    f"This {type(self).__name__} estimator requires y to be "
                    "passed, but the target y is None."
                )
            X, y = self._validate_array_Xy(X, y)
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
        # The encoder is fitted once here and frozen for all of boosting.
        encoder.fit(X_num, y=self._pretrain_target(dataset))
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

    def _pretrain_target(self, dataset: RepLeafDataset) -> np.ndarray | None:
        """Supervised pretraining target for learned encoders: the Newton
        residual at the initial score (y - mean for regression, y - sigmoid(F0)
        for binary). Unlearned encoders ignore it. The classifier overrides
        this to return None for multiclass targets (the residual is a matrix
        there; learned-encoder pretraining stays scalar-target for now)."""
        objective = self._build_objective()
        f0 = np.full(dataset.n_rows, objective.init_score(dataset.y))
        grad0, _ = objective.grad_hess(dataset.y, f0)
        return -grad0

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #
    def _predict_raw(self, X: Any) -> np.ndarray:
        self._check_is_fitted()
        if isinstance(X, RepLeafDataset):
            self._check_metadata_compatible(X, context="predict input")
            dataset = X
        else:
            if not self._is_dataframe(X):
                X = self._validate_array_X(X)
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
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet; call "
                "fit() with appropriate arguments before using this estimator."
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
        # Informational only; the loader never reads it.
        (Path(path) / "summary.txt").write_text(self.summary() + "\n")

    def summary(self, top_features: int = 10) -> str:
        """Human-readable description of the fitted model.

        Also written to ``summary.txt`` inside the model directory by
        :meth:`save_model` so a saved model can be inspected without loading
        it.
        """
        self._check_is_fitted()
        booster = self.booster_
        n_trees = len(booster.trees_)
        md = self.metadata_

        lines = [f"{type(self).__name__} (objective: {booster.objective.name})"]
        n_classes = getattr(booster, "n_classes", None)
        if n_classes is not None:  # multiclass: best_iteration_ counts rounds
            n_rounds = n_trees // n_classes
            used_rounds = booster.best_iteration_ or n_rounds
            used = (
                f"trees: {n_trees} grown ({n_rounds} rounds x {n_classes} "
                f"classes), {used_rounds * n_classes} used for prediction"
            )
        else:
            n_used = booster.best_iteration_ or n_trees
            used = f"trees: {n_trees} grown, {n_used} used for prediction"
        if booster.best_iteration_ is not None:
            used += f" (early stopping, best_score={booster.best_score_:.6g})"
        lines.append(used)
        # Subclasses (e.g. router extraction) drop some hyperparameters, so
        # describe only what this estimator actually has.
        params = self.get_params()
        leaf_line = f"leaf_model: {params.get('leaf_model', 'constant')}"
        if self.encoder_ is not None:
            leaf_line += (
                f", encoder: {self.encoder_.name} "
                f"(output_dim={self.encoder_.output_dim})"
            )
        lines.append(leaf_line)
        feats = (
            f"features: {md.n_features} ({len(md.numerical_features)} numerical, "
            f"{len(md.categorical_features)} categorical"
        )
        if md.frequency_maps:
            feats += f", {len(md.frequency_maps)} frequency-encoded"
        lines.append(feats + ")")
        shown = [
            f"{k}={params[k]}"
            for k in ("num_leaves", "learning_rate", "l2_leaf",
                      "min_samples_leaf", "max_bins")
            if k in params
        ]
        if shown:
            lines.append(", ".join(shown))

        importance = self.feature_importances_
        order = np.argsort(importance)[::-1]
        top = [i for i in order[:top_features] if importance[i] > 0]
        if top:
            lines.append(f"top features by gain (of {md.n_features}):")
            lines += [
                f"  {self.feature_names_in_[i]}: {importance[i]:.1%}" for i in top
            ]
        return "\n".join(lines)

    def _serializable_config(self, config: dict) -> dict:
        """Make get_params() JSON-safe; subclasses extend for their params."""
        # Encoder instances are not JSON; persist the fitted encoder separately
        # and record only its registry name here.
        if isinstance(config.get("encoder"), BaseEncoder):
            config["encoder"] = config["encoder"].name
        # Metric objects/callables are saved by name only; registered names
        # resolve on refit, custom metrics must be passed in again.
        metric = config.get("eval_metric")
        if isinstance(metric, BaseMetric):
            config["eval_metric"] = metric.name
        elif metric is not None and not isinstance(metric, str):
            config["eval_metric"] = getattr(metric, "__name__", str(metric))
        # Objective instances likewise persist by registry name; the fitted
        # booster's objective governs prediction either way.
        if isinstance(config.get("objective"), BaseObjective):
            config["objective"] = config["objective"].name
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
