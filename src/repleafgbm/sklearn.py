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
from repleafgbm.core.profiling import profiler_from_env, timed
from repleafgbm.core.serialization import load_model_dir, save_model_dir
from repleafgbm.data import RepLeafDataset
from repleafgbm.encoders import RandomProjectionEncoder, make_encoder
from repleafgbm.encoders.base import BaseEncoder
from repleafgbm.utils.validation import as_sample_weight

# scikit-learn renamed ``force_all_finite`` -> ``ensure_all_finite`` in 1.6 and
# removed the old name in 1.8. Pick whichever the installed version accepts so
# the estimators work across the supported range (scikit-learn >= 1.2).
_FINITE_KW = (
    "ensure_all_finite"
    if "ensure_all_finite" in inspect.signature(check_array).parameters
    else "force_all_finite"
)


def _allow_nan_kwargs() -> dict:
    """Validation kwargs that permit NaN (it routes left) but reject inf."""
    return {_FINITE_KW: "allow-nan"}


class BaseRepLeafModel(BaseEstimator):
    """Shared implementation for RepLeafRegressor / RepLeafClassifier.

    Args:
        n_estimators: Number of boosting rounds (trees).
        learning_rate: Shrinkage applied to every tree's contribution.
        num_leaves: Maximum leaves per tree (leaf-wise growth).
        max_depth: Maximum tree depth; -1 means unlimited.
        grow_policy: Tree growth strategy. "leafwise" (default) grows
            best-gain-first (controlled by num_leaves, like LightGBM);
            "depthwise" grows level-order to max_depth (like XGBoost);
            "symmetric" grows CatBoost-style oblivious trees where every node at
            a level shares one split, giving up to 2**max_depth leaves and strong
            implicit regularization. "depthwise" and "symmetric" require
            max_depth >= 1. "symmetric" is numeric/scalar-only in v0 (categorical
            features route as ordered thresholds, not subset splits, and
            multi-output targets are unsupported).
        min_samples_leaf: Minimum rows per leaf for a split to be valid.
        leaf_model: "constant", "embedded_linear", "raw_linear", or "adaptive".
            Defaults to "embedded_linear"; on unknown real-world tabular data the
            OpenML benchmark (docs/roadmap.md, Phase 25) found "constant" the
            honest practical choice — embedded leaves help mainly on
            smooth/periodic structure. "adaptive" (experimental, opt-in) chooses
            constant vs embedded_linear *per leaf* via a weighted leave-one-out
            test (``leaf_gate_margin``), keeping the linear fit only where it
            generalizes and falling back to constant elsewhere. In multi-seed
            real-data benchmarks it tracked the better of the two within seed
            noise — a robust per-leaf hedge, not a statistically separated
            accuracy gain (it is per-leaf model *selection*, not jointly-trained
            leaf embeddings). The default is kept for backwards compatibility and
            may change only in a major release.
        leaf_gate_margin: For leaf_model="adaptive" only (inert otherwise). A
            leaf keeps its linear fit only if its weighted leave-one-out error
            beats the constant leaf's by this relative margin: 0.0 keeps any
            non-worse linear fit, larger values demand a larger held-out
            improvement. Must be >= 0. Default 0.01.
        leaf_gate: For leaf_model="adaptive" only. "loo" (default,
            leverage-corrected held-out error) or "insample" (drops the leverage
            correction — a deliberately weak baseline for diagnostics).
        leaf_fit_precision: "float64" (default) or "float32_gram". A perf knob for
            the linear leaf models on **wide embeddings** (emb_dim > 128, the BLAS
            leaf-fit path): "float32_gram" accumulates only the per-leaf Gram and
            gradient projection in float32 (~1.6-2x faster on those reductions,
            ~15% faster wide-emb fit; also covers multi-output vector leaves) while
            the solve stays float64. Inert for "constant" and for narrow embeddings
            (the native path). Trade-off:
            "float32_gram" is allclose-not-bitwise and not guaranteed
            reproducible across BLAS vendors/platforms; "float64" is the
            reproducible, NumPy<->Rust bitwise-parity default. Opt-in only.
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
        verbose: Print eval_set scores to stdout every ``verbose`` boosting
            rounds (LightGBM-style ``[10]  valid_0's rmse: 0.123456`` lines,
            plus a best-iteration line when early stopping triggers). 0
            (default) is silent. Without an eval_set there is nothing to
            report, so training stays silent for any value.
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
        class_weight: Classification only (ignored by the regressor). Per-class
            weights folded into the per-row sample weight before boosting: a
            ``{label: weight}`` dict, or "balanced" to weight each class
            inversely to its frequency (``n_samples / (n_classes * count)``,
            via sklearn). Combined multiplicatively with any ``sample_weight``
            passed to :meth:`fit`. None (default) leaves the classes unweighted.
            Use it to optimize imbalanced targets toward balanced accuracy
            (pair with ``eval_metric="balanced_accuracy"``). Serialized with
            the model config.
        random_state: Seed controlling all internal randomness.
    """

    _objective_name: str = "squared_error"
    _eval_metric_name: str = "rmse"
    #: Whether this estimator can honor ``sample_weight`` / ``class_weight``.
    #: The native boosting path always can (it owns the gradients), so this is
    #: True here. Subclasses whose training cannot reweight rows — frozen-route
    #: replay (``RouterExtraction*``) — set it False; weights are then dropped
    #: with a UserWarning instead of raising, and the documented fallback is to
    #: train the plain loss, early-stop on a built-in metric, and compute
    #: balanced accuracy externally (docs/weighting_and_metrics.md).
    _supports_sample_weight: bool = True
    #: Whether the target must be numeric (regressor) or may be labels
    #: (classifier overrides to False). Drives array validation of ``y``.
    _y_numeric: bool = True

    def _more_tags(self) -> dict:
        # scikit-learn < 1.6 tag API. NaN is a supported value (missing routes
        # left), so check_estimator's finiteness checks should test
        # inf-rejection only, not NaN-rejection. Both estimators require y.
        return {"allow_nan": True, "requires_y": True}

    def __sklearn_tags__(self):
        # scikit-learn >= 1.6 tag API (the dataclass form). Implementing both
        # keeps the estimator working across sklearn versions (1.6 raises if
        # only ``_more_tags`` is defined). On < 1.6 ``BaseEstimator`` has no
        # ``__sklearn_tags__`` (it reads ``_more_tags`` via ``_get_tags``);
        # sklearn never calls this method there, but guard the super() call so
        # a direct invocation fails with a clear message instead of an opaque
        # AttributeError about the missing base implementation.
        if not hasattr(super(), "__sklearn_tags__"):
            raise AttributeError(
                "__sklearn_tags__ is only available with scikit-learn >= 1.6"
            )
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.target_tags.required = True
        return tags

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        max_depth: int = -1,
        grow_policy: str = "leafwise",
        min_samples_leaf: int = 20,
        leaf_model: str = "embedded_linear",
        leaf_gate_margin: float = 0.01,
        leaf_gate: str = "loo",
        leaf_fit_precision: str = "float64",
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
        verbose: int = 0,
        objective: str | BaseObjective | None = None,
        label_smoothing: float = 0.0,
        class_weight: dict | str | None = None,
        random_state: int | None = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.grow_policy = grow_policy
        self.min_samples_leaf = min_samples_leaf
        self.leaf_model = leaf_model
        self.leaf_gate_margin = leaf_gate_margin
        self.leaf_gate = leaf_gate
        self.leaf_fit_precision = leaf_fit_precision
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
        self.verbose = verbose
        self.objective = objective
        self.label_smoothing = label_smoothing
        self.class_weight = class_weight
        self.random_state = random_state

    # ------------------------------------------------------------------ #
    # Fitting
    # ------------------------------------------------------------------ #
    def fit(
        self,
        X: Any,
        y: Any | None = None,
        sample_weight: Any | None = None,
        eval_set: list[RepLeafDataset | tuple] | None = None,
    ) -> BaseRepLeafModel:
        """Fit on (X, y) arrays/DataFrames or a RepLeafDataset.

        Args:
            X: Feature matrix or a RepLeafDataset that already contains y.
            y: Target vector; must be None when X is a RepLeafDataset.
            sample_weight: Optional per-row weights (length n_rows). Non-negative
                and finite; they scale each row's gradient/Hessian (and the init
                score) during boosting. The classifier multiplies these by the
                weights implied by ``class_weight``. None means uniform weights.
            eval_set: Optional list of RepLeafDataset (or (X, y) tuples)
                evaluated after every boosting round; results are stored in
                ``evals_result_``.
        """
        if not self.freeze_encoder:
            raise NotImplementedError(
                "freeze_encoder=False (encoder updates during boosting) is not "
                "supported in v0; see docs/roadmap.md"
            )

        # Internal phase profiler (None unless REPLEAFGBM_PROFILE is set); threaded
        # through preprocessing, the encoder, and the booster, then exposed as the
        # fitted ``phase_seconds_`` attribute. Off by default — see core.profiling.
        profiler = profiler_from_env()
        with timed(profiler, "preprocessing"):
            dataset = self._build_dataset(X, y)
            dataset = self._prepare_target(dataset, is_train=True)
            weight = self._resolve_sample_weight(dataset, sample_weight)
            dataset.sample_weight = self._enforce_weight_capability(weight)
        # Robust regression objectives (huber/quantile) are fit in standardized
        # target space (so a fixed delta=1 is ~1 sigma); identity otherwise. Done
        # before the encoder so a learned encoder's supervised pretrain target is
        # coherent. The booster un-standardizes predictions/eval (set below).
        target_transform = self._resolve_target_transform(dataset)
        if target_transform is not None:
            loc, scale = target_transform
            dataset.y = (dataset.y - loc) / scale
        self.metadata_ = dataset.metadata
        self.n_features_in_ = dataset.n_features
        self.feature_names_in_ = np.asarray(dataset.feature_names, dtype=object)

        leaf_model = make_leaf_model(
            self.leaf_model,
            l2=self.l2_leaf,
            min_samples_linear=2 * self.min_samples_leaf,
            leaf_gate_margin=self.leaf_gate_margin,
            leaf_gate=self.leaf_gate,
            leaf_fit_precision=self.leaf_fit_precision,
        )
        with timed(profiler, "encoder"):
            self.encoder_ = (
                self._build_and_fit_encoder(dataset)
                if leaf_model.uses_embeddings
                else None
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
            grow_policy=self.grow_policy,
            min_samples_leaf=self.min_samples_leaf,
            l2_leaf=self.l2_leaf,
            max_bins=self.max_bins,
            cat_smooth=self.cat_smooth,
            min_data_per_group=self.min_data_per_group,
            max_cat_threshold=self.max_cat_threshold,
            split_backend=self.split_backend,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose=self.verbose,
        )
        self.booster_ = self._make_booster(params)
        if target_transform is not None:
            self.booster_.target_loc_, self.booster_.target_scale_ = target_transform
        self.booster_.fit(
            dataset,
            self.encoder_,
            leaf_model,
            eval_sets=eval_sets or None,
            eval_metric=self._resolve_eval_metric(),
            profiler=profiler,
        )
        self.evals_result_ = self.booster_.evals_result_
        if profiler is not None:
            #: Per-phase fit wall-clock seconds (only when profiling is enabled);
            #: ``_predict_raw`` adds a "predict" entry. Internal/benchmark aid.
            self.phase_seconds_ = profiler.as_dict()
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

    def _resolve_target_transform(
        self, dataset: RepLeafDataset
    ) -> tuple[float | np.ndarray, float | np.ndarray] | None:
        """Hook: per-output target standardization ``(loc, scale)`` applied to
        the training target before boosting, or ``None`` for no transform (the
        base/classifier default). The regressor returns ``(median, 1.4826*MAD)``
        for the saturated-gradient robust objectives (huber/quantile) so a fixed
        ``delta=1`` lands at ~1 sigma across target scales — see
        docs/proposals/robust-target-standardization.md."""
        return None

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

    def _resolve_sample_weight(
        self, dataset: RepLeafDataset, sample_weight: Any | None
    ) -> np.ndarray | None:
        """Validate the per-row sample weights to attach to the dataset.

        An explicit ``sample_weight`` passed to :meth:`fit` overrides any
        weights already carried by a RepLeafDataset; None falls back to the
        dataset's own. The classifier overrides this to fold in ``class_weight``.
        """
        if sample_weight is None:
            return dataset.sample_weight
        return as_sample_weight(sample_weight, n_rows=dataset.n_rows)

    def _enforce_weight_capability(
        self, weight: np.ndarray | None
    ) -> np.ndarray | None:
        """Drop resolved weights (with a UserWarning) for estimators that
        cannot honor them, rather than raising. Estimators that can
        (``_supports_sample_weight``, the default) return the weight unchanged.
        """
        if weight is not None and not self._supports_sample_weight:
            warnings.warn(
                f"{type(self).__name__} cannot apply sample_weight/class_weight "
                "(it replays frozen routes); the weights are ignored. Train on "
                "the plain loss, early-stop on a built-in metric, and compute "
                "balanced accuracy externally for reporting "
                "(docs/weighting_and_metrics.md).",
                UserWarning,
                stacklevel=3,
            )
            return None
        return weight

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
            X, accept_sparse=False, dtype="numeric", **_allow_nan_kwargs()
        )

    def _validate_array_Xy(self, X: Any, y: Any) -> tuple[Any, Any]:
        X, y = check_X_y(
            X,
            y,
            accept_sparse=False,
            dtype="numeric",
            multi_output=self._multi_output,
            y_numeric=self._y_numeric,
            **_allow_nan_kwargs(),
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
        # The encoder is fitted once here and frozen for all of boosting. The
        # effective per-row weight (sample x class weights) flows into the
        # supervised pretraining target and loss so learned encoders honor it;
        # fixed encoders ignore it.
        weight = dataset.sample_weight
        encoder.fit(
            X_num,
            y=self._pretrain_target(dataset, sample_weight=weight),
            sample_weight=weight,
        )
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

    def _pretrain_target(
        self, dataset: RepLeafDataset, sample_weight: np.ndarray | None = None
    ) -> np.ndarray | None:
        """Supervised pretraining target for learned encoders: the Newton
        residual at the initial score (y - mean for regression, y - sigmoid(F0)
        for binary). ``sample_weight`` makes F0 the *weighted* initial score so
        the residual matches the booster's weighted starting point (None leaves
        it unweighted, identical to before). Unlearned encoders ignore it. The
        classifier and regressor override this to return the ``(n_rows, K)``
        residual *matrix* for multiclass / multi-output targets, which learned
        encoders pretrain a K-output head on (docs/math.md)."""
        objective = self._build_objective()
        f0 = np.full(
            dataset.n_rows, objective.init_score(dataset.y, weight=sample_weight)
        )
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
                if X.shape[1] != self.n_features_in_:
                    # Standard scikit-learn message (matches check_estimator).
                    raise ValueError(
                        f"X has {X.shape[1]} features, but "
                        f"{type(self).__name__} is expecting "
                        f"{self.n_features_in_} features as input."
                    )
                # A numeric ndarray cannot carry category labels: its float
                # codes never match the training string category maps, so every
                # row would silently route through the missing branch. Refuse it
                # and point at the supported categorical inputs instead of
                # returning quietly wrong predictions.
                if self.metadata_.categorical_features:
                    raise ValueError(
                        f"{type(self).__name__} was trained with categorical "
                        f"features {self.metadata_.categorical_features}, which a "
                        "numeric array cannot represent. Pass a pandas DataFrame "
                        "with the original category labels, or a RepLeafDataset "
                        "built with this model's metadata."
                    )
            dataset = RepLeafDataset(X, metadata=self.metadata_)
        X_raw = dataset.get_raw_features()
        Z = dataset.get_embeddings(self.encoder_) if self.encoder_ is not None else None
        profiler = profiler_from_env()
        if profiler is None:
            return self.booster_.predict_raw(X_raw, Z)
        with profiler.phase("predict"):
            out = self.booster_.predict_raw(X_raw, Z)
        # Merge into any fit-time phases so a profiled fit+predict reports both.
        self.phase_seconds_ = {**getattr(self, "phase_seconds_", {}), **profiler.as_dict()}
        return out

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
