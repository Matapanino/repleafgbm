"""router_extraction mode: frozen external routes + RepLeaf leaf models.

Implements milestones 2-3 of docs/adr/0002-router-extraction.md: LightGBM
trees are mapped into native :class:`~repleafgbm.core.tree.Tree` arrays
(including per-node missing direction), then leaf models are refit by
sequential replay (``Booster.fit_with_routes``), preserving the stage-wise
additive structure.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from repleafgbm.classifier import RepLeafClassifier
from repleafgbm.core.booster import Booster, BoosterParams
from repleafgbm.core.leaf_models import make_leaf_model
from repleafgbm.core.metrics import get_metric
from repleafgbm.core.objectives import get_objective
from repleafgbm.core.tree import Tree
from repleafgbm.external.lightgbm_model import LightGBMExternalModel
from repleafgbm.regressor import RepLeafRegressor


def extract_routes(model: Any) -> tuple[list[Tree], list[np.ndarray]]:
    """Map a fitted LightGBM model's trees into native routing trees.

    Args:
        model: A fitted LightGBMExternalModel, LightGBM sklearn estimator,
            or raw ``lgb.Booster``.

    Returns:
        ``(trees, leaf_values)``: native trees (with LightGBM's learned
        ``default_left`` as per-node missing direction) and, per tree, the
        original LightGBM leaf values indexed by our dense leaf ids.
        Summing the original leaf values over trees reproduces LightGBM's
        raw prediction exactly (verified in tests), which is the
        correctness guarantee for the structure mapping.

    Raises:
        ValueError: If the model contains categorical (``==``) splits;
            train the base model on ordinal-encoded data (e.g. via
            RepLeafDataset) instead.
    """
    booster, num_iteration = _unwrap_booster(model)
    # When the base was early-stopped, only its best-iteration prefix is
    # extracted — those are the trees its predictions actually use.
    dump = booster.dump_model(num_iteration=num_iteration or 0)
    trees: list[Tree] = []
    leaf_values: list[np.ndarray] = []
    for info in dump["tree_info"]:
        tree, values = _convert_tree(info["tree_structure"])
        trees.append(tree)
        leaf_values.append(values)
    return trees, leaf_values


def _unwrap_booster(model: Any) -> tuple[Any, int | None]:
    """Return (lgb.Booster, best_iteration or None)."""
    if isinstance(model, LightGBMExternalModel):
        model._check_fitted()
        return model.model_.booster_, model.best_iteration_
    if hasattr(model, "booster_"):  # lgb sklearn estimator
        best = getattr(model, "_best_iteration", None) or None
        return model.booster_, best
    if hasattr(model, "dump_model"):  # raw lgb.Booster
        best = getattr(model, "best_iteration", 0) or None
        return model, best
    raise TypeError(
        f"Cannot extract routes from {type(model).__name__}; expected a fitted "
        "LightGBMExternalModel, LGBMRegressor/LGBMClassifier, or lgb.Booster"
    )


def _convert_tree(root: dict) -> tuple[Tree, np.ndarray]:
    feature: list[int] = []
    threshold: list[float] = []
    left: list[int] = []
    right: list[int] = []
    leaf_id: list[int] = []
    missing_left: list[bool] = []
    gain: list[float] = []
    values: list[float] = []

    def add(node: dict) -> int:
        index = len(feature)
        feature.append(-1)
        threshold.append(np.nan)
        left.append(-1)
        right.append(-1)
        leaf_id.append(-1)
        missing_left.append(True)
        gain.append(0.0)
        if "split_feature" in node:
            if node["decision_type"] != "<=":
                raise ValueError(
                    f"Unsupported split type {node['decision_type']!r} (categorical "
                    "split). Train the base model on ordinal-encoded features, "
                    "e.g. by fitting it on a RepLeafDataset."
                )
            feature[index] = int(node["split_feature"])
            threshold[index] = float(node["threshold"])
            missing_left[index] = bool(node["default_left"])
            gain[index] = float(node.get("split_gain", 0.0))
            left[index] = add(node["left_child"])
            right[index] = add(node["right_child"])
        else:
            leaf_id[index] = len(values)
            values.append(float(node["leaf_value"]))
        return index

    add(root)
    tree = Tree(
        feature=np.asarray(feature, dtype=np.int32),
        threshold=np.asarray(threshold, dtype=np.float64),
        left=np.asarray(left, dtype=np.int32),
        right=np.asarray(right, dtype=np.int32),
        leaf_id=np.asarray(leaf_id, dtype=np.int32),
        missing_left=np.asarray(missing_left, dtype=bool),
        gain=np.asarray(gain, dtype=np.float64),
    )
    return tree, np.asarray(values, dtype=np.float64)


class _RouterExtractionMixin:
    """Shared fit logic: extract routes from a LightGBM base, replay leaves.

    The base model provides the routes; leaf parameters are refit by
    sequential replay on Newton targets, so all leaf-model machinery
    (embedded linear fits, ridge, constant fallback) applies unchanged.
    Prediction and save/load use the native ensemble representation — the
    saved model does not depend on LightGBM.

    Args:
        base: LightGBMExternalModel with the task matching the estimator.
            Unfitted bases are trained on the fit() data (with the first
            eval_set entry and ``early_stopping_rounds`` forwarded, so the
            base's capacity is tuned too); pre-fitted bases are used as-is.
            None creates a default one.
        early_stopping_rounds: Replay-stage patience — stops *consuming
            extracted routes* when the eval metric stalls; ``best_iteration_``
            applies to prediction as usual. Requires eval_set.
        eval_metric: Metric name for eval monitoring (task default if None).
        Other arguments: as in RepLeafRegressor / RepLeafClassifier.

    The replay learning rate is taken from the base model (ADR 0002).
    """

    _base_task: str = "regression"

    def __init__(
        self,
        base: LightGBMExternalModel | None = None,
        leaf_model: str = "embedded_linear",
        encoder: str = "identity",
        encoder_params: dict | None = None,
        max_leaf_emb_dim: int = 64,
        l2_leaf: float = 1.0,
        min_samples_leaf: int = 20,
        early_stopping_rounds: int | None = None,
        eval_metric: str | None = None,
        random_state: int | None = 42,
    ) -> None:
        self.base = base
        self.leaf_model = leaf_model
        self.encoder = encoder
        self.encoder_params = encoder_params
        self.max_leaf_emb_dim = max_leaf_emb_dim
        self.l2_leaf = l2_leaf
        self.min_samples_leaf = min_samples_leaf
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric
        self.random_state = random_state

    def fit(self, X: Any, y: Any | None = None, eval_set: Any = None):
        dataset = self._build_dataset(X, y)
        dataset = self._prepare_target(dataset, is_train=True)
        self.metadata_ = dataset.metadata
        self.n_features_in_ = dataset.n_features
        self.feature_names_in_ = np.asarray(dataset.feature_names, dtype=object)

        eval_sets = self._prepare_eval_sets(eval_set)
        if self.early_stopping_rounds is not None and not eval_sets:
            raise ValueError(
                "early_stopping_rounds requires eval_set; pass eval_set=[...] to fit()"
            )

        leaf_model = make_leaf_model(
            self.leaf_model, l2=self.l2_leaf, min_samples_linear=2 * self.min_samples_leaf
        )
        self.encoder_ = (
            self._build_and_fit_encoder(dataset) if leaf_model.uses_embeddings else None
        )

        base = self.base if self.base is not None else LightGBMExternalModel(
            task=self._base_task, random_state=self.random_state or 0
        )
        if base.task != self._base_task:
            raise ValueError(
                f"{type(self).__name__} requires a base model with "
                f"task={self._base_task!r}, got task={base.task!r}"
            )
        self.base_ = copy.deepcopy(base)
        if self.base_.model_ is None:
            if eval_sets and self.early_stopping_rounds is not None:
                # Tune the base's capacity on the same validation data.
                _, valid_ds = eval_sets[0]
                self.base_.fit(
                    dataset,
                    eval_set=[(valid_ds.get_raw_features(), valid_ds.y)],
                    early_stopping_rounds=self.early_stopping_rounds,
                )
            else:
                self.base_.fit(dataset)

        trees, _ = extract_routes(self.base_)
        replay_rate = float(self.base_.model_.get_params().get("learning_rate", 0.1))
        params = BoosterParams(
            n_estimators=len(trees),
            learning_rate=replay_rate,
            min_samples_leaf=self.min_samples_leaf,
            l2_leaf=self.l2_leaf,
            early_stopping_rounds=self.early_stopping_rounds,
        )
        self.booster_ = Booster(params, get_objective(self._objective_name))
        self.booster_.fit_with_routes(
            dataset,
            self.encoder_,
            leaf_model,
            trees,
            eval_sets=eval_sets or None,
            eval_metric=get_metric(self.eval_metric or self._eval_metric_name),
        )
        self.evals_result_ = self.booster_.evals_result_
        return self

    def _serializable_config(self, config: dict) -> dict:
        config = super()._serializable_config(config)
        # The base model is not needed for prediction; record its settings
        # for provenance only.
        if config.get("base") is not None:
            base = config["base"]
            config["base"] = None
            config["base_provenance"] = {
                "task": base.task,
                "random_state": base.random_state,
                "lgb_params": base.lgb_params,
            }
        return config


class RouterExtractionRegressor(_RouterExtractionMixin, RepLeafRegressor):
    """Regressor with LightGBM routing and RepLeaf leaf models.

    See :class:`_RouterExtractionMixin` for the fit contract and arguments.
    """

    _base_task = "regression"


class RouterExtractionClassifier(_RouterExtractionMixin, RepLeafClassifier):
    """Binary classifier with LightGBM routing and RepLeaf leaf models.

    The base LightGBM model must use task="binary"; replay refits leaves on
    logistic Newton targets, and predict/predict_proba behave exactly like
    RepLeafClassifier. See :class:`_RouterExtractionMixin` for arguments.
    """

    _base_task = "binary"
