"""The boosting loop: stage-wise additive training of routing trees with
representation-conditioned leaf models.

Responsibilities are split as documented in docs/design.md:

* :class:`~repleafgbm.data.RepLeafDataset` owns data and metadata,
* the encoder owns the representation z_theta(x) (frozen during boosting),
* the Booster owns gradients, tree growth, and leaf fitting.

The encoder is fitted once before boosting and never updated afterwards.
Updating it mid-boosting would silently change the outputs of already-built
trees and break the stage-wise additive assumption (docs/math.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from repleafgbm.backends import make_split_backend
from repleafgbm.core.leaf_models import BaseLeafModel, LeafValues
from repleafgbm.core.metrics import BaseMetric
from repleafgbm.core.objectives import BaseObjective
from repleafgbm.core.prediction import predict_raw
from repleafgbm.core.splitter import Splitter
from repleafgbm.core.tree import Tree, TreeGrower
from repleafgbm.data import RepLeafDataset
from repleafgbm.encoders.base import BaseEncoder


@dataclass
class BoosterParams:
    """Hyperparameters consumed by the boosting loop."""

    n_estimators: int = 100
    learning_rate: float = 0.1
    num_leaves: int = 31
    max_depth: int = -1
    min_samples_leaf: int = 20
    l2_leaf: float = 1.0
    max_bins: int = 256
    #: Categorical subset-split guards (LightGBM semantics/defaults).
    cat_smooth: float = 10.0
    min_data_per_group: int = 100
    max_cat_threshold: int = 32
    #: Split kernel implementation: "auto" (Rust extension when installed,
    #: NumPy otherwise), "numpy", or "rust".
    split_backend: str = "auto"
    #: Stop when the first eval set's metric has not improved for this many
    #: rounds. None disables early stopping. Requires eval_sets + eval_metric.
    early_stopping_rounds: int | None = None


class Booster:
    """Trains and stores the additive ensemble.

    The booster is constructed unfitted; ``fit`` consumes a dataset, an
    already-fitted (frozen) encoder, a leaf model, and an objective.
    """

    def __init__(self, params: BoosterParams, objective: BaseObjective) -> None:
        self.params = params
        self.objective = objective
        self.init_score_: float = 0.0
        self.trees_: list[Tree] = []
        self.leaf_values_: list[LeafValues] = []
        self.evals_result_: dict[str, dict[str, list[float]]] = {}
        #: Number of trees of the best model found by early stopping
        #: (None when early stopping was not used). All grown trees are kept;
        #: prediction uses the first ``best_iteration_`` trees by default.
        self.best_iteration_: int | None = None
        self.best_score_: float | None = None

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def fit(
        self,
        dataset: RepLeafDataset,
        encoder: BaseEncoder | None,
        leaf_model: BaseLeafModel,
        eval_sets: list[tuple[str, RepLeafDataset]] | None = None,
        eval_metric: BaseMetric | None = None,
    ) -> Booster:
        """Grow trees natively; see :meth:`_run_boosting` for the loop."""
        p = self.params
        splitter = Splitter(
            dataset.get_raw_features(),
            max_bins=p.max_bins,
            min_samples_leaf=p.min_samples_leaf,
            l2=p.l2_leaf,
            backend=make_split_backend(p.split_backend),
            categorical_indices=dataset.metadata.categorical_indices,
            cat_smooth=p.cat_smooth,
            min_data_per_group=p.min_data_per_group,
            max_cat_threshold=p.max_cat_threshold,
        )
        grower = TreeGrower(splitter, num_leaves=p.num_leaves, max_depth=p.max_depth)
        return self._run_boosting(
            dataset, encoder, leaf_model,
            next_tree=grower.grow,
            n_rounds=p.n_estimators,
            eval_sets=eval_sets,
            eval_metric=eval_metric,
        )

    def fit_with_routes(
        self,
        dataset: RepLeafDataset,
        encoder: BaseEncoder | None,
        leaf_model: BaseLeafModel,
        trees: list[Tree],
        eval_sets: list[tuple[str, RepLeafDataset]] | None = None,
        eval_metric: BaseMetric | None = None,
    ) -> Booster:
        """Sequential replay over frozen routing trees (router_extraction).

        Identical to :meth:`fit` except tree *growth* is replaced by looking
        up the next frozen tree: leaf models are still fitted stage-wise on
        the current Newton targets, so the stage-wise additive structure is
        preserved (docs/adr/0002-router-extraction.md). ``n_estimators`` is
        ignored (the route list bounds the rounds); ``learning_rate`` must be
        the *base model's* rate. Eval sets and early stopping work exactly as
        in native training — replay simply stops consuming routes.
        """
        X_raw = dataset.get_raw_features()
        tree_iter = iter(trees)

        def next_tree(grad: np.ndarray, hess: np.ndarray) -> tuple[Tree, list[np.ndarray]]:
            tree = next(tree_iter)
            leaf_idx = tree.apply(X_raw)
            leaf_rows = [np.where(leaf_idx == i)[0] for i in range(tree.n_leaves)]
            return tree, leaf_rows

        return self._run_boosting(
            dataset, encoder, leaf_model,
            next_tree=next_tree,
            n_rounds=len(trees),
            eval_sets=eval_sets,
            eval_metric=eval_metric,
        )

    def _run_boosting(
        self,
        dataset: RepLeafDataset,
        encoder: BaseEncoder | None,
        leaf_model: BaseLeafModel,
        next_tree,
        n_rounds: int,
        eval_sets: list[tuple[str, RepLeafDataset]] | None,
        eval_metric: BaseMetric | None,
    ) -> Booster:
        """The boosting loop, generic over where trees come from.

        ``next_tree(grad, hess) -> (tree, leaf_rows)`` either grows a tree
        (native fit) or looks up the next frozen route (replay).
        """
        if dataset.y is None:
            raise ValueError("Training dataset must contain a target (y)")
        y = dataset.y
        Z = dataset.get_embeddings(encoder) if leaf_model.uses_embeddings else None

        p = self.params
        if p.early_stopping_rounds is not None and not eval_sets:
            raise ValueError(
                "early_stopping_rounds requires at least one eval_set to monitor"
            )

        self.init_score_ = self.objective.init_score(y)
        F = np.full(y.shape[0], self.init_score_, dtype=np.float64)

        # Each eval set keeps an incrementally updated raw-score cache, so
        # per-round evaluation costs one tree, not the whole ensemble.
        evals: list[tuple[str, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]] = []
        if eval_sets:
            for name, ds in eval_sets:
                if ds.y is None:
                    raise ValueError(f"eval_set {name!r} must contain a target (y)")
                Ze = ds.get_embeddings(encoder) if leaf_model.uses_embeddings else None
                Fe = np.full(ds.n_rows, self.init_score_, dtype=np.float64)
                evals.append((name, ds.get_raw_features(), ds.y, Ze, Fe))
            self.evals_result_ = {name: {eval_metric.name: []} for name, *_ in evals}

        leaf_idx = np.empty(y.shape[0], dtype=np.int64)
        best_score: float | None = None
        rounds_since_best = 0
        for _ in range(n_rounds):
            grad, hess = self.objective.grad_hess(y, F)
            tree, leaf_rows = next_tree(grad, hess)
            leaf_values = leaf_model.fit_leaves(leaf_rows, grad, hess, Z)
            self.trees_.append(tree)
            self.leaf_values_.append(leaf_values)

            # Update the training-score cache with the new tree only; the
            # row partition is already known, no re-routing needed.
            for i, rows in enumerate(leaf_rows):
                leaf_idx[rows] = i
            F += p.learning_rate * leaf_values.predict(leaf_idx, Z)

            if evals and eval_metric is not None:
                for name, Xe, ye, Ze, Fe in evals:
                    Fe += p.learning_rate * leaf_values.predict(tree.apply(Xe), Ze)
                    pred = self.objective.transform(Fe)
                    self.evals_result_[name][eval_metric.name].append(
                        eval_metric(ye, pred)
                    )
                if p.early_stopping_rounds is not None:
                    # Monitor the first eval set, honoring the metric direction.
                    score = self.evals_result_[evals[0][0]][eval_metric.name][-1]
                    improved = best_score is None or (
                        score < best_score if eval_metric.minimize else score > best_score
                    )
                    if improved:
                        best_score = score
                        self.best_iteration_ = self.n_trees
                        self.best_score_ = score
                        rounds_since_best = 0
                    else:
                        rounds_since_best += 1
                        if rounds_since_best >= p.early_stopping_rounds:
                            break
        return self

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #
    def predict_raw(
        self, X_raw: np.ndarray, Z: np.ndarray | None, n_trees: int | None = None
    ) -> np.ndarray:
        """Raw scores. Uses the early-stopping best iteration by default."""
        if n_trees is None:
            n_trees = self.best_iteration_  # None -> all trees
        return predict_raw(
            self.trees_,
            self.leaf_values_,
            self.init_score_,
            self.params.learning_rate,
            X_raw,
            Z,
            n_trees=n_trees,
        )

    def feature_importance(
        self, n_features: int, importance_type: str = "gain"
    ) -> np.ndarray:
        """Per-feature importance aggregated over the predicting trees.

        Uses the first ``best_iteration_`` trees when early stopping was
        active — the same trees prediction uses.

        Args:
            importance_type: "gain" (total split gain) or "split" (number of
                times the feature was split on).
        """
        if importance_type not in ("gain", "split"):
            raise ValueError(
                f"importance_type must be 'gain' or 'split', got {importance_type!r}"
            )
        importance = np.zeros(n_features, dtype=np.float64)
        n_trees = self.best_iteration_ or self.n_trees
        for tree in self.trees_[:n_trees]:
            internal = tree.feature >= 0
            weights = tree.gain[internal] if importance_type == "gain" else 1.0
            np.add.at(importance, tree.feature[internal], weights)
        return importance

    @property
    def n_trees(self) -> int:
        return len(self.trees_)
