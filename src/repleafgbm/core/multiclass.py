"""Multiclass boosting: one routing tree per class per round (softmax).

Kept separate from :mod:`repleafgbm.core.booster` so the scalar boosting
loop stays readable. The structure mirrors ``Booster._run_boosting`` exactly,
lifted to (n_rows, n_classes) score/gradient matrices: each round computes
softmax gradients once, then grows ``n_classes`` trees — class k's tree is
fitted on column k with the same Newton-target leaf machinery as binary and
regression training (docs/math.md). The encoder stays frozen; every class
shares the same embedding matrix Z.

Trees and leaf parameters are stored round-major in flat lists (round r,
class k at index ``r * n_classes + k``) so serialization and feature
importance reuse the per-tree code paths unchanged. ``best_iteration_``
counts *rounds*, matching the scalar booster's semantics of "size of the
best model".
"""

from __future__ import annotations

import numpy as np

from repleafgbm.backends import make_split_backend
from repleafgbm.backends.base import BaseSplitBackend
from repleafgbm.core.booster import BoosterParams, weight_grad_hess
from repleafgbm.core.leaf_models import BaseLeafModel, LeafValues
from repleafgbm.core.metrics import BaseMetric
from repleafgbm.core.objectives import MulticlassSoftmax
from repleafgbm.core.prediction import predict_raw_multiclass
from repleafgbm.core.splitter import Splitter
from repleafgbm.core.tree import Tree, TreeGrower
from repleafgbm.data import RepLeafDataset
from repleafgbm.encoders.base import BaseEncoder


class MulticlassBooster:
    """Trains and stores the K-trees-per-round softmax ensemble.

    The target ``y`` must contain integer class indices in ``[0, n_classes)``
    (the sklearn wrapper maps arbitrary labels). Construction, fitting, and
    prediction mirror :class:`~repleafgbm.core.booster.Booster`.
    """

    def __init__(self, params: BoosterParams, objective: MulticlassSoftmax) -> None:
        self.params = params
        self.objective = objective
        #: Per-class init scores (log priors), shape (n_classes,).
        self.init_score_: np.ndarray = np.zeros(objective.n_classes)
        self.trees_: list[Tree] = []
        self.leaf_values_: list[LeafValues] = []
        self.evals_result_: dict[str, dict[str, list[float]]] = {}
        #: Best number of *rounds* found by early stopping (None when unused).
        self.best_iteration_: int | None = None
        self.best_score_: float | None = None
        #: Split backend from the last ``fit`` (runtime-only introspection
        #: handle, never serialized); see :class:`Booster.split_backend_`.
        self.split_backend_: BaseSplitBackend | None = None

    def __getstate__(self) -> dict:
        # Drop the runtime split-backend handle so the model stays picklable;
        # see :meth:`Booster.__getstate__`.
        return {**self.__dict__, "split_backend_": None}

    @property
    def n_classes(self) -> int:
        return self.objective.n_classes

    @property
    def n_trees(self) -> int:
        return len(self.trees_)

    @property
    def n_rounds(self) -> int:
        return len(self.trees_) // self.n_classes

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
    ) -> MulticlassBooster:
        """Grow ``n_estimators`` rounds of ``n_classes`` trees each."""
        if dataset.y is None:
            raise ValueError("Training dataset must contain a target (y)")
        y = dataset.y.astype(np.int64)
        w = dataset.sample_weight
        Z = dataset.get_embeddings(encoder) if leaf_model.uses_embeddings else None

        p = self.params
        if p.early_stopping_rounds is not None and not eval_sets:
            raise ValueError(
                "early_stopping_rounds requires at least one eval_set to monitor"
            )
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
        self.split_backend_ = splitter.backend
        grower = TreeGrower(splitter, num_leaves=p.num_leaves, max_depth=p.max_depth)

        n_classes = self.n_classes
        self.init_score_ = self.objective.init_score(y, weight=w)
        F = np.tile(self.init_score_, (y.shape[0], 1))

        # Incrementally updated raw-score caches per eval set, as in Booster.
        evals: list[tuple[str, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]] = []
        if eval_sets:
            for name, ds in eval_sets:
                if ds.y is None:
                    raise ValueError(f"eval_set {name!r} must contain a target (y)")
                Ze = ds.get_embeddings(encoder) if leaf_model.uses_embeddings else None
                Fe = np.tile(self.init_score_, (ds.n_rows, 1))
                evals.append((name, ds.get_raw_features(), ds.y.astype(np.int64), Ze, Fe))
            self.evals_result_ = {name: {eval_metric.name: []} for name, *_ in evals}

        leaf_idx = np.empty(y.shape[0], dtype=np.int64)
        best_score: float | None = None
        rounds_since_best = 0
        for _ in range(p.n_estimators):
            grad, hess = self.objective.grad_hess(y, F)
            grad, hess = weight_grad_hess(grad, hess, w)
            round_trees: list[tuple[Tree, LeafValues]] = []
            for k in range(n_classes):
                tree, leaf_rows = grower.grow(grad[:, k], hess[:, k])
                leaf_values = leaf_model.fit_leaves(
                    leaf_rows, grad[:, k], hess[:, k], Z
                )
                self.trees_.append(tree)
                self.leaf_values_.append(leaf_values)
                round_trees.append((tree, leaf_values))

                for i, rows in enumerate(leaf_rows):
                    leaf_idx[rows] = i
                # clip=False is exact on training rows (see Booster).
                F[:, k] += p.learning_rate * leaf_values.predict(leaf_idx, Z, clip=False)

            if evals and eval_metric is not None:
                for name, Xe, ye, Ze, Fe in evals:
                    for k, (tree, leaf_values) in enumerate(round_trees):
                        Fe[:, k] += p.learning_rate * leaf_values.predict(
                            tree.apply(Xe), Ze
                        )
                    pred = self.objective.transform(Fe)
                    self.evals_result_[name][eval_metric.name].append(
                        eval_metric(ye, pred)
                    )
                if p.early_stopping_rounds is not None:
                    score = self.evals_result_[evals[0][0]][eval_metric.name][-1]
                    improved = best_score is None or (
                        score < best_score if eval_metric.minimize else score > best_score
                    )
                    if improved:
                        best_score = score
                        self.best_iteration_ = self.n_rounds
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
        self, X_raw: np.ndarray, Z: np.ndarray | None, n_rounds: int | None = None
    ) -> np.ndarray:
        """Raw score matrix (n_rows, n_classes); best rounds by default."""
        if n_rounds is None:
            n_rounds = self.best_iteration_  # None -> all rounds
        return predict_raw_multiclass(
            self.trees_,
            self.leaf_values_,
            self.init_score_,
            self.params.learning_rate,
            X_raw,
            Z,
            n_classes=self.n_classes,
            n_rounds=n_rounds,
        )

    def feature_importance(
        self, n_features: int, importance_type: str = "gain"
    ) -> np.ndarray:
        """Per-feature importance over the predicting trees (all classes)."""
        if importance_type not in ("gain", "split"):
            raise ValueError(
                f"importance_type must be 'gain' or 'split', got {importance_type!r}"
            )
        importance = np.zeros(n_features, dtype=np.float64)
        n_rounds = self.best_iteration_ or self.n_rounds
        for tree in self.trees_[: n_rounds * self.n_classes]:
            internal = tree.feature >= 0
            weights = tree.gain[internal] if importance_type == "gain" else 1.0
            np.add.at(importance, tree.feature[internal], weights)
        return importance
