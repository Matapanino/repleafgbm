"""Prediction over a fitted ensemble.

Kept separate from the booster so that alternative backends (and eventually
compiled predictors) can reuse the same ensemble representation: a list of
routing trees plus per-tree leaf parameters.
"""

from __future__ import annotations

import numpy as np

from repleafgbm.core.leaf_models import LeafValues
from repleafgbm.core.tree import Tree


def predict_raw(
    trees: list[Tree],
    leaf_values: list[LeafValues],
    init_score: float,
    learning_rate: float,
    X_raw: np.ndarray,
    Z: np.ndarray | None,
    n_trees: int | None = None,
) -> np.ndarray:
    """Raw additive score F(x) = F_0 + lr * sum_t f_t(x).

    Args:
        n_trees: Optionally use only the first ``n_trees`` trees (staged
            prediction / future early stopping support).
    """
    n_rows = X_raw.shape[0]
    out = np.full(n_rows, init_score, dtype=np.float64)
    if n_trees is None:
        n_trees = len(trees)
    for tree, lv in zip(trees[:n_trees], leaf_values[:n_trees]):
        leaf_idx = tree.apply(X_raw)
        out += learning_rate * lv.predict(leaf_idx, Z)
    return out


def predict_raw_multiclass(
    trees: list[Tree],
    leaf_values: list[LeafValues],
    init_scores: np.ndarray,
    learning_rate: float,
    X_raw: np.ndarray,
    Z: np.ndarray | None,
    n_classes: int,
    n_rounds: int | None = None,
) -> np.ndarray:
    """Raw score matrix (n_rows, n_classes) for the K-trees-per-round ensemble.

    Trees are stored round-major: round r, class k at index
    ``r * n_classes + k`` (see core/multiclass.py).
    """
    n_rows = X_raw.shape[0]
    out = np.tile(np.asarray(init_scores, dtype=np.float64), (n_rows, 1))
    n_trees = len(trees) if n_rounds is None else n_rounds * n_classes
    for i, (tree, lv) in enumerate(zip(trees[:n_trees], leaf_values[:n_trees])):
        leaf_idx = tree.apply(X_raw)
        out[:, i % n_classes] += learning_rate * lv.predict(leaf_idx, Z)
    return out


def predict_raw_multioutput(
    trees: list[Tree],
    leaf_values: list[LeafValues],
    init_scores: np.ndarray,
    learning_rate: float,
    X_raw: np.ndarray,
    Z: np.ndarray | None,
    n_trees: int | None = None,
) -> np.ndarray:
    """Raw score matrix (n_rows, n_outputs) for shared-routing vector leaves.

    Unlike the multiclass ensemble there is one tree per round; each leaf emits
    an (n_outputs,) vector, so every tree contributes to all output columns.
    """
    n_rows = X_raw.shape[0]
    out = np.tile(np.asarray(init_scores, dtype=np.float64), (n_rows, 1))
    if n_trees is None:
        n_trees = len(trees)
    for tree, lv in zip(trees[:n_trees], leaf_values[:n_trees]):
        out += learning_rate * lv.predict(tree.apply(X_raw), Z)
    return out
