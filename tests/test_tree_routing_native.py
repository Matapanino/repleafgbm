"""Parity tests: native ``apply_tree`` router vs the NumPy reference.

The compiled router in ``Tree.apply`` must produce **index-identical** leaf ids
to the NumPy level-synchronous fallback (``Tree._apply_numpy``) — integer
routing, so ``assert_array_equal``, not allclose. Each case is also checked
against an independent scalar oracle so a shared bug in the two production paths
cannot hide. Skipped when the optional extension is not built (the fallback
itself is exercised throughout the rest of the suite).
"""

from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

from repleafgbm import RepLeafClassifier, RepLeafDataset, RepLeafRegressor
from repleafgbm.core import tree as tree_mod
from repleafgbm.core.tree import Tree

pytest.importorskip("repleafgbm_native", reason="Rust extension not built")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _tree(feature, threshold, left, right, leaf_id, missing_left=None,
          left_categories=None) -> Tree:
    """Build a Tree from plain lists (``threshold`` uses None for NaN)."""
    n = len(feature)
    return Tree(
        feature=np.array(feature, dtype=np.int32),
        threshold=np.array([np.nan if t is None else t for t in threshold],
                           dtype=np.float64),
        left=np.array(left, dtype=np.int32),
        right=np.array(right, dtype=np.int32),
        leaf_id=np.array(leaf_id, dtype=np.int32),
        missing_left=np.array([True] * n if missing_left is None else missing_left,
                              dtype=bool),
        gain=np.zeros(n, dtype=np.float64),
        left_categories=left_categories,
    )


def _route_ref(tree: Tree, X: np.ndarray) -> np.ndarray:
    """Independent scalar oracle: one explicit root-to-leaf descent per row.

    Precedence mirrors Tree.apply: NaN follows ``missing_left``; else a
    categorical node tests subset membership; else ``x <= threshold``.
    """
    out = np.empty(X.shape[0], dtype=np.int64)
    for i in range(X.shape[0]):
        node = 0
        while tree.leaf_id[node] < 0:
            f = int(tree.feature[node])
            x = X[i, f]
            if np.isnan(x):
                go_left = bool(tree.missing_left[node])
            elif (tree.left_categories is not None
                  and tree.left_categories[node] is not None):
                go_left = bool(np.isin(x, tree.left_categories[node]))
            else:
                go_left = bool(x <= tree.threshold[node])
            node = int(tree.left[node]) if go_left else int(tree.right[node])
        out[i] = tree.leaf_id[node]
    return out


def _assert_routing(tree: Tree, X: np.ndarray) -> None:
    """native ``apply`` == NumPy ``_apply_numpy`` == scalar oracle (exact)."""
    oracle = _route_ref(tree, X)
    native = tree.apply(X)          # compiled apply_tree (extension is built)
    numpy_ = tree._apply_numpy(X)   # vectorized reference
    assert native.dtype == np.int64
    np.testing.assert_array_equal(native, oracle)
    np.testing.assert_array_equal(numpy_, oracle)


# --------------------------------------------------------------------------- #
# Hand-built trees: each routing rule in isolation
# --------------------------------------------------------------------------- #
def test_numeric_threshold_and_missing():
    t = _tree([0, -1, -1], [0.0, None, None], [1, -1, -1], [2, -1, -1],
              [-1, 0, 1])
    # <=0 (incl. exactly 0) -> left leaf 0; >0 -> right leaf 1; NaN -> left.
    X = np.array([[-1.0], [0.0], [0.5], [np.nan]])
    _assert_routing(t, X)


def test_missing_left_false_external():
    """External-style route (LightGBM default_left=False): NaN goes right."""
    t = _tree([0, -1, -1], [0.0, None, None], [1, -1, -1], [2, -1, -1],
              [-1, 0, 1], missing_left=[False, False, False])
    X = np.array([[np.nan], [-1.0], [2.0]])  # nan -> right(1); -1 -> left(0)
    _assert_routing(t, X)
    assert t.apply(np.array([[np.nan]]))[0] == 1


def test_categorical_subset_and_missing():
    t = _tree([0, -1, -1], [None, None, None], [1, -1, -1], [2, -1, -1],
              [-1, 0, 1], left_categories=[np.array([1.0, 3.0]), None, None])
    # codes in {1,3} -> left; others -> right; NaN -> missing_left(True) -> left.
    X = np.array([[1.0], [3.0], [2.0], [0.0], [5.0], [np.nan]])
    _assert_routing(t, X)


def test_mixed_numeric_and_categorical_levels():
    """Root numeric, right child categorical (mixed kinds at different depths)."""
    # node0: x0 <= 0 ? -> node1(leaf0) : node2(categorical on x1)
    # node2: x1 in {2,4} -> leaf1 : leaf2 ; NaN at node2 -> missing_left.
    t = _tree(
        feature=[0, -1, 1, -1, -1],
        threshold=[0.0, None, None, None, None],
        left=[1, -1, 3, -1, -1],
        right=[2, -1, 4, -1, -1],
        leaf_id=[-1, 0, -1, 1, 2],
        left_categories=[None, None, np.array([2.0, 4.0]), None, None],
    )
    rng = np.random.default_rng(0)
    x0 = rng.normal(size=400)
    x1 = rng.integers(0, 6, size=400).astype(np.float64)
    x1[rng.random(400) < 0.1] = np.nan
    _assert_routing(t, np.column_stack([x0, x1]))


def test_root_only_leaf():
    t = _tree([-1], [None], [-1], [-1], [0])  # single node == leaf 0
    X = np.array([[1.0], [np.nan], [-2.0]])
    np.testing.assert_array_equal(t.apply(X), [0, 0, 0])
    _assert_routing(t, X)


def test_empty_rows():
    t = _tree([0, -1, -1], [0.0, None, None], [1, -1, -1], [2, -1, -1],
              [-1, 0, 1])
    out = t.apply(np.empty((0, 1), dtype=np.float64))
    assert out.shape == (0,) and out.dtype == np.int64


def test_all_left_all_right_and_singleton():
    t = _tree([0, -1, -1], [0.0, None, None], [1, -1, -1], [2, -1, -1],
              [-1, 0, 1])
    _assert_routing(t, np.array([[-5.0]]))         # singleton -> left
    _assert_routing(t, np.array([[5.0]]))          # singleton -> right
    _assert_routing(t, np.full((64, 1), -1.0))     # all left
    _assert_routing(t, np.full((64, 1), 1.0))      # all right


def test_singleton_left_category():
    t = _tree([0, -1, -1], [None, None, None], [1, -1, -1], [2, -1, -1],
              [-1, 0, 1], left_categories=[np.array([2.0]), None, None])
    X = np.array([[2.0], [0.0], [1.0], [3.0], [np.nan]])
    _assert_routing(t, X)


# --------------------------------------------------------------------------- #
# Fitted trees: realistic depth / multiple features / parallel branch
# --------------------------------------------------------------------------- #
def _inject_nans(X, frac, seed):
    rng = np.random.default_rng(seed)
    X = X.copy()
    X[rng.random(X.shape) < frac] = np.nan
    return X


def test_fitted_regressor_trees_numeric_with_missing():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(1500, 10))
    y = X[:, 0] + np.sin(2 * X[:, 3]) + 0.5 * X[:, 5] ** 2 + 0.1 * rng.normal(size=1500)
    model = RepLeafRegressor(
        n_estimators=15, num_leaves=16, leaf_model="embedded_linear",
        split_backend="numpy", random_state=0,
    ).fit(X, y)
    Xte = _inject_nans(rng.normal(size=(1200, 10)), 0.05, 2)
    for tree in model.booster_.trees_:
        _assert_routing(tree, Xte)


def test_fitted_multiclass_trees():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(1500, 8))
    y = rng.integers(0, 4, size=1500)
    model = RepLeafClassifier(
        n_estimators=10, num_leaves=12, leaf_model="constant",
        split_backend="numpy", random_state=0,
    ).fit(X, y)
    Xte = _inject_nans(rng.normal(size=(1000, 8)), 0.05, 6)
    for tree in model.booster_.trees_:
        _assert_routing(tree, Xte)


def test_fitted_categorical_and_missing_end_to_end():
    rng = np.random.default_rng(3)
    n = 1500
    cat = rng.choice(list("abcde"), size=n).astype(object)
    cat[rng.random(n) < 0.05] = None
    x = rng.normal(size=n)
    high = pd.Series(cat).isin(["a", "d"]).to_numpy()
    y = np.where(high, 3.0, -3.0) + x + rng.normal(0, 0.2, n)
    df = pd.DataFrame({"c": cat, "x": x})
    train = RepLeafDataset(df, y, categorical_features=["c"])
    model = RepLeafRegressor(
        n_estimators=20, num_leaves=8, min_samples_leaf=10, min_data_per_group=20,
        split_backend="numpy", random_state=42,
    ).fit(train)
    # Route the raw predict-time matrix (ordinal codes + NaN) through each tree.
    X_raw = RepLeafDataset(df, metadata=model.metadata_).get_raw_features()
    trees = model.booster_.trees_
    assert any(t.left_categories is not None for t in trees)  # categorical splits exist
    for tree in trees:
        _assert_routing(tree, X_raw)


def test_parallel_branch_large_node():
    """> APPLY_PARALLEL_MIN (1<<14) rows exercises the rayon routing branch;
    integer leaf ids stay exact regardless of thread count."""
    rng = np.random.default_rng(9)
    X = rng.normal(size=(4000, 6))
    y = (X[:, 0] > 0).astype(float) * 2 + X[:, 1] + 0.1 * rng.normal(size=4000)
    model = RepLeafRegressor(
        n_estimators=5, num_leaves=16, split_backend="numpy", random_state=0,
    ).fit(X, y)
    Xte = _inject_nans(rng.normal(size=(20_000, 6)), 0.03, 10)  # > 16384 -> parallel
    for tree in model.booster_.trees_:
        np.testing.assert_array_equal(tree.apply(Xte), tree._apply_numpy(Xte))


# --------------------------------------------------------------------------- #
# Fallback: missing / older extension uses the NumPy path transparently
# --------------------------------------------------------------------------- #
def test_older_or_absent_native_falls_back(monkeypatch):
    t = _tree([0, -1, -1], [0.0, None, None], [1, -1, -1], [2, -1, -1],
              [-1, 0, 1], left_categories=None)
    X = np.array([[-1.0], [2.0], [np.nan]])
    expected = _route_ref(t, X)

    # Older extension: module present but without the apply_tree symbol.
    monkeypatch.setattr(tree_mod, "_native", types.SimpleNamespace())
    np.testing.assert_array_equal(t.apply(X), expected)

    # Extension entirely absent.
    monkeypatch.setattr(tree_mod, "_native", None)
    np.testing.assert_array_equal(t.apply(X), expected)
