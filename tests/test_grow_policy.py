"""Tree growth policies: leafwise (default), depthwise, symmetric (oblivious).

Covers the functional matrix (3 policies x {constant, embedded_linear} x
{regression, binary, multiclass}, plus depthwise multi-output), determinism,
save/load round-trip, NumPy/Rust parity, the structural guarantees of each
policy (depthwise depth bound; symmetric one-split-per-level + completeness +
2**depth leaves + oblivious routing), and the validation errors.
"""

from __future__ import annotations

import json
from collections import deque

import numpy as np
import pytest

from repleafgbm import RepLeafClassifier, RepLeafRegressor

POLICIES = ("leafwise", "depthwise", "symmetric")
DEPTH_POLICIES = ("depthwise", "symmetric")


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #
def _multiclass_data(n: int = 600, seed: int = 3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    logits = np.column_stack(
        [X[:, 0] + X[:, 1], -X[:, 0] + X[:, 2], X[:, 3] - X[:, 1]]
    )
    y = np.argmax(logits + rng.normal(0, 0.3, logits.shape), axis=1)
    return X, y


def _policy_kwargs(policy: str, **extra) -> dict:
    """Common estimator kwargs; depth policies need max_depth set."""
    kw = dict(n_estimators=12, random_state=42, min_samples_leaf=5, **extra)
    if policy in DEPTH_POLICIES:
        kw["max_depth"] = 3
        # symmetric ignores num_leaves; give depthwise room for a full tree.
        kw["num_leaves"] = 64
    return kw


# --------------------------------------------------------------------------- #
# Functional matrix
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("leaf_model", ["constant", "embedded_linear"])
def test_regression_fit_predict(policy, leaf_model, regression_data):
    Xtr, ytr, Xte, _ = regression_data
    model = RepLeafRegressor(
        grow_policy=policy, leaf_model=leaf_model, encoder="plr",
        **_policy_kwargs(policy),
    ).fit(Xtr, ytr)
    pred = model.predict(Xte)
    assert pred.shape == (Xte.shape[0],)
    assert np.all(np.isfinite(pred))


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("leaf_model", ["constant", "embedded_linear"])
def test_binary_fit_predict(policy, leaf_model, classification_data):
    Xtr, ytr, Xte, yte = classification_data
    model = RepLeafClassifier(
        grow_policy=policy, leaf_model=leaf_model, encoder="plr",
        **_policy_kwargs(policy),
    ).fit(Xtr, ytr)
    pred = model.predict(Xte)
    assert set(np.unique(pred)) <= set(model.classes_)
    assert model.predict_proba(Xte).shape == (Xte.shape[0], 2)


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("leaf_model", ["constant", "embedded_linear"])
def test_multiclass_fit_predict(policy, leaf_model):
    X, y = _multiclass_data()
    Xtr, ytr, Xte = X[:450], y[:450], X[450:]
    model = RepLeafClassifier(
        grow_policy=policy, leaf_model=leaf_model, encoder="plr",
        **_policy_kwargs(policy),
    ).fit(Xtr, ytr)
    assert model.n_classes_ == 3
    # n_estimators counts rounds; multiclass grows one tree per class per round.
    assert len(model.booster_.trees_) == 12 * 3
    assert model.predict_proba(Xte).shape == (Xte.shape[0], 3)


@pytest.mark.parametrize("policy", ["leafwise", "depthwise"])
def test_multioutput_supported(policy):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 5))
    Y = np.column_stack([X[:, 0] + X[:, 1] ** 2, X[:, 2] - X[:, 3]])
    model = RepLeafRegressor(
        grow_policy=policy, leaf_model="embedded_linear", encoder="plr",
        **_policy_kwargs(policy),
    ).fit(X, Y)
    assert model.predict(X).shape == (400, 2)


def test_symmetric_multioutput_raises():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(300, 4))
    Y = np.column_stack([X[:, 0], X[:, 1]])
    with pytest.raises(NotImplementedError, match="multi-output"):
        RepLeafRegressor(
            grow_policy="symmetric", max_depth=3, n_estimators=5,
            leaf_model="constant",
        ).fit(X, Y)


# --------------------------------------------------------------------------- #
# Determinism / round-trip / wiring
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("policy", POLICIES)
def test_determinism(policy, regression_data):
    Xtr, ytr, Xte, _ = regression_data
    kw = dict(
        grow_policy=policy, leaf_model="embedded_linear", encoder="plr",
        max_leaf_emb_dim=8, **_policy_kwargs(policy),
    )
    p1 = RepLeafRegressor(**kw).fit(Xtr, ytr).predict(Xte)
    p2 = RepLeafRegressor(**kw).fit(Xtr, ytr).predict(Xte)
    np.testing.assert_allclose(p1, p2)


def test_leafwise_matches_default(regression_data):
    """grow_policy='leafwise' is the historical default path, byte-for-byte."""
    Xtr, ytr, Xte, _ = regression_data
    kw = dict(n_estimators=15, num_leaves=8, leaf_model="embedded_linear",
              encoder="plr", random_state=42)
    default = RepLeafRegressor(**kw).fit(Xtr, ytr).predict(Xte)
    explicit = RepLeafRegressor(grow_policy="leafwise", **kw).fit(Xtr, ytr).predict(Xte)
    np.testing.assert_array_equal(default, explicit)


@pytest.mark.parametrize("policy", POLICIES)
def test_save_load_roundtrip(policy, tmp_path, regression_data):
    Xtr, ytr, Xte, _ = regression_data
    model = RepLeafRegressor(
        grow_policy=policy, leaf_model="embedded_linear", encoder="plr",
        **_policy_kwargs(policy),
    ).fit(Xtr, ytr)
    before = model.predict(Xte)

    model.save_model(tmp_path)
    saved = json.loads((tmp_path / "model_config.json").read_text())
    # Decision A: symmetric expands into the existing flat Tree, so grow_policy
    # does not bump the tree format — a single-output regression model still
    # writes v3 (v5/v6 are reserved for multiclass / multi-output ensembles).
    assert saved["format_version"] == 3
    assert saved["config"]["grow_policy"] == policy

    loaded = RepLeafRegressor.load_model(tmp_path)
    assert loaded.get_params()["grow_policy"] == policy
    np.testing.assert_allclose(loaded.predict(Xte), before)


# --------------------------------------------------------------------------- #
# Structural guarantees
# --------------------------------------------------------------------------- #
def _node_depths(tree) -> dict[int, int]:
    """Depth of every node (root = 0) via BFS over the flat arrays."""
    depth = {0: 0}
    queue = deque([0])
    while queue:
        i = queue.popleft()
        if tree.feature[i] < 0:  # leaf
            continue
        for child in (int(tree.left[i]), int(tree.right[i])):
            depth[child] = depth[i] + 1
            queue.append(child)
    return depth


def _oblivious_levels(tree):
    """Per-level (feature, threshold) for a symmetric tree, with assertions.

    Verifies every internal node at a given depth shares one (feature, threshold)
    and that each depth is uniformly all-internal or all-leaf (completeness).
    Returns (rules, achieved_depth).
    """
    depth = _node_depths(tree)
    by_depth: dict[int, list[int]] = {}
    for node, d in depth.items():
        by_depth.setdefault(d, []).append(node)

    rules = []
    achieved = max(by_depth)
    for d in range(achieved):
        nodes = by_depth[d]
        is_leaf = [int(tree.feature[i]) < 0 for i in nodes]
        assert not any(is_leaf), f"depth {d} mixes internal and leaf nodes"
        feats = {int(tree.feature[i]) for i in nodes}
        threshs = {float(tree.threshold[i]) for i in nodes}
        assert len(feats) == 1, f"depth {d} has >1 split feature: {feats}"
        assert len(threshs) == 1, f"depth {d} has >1 threshold: {threshs}"
        rules.append((feats.pop(), threshs.pop()))
    # The deepest level is all leaves.
    assert all(int(tree.feature[i]) < 0 for i in by_depth[achieved])
    return rules, achieved


def test_symmetric_is_oblivious_and_complete():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(800, 5))
    y = X[:, 0] + X[:, 1] ** 2 + 0.1 * rng.normal(size=800)
    model = RepLeafRegressor(
        grow_policy="symmetric", max_depth=3, min_samples_leaf=5,
        n_estimators=8, leaf_model="constant", random_state=42,
    ).fit(X, y)

    for tree in model.booster_.trees_:
        rules, depth = _oblivious_levels(tree)
        assert len(rules) == depth
        # Complete tree: 2**depth leaves.
        assert tree.n_leaves == 2 ** depth

        # Independent oblivious routing: the leaf a row reaches is a pure
        # function of the per-level bit vector (x[feat] <= thresh, missing-left),
        # i.e. leaf <-> bitcode is a bijection.
        code = np.zeros(X.shape[0], dtype=np.int64)
        for feat, thresh in rules:
            x = X[:, feat]
            go_left = np.where(np.isnan(x), True, x <= thresh)
            code = (code << 1) | (~go_left).astype(np.int64)
        leaves = tree.apply(X)
        pairs = set(zip(leaves.tolist(), code.tolist()))
        leaf_to_code: dict[int, set] = {}
        code_to_leaf: dict[int, set] = {}
        for lf, cd in pairs:
            leaf_to_code.setdefault(lf, set()).add(cd)
            code_to_leaf.setdefault(cd, set()).add(lf)
        assert all(len(s) == 1 for s in leaf_to_code.values())
        assert all(len(s) == 1 for s in code_to_leaf.values())


def test_symmetric_depth_at_least_two():
    """Guard the structure test is meaningful: the tree actually grows deep."""
    rng = np.random.default_rng(6)
    X = rng.normal(size=(800, 5))
    y = X[:, 0] + X[:, 1] ** 2
    model = RepLeafRegressor(
        grow_policy="symmetric", max_depth=3, min_samples_leaf=5,
        n_estimators=3, leaf_model="constant", random_state=0,
    ).fit(X, y)
    depths = [max(_node_depths(t).values()) for t in model.booster_.trees_]
    assert max(depths) >= 2


def test_depthwise_respects_max_depth(regression_data):
    Xtr, ytr, _, _ = regression_data
    max_depth = 3
    model = RepLeafRegressor(
        grow_policy="depthwise", max_depth=max_depth, num_leaves=64,
        min_samples_leaf=5, n_estimators=10, leaf_model="constant",
        random_state=42,
    ).fit(Xtr, ytr)
    for tree in model.booster_.trees_:
        assert max(_node_depths(tree).values()) <= max_depth


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("policy", DEPTH_POLICIES)
def test_depth_policies_require_max_depth(policy, regression_data):
    Xtr, ytr, _, _ = regression_data
    with pytest.raises(ValueError, match="max_depth"):
        RepLeafRegressor(grow_policy=policy, n_estimators=5).fit(Xtr, ytr)


def test_invalid_grow_policy_rejected(regression_data):
    Xtr, ytr, _, _ = regression_data
    with pytest.raises(ValueError, match="grow_policy"):
        RepLeafRegressor(grow_policy="bogus", max_depth=3, n_estimators=5).fit(Xtr, ytr)


# --------------------------------------------------------------------------- #
# NumPy / Rust parity (skipped when the native extension is not built)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("policy", POLICIES)
def test_backend_parity(policy, regression_data):
    pytest.importorskip("repleafgbm_native", reason="Rust extension not built")
    Xtr, ytr, Xte, _ = regression_data
    preds = {}
    for backend in ("numpy", "rust"):
        model = RepLeafRegressor(
            grow_policy=policy, leaf_model="embedded_linear",
            split_backend=backend, **_policy_kwargs(policy),
        ).fit(Xtr, ytr)
        preds[backend] = model.predict(Xte)
    np.testing.assert_allclose(preds["numpy"], preds["rust"], rtol=1e-6, atol=1e-8)
