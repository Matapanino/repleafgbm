"""MatGBM leaf-only features, expert bundles, and select leaves."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from repleafgbm import RepLeafDataset, RepLeafRegressor


def test_leaf_only_feature_is_encoder_visible_but_not_routable():
    rng = np.random.RandomState(11)
    n = 240
    route = rng.normal(size=n)
    leaf_only = np.linspace(-5.0, 5.0, n)
    noise = rng.normal(size=n)
    X = pd.DataFrame({"route": route, "teacher": leaf_only, "noise": noise})
    y = 8.0 * leaf_only + 0.1 * route

    ds = RepLeafDataset(X, y, leaf_only_features=["teacher"])
    assert ds.get_raw_features().shape == (n, 2)
    assert ds.metadata.routing_feature_names == ["route", "noise"]
    assert ds.metadata.ordered_numerical_features == ["route", "noise", "teacher"]
    np.testing.assert_array_equal(ds.get_numerical_features()[:, -1], leaf_only)

    model = RepLeafRegressor(
        n_estimators=8,
        num_leaves=6,
        min_samples_leaf=10,
        leaf_model="embedded_linear",
        encoder="identity",
        leaf_only_features=["teacher"],
        split_backend="numpy",
        random_state=3,
    ).fit(X, y)
    assert model.routing_feature_names_.tolist() == ["route", "noise"]
    assert model.get_feature_importance().shape == (2,)
    for tree in model.booster_.trees_:
        used = tree.feature[tree.feature >= 0]
        assert np.all(used < 2)
    for sampled in model.booster_.sampled_features_:
        assert np.all(sampled < 2)


@pytest.mark.parametrize(
    ("X", "leaf_only", "match"),
    [
        (
            pd.DataFrame({"x": [0.0, 1.0], "teacher": [1.0, np.nan]}),
            ["teacher"],
            "leaf-only feature 'teacher'.*finite",
        ),
        (
            pd.DataFrame({"x": [0.0, 1.0], "cat": ["a", "b"]}),
            ["cat"],
            "leaf-only features.*categorical",
        ),
    ],
)
def test_leaf_only_features_reject_nonfinite_and_categorical(X, leaf_only, match):
    with pytest.raises(ValueError, match=match):
        RepLeafDataset(X, [0.0, 1.0], leaf_only_features=leaf_only)


def test_leaf_only_integer_indices_follow_declared_order():
    X = np.arange(24, dtype=np.float64).reshape(6, 4)
    ds = RepLeafDataset(X, np.arange(6), leaf_only_features=[3, 1])
    assert ds.metadata.leaf_only_features == ["f3", "f1"]
    assert ds.metadata.ordered_numerical_features == ["f0", "f2", "f3", "f1"]
    np.testing.assert_array_equal(ds.get_raw_features(), X[:, [0, 2]])
    np.testing.assert_array_equal(ds.get_numerical_features(), X[:, [0, 2, 3, 1]])


def test_default_path_matches_origin_main_bitwise_fixture():
    rng = np.random.RandomState(123)
    X = rng.normal(size=(80, 4))
    y = 1.5 * X[:, 0] - 0.7 * X[:, 2] + 0.2 * rng.normal(size=80)
    model = RepLeafRegressor(
        n_estimators=4,
        num_leaves=4,
        min_samples_leaf=5,
        leaf_model="adaptive",
        encoder="identity",
        split_backend="numpy",
        random_state=7,
    ).fit(X, y)
    expected = np.array(
        [
            float.fromhex("-0x1.31e6a3df7b79fp-1"),
            float.fromhex("0x1.4cb06d8ab3ba9p-3"),
            float.fromhex("0x1.aba9789703040p-1"),
            float.fromhex("0x1.c5423830c3dfap-1"),
            float.fromhex("0x1.bc50b6f265908p-1"),
            float.fromhex("0x1.4ece51ae9f6c5p-1"),
            float.fromhex("-0x1.a83da04c1ce62p-1"),
            float.fromhex("0x1.ca2b037a6b336p-6"),
            float.fromhex("-0x1.fec80efdc0fbcp-1"),
            float.fromhex("0x1.aa7fdf8f89668p-3"),
        ]
    )
    assert np.array_equal(model.predict(X[:10]), expected)
