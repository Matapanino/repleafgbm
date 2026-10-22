"""MatGBM leaf-only features, expert bundles, and select leaves."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from repleafgbm import LeafExpert, RepLeafDataset, RepLeafRegressor
from repleafgbm.encoders import ExpertBundleEncoder, make_encoder
from repleafgbm.encoders.base import BaseEncoder


class _DifferentTrainEncoder(BaseEncoder):
    """Test encoder whose OOF-like train representation differs from inference."""

    name = "different_train_test"
    transform_row_counts: list[int] = []

    def __init__(self):
        self.n_features_in_: int | None = None

    def fit(self, X_num, y=None, sample_weight=None):
        self.n_features_in_ = X_num.shape[1]
        return self

    def fit_transform_train(self, X_num, y=None, sample_weight=None):
        self.fit(X_num, y, sample_weight)
        return np.asarray(X_num, dtype=np.float64) + 100.0

    def transform(self, X_num):
        self._check_fitted("n_features_in_")
        type(self).transform_row_counts.append(len(X_num))
        return np.asarray(X_num, dtype=np.float64) - 100.0

    @property
    def output_dim(self):
        self._check_fitted("n_features_in_")
        return self.n_features_in_

    def get_config(self):
        return {}

    def get_state(self):
        return {"n_features_in": np.asarray(self.n_features_in_, dtype=np.int64)}

    def set_state(self, state):
        self.n_features_in_ = int(np.asarray(state["n_features_in"]).item())


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


def test_expert_bundle_concatenates_named_column_subsets_in_order():
    X = np.arange(40, dtype=np.float64).reshape(10, 4)
    bundle = ExpertBundleEncoder(
        expert_names=["right", "left"],
        encoders=[
            make_encoder("identity", standardize=False),
            make_encoder("identity", standardize=False),
        ],
        column_subsets=[[2, 3], [0]],
    )
    Z = bundle.fit_transform_train(X)
    assert bundle.expert_names == ["right", "left"]
    assert bundle.block_slices == [slice(0, 2), slice(2, 3)]
    np.testing.assert_array_equal(Z, np.column_stack([X[:, 2:4], X[:, 0]]))
    np.testing.assert_array_equal(bundle.transform(X), Z)


def test_bundle_training_cache_uses_fit_transform_train_but_eval_uses_transform():
    _DifferentTrainEncoder.transform_row_counts.clear()
    rng = np.random.RandomState(5)
    X = rng.normal(size=(91, 2))
    y = X[:, 0] - X[:, 1]
    X_eval = rng.normal(size=(17, 2))
    y_eval = X_eval[:, 0] - X_eval[:, 1]

    model = RepLeafRegressor(
        n_estimators=2,
        num_leaves=3,
        min_samples_leaf=5,
        leaf_model="embedded_linear",
        leaf_experts=[LeafExpert("oof", _DifferentTrainEncoder(), columns=[0])],
        split_backend="numpy",
        random_state=2,
    ).fit(X, y, eval_set=[(X_eval, y_eval)])

    assert model.encoder_.expert_names == ["oof"]
    assert 91 not in _DifferentTrainEncoder.transform_row_counts
    assert 17 in _DifferentTrainEncoder.transform_row_counts


def test_leaf_experts_and_nondefault_encoder_are_rejected():
    X = np.arange(40, dtype=np.float64).reshape(20, 2)
    with pytest.raises(ValueError, match="leaf_experts.*encoder"):
        RepLeafRegressor(
            n_estimators=1,
            leaf_experts=[{"name": "x", "encoder": "identity"}],
            encoder="plr",
        ).fit(X, X[:, 0])
