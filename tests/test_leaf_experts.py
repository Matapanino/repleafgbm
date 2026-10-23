"""MatGBM leaf-only features and expert bundles."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from repleafgbm import LeafExpert, RepLeafClassifier, RepLeafDataset, RepLeafRegressor
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


def test_default_path_matches_origin_main_fixture():
    rng = np.random.RandomState(123)
    X = rng.normal(size=(80, 4))
    y = 1.5 * X[:, 0] - 0.7 * X[:, 2] + 0.2 * rng.normal(size=80)
    model = RepLeafRegressor(
        n_estimators=4,
        num_leaves=4,
        min_samples_leaf=5,
        leaf_model="adaptive",
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
    # Linear solves vary by a few ULP across BLAS implementations and platforms.
    np.testing.assert_allclose(model.predict(X[:10]), expected, rtol=1e-14, atol=1e-15)
    implicit_identity = RepLeafRegressor(
        n_estimators=4,
        num_leaves=4,
        min_samples_leaf=5,
        leaf_model="adaptive",
        split_backend="numpy",
        random_state=7,
    ).fit(X, y)
    np.testing.assert_allclose(
        implicit_identity.predict(X[:10]), expected, rtol=1e-14, atol=1e-15
    )


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


def test_max_leaf_embedding_dimension_is_applied_per_expert():
    rng = np.random.RandomState(3)
    X = rng.normal(size=(100, 2))
    y = X[:, 0] - X[:, 1]
    with pytest.warns(UserWarning, match="Expert"):
        model = RepLeafRegressor(
            n_estimators=1,
            min_samples_leaf=5,
            leaf_model="embedded_linear",
            leaf_experts=[
                {
                    "name": "first",
                    "encoder": "plr",
                    "columns": [0],
                    "encoder_params": {"n_bins": 4},
                },
                {
                    "name": "second",
                    "encoder": "plr",
                    "columns": [1],
                    "encoder_params": {"n_bins": 4},
                },
            ],
            max_leaf_emb_dim=2,
            random_state=1,
        ).fit(X, y)
    assert model.encoder_.block_slices == [slice(0, 2), slice(2, 4)]
    assert model.leaf_embedding_input_dim_ == 10
    assert model.leaf_embedding_dim_ == 4


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


def test_encoder_public_default_is_identity():
    assert RepLeafRegressor().get_params()["encoder"] == "identity"
    from repleafgbm.external import RouterExtractionRegressor

    assert RouterExtractionRegressor().get_params()["encoder"] == "identity"


def test_leaf_experts_accept_public_default_encoder():
    X = np.arange(40, dtype=np.float64).reshape(20, 2)
    model = RepLeafRegressor(
        n_estimators=1,
        leaf_experts=[{"name": "x", "encoder": "identity"}],
    ).fit(X, X[:, 0])
    assert model.encoder_.expert_names == ["x"]


def test_leaf_experts_and_nondefault_encoder_are_rejected():
    X = np.arange(40, dtype=np.float64).reshape(20, 2)
    with pytest.raises(ValueError, match="leaf_experts.*encoder"):
        RepLeafRegressor(
            n_estimators=1,
            leaf_experts=[{"name": "x", "encoder": "identity"}],
            encoder="plr",
        ).fit(X, X[:, 0])


def test_leaf_expert_estimator_clone_preserves_omitted_encoder():
    estimator = RepLeafRegressor(
        n_estimators=1,
        leaf_experts=[{"name": "x", "encoder": "identity"}],
    )
    cloned = clone(estimator)
    X = np.arange(40, dtype=np.float64).reshape(20, 2)
    cloned.fit(X, X[:, 0])
    assert cloned.encoder_.expert_names == ["x"]


def test_constant_is_a_reserved_expert_name():
    X = np.arange(40, dtype=np.float64).reshape(20, 2)
    with pytest.raises(ValueError, match="reserved"):
        RepLeafRegressor(
            n_estimators=1,
            leaf_experts=[{"name": "constant", "encoder": "identity"}],
        ).fit(X, X[:, 0])


def _planted_expert_data(seed, n):
    rng = np.random.RandomState(seed)
    route = rng.uniform(-1.0, 1.0, size=n)
    noise = rng.normal(size=n)
    a = rng.normal(size=(n, 2))
    b = rng.normal(size=(n, 2))
    y = (
        np.where(route < 0, -3.0, 3.0)
        + np.where(route < 0, 4.0 * a[:, 0] - 2.0 * a[:, 1],
                   -3.0 * b[:, 0] + 1.5 * b[:, 1])
        + rng.normal(scale=0.05, size=n)
    )
    X = pd.DataFrame(
        {
            "route": route,
            "noise": noise,
            "a0": a[:, 0],
            "a1": a[:, 1],
            "b0": b[:, 0],
            "b1": b[:, 1],
        }
    )
    return X, y


def test_router_extraction_accepts_leaf_only_bundle():
    pytest.importorskip("lightgbm")
    from repleafgbm.external import LightGBMExternalModel, RouterExtractionRegressor

    rng = np.random.RandomState(62)
    route = rng.uniform(-1, 1, size=180)
    teacher = rng.normal(size=180)
    X = pd.DataFrame({"route": route, "teacher": teacher})
    y = np.where(route < 0, -2.0, 2.0) + 3.0 * teacher
    model = RouterExtractionRegressor(
        base=LightGBMExternalModel(
            task="regression",
            random_state=4,
            n_estimators=3,
            num_leaves=3,
            min_child_samples=10,
        ),
        leaf_model="embedded_linear",
        leaf_experts=[
            {"name": "teacher", "encoder": "identity", "columns": ["teacher"]}
        ],
        leaf_only_features=["teacher"],
        min_samples_leaf=10,
        random_state=4,
    ).fit(X, y)
    assert model.routing_feature_names_.tolist() == ["route"]
    assert model.encoder_.expert_names == ["teacher"]


def test_legacy_configuration_keeps_pre_v9_format(tmp_path):
    rng = np.random.RandomState(67)
    X = rng.normal(size=(100, 2))
    y = X[:, 0]
    model = RepLeafRegressor(
        n_estimators=2, leaf_model="adaptive", split_backend="numpy"
    ).fit(X, y)
    path = tmp_path / "legacy"
    model.save_model(path)
    config = json.loads((path / "model_config.json").read_text())
    assert config["format_version"] == 3


def test_leaf_only_bundle_same_seed_is_deterministic():
    X, y = _planted_expert_data(71, 320)
    params = dict(
        n_estimators=5,
        num_leaves=4,
        min_samples_leaf=20,
        leaf_model="embedded_linear",
        leaf_experts=[
            {"name": "a", "encoder": "identity", "columns": ["a0", "a1"]},
            {"name": "b", "encoder": "identity", "columns": ["b0", "b1"]},
        ],
        leaf_only_features=["a0", "a1", "b0", "b1"],
        split_backend="numpy",
        random_state=13,
    )
    first = RepLeafRegressor(**params).fit(X, y)
    second = RepLeafRegressor(**params).fit(X, y)
    assert np.array_equal(first.predict(X), second.predict(X))


def test_routing_matrix_is_not_copied_per_call():
    X = np.arange(24, dtype=np.float64).reshape(6, 4)
    ds = RepLeafDataset(X, np.arange(6))
    assert ds.get_raw_features() is ds.get_raw_features()
    assert ds.get_raw_features().flags["C_CONTIGUOUS"]
    ds_leaf = RepLeafDataset(X, np.arange(6), leaf_only_features=[1])
    routing = ds_leaf.get_raw_features()
    assert routing is ds_leaf.get_raw_features()
    assert routing.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(routing, X[:, [0, 2, 3]])


class _OOFEncoder(BaseEncoder):
    name = "review_oof"

    def __init__(self):
        self.n_features_in_ = None

    def fit(self, X_num, y=None, sample_weight=None):
        self.n_features_in_ = X_num.shape[1]
        return self

    def fit_transform_train(self, X_num, y=None, sample_weight=None):
        self.fit(X_num)
        return np.asarray(X_num, dtype=np.float64) + 100.0

    def transform(self, X_num):
        return np.asarray(X_num, dtype=np.float64) - 100.0

    @property
    def output_dim(self):
        return self.n_features_in_

    def get_config(self):
        return {}

    def get_state(self):
        return {"n": np.asarray(self.n_features_in_)}

    def set_state(self, state):
        self.n_features_in_ = int(state["n"])


def test_training_representation_does_not_leak_into_predict():
    rng = np.random.RandomState(1)
    X = rng.normal(size=(120, 2))
    y = X[:, 0] - X[:, 1]
    ds = RepLeafDataset(X, y)
    model = RepLeafRegressor(
        n_estimators=2,
        num_leaves=3,
        min_samples_leaf=5,
        leaf_model="embedded_linear",
        leaf_experts=[LeafExpert("oof", _OOFEncoder(), columns=[0])],
        split_backend="numpy",
        random_state=2,
    ).fit(ds)
    np.testing.assert_array_equal(model.predict(ds), model.predict(X))


class _CountingOOFEncoder(_OOFEncoder):
    name = "review_oof_counting"
    transform_rows: list[int] = []

    def transform(self, X_num):
        type(self).transform_rows.append(len(X_num))
        return super().transform(X_num)


def test_aliased_eval_set_uses_transform_not_training_representation():
    _CountingOOFEncoder.transform_rows.clear()
    rng = np.random.RandomState(3)
    X = rng.normal(size=(130, 2))
    y = X[:, 0] - X[:, 1]
    ds = RepLeafDataset(X, y)
    RepLeafRegressor(
        n_estimators=2,
        num_leaves=3,
        min_samples_leaf=5,
        leaf_model="embedded_linear",
        leaf_experts=[LeafExpert("oof", _CountingOOFEncoder(), columns=[0])],
        split_backend="numpy",
        random_state=2,
    ).fit(ds, eval_set=[ds])
    assert 130 in _CountingOOFEncoder.transform_rows


def test_router_extraction_rejects_base_fitted_with_leaf_only_columns():
    pytest.importorskip("lightgbm")
    from repleafgbm.external import LightGBMExternalModel, RouterExtractionRegressor

    rng = np.random.RandomState(62)
    route = rng.uniform(-1, 1, size=180)
    teacher = rng.normal(size=180)
    X = pd.DataFrame({"teacher": teacher, "route": route})
    y = np.where(route < 0, -2.0, 2.0) + 3.0 * teacher
    base = LightGBMExternalModel(
        task="regression", random_state=4, n_estimators=3, num_leaves=3, min_child_samples=10
    ).fit(X.to_numpy(), y)
    with pytest.raises(ValueError, match="routing"):
        RouterExtractionRegressor(
            base=base,
            leaf_model="embedded_linear",
            leaf_experts=[{"name": "teacher", "encoder": "identity", "columns": ["teacher"]}],
            leaf_only_features=["teacher"],
            min_samples_leaf=10,
            random_state=4,
        ).fit(X, y)


@pytest.mark.parametrize("leaf_model", ["constant", "raw_linear"])
def test_router_extraction_rejects_experts_for_unsupported_leaf_models(leaf_model):
    pytest.importorskip("lightgbm")
    from repleafgbm.external import LightGBMExternalModel, RouterExtractionRegressor

    X = np.arange(120, dtype=np.float64).reshape(60, 2)
    y = X[:, 0]
    with pytest.raises(ValueError, match="leaf_experts requires leaf_model"):
        RouterExtractionRegressor(
            base=LightGBMExternalModel(
                task="regression", random_state=4, n_estimators=2, num_leaves=3, min_child_samples=5
            ),
            leaf_model=leaf_model,
            leaf_experts=[{"name": "a", "encoder": "identity", "columns": [0]}],
            min_samples_leaf=5,
        ).fit(X, y)


class _WideTrainEncoder(_OOFEncoder):
    name = "review_wide_train"

    def fit_transform_train(self, X_num, y=None, sample_weight=None):
        self.fit(X_num)
        X_num = np.asarray(X_num, dtype=np.float64)
        return np.column_stack([X_num, X_num])


def test_bundle_rejects_training_representation_width_mismatch():
    X = np.arange(60, dtype=np.float64).reshape(30, 2)
    with pytest.raises(ValueError, match="wide.*2 columns.*output_dim.*1"):
        RepLeafRegressor(
            n_estimators=1,
            min_samples_leaf=5,
            leaf_experts=[LeafExpert("wide", _WideTrainEncoder(), columns=[0])],
            split_backend="numpy",
        ).fit(X, X[:, 0])


def test_numpy_integer_column_specs_survive_save_model(tmp_path):
    X = np.arange(60, dtype=np.float64).reshape(20, 3)
    y = X[:, 0]
    model = RepLeafRegressor(
        n_estimators=1,
        min_samples_leaf=5,
        leaf_only_features=[np.int64(1)],
        leaf_experts=[{"name": "a", "encoder": "identity", "columns": [np.int64(0)]}],
        split_backend="numpy",
    ).fit(X, y)
    model.save_model(tmp_path / "m")
    config = json.loads((tmp_path / "m" / "model_config.json").read_text())["config"]
    assert config["leaf_only_features"] == [1]
    assert config["leaf_experts"][0]["columns"] == [0]
    loaded = RepLeafRegressor.load_model(tmp_path / "m")
    np.testing.assert_array_equal(loaded.predict(X), model.predict(X))


def test_loaded_bundle_restores_per_expert_reduction_diagnostics(tmp_path):
    rng = np.random.RandomState(3)
    X = rng.normal(size=(150, 3))
    y = X[:, 0] - X[:, 1]
    model = RepLeafRegressor(
        n_estimators=2,
        num_leaves=4,
        min_samples_leaf=8,
        leaf_model="embedded_linear",
        leaf_experts=[
            {"name": "a", "encoder": "plr", "columns": [0], "encoder_params": {"n_bins": 4}},
            {"name": "b", "encoder": "identity", "columns": [1, 2]},
        ],
        max_leaf_emb_dim=2,
        leaf_reduction="random_projection",
        split_backend="numpy",
        random_state=1,
    ).fit(X, y)
    model.save_model(tmp_path / "m")
    loaded = RepLeafRegressor.load_model(tmp_path / "m")
    assert loaded.leaf_embedding_input_dim_ == model.leaf_embedding_input_dim_ == 7
    assert loaded.leaf_reduction_ == model.leaf_reduction_ == "per_expert"
    assert loaded.leaf_embedding_dim_ == model.leaf_embedding_dim_ == 4


def test_all_columns_leaf_only_is_rejected_clearly():
    X = np.arange(40, dtype=np.float64).reshape(20, 2)
    with pytest.raises(ValueError, match="every input column"):
        RepLeafDataset(X, X[:, 0], leaf_only_features=[0, 1])


def test_bundle_passed_as_encoder_keeps_training_representation_private():
    from repleafgbm.encoders import ExpertBundleEncoder

    rng = np.random.RandomState(4)
    X = rng.normal(size=(110, 2))
    y = X[:, 0] - X[:, 1]
    ds = RepLeafDataset(X, y)
    bundle = ExpertBundleEncoder(
        expert_names=["oof"], encoders=[_OOFEncoder()], column_subsets=[[0]]
    )
    model = RepLeafRegressor(
        n_estimators=2,
        num_leaves=3,
        min_samples_leaf=5,
        leaf_model="embedded_linear",
        encoder=bundle,
        split_backend="numpy",
        random_state=2,
    ).fit(ds)
    np.testing.assert_array_equal(model.predict(ds), model.predict(X))


def test_wrapped_bundle_encoder_writes_v9(tmp_path):
    from repleafgbm.encoders import ExpertBundleEncoder, RandomProjectionEncoder, make_encoder

    rng = np.random.RandomState(6)
    X = rng.normal(size=(120, 2))
    y = X[:, 0]
    bundle = ExpertBundleEncoder(
        expert_names=["p"],
        encoders=[make_encoder("plr", n_bins=6)],
        column_subsets=[[0, 1]],
    )
    model = RepLeafRegressor(
        n_estimators=1,
        min_samples_leaf=8,
        leaf_model="embedded_linear",
        encoder=RandomProjectionEncoder(bundle, out_dim=3, random_state=0),
        split_backend="numpy",
    ).fit(X, y)
    model.save_model(tmp_path / "m")
    config = json.loads((tmp_path / "m" / "model_config.json").read_text())
    assert config["format_version"] == 9
    np.testing.assert_array_equal(
        RepLeafRegressor.load_model(tmp_path / "m").predict(X), model.predict(X)
    )


def test_leaf_only_scalar_string_is_rejected():
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [1.0, 2.0, 3.0], "z": [2.0, 3.0, 4.0]})
    with pytest.raises(ValueError, match="list of column names"):
        RepLeafDataset(X, [0.0, 1.0, 2.0], leaf_only_features="xy")


def test_feature_metadata_resolves_leaf_only_indices_and_none():
    from repleafgbm.data.metadata import FeatureMetadata

    kwargs = dict(feature_names=["a", "b"], numerical_features=["a", "b"], categorical_features=[])
    assert FeatureMetadata(leaf_only_features=[1], **kwargs).leaf_only_features == ["b"]
    assert FeatureMetadata(leaf_only_features=None, **kwargs).leaf_only_features == []


@pytest.mark.parametrize("spec", [[-1], [99]])
def test_leaf_only_compat_path_reports_index_errors(spec):
    X = np.arange(40, dtype=np.float64).reshape(20, 2)
    ds = RepLeafDataset(X, X[:, 0], leaf_only_features=[1])
    with pytest.raises(ValueError, match="out of bounds"):
        RepLeafRegressor(n_estimators=1, leaf_only_features=spec).fit(ds)


def test_bundle_does_not_copy_full_or_contiguous_column_subsets():
    from repleafgbm.encoders import ExpertBundleEncoder, make_encoder

    X = np.arange(40, dtype=np.float64).reshape(10, 4)
    bundle = ExpertBundleEncoder(
        expert_names=["all", "mid", "gap"],
        encoders=[make_encoder("identity") for _ in range(3)],
        column_subsets=[[0, 1, 2, 3], [1, 2], [0, 3]],
    )
    assert np.shares_memory(bundle._select(X, (0, 1, 2, 3)), X)
    assert np.shares_memory(bundle._select(X, (1, 2)), X)
    np.testing.assert_array_equal(bundle._select(X, (0, 3)), X[:, [0, 3]])
    Z = bundle.fit_transform_train(X)
    np.testing.assert_array_equal(bundle.transform(X), Z)
    assert Z.shape == (10, 8)


def test_router_rejects_base_fitted_on_other_named_columns():
    pytest.importorskip("lightgbm")
    from repleafgbm.external import LightGBMExternalModel, RouterExtractionRegressor

    rng = np.random.RandomState(8)
    X = pd.DataFrame(
        {"r0": rng.normal(size=160), "r1": rng.normal(size=160), "teacher": rng.normal(size=160)}
    )
    y = X["r0"].to_numpy() + 2.0 * X["teacher"].to_numpy()
    base = LightGBMExternalModel(
        task="regression", random_state=4, n_estimators=3, num_leaves=3, min_child_samples=10
    ).fit(X[["r0", "teacher"]], y)
    with pytest.raises(ValueError, match="routing"):
        RouterExtractionRegressor(
            base=base,
            leaf_model="embedded_linear",
            leaf_only_features=["teacher"],
            min_samples_leaf=10,
            random_state=4,
        ).fit(X, y)


def test_router_rejects_base_fitted_on_unrelated_named_columns():
    pytest.importorskip("lightgbm")
    from repleafgbm.external import LightGBMExternalModel, RouterExtractionRegressor

    rng = np.random.RandomState(9)
    X = pd.DataFrame(
        {"r0": rng.normal(size=160), "r1": rng.normal(size=160), "teacher": rng.normal(size=160)}
    )
    y = X["r0"].to_numpy() + 2.0 * X["teacher"].to_numpy()
    other = X[["r0", "teacher"]].rename(columns={"teacher": "foreign"})
    base = LightGBMExternalModel(
        task="regression", random_state=4, n_estimators=3, num_leaves=3, min_child_samples=10
    ).fit(other, y)
    with pytest.raises(ValueError, match="routing"):
        RouterExtractionRegressor(
            base=base,
            leaf_model="embedded_linear",
            leaf_only_features=["teacher"],
            min_samples_leaf=10,
            random_state=4,
        ).fit(X, y)


def test_expert_columns_scalar_string_is_rejected():
    X = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0] * 5, "b": [1.0, 2.0, 3.0, 4.0] * 5})
    with pytest.raises(ValueError, match="list of column names"):
        RepLeafRegressor(
            n_estimators=1,
            min_samples_leaf=5,
            leaf_experts=[{"name": "e", "encoder": "identity", "columns": "ab"}],
        ).fit(X, X["a"])


def test_registry_name_bundle_keeps_training_representation_private():
    from repleafgbm.encoders import _ENCODER_REGISTRY

    rng = np.random.RandomState(10)
    X = rng.normal(size=(100, 2))
    y = X[:, 0] - X[:, 1]
    ds = RepLeafDataset(X, y)
    _ENCODER_REGISTRY.setdefault(_OOFEncoder.name, _OOFEncoder)
    try:
        model = RepLeafRegressor(
            n_estimators=2,
            num_leaves=3,
            min_samples_leaf=5,
            leaf_model="embedded_linear",
            encoder="expert_bundle",
            encoder_params={
                "expert_names": ["oof"],
                "column_subsets": [[0]],
                "children": [{"name": _OOFEncoder.name, "config": {}}],
            },
            split_backend="numpy",
            random_state=2,
        ).fit(ds)
    finally:
        _ENCODER_REGISTRY.pop(_OOFEncoder.name, None)
    np.testing.assert_array_equal(model.predict(ds), model.predict(X))


def test_expert_encoder_instance_config_survives_save_load(tmp_path):
    from sklearn.base import clone

    from repleafgbm.encoders import make_encoder

    rng = np.random.RandomState(12)
    X = rng.normal(size=(120, 2))
    y = X[:, 0] - X[:, 1]
    plr = make_encoder("plr", n_bins=9, add_linear=False)
    model = RepLeafRegressor(
        n_estimators=1,
        min_samples_leaf=8,
        leaf_experts=[LeafExpert("p", plr, columns=[0])],
        split_backend="numpy",
        random_state=1,
    ).fit(X, y)
    model.save_model(tmp_path / "m")
    loaded = RepLeafRegressor.load_model(tmp_path / "m")
    assert loaded.leaf_experts[0]["encoder_params"] == plr.get_config()
    refit = clone(loaded).fit(X, y)
    assert refit.encoder_.encoders[0].n_bins == 9
    assert refit.encoder_.encoders[0].add_linear is False


def test_direct_bundle_round_trips_and_refits(tmp_path):
    from sklearn.base import clone

    from repleafgbm.encoders import ExpertBundleEncoder, make_encoder

    rng = np.random.RandomState(13)
    X = rng.normal(size=(120, 2))
    y = X[:, 0] - X[:, 1]
    bundle = ExpertBundleEncoder(
        expert_names=["p"],
        encoders=[make_encoder("plr", n_bins=5)],
        column_subsets=[[0]],
        max_output_dim=3,
        reduction="random_projection",
        random_state=0,
    )
    model = RepLeafRegressor(
        n_estimators=1, min_samples_leaf=8, encoder=bundle, split_backend="numpy", random_state=1
    ).fit(X, y)
    model.save_model(tmp_path / "m")
    loaded = RepLeafRegressor.load_model(tmp_path / "m")
    assert loaded.encoder == "expert_bundle"
    assert loaded.encoder_params["children"][0]["name"] == "plr"
    refit = clone(loaded).fit(X, y)
    assert refit.encoder_.max_output_dim == 3
    assert refit.encoder_.reduction == "random_projection"
    assert refit.leaf_embedding_dim_ == 3


def test_bundle_constructor_validates_names_and_subsets():
    from repleafgbm.encoders import ExpertBundleEncoder, make_encoder

    with pytest.raises(ValueError, match="reserved"):
        ExpertBundleEncoder(
            expert_names=["constant"], encoders=[make_encoder("identity")], column_subsets=[[0]]
        )
    with pytest.raises(ValueError, match="integer"):
        ExpertBundleEncoder(
            expert_names=["e"], encoders=[make_encoder("identity")], column_subsets=[[0.1, 0.9]]
        )
    with pytest.raises(ValueError, match="duplicates"):
        ExpertBundleEncoder(
            expert_names=["e"], encoders=[make_encoder("identity")], column_subsets=[[0, 0]]
        )


def test_training_view_shares_routing_matrix():
    from repleafgbm.core.leaf_models import make_leaf_model

    X = np.arange(60, dtype=np.float64).reshape(20, 3)
    ds = RepLeafDataset(X, X[:, 0], leaf_only_features=[2])
    model = RepLeafRegressor(
        n_estimators=1,
        min_samples_leaf=5,
        leaf_experts=[{"name": "t", "encoder": "identity", "columns": ["f2"]}],
        split_backend="numpy",
    )
    leaf_model = make_leaf_model("embedded_linear", l2=1.0, min_samples_linear=10)
    train = model._fit_leaf_encoder(ds, leaf_model)
    assert train is not ds
    assert train.get_raw_features() is ds.get_raw_features()


# --------------------------------------------------------------------- #
# leaf_model="select" removal (1.14.0, ADR 0012)
# --------------------------------------------------------------------- #
_SELECT_REMOVED = r"leaf_model='select' was removed in repleafgbm 1\.14\.0"


@pytest.mark.parametrize("estimator_cls", [RepLeafRegressor, RepLeafClassifier])
def test_select_leaf_model_is_removed_with_actionable_error(estimator_cls):
    from repleafgbm.core import leaf_models
    from repleafgbm.core.leaf_models import make_leaf_model

    rng = np.random.RandomState(83)
    X = rng.normal(size=(80, 3))
    y = (X[:, 0] > 0).astype(int)
    with pytest.raises(ValueError, match=_SELECT_REMOVED + r".*embedded_linear"):
        estimator_cls(n_estimators=1, min_samples_leaf=5, leaf_model="select").fit(X, y)
    with pytest.raises(ValueError, match=_SELECT_REMOVED):
        make_leaf_model("select", l2=1.0, min_samples_linear=10)
    assert not hasattr(leaf_models, "SelectLeafModel")


@pytest.mark.parametrize("with_experts", [False, True])
def test_router_extraction_rejects_removed_select_leaf_model(with_experts):
    pytest.importorskip("lightgbm")
    from repleafgbm.external import LightGBMExternalModel, RouterExtractionRegressor

    X = np.arange(120, dtype=np.float64).reshape(60, 2)
    experts = (
        dict(
            leaf_experts=[{"name": "teacher", "encoder": "identity", "columns": [1]}],
            leaf_only_features=[1],
        )
        if with_experts
        else {}
    )
    # The removal error must win over the leaf_experts compatibility check.
    with pytest.raises(ValueError, match=_SELECT_REMOVED):
        RouterExtractionRegressor(
            base=LightGBMExternalModel(
                task="regression", random_state=4, n_estimators=2, num_leaves=3,
                min_child_samples=5,
            ),
            leaf_model="select",
            min_samples_leaf=5,
            **experts,
        ).fit(X, X[:, 0])


@pytest.mark.parametrize("bundled", [True, False])
def test_loading_a_select_model_saved_by_1_13_raises_clear_error(tmp_path, bundled):
    """A 1.13.x ``select`` model must not load silently as another leaf model.

    1.13.x stored ``leaf_model="select"`` in ``model_config.json`` and its leaf
    arrays in the ordinary ``LeafValues`` layout, so such a directory is
    emulated by saving an ``embedded_linear`` model and rewriting that one
    field. The format version depends on the encoder, not on ``select``:
    a bundle with leaf-only columns writes v9, a single plain encoder v3.
    """
    X, y = _planted_expert_data(89, 200)
    params = dict(
        n_estimators=2,
        num_leaves=3,
        min_samples_leaf=20,
        leaf_model="embedded_linear",
        split_backend="numpy",
        random_state=3,
    )
    if bundled:
        params.update(
            leaf_experts=[
                {"name": "a", "encoder": "identity", "columns": ["a0", "a1"]},
                {"name": "b", "encoder": "identity", "columns": ["b0", "b1"]},
            ],
            leaf_only_features=["a0", "a1", "b0", "b1"],
            leaf_reduction="none",
        )
    path = tmp_path / "select_model"
    RepLeafRegressor(**params).fit(X, y).save_model(path)
    config_path = path / "model_config.json"
    saved = json.loads(config_path.read_text())
    assert saved["format_version"] == (9 if bundled else 3)
    saved["config"]["leaf_model"] = "select"
    config_path.write_text(json.dumps(saved))

    with pytest.raises(
        ValueError, match=r"leaf_model='select'.*repleafgbm 1\.13\.x.*embedded_linear"
    ):
        RepLeafRegressor.load_model(path)


@pytest.mark.parametrize("estimator_cls", [RepLeafRegressor, RepLeafClassifier])
def test_two_block_embedded_linear_experts_fit_predict_save_load(
    tmp_path, monkeypatch, estimator_cls
):
    """The fixed-block recipe that replaces ``select`` (S8 + O in wave 4)."""
    X, y = _planted_expert_data(97, 320)
    if estimator_cls is RepLeafClassifier:
        y = (y > np.median(y)).astype(int)
    model = estimator_cls(
        n_estimators=6,
        num_leaves=4,
        min_samples_leaf=20,
        leaf_model="embedded_linear",
        leaf_experts=[
            {"name": "a", "encoder": "identity", "columns": ["a0", "a1"]},
            {"name": "b", "encoder": "identity", "columns": ["b0", "b1"]},
        ],
        leaf_only_features=["a0", "a1", "b0", "b1"],
        leaf_reduction="none",
        split_backend="numpy",
        random_state=14,
    ).fit(X, y)
    assert model.routing_feature_names_.tolist() == ["route", "noise"]
    assert model.encoder_.expert_names == ["a", "b"]
    assert model.encoder_.block_slices == [slice(0, 2), slice(2, 4)]
    assert model.linear_leaf_fraction_ > 0.0
    predict = model.predict_proba if estimator_cls is RepLeafClassifier else model.predict
    expected = predict(X)
    assert np.all(np.isfinite(expected))

    path = tmp_path / "two_block"
    model.save_model(path)
    config = json.loads((path / "model_config.json").read_text())
    metadata = json.loads((path / "feature_metadata.json").read_text())
    assert config["format_version"] == 9
    assert metadata["leaf_only_features"] == ["a0", "a1", "b0", "b1"]
    loaded = estimator_cls.load_model(path)
    loaded_predict = (
        loaded.predict_proba if estimator_cls is RepLeafClassifier else loaded.predict
    )
    np.testing.assert_array_equal(loaded_predict(X), expected)

    import repleafgbm.core.serialization as serialization

    monkeypatch.setattr(serialization, "READABLE_VERSIONS", tuple(range(1, 9)))
    with pytest.raises(ValueError, match="Unsupported model format version 9"):
        estimator_cls.load_model(path)
