"""MatGBM leaf-only features, expert bundles, and select leaves."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from repleafgbm import LeafExpert, RepLeafClassifier, RepLeafDataset, RepLeafRegressor
from repleafgbm.core.leaf_models import AdaptiveLeafModel, SelectLeafModel
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
    implicit_identity = RepLeafRegressor(
        n_estimators=4,
        num_leaves=4,
        min_samples_leaf=5,
        leaf_model="adaptive",
        split_backend="numpy",
        random_state=7,
    ).fit(X, y)
    assert np.array_equal(implicit_identity.predict(X[:10]), expected)


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


def test_select_single_block_matches_adaptive_leaf_fit():
    rng = np.random.RandomState(19)
    Z = rng.normal(size=(180, 3))
    target = 2.0 * Z[:, 0] - 0.8 * Z[:, 2] + rng.normal(scale=0.05, size=180)
    grad = -target
    hess = np.ones(180)
    leaves = [np.arange(90), np.arange(90, 180)]
    adaptive = AdaptiveLeafModel(
        l2=0.2, min_samples_linear=10, leaf_gate_margin=0.01
    ).fit_leaves(leaves, grad, hess, Z)
    select = SelectLeafModel(
        l2=0.2, min_samples_linear=10, leaf_gate_margin=0.01
    )
    select.blocks = [slice(0, 3)]
    selected = select.fit_leaves(leaves, grad, hess, Z)
    np.testing.assert_allclose(selected.bias, adaptive.bias, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        selected.weights, adaptive.weights, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(selected.z_min, adaptive.z_min)
    np.testing.assert_allclose(selected.z_max, adaptive.z_max)


def test_select_chooses_different_planted_blocks_and_zeroes_other_weights():
    rng = np.random.RandomState(23)
    n = 300
    route = np.repeat([0, 1], n // 2)
    Z = rng.normal(size=(n, 4))
    target = np.where(
        route == 0,
        4.0 * Z[:, 0] - 2.0 * Z[:, 1],
        -3.0 * Z[:, 2] + 1.5 * Z[:, 3],
    ) + rng.normal(scale=0.03, size=n)
    leaves = [np.flatnonzero(route == 0), np.flatnonzero(route == 1)]
    select = SelectLeafModel(
        l2=0.1, min_samples_linear=20, leaf_gate_margin=0.01
    )
    select.blocks = [slice(0, 2), slice(2, 4)]
    values = select.fit_leaves(leaves, -target, np.ones(n), Z)
    assert select.last_selections_ == [0, 1]
    assert np.all(values.weights[0, 2:] == 0.0)
    assert np.all(values.weights[1, :2] == 0.0)
    assert np.all(np.isfinite(values.z_min))


def test_select_estimator_exposes_named_selection_diagnostics():
    rng = np.random.RandomState(29)
    X = rng.normal(size=(260, 3))
    y = np.where(X[:, 0] < 0, 3.0 * X[:, 1], -4.0 * X[:, 2])
    model = RepLeafRegressor(
        n_estimators=5,
        num_leaves=4,
        min_samples_leaf=15,
        leaf_model="select",
        leaf_gate_margin=0.0,
        leaf_experts=[
            {"name": "left_signal", "encoder": "identity", "columns": [1]},
            {"name": "right_signal", "encoder": "identity", "columns": [2]},
        ],
        split_backend="numpy",
        random_state=4,
    ).fit(X, y)
    assert set(model.expert_selection_counts_) == {
        "constant",
        "left_signal",
        "right_signal",
    }
    assert sum(model.expert_selection_counts_.values()) == sum(
        tree.n_leaves for tree in model.booster_.trees_
    )
    assert sum(model.expert_selection_fraction_.values()) == pytest.approx(1.0)


@pytest.mark.parametrize("backend", ["numpy", "rust"])
def test_select_single_expert_matches_adaptive_prediction(backend):
    rng = np.random.RandomState(31)
    X = rng.normal(size=(280, 3))
    y = 2.5 * X[:, 0] - X[:, 2] + rng.normal(scale=0.1, size=280)
    common = dict(
        n_estimators=7,
        num_leaves=5,
        min_samples_leaf=12,
        leaf_gate_margin=0.01,
        split_backend=backend,
        random_state=8,
    )
    adaptive = RepLeafRegressor(
        leaf_model="adaptive", encoder="identity", **common
    ).fit(X, y)
    select = RepLeafRegressor(
        leaf_model="select",
        leaf_experts=[{"name": "identity", "encoder": "identity"}],
        **common,
    ).fit(X, y)
    np.testing.assert_allclose(
        select.predict(X), adaptive.predict(X), rtol=1e-9, atol=1e-11
    )


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


@pytest.mark.parametrize("backend", ["numpy", "rust"])
def test_select_recovers_planted_blocks_and_beats_concatenated_ridge(backend):
    X_train, y_train = _planted_expert_data(37, 600)
    X_valid, y_valid = _planted_expert_data(41, 300)
    experts = [
        {"name": "a", "encoder": "identity", "columns": ["a0", "a1"]},
        {"name": "b", "encoder": "identity", "columns": ["b0", "b1"]},
    ]
    common = dict(
        n_estimators=8,
        num_leaves=4,
        max_depth=2,
        min_samples_leaf=35,
        leaf_experts=experts,
        leaf_only_features=["a0", "a1", "b0", "b1"],
        leaf_reduction="none",
        leaf_gate_margin=0.05,
        split_backend=backend,
        random_state=9,
    )
    select = RepLeafRegressor(leaf_model="select", **common).fit(X_train, y_train)
    concat = RepLeafRegressor(leaf_model="embedded_linear", **common).fit(
        X_train, y_train
    )
    select_rmse = np.sqrt(np.mean((select.predict(X_valid) - y_valid) ** 2))
    concat_rmse = np.sqrt(np.mean((concat.predict(X_valid) - y_valid) ** 2))
    selected_true = (
        select.expert_selection_counts_["a"]
        + select.expert_selection_counts_["b"]
    )
    assert selected_true > select.expert_selection_counts_["constant"]
    assert select_rmse <= concat_rmse


def test_permuted_false_block_is_selected_less_and_does_not_change_validation_rmse():
    X_train, y_train = _planted_expert_data(43, 700)
    X_valid, y_valid = _planted_expert_data(47, 350)
    rng = np.random.RandomState(53)
    X_train["fake"] = X_train["a0"].to_numpy()[rng.permutation(len(X_train))]
    X_valid["fake"] = X_valid["a0"].to_numpy()[rng.permutation(len(X_valid))]
    true_experts = [
        {"name": "a", "encoder": "identity", "columns": ["a0", "a1"]},
        {"name": "b", "encoder": "identity", "columns": ["b0", "b1"]},
    ]
    common = dict(
        n_estimators=6,
        num_leaves=4,
        max_depth=2,
        min_samples_leaf=40,
        leaf_model="select",
        leaf_only_features=["a0", "a1", "b0", "b1", "fake"],
        leaf_reduction="none",
        leaf_gate_margin=0.1,
        split_backend="numpy",
        random_state=10,
    )
    baseline = RepLeafRegressor(leaf_experts=true_experts, **common).fit(
        X_train, y_train
    )
    with_fake = RepLeafRegressor(
        leaf_experts=true_experts
        + [{"name": "fake", "encoder": "identity", "columns": ["fake"]}],
        **common,
    ).fit(X_train, y_train)
    true_count = (
        with_fake.expert_selection_counts_["a"]
        + with_fake.expert_selection_counts_["b"]
    )
    assert with_fake.expert_selection_counts_["fake"] < true_count
    rmse_baseline = np.sqrt(np.mean((baseline.predict(X_valid) - y_valid) ** 2))
    rmse_fake = np.sqrt(np.mean((with_fake.predict(X_valid) - y_valid) ** 2))
    assert rmse_fake == pytest.approx(rmse_baseline, rel=1e-3, abs=1e-3)


def test_select_rejects_multiclass_and_multioutput():
    rng = np.random.RandomState(59)
    X = rng.normal(size=(90, 3))
    with pytest.raises(NotImplementedError, match="adaptive.*embedded_linear"):
        RepLeafClassifier(n_estimators=1, leaf_model="select").fit(
            X, np.arange(90) % 3
        )
    with pytest.raises(NotImplementedError, match="adaptive.*embedded_linear"):
        RepLeafRegressor(n_estimators=1, leaf_model="select").fit(
            X, np.column_stack([X[:, 0], X[:, 1]])
        )


def test_select_supports_binary_classification():
    rng = np.random.RandomState(60)
    X = rng.normal(size=(180, 3))
    y = (2.0 * X[:, 1] - X[:, 2] > 0).astype(int)
    model = RepLeafClassifier(
        n_estimators=3,
        num_leaves=4,
        min_samples_leaf=10,
        leaf_model="select",
        leaf_experts=[{"name": "signal", "encoder": "identity", "columns": [1, 2]}],
        split_backend="numpy",
        random_state=6,
    ).fit(X, y)
    assert model.predict_proba(X).shape == (180, 2)
    assert sum(model.expert_selection_counts_.values()) == sum(
        tree.n_leaves for tree in model.booster_.trees_
    )


def test_router_extraction_accepts_leaf_only_bundle_select():
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
        leaf_model="select",
        leaf_experts=[
            {"name": "teacher", "encoder": "identity", "columns": ["teacher"]}
        ],
        leaf_only_features=["teacher"],
        min_samples_leaf=10,
        random_state=4,
    ).fit(X, y)
    assert model.routing_feature_names_.tolist() == ["route"]
    assert set(model.expert_selection_counts_) == {"constant", "teacher"}


def test_leaf_only_bundle_select_roundtrip_writes_v9(tmp_path, monkeypatch):
    X, y = _planted_expert_data(61, 260)
    model = RepLeafRegressor(
        n_estimators=4,
        num_leaves=4,
        min_samples_leaf=20,
        leaf_model="select",
        leaf_experts=[
            {"name": "a", "encoder": "identity", "columns": ["a0", "a1"]},
            {"name": "b", "encoder": "identity", "columns": ["b0", "b1"]},
        ],
        leaf_only_features=["a0", "a1", "b0", "b1"],
        leaf_reduction="none",
        split_backend="numpy",
        random_state=12,
    ).fit(X, y)
    expected = model.predict(X)
    path = tmp_path / "v9_model"
    model.save_model(path)
    config = json.loads((path / "model_config.json").read_text())
    metadata = json.loads((path / "feature_metadata.json").read_text())
    assert config["format_version"] == 9
    assert metadata["leaf_only_features"] == ["a0", "a1", "b0", "b1"]
    loaded = RepLeafRegressor.load_model(path)
    np.testing.assert_allclose(loaded.predict(X), expected, rtol=0, atol=0)
    assert loaded.expert_selection_counts_ == model.expert_selection_counts_

    import repleafgbm.core.serialization as serialization

    monkeypatch.setattr(serialization, "READABLE_VERSIONS", tuple(range(1, 9)))
    with pytest.raises(ValueError, match="Unsupported model format version 9"):
        RepLeafRegressor.load_model(path)


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


def test_select_same_seed_is_deterministic():
    X, y = _planted_expert_data(71, 320)
    params = dict(
        n_estimators=5,
        num_leaves=4,
        min_samples_leaf=20,
        leaf_model="select",
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
    assert first.expert_selection_counts_ == second.expert_selection_counts_


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
            leaf_model="select",
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


def test_select_ignores_zero_hessian_rows():
    from repleafgbm.core.leaf_models import SelectLeafModel

    rng = np.random.RandomState(5)
    Z = rng.normal(size=(80, 2))
    target = 3.0 * Z[:, 0] + rng.normal(scale=0.05, size=80)
    grad, hess = -target, np.ones(80)
    reference = SelectLeafModel(l2=0.5, min_samples_linear=5, leaf_gate_margin=0.0)
    reference.blocks = [slice(0, 2)]
    expected = reference.fit_leaves([np.arange(79)], grad[:79], hess[:79], Z[:79])
    assert reference.last_selections_ == [0]
    grad_z, hess_z = grad.copy(), hess.copy()
    grad_z[79] = 0.0
    hess_z[79] = 0.0
    select = SelectLeafModel(l2=0.5, min_samples_linear=5, leaf_gate_margin=0.0)
    select.blocks = [slice(0, 2)]
    values = select.fit_leaves([np.arange(80)], grad_z, hess_z, Z)
    assert select.last_selections_ == [0]
    np.testing.assert_allclose(values.weights, expected.weights, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(values.bias, expected.bias, rtol=1e-12, atol=1e-12)


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
        leaf_model="select",
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


def test_select_rejects_insample_gate():
    from repleafgbm.core.leaf_models import SelectLeafModel

    with pytest.raises(ValueError, match="select.*LOO.*insample"):
        SelectLeafModel(l2=0.5, min_samples_linear=5, leaf_gate="insample")
    X = np.arange(60, dtype=np.float64).reshape(30, 2)
    with pytest.raises(ValueError, match="select.*LOO.*insample"):
        RepLeafRegressor(
            n_estimators=1, min_samples_leaf=5, leaf_model="select", leaf_gate="insample"
        ).fit(X, X[:, 0])


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


def test_router_extraction_rejects_select_multi_output_target():
    pytest.importorskip("lightgbm")
    from repleafgbm.external import RouterExtractionRegressor

    X = np.arange(120, dtype=np.float64).reshape(60, 2)
    with pytest.raises(NotImplementedError, match="adaptive.*embedded_linear"):
        RouterExtractionRegressor(leaf_model="select", min_samples_leaf=5).fit(
            X, np.column_stack([X[:, 0], X[:, 1]])
        )


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
