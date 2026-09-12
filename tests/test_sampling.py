"""LightGBM-style row and column sampling tests."""

import numpy as np
import pytest

from repleafgbm import RepLeafClassifier, RepLeafRegressor


def _data(seed=0, n_rows=320, n_features=10):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_rows, n_features))
    y = 2 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2] ** 2 + rng.normal(
        scale=0.1, size=n_rows
    )
    return X, y


def _tree_signature(model):
    return [tree.to_dict() for tree in model.booster_.trees_]


def test_sampling_parameter_defaults_and_validation():
    defaults = RepLeafRegressor().get_params()
    assert defaults["subsample"] == 1.0
    assert defaults["subsample_freq"] == 0
    assert defaults["colsample_bytree"] == 1.0

    X, y = _data(n_rows=80)
    for kwargs, match in (
        ({"subsample": 0.0}, "subsample"),
        ({"subsample": 1.1}, "subsample"),
        ({"subsample_freq": -1}, "subsample_freq"),
        ({"colsample_bytree": 0.0}, "colsample_bytree"),
        ({"colsample_bytree": 1.1}, "colsample_bytree"),
    ):
        with pytest.raises(ValueError, match=match):
            RepLeafRegressor(n_estimators=1, **kwargs).fit(X, y)


def test_sampling_is_exactly_reproducible_at_fixed_seed():
    X, y = _data()
    params = dict(
        n_estimators=6,
        num_leaves=8,
        min_samples_leaf=8,
        leaf_model="constant",
        subsample=0.65,
        subsample_freq=2,
        colsample_bytree=0.6,
        random_state=17,
        split_backend="numpy",
    )
    first = RepLeafRegressor(**params).fit(X, y)
    second = RepLeafRegressor(**params).fit(X, y)
    assert _tree_signature(first) == _tree_signature(second)
    np.testing.assert_array_equal(first.predict(X), second.predict(X))
    for a, b in zip(first.booster_.sampled_rows_, second.booster_.sampled_rows_):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(first.booster_.sampled_features_, second.booster_.sampled_features_):
        np.testing.assert_array_equal(a, b)


def test_disabled_sampling_is_bitwise_identical_to_previous_path():
    X, y = _data()
    common = dict(
        n_estimators=5,
        num_leaves=8,
        min_samples_leaf=8,
        leaf_model="constant",
        random_state=23,
        split_backend="numpy",
    )
    default = RepLeafRegressor(**common).fit(X, y)
    explicit = RepLeafRegressor(
        **common, subsample=1.0, subsample_freq=0, colsample_bytree=1.0
    ).fit(X, y)
    assert _tree_signature(default) == _tree_signature(explicit)
    np.testing.assert_array_equal(default.predict(X), explicit.predict(X))


def test_rows_refresh_by_frequency_and_columns_refresh_per_tree():
    X, y = _data(n_rows=300, n_features=10)
    model = RepLeafRegressor(
        n_estimators=5,
        num_leaves=8,
        min_samples_leaf=8,
        leaf_model="constant",
        subsample=0.5,
        subsample_freq=2,
        colsample_bytree=0.4,
        random_state=29,
        split_backend="numpy",
    ).fit(X, y)
    rows = model.booster_.sampled_rows_
    features = model.booster_.sampled_features_
    assert [len(value) for value in rows] == [150] * 5
    np.testing.assert_array_equal(rows[0], rows[1])
    np.testing.assert_array_equal(rows[2], rows[3])
    assert not np.array_equal(rows[0], rows[2])
    assert [len(value) for value in features] == [4] * 5
    assert any(not np.array_equal(features[0], value) for value in features[1:])
    for tree, allowed in zip(model.booster_.trees_, features):
        used = tree.feature[tree.feature >= 0]
        assert np.isin(used, allowed).all()


def test_min_samples_leaf_is_counted_on_sampled_rows():
    X, y = _data(n_rows=100, n_features=4)
    model = RepLeafRegressor(
        n_estimators=1,
        num_leaves=8,
        min_samples_leaf=15,
        leaf_model="constant",
        subsample=0.2,
        subsample_freq=1,
        random_state=3,
        split_backend="numpy",
    ).fit(X, y)
    assert len(model.booster_.sampled_rows_[0]) == 20
    assert model.booster_.trees_[0].n_leaves == 1


def test_subsample_freq_zero_disables_row_sampling():
    X, y = _data(n_rows=120)
    model = RepLeafRegressor(
        n_estimators=3,
        leaf_model="constant",
        subsample=0.2,
        subsample_freq=0,
        random_state=5,
    ).fit(X, y)
    for rows in model.booster_.sampled_rows_:
        np.testing.assert_array_equal(rows, np.arange(120))


def test_sampling_numpy_rust_parity():
    pytest.importorskip("repleafgbm_native", reason="Rust extension not built")
    X, y = _data(n_rows=400)
    predictions = {}
    signatures = {}
    for backend in ("numpy", "rust"):
        model = RepLeafRegressor(
            n_estimators=8,
            num_leaves=8,
            min_samples_leaf=8,
            leaf_model="constant",
            subsample=0.7,
            subsample_freq=1,
            colsample_bytree=0.5,
            random_state=31,
            split_backend=backend,
        ).fit(X, y)
        predictions[backend] = model.predict(X)
        signatures[backend] = _tree_signature(model)
    assert signatures["numpy"] == signatures["rust"]
    np.testing.assert_array_equal(predictions["numpy"], predictions["rust"])


def test_sampling_save_load_roundtrip(tmp_path):
    X, y = _data()
    model = RepLeafRegressor(
        n_estimators=5,
        num_leaves=8,
        min_samples_leaf=8,
        leaf_model="constant",
        subsample=0.7,
        subsample_freq=2,
        colsample_bytree=0.6,
        random_state=37,
    ).fit(X, y)
    prediction = model.predict(X)
    model.save_model(tmp_path / "sampled")
    loaded = RepLeafRegressor.load_model(tmp_path / "sampled")
    np.testing.assert_array_equal(loaded.predict(X), prediction)
    assert loaded.subsample == 0.7
    assert loaded.subsample_freq == 2
    assert loaded.colsample_bytree == 0.6


def test_classifier_uses_same_sampling_semantics():
    X, y = _data(n_rows=240, n_features=8)
    model = RepLeafClassifier(
        n_estimators=3,
        num_leaves=4,
        min_samples_leaf=8,
        leaf_model="constant",
        subsample=0.5,
        subsample_freq=1,
        colsample_bytree=0.5,
        random_state=41,
        split_backend="numpy",
    ).fit(X, y > np.median(y))
    assert len(model.booster_.sampled_rows_) == 3
    assert all(len(rows) == 120 for rows in model.booster_.sampled_rows_)
    assert len(model.booster_.sampled_features_) == 3
    assert all(len(features) == 4 for features in model.booster_.sampled_features_)
