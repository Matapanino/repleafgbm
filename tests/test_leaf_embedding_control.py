"""Leaf embedding width, reduction, and numerical-column routing tests."""

import warnings

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from repleafgbm import RepLeafClassifier, RepLeafRegressor
from repleafgbm.encoders import (
    ColumnSubsetEncoder,
    RandomProjectionEncoder,
    SimplePLREncoder,
    TargetCorrelationEncoder,
)


def _linear_leaf_fraction(model) -> float:
    """Independent check for the public fitted diagnostic."""
    total = linear = 0
    n_trees = model.booster_.best_iteration_ or model.booster_.n_trees
    for values in model.booster_.leaf_values_[:n_trees]:
        total += len(values.bias)
        if values.weights.shape[1]:
            linear += int(np.any(values.weights != 0.0, axis=1).sum())
    return linear / max(total, 1)


def _wide_data(seed=0, n_rows=90):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_rows, 133))
    y = X[:, :8].sum(axis=1) + rng.normal(scale=0.05, size=n_rows)
    return X, y


def test_665_to_64_reduction_modes_warning_and_diagnostics():
    X, y = _wide_data()
    common = dict(
        n_estimators=1,
        num_leaves=2,
        min_samples_leaf=5,
        leaf_model="embedded_linear",
        encoder="plr",
        encoder_params={"n_bins": 4, "add_linear": True},
        max_leaf_emb_dim=64,
        random_state=7,
    )

    with pytest.warns(UserWarning, match=r"665.*64.*random_projection"):
        projected = RepLeafRegressor(**common).fit(X, y)
    assert projected.leaf_embedding_input_dim_ == 665
    assert projected.leaf_embedding_dim_ == 64
    assert projected.leaf_reduction_ == "random_projection"

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        unreduced = RepLeafRegressor(**common, leaf_reduction="none").fit(X, y)
    assert unreduced.leaf_embedding_input_dim_ == 665
    assert unreduced.leaf_embedding_dim_ == 665
    assert unreduced.leaf_reduction_ == "none"

    with pytest.warns(UserWarning, match=r"665.*64.*target_correlation"):
        learned = RepLeafRegressor(
            **common, leaf_reduction="target_correlation"
        ).fit(X, y)
    assert learned.leaf_embedding_dim_ == 64
    assert learned.leaf_reduction_ == "target_correlation"

    for model in (projected, unreduced, learned):
        assert model.linear_leaf_fraction_ == _linear_leaf_fraction(model)


def test_reduction_warning_only_fires_when_width_is_reduced():
    X, y = _wide_data(n_rows=70)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        model = RepLeafRegressor(
            n_estimators=1,
            num_leaves=2,
            min_samples_leaf=5,
            leaf_model="embedded_linear",
            encoder="identity",
            max_leaf_emb_dim=256,
            leaf_reduction="target_correlation",
            random_state=3,
        ).fit(X, y)
    assert model.leaf_reduction_ == "none"
    assert model.leaf_embedding_input_dim_ == model.leaf_embedding_dim_ == 133


def test_target_reduction_recovers_wide_signal_lost_by_random_projection():
    """The released 665->64 path loses a distributed PLR signal."""
    rng = np.random.default_rng(11)
    X = rng.normal(size=(1200, 133))
    # Eighty independent threshold effects need more than a random 64-D
    # subspace, while supervised selection can retain their strongest bases.
    y = (X[:, :80] > 0.0).sum(axis=1) + rng.normal(scale=0.05, size=1200)
    X_train, X_test = X[:900], X[900:]
    y_train, y_test = y[:900], y[900:]

    base = SimplePLREncoder(n_bins=4, add_linear=True)
    none = base.fit(X_train, y_train)
    random = RandomProjectionEncoder(
        SimplePLREncoder(n_bins=4, add_linear=True), out_dim=64, random_state=4
    ).fit(X_train, y_train)
    learned = TargetCorrelationEncoder(
        SimplePLREncoder(n_bins=4, add_linear=True), out_dim=64
    ).fit(X_train, y_train)

    def rmse(encoder):
        reg = Ridge(alpha=1.0).fit(encoder.transform(X_train), y_train)
        return mean_squared_error(y_test, reg.predict(encoder.transform(X_test))) ** 0.5

    scores = {name: rmse(enc) for name, enc in (
        ("none", none), ("random_projection", random),
        ("target_correlation", learned),
    )}
    assert scores["none"] < scores["random_projection"] * 0.8
    assert scores["target_correlation"] < scores["random_projection"] * 0.8


def test_column_subset_plr_beats_all_columns_on_noise_bank():
    rng = np.random.default_rng(21)
    X = rng.normal(size=(900, 22))
    y = (
        2.0 * np.maximum(X[:, 0], 0.0)
        - 1.5 * np.maximum(-X[:, 1], 0.0)
        + rng.normal(scale=0.08, size=900)
    )
    X_train, X_test = X[:700], X[700:]
    y_train, y_test = y[:700], y[700:]
    common = dict(
        n_estimators=12,
        learning_rate=0.15,
        num_leaves=8,
        min_samples_leaf=20,
        leaf_model="embedded_linear",
        max_leaf_emb_dim=256,
        leaf_reduction="none",
        random_state=5,
    )
    all_columns = RepLeafRegressor(
        **common, encoder="plr", encoder_params={"n_bins": 4}
    ).fit(X_train, y_train)
    routed = RepLeafRegressor(
        **common,
        encoder="column_subset",
        encoder_params={
            "columns": [0, 1],
            "base_name": "plr",
            "base_config": {"n_bins": 4},
        },
    ).fit(X_train, y_train)
    all_rmse = mean_squared_error(y_test, all_columns.predict(X_test)) ** 0.5
    routed_rmse = mean_squared_error(y_test, routed.predict(X_test)) ** 0.5
    assert routed_rmse < all_rmse * 0.9
    assert routed.linear_leaf_fraction_ > all_columns.linear_leaf_fraction_
    assert routed.encoder_.columns == (0, 1)
    assert routed.encoder_.output_dim == 10


@pytest.mark.parametrize("reduction", ["random_projection", "target_correlation"])
def test_reduction_determinism_and_save_load(tmp_path, reduction):
    X, y = _wide_data(n_rows=100)
    params = dict(
        n_estimators=2,
        num_leaves=2,
        min_samples_leaf=5,
        leaf_model="embedded_linear",
        encoder="plr",
        encoder_params={"n_bins": 4},
        max_leaf_emb_dim=64,
        leaf_reduction=reduction,
        random_state=19,
    )
    with pytest.warns(UserWarning):
        first = RepLeafRegressor(**params).fit(X, y)
    with pytest.warns(UserWarning):
        second = RepLeafRegressor(**params).fit(X, y)
    np.testing.assert_array_equal(first.encoder_.transform(X), second.encoder_.transform(X))
    pred = first.predict(X)
    first.save_model(tmp_path / reduction)
    loaded = RepLeafRegressor.load_model(tmp_path / reduction)
    np.testing.assert_allclose(loaded.predict(X), pred)
    assert loaded.leaf_reduction == reduction


def test_column_subset_encoder_state_roundtrip():
    X, y = _wide_data(n_rows=40)
    encoder = ColumnSubsetEncoder(
        SimplePLREncoder(n_bins=3), columns=[1, 4, 7]
    ).fit(X, y)
    fresh = ColumnSubsetEncoder(
        SimplePLREncoder(n_bins=3), columns=[1, 4, 7]
    )
    fresh.set_state(encoder.get_state())
    np.testing.assert_allclose(fresh.transform(X), encoder.transform(X))


def test_invalid_reduction_and_subset_are_actionable():
    X, y = _wide_data(n_rows=40)
    with pytest.raises(ValueError, match="leaf_reduction"):
        RepLeafRegressor(leaf_reduction="magic").fit(X, y)
    with pytest.raises(ValueError, match="numerical-feature index"):
        ColumnSubsetEncoder(SimplePLREncoder(), columns=[133]).fit(X, y)


def test_classifier_exposes_same_reduction_controls():
    X, y = _wide_data(n_rows=100)
    clf = RepLeafClassifier(
        n_estimators=1,
        num_leaves=2,
        min_samples_leaf=5,
        max_leaf_emb_dim=64,
        leaf_reduction="none",
        encoder="plr",
        encoder_params={"n_bins": 4},
    ).fit(X, y > np.median(y))
    assert clf.get_params()["leaf_reduction"] == "none"
    assert clf.leaf_embedding_dim_ == 665
    assert clf.linear_leaf_fraction_ == _linear_leaf_fraction(clf)
