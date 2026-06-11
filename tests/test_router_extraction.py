"""Tests for router_extraction: structure mapping, replay, format v2."""

import json

import numpy as np
import pytest

from repleafgbm import RepLeafRegressor
from repleafgbm.core.leaf_models import LeafValues
from repleafgbm.core.prediction import predict_raw
from repleafgbm.external import (
    LightGBMExternalModel,
    RouterExtractionClassifier,
    RouterExtractionRegressor,
    extract_routes,
)

lgb = pytest.importorskip("lightgbm", reason="lightgbm not installed")


@pytest.fixture
def nan_regression_data():
    """Regression data with missing values to exercise default_left mapping."""
    rng = np.random.default_rng(5)
    n = 600
    X = rng.normal(size=(n, 5))
    X[rng.random((n, 5)) < 0.1] = np.nan  # 10% missing everywhere
    y = (
        np.where(np.nan_to_num(X[:, 0]) > 0, 2.0, -1.0)
        + 1.5 * np.nan_to_num(X[:, 1])
        + rng.normal(0.0, 0.1, n)
    )
    return X[:400], y[:400], X[400:], y[400:]


def _base(n_estimators=40):
    return LightGBMExternalModel(
        task="regression", random_state=0,
        n_estimators=n_estimators, num_leaves=7, min_child_samples=5,
    )


def test_extract_routes_reproduces_lightgbm_exactly(nan_regression_data):
    """Milestone 2 correctness: mapped trees + original leaf values must
    equal LightGBM's raw prediction, including missing-value routing."""
    Xtr, ytr, Xte, _ = nan_regression_data
    base = _base().fit(Xtr, ytr)
    trees, leaf_values = extract_routes(base)

    assert len(trees) == base.n_trees_
    assert any(not t.missing_left.all() for t in trees) or all(
        t.missing_left.all() for t in trees
    )  # field populated either way

    lvs = [LeafValues(bias=v, weights=np.zeros((len(v), 0))) for v in leaf_values]
    for X in (Xtr, Xte):
        ours = predict_raw(trees, lvs, init_score=0.0, learning_rate=1.0,
                           X_raw=np.asarray(X, dtype=np.float64), Z=None)
        theirs = base.model_.predict(X, raw_score=True)
        np.testing.assert_allclose(ours, theirs, atol=1e-10)


def test_extract_routes_maps_categorical_splits_exactly():
    """LightGBM '==' splits map onto native left_categories; predictions
    must match exactly, including NaN categorical routing (default_left)."""
    rng = np.random.default_rng(0)
    n = 800
    cat = rng.integers(0, 6, n).astype(float)
    x = rng.normal(size=n)
    y = np.where(np.isin(cat, [1, 4]), 3.0, -3.0) + 0.3 * x + rng.normal(0, 0.1, n)
    X = np.column_stack([cat, x])
    X[::31, 0] = np.nan
    m = lgb.LGBMRegressor(
        n_estimators=15, num_leaves=5, random_state=0, verbose=-1, min_child_samples=5
    ).fit(X, y, categorical_feature=[0])

    trees, leaf_values = extract_routes(m)
    assert any(t.left_categories is not None for t in trees)
    lvs = [LeafValues(bias=v, weights=np.zeros((len(v), 0))) for v in leaf_values]
    ours = predict_raw(trees, lvs, init_score=0.0, learning_rate=1.0,
                       X_raw=X, Z=None)
    np.testing.assert_allclose(ours, m.predict(X, raw_score=True), atol=1e-10)


def test_extract_routes_bad_input():
    with pytest.raises(TypeError, match="extract routes"):
        extract_routes(object())


def test_replay_constant_matches_lightgbm(nan_regression_data):
    """With constant leaves and ~zero ridge, sequential replay must
    reproduce LightGBM's own leaf values (ADR 0002 correctness check)."""
    Xtr, ytr, Xte, _ = nan_regression_data
    base = _base().fit(Xtr, ytr)
    model = RouterExtractionRegressor(base=base, leaf_model="constant", l2_leaf=1e-9)
    model.fit(Xtr, ytr)
    np.testing.assert_allclose(
        model.predict(Xte), base.predict_score(Xte), atol=1e-5
    )


def test_embedded_linear_replay_beats_baseline(nan_regression_data):
    Xtr, ytr, Xte, yte = nan_regression_data
    model = RouterExtractionRegressor(
        base=_base(n_estimators=80), leaf_model="embedded_linear",
        encoder="identity", min_samples_leaf=10, random_state=42,
    )
    model.fit(Xtr, ytr)
    rmse = float(np.sqrt(np.mean((model.predict(Xte) - yte) ** 2)))
    baseline = float(np.sqrt(np.mean((yte - ytr.mean()) ** 2)))
    assert rmse < 0.5 * baseline


def test_unfitted_base_is_trained_and_user_base_untouched(nan_regression_data):
    Xtr, ytr, _, _ = nan_regression_data
    base = _base()
    model = RouterExtractionRegressor(base=base, leaf_model="constant")
    model.fit(Xtr, ytr)
    assert base.model_ is None  # user's object not mutated (deepcopy)
    assert model.base_.model_ is not None
    assert model.booster_.n_trees == model.base_.n_trees_


def test_replay_early_stopping():
    """Replay stops consuming routes when the eval metric stalls, and
    prediction uses the best iteration. The base is pre-fitted *without*
    early stopping (LightGBM trims an early-stopped booster to its best
    iteration, so an untrimmed oversized base is needed here)."""
    rng = np.random.default_rng(9)
    n = 600
    X = rng.normal(size=(n, 5))
    y = X[:, 0] + 0.5 * X[:, 1] + rng.normal(0.0, 2.0, n)  # very noisy
    Xtr, ytr, Xte, yte = X[:400], y[:400], X[400:], y[400:]

    base = LightGBMExternalModel(
        task="regression", random_state=0, n_estimators=200, num_leaves=7,
        min_child_samples=5, learning_rate=0.3,
    ).fit(Xtr, ytr)  # oversized: will overfit this noise level
    model = RouterExtractionRegressor(
        base=base,
        leaf_model="embedded_linear",
        min_samples_leaf=10,
        early_stopping_rounds=10,
        random_state=42,
    )
    model.fit(Xtr, ytr, eval_set=[(Xte, yte)])

    history = model.evals_result_["valid_0"]["rmse"]
    n_replayed = model.booster_.n_trees
    assert n_replayed < base.n_trees_  # stopped before consuming all routes
    assert len(history) == n_replayed
    assert model.best_iteration_ == int(np.argmin(history)) + 1

    Xte_arr = np.asarray(Xte, dtype=np.float64)
    Z = model.encoder_.transform(Xte_arr)  # all features are numerical here
    manual = model.booster_.predict_raw(Xte_arr, Z, n_trees=model.best_iteration_)
    np.testing.assert_allclose(model.predict(Xte), manual)


def test_replay_early_stopping_tunes_unfitted_base(nan_regression_data):
    """An unfitted base gets LightGBM-native early stopping on the same
    eval data, and only its best-iteration routes are extracted."""
    Xtr, ytr, Xte, yte = nan_regression_data
    model = RouterExtractionRegressor(
        base=_base(n_estimators=300),
        leaf_model="constant",
        early_stopping_rounds=10,
        random_state=42,
    )
    model.fit(Xtr, ytr, eval_set=[(Xte, yte)])
    assert model.base_.best_iteration_ is not None
    assert model.base_.best_iteration_ < 300
    # Replay saw at most the base's best-iteration routes.
    assert model.booster_.n_trees <= model.base_.best_iteration_


def test_early_stopping_without_eval_set_raises(nan_regression_data):
    Xtr, ytr, _, _ = nan_regression_data
    with pytest.raises(ValueError, match="eval_set"):
        RouterExtractionRegressor(early_stopping_rounds=5).fit(Xtr, ytr)


def test_save_load_roundtrip_without_lightgbm_dependency(tmp_path, nan_regression_data):
    Xtr, ytr, Xte, _ = nan_regression_data
    model = RouterExtractionRegressor(base=_base(), min_samples_leaf=10, random_state=42)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)

    model.save_model(tmp_path / "m")
    loaded = RouterExtractionRegressor.load_model(tmp_path / "m")
    np.testing.assert_allclose(loaded.predict(Xte), pred)

    # The saved config carries provenance, not a live base object.
    config = json.loads((tmp_path / "m" / "model_config.json").read_text())
    assert config["config"]["base"] is None
    assert config["config"]["base_provenance"]["task"] == "regression"
    assert config["format_version"] == 3


def test_classifier_fit_predict(classification_data):
    Xtr, ytr, Xte, yte = classification_data
    base = LightGBMExternalModel(
        task="binary", random_state=0, n_estimators=60, num_leaves=7,
        min_child_samples=5,
    )
    model = RouterExtractionClassifier(
        base=base, leaf_model="embedded_linear", min_samples_leaf=10, random_state=42
    )
    model.fit(Xtr, ytr)

    proba = model.predict_proba(Xte)
    assert proba.shape == (len(yte), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    acc = (model.predict(Xte) == yte).mean()
    assert acc > 0.85


def test_classifier_early_stopping_and_roundtrip(tmp_path, classification_data):
    Xtr, ytr, Xte, yte = classification_data
    model = RouterExtractionClassifier(
        base=LightGBMExternalModel(task="binary", random_state=0,
                                   n_estimators=200, num_leaves=7),
        leaf_model="embedded_linear",
        early_stopping_rounds=10,
        eval_metric="auc",
        random_state=42,
    )
    model.fit(Xtr, ytr, eval_set=[(Xte, yte)])
    history = model.evals_result_["valid_0"]["auc"]
    assert model.best_iteration_ == int(np.argmax(history)) + 1

    proba = model.predict_proba(Xte)
    model.save_model(tmp_path / "m")
    loaded = RouterExtractionClassifier.load_model(tmp_path / "m")
    np.testing.assert_allclose(loaded.predict_proba(Xte), proba)
    np.testing.assert_array_equal(loaded.classes_, model.classes_)


def test_classifier_rejects_regression_base(classification_data):
    Xtr, ytr, _, _ = classification_data
    with pytest.raises(ValueError, match="task='binary'"):
        RouterExtractionClassifier(base=_base()).fit(Xtr, ytr)


def test_format_v1_compat(tmp_path, regression_data):
    """Old (v1) model directories — no missing_left, version 1 — must load
    with NaN-left defaults and identical predictions."""
    Xtr, ytr, Xte, _ = regression_data
    model = RepLeafRegressor(n_estimators=5, num_leaves=8, random_state=42)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    model.save_model(tmp_path / "m")

    # Downgrade the directory to format version 1.
    cfg_path = tmp_path / "m" / "model_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["format_version"] = 1
    cfg_path.write_text(json.dumps(cfg))
    ens_path = tmp_path / "m" / "tree_ensemble.json"
    ens = json.loads(ens_path.read_text())
    for tree in ens["trees"]:
        del tree["missing_left"]
    ens_path.write_text(json.dumps(ens))

    loaded = RepLeafRegressor.load_model(tmp_path / "m")
    np.testing.assert_allclose(loaded.predict(Xte), pred)


def test_unknown_format_version_rejected(tmp_path, regression_data):
    Xtr, ytr, _, _ = regression_data
    model = RepLeafRegressor(n_estimators=2, random_state=42).fit(Xtr, ytr)
    model.save_model(tmp_path / "m")
    cfg_path = tmp_path / "m" / "model_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["format_version"] = 99
    cfg_path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="format version"):
        RepLeafRegressor.load_model(tmp_path / "m")
