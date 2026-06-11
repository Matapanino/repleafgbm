"""Early stopping behavior tests."""

import numpy as np
import pytest

from repleafgbm import RepLeafClassifier, RepLeafDataset, RepLeafRegressor


@pytest.fixture
def noisy_split():
    """Small signal, lots of noise: validation error turns upward early."""
    rng = np.random.default_rng(3)
    n = 600
    X = rng.normal(size=(n, 5))
    y = X[:, 0] + 0.5 * X[:, 1] + rng.normal(0.0, 2.0, n)
    return X[:400], y[:400], X[400:], y[400:]


def test_early_stopping_stops_and_sets_best_iteration(noisy_split):
    Xtr, ytr, Xva, yva = noisy_split
    model = RepLeafRegressor(
        n_estimators=300,
        learning_rate=0.3,
        num_leaves=16,
        min_samples_leaf=5,
        leaf_model="constant",
        early_stopping_rounds=10,
        random_state=42,
    )
    train = RepLeafDataset(Xtr, ytr)
    model.fit(train, eval_set=[RepLeafDataset(Xva, yva, metadata=train.metadata)])

    history = model.evals_result_["valid_0"]["rmse"]
    n_grown = model.booster_.n_trees
    assert n_grown < 300  # stopped early on this noisy problem
    assert len(history) == n_grown
    assert model.best_iteration_ == int(np.argmin(history)) + 1
    assert model.best_score_ == pytest.approx(min(history))
    # The monitored metric did not improve in the last `early_stopping_rounds`.
    assert n_grown - model.best_iteration_ == 10


def test_predict_uses_best_iteration(noisy_split):
    Xtr, ytr, Xva, yva = noisy_split
    model = RepLeafRegressor(
        n_estimators=300,
        learning_rate=0.3,
        num_leaves=16,
        min_samples_leaf=5,
        leaf_model="constant",
        early_stopping_rounds=10,
        random_state=42,
    )
    train = RepLeafDataset(Xtr, ytr)
    model.fit(train, eval_set=[RepLeafDataset(Xva, yva, metadata=train.metadata)])

    pred = model.predict(Xva)
    manual = model.booster_.predict_raw(
        np.asarray(Xva, dtype=np.float64), None, n_trees=model.best_iteration_
    )
    np.testing.assert_allclose(pred, manual)
    # And it differs from the full (overfit) ensemble.
    full = model.booster_.predict_raw(
        np.asarray(Xva, dtype=np.float64), None, n_trees=model.booster_.n_trees
    )
    assert not np.allclose(pred, full)


def test_early_stopping_roundtrip(tmp_path, noisy_split):
    Xtr, ytr, Xva, yva = noisy_split
    model = RepLeafRegressor(
        n_estimators=300,
        learning_rate=0.3,
        num_leaves=16,
        min_samples_leaf=5,
        leaf_model="constant",
        early_stopping_rounds=10,
        random_state=42,
    )
    train = RepLeafDataset(Xtr, ytr)
    model.fit(train, eval_set=[RepLeafDataset(Xva, yva, metadata=train.metadata)])

    model.save_model(tmp_path / "m")
    loaded = RepLeafRegressor.load_model(tmp_path / "m")
    assert loaded.best_iteration_ == model.best_iteration_
    np.testing.assert_allclose(loaded.predict(Xva), model.predict(Xva))


def test_early_stopping_without_eval_set_raises(noisy_split):
    Xtr, ytr, _, _ = noisy_split
    model = RepLeafRegressor(early_stopping_rounds=5)
    with pytest.raises(ValueError, match="eval_set"):
        model.fit(Xtr, ytr)


def test_classifier_early_stopping_with_auc(classification_data):
    Xtr, ytr, Xva, yva = classification_data
    model = RepLeafClassifier(
        n_estimators=200,
        learning_rate=0.3,
        num_leaves=8,
        early_stopping_rounds=10,
        eval_metric="auc",
        random_state=42,
    )
    train = RepLeafDataset(Xtr, ytr)
    model.fit(train, eval_set=[RepLeafDataset(Xva, yva, metadata=train.metadata)])

    history = model.evals_result_["valid_0"]["auc"]
    # AUC is maximized: best_iteration_ tracks the argmax.
    assert model.best_iteration_ == int(np.argmax(history)) + 1
    assert model.best_score_ == pytest.approx(max(history))
    assert model.best_score_ > 0.9


def test_no_early_stopping_keeps_all_trees(regression_data):
    Xtr, ytr, Xva, yva = regression_data
    model = RepLeafRegressor(n_estimators=15, num_leaves=8, random_state=42)
    train = RepLeafDataset(Xtr, ytr)
    model.fit(train, eval_set=[RepLeafDataset(Xva, yva, metadata=train.metadata)])
    assert model.booster_.n_trees == 15
    assert model.best_iteration_ is None


def test_eval_metric_override_recorded(regression_data):
    Xtr, ytr, Xva, yva = regression_data
    model = RepLeafRegressor(n_estimators=10, num_leaves=8, eval_metric="mae", random_state=42)
    train = RepLeafDataset(Xtr, ytr)
    model.fit(train, eval_set=[RepLeafDataset(Xva, yva, metadata=train.metadata)])
    assert "mae" in model.evals_result_["valid_0"]
