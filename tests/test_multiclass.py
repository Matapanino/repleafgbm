"""Multiclass classification (softmax, one tree per class per round)."""

import json

import numpy as np
import pytest

from repleafgbm import RepLeafClassifier
from repleafgbm.core.metrics import get_metric
from repleafgbm.core.objectives import MulticlassSoftmax, _softmax


def make_blobs(n: int, seed: int, n_classes: int = 3):
    """Well-separated Gaussian blobs, one per class."""
    rng = np.random.default_rng(seed)
    centers = 4.0 * np.column_stack(
        [np.cos(2 * np.pi * np.arange(n_classes) / n_classes),
         np.sin(2 * np.pi * np.arange(n_classes) / n_classes)]
    )
    y = rng.integers(0, n_classes, n)
    X = centers[y] + rng.normal(0.0, 0.8, (n, 2))
    return X, y


def test_multiclass_fit_predict():
    X_train, y_train = make_blobs(400, seed=0)
    X_test, y_test = make_blobs(200, seed=1)
    model = RepLeafClassifier(n_estimators=20, num_leaves=8, random_state=42)
    model.fit(X_train, y_train)

    assert model.n_classes_ == 3
    # n_estimators counts rounds: 3 trees per round.
    assert len(model.booster_.trees_) == 20 * 3
    pred = model.predict(X_test)
    assert set(np.unique(pred)) <= set(model.classes_)
    assert (pred == y_test).mean() > 0.9


def test_predict_proba_rows_sum_to_one():
    X, y = make_blobs(300, seed=0, n_classes=4)
    model = RepLeafClassifier(n_estimators=10, num_leaves=4, random_state=0)
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (300, 4)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert (proba >= 0).all()


def test_string_labels_roundtrip():
    X, y_int = make_blobs(300, seed=0)
    labels = np.array(["ant", "bee", "cat"], dtype=object)
    y = labels[y_int]
    model = RepLeafClassifier(n_estimators=15, num_leaves=8, random_state=0)
    model.fit(X, y)
    assert list(model.classes_) == ["ant", "bee", "cat"]
    pred = model.predict(X)
    assert set(np.unique(pred)) <= set(labels)
    assert (pred == y).mean() > 0.9


def test_save_load_roundtrip(tmp_path):
    X, y = make_blobs(300, seed=0)
    model = RepLeafClassifier(
        n_estimators=10, num_leaves=8, leaf_model="embedded_linear",
        encoder="identity", random_state=0,
    )
    model.fit(X, y)
    model.save_model(tmp_path / "m")
    loaded = RepLeafClassifier.load_model(tmp_path / "m")
    np.testing.assert_allclose(
        loaded.predict_proba(X), model.predict_proba(X), atol=1e-12
    )
    assert (loaded.predict(X) == model.predict(X)).all()


def test_multiclass_models_write_format_v5(tmp_path):
    X, y = make_blobs(200, seed=0)
    model = RepLeafClassifier(n_estimators=5, num_leaves=4, random_state=0)
    model.fit(X, y)
    model.save_model(tmp_path / "m")
    config = json.loads((tmp_path / "m" / "model_config.json").read_text())
    assert config["format_version"] == 5
    assert config["objective"] == "multiclass_softmax"
    ensemble = json.loads((tmp_path / "m" / "tree_ensemble.json").read_text())
    assert ensemble["n_classes"] == 3
    assert len(ensemble["init_score"]) == 3


def test_binary_models_keep_format_v3(tmp_path):
    X, y = make_blobs(200, seed=0)
    model = RepLeafClassifier(n_estimators=5, num_leaves=4, random_state=0)
    model.fit(X, (y == 0).astype(int))
    model.save_model(tmp_path / "m")
    config = json.loads((tmp_path / "m" / "model_config.json").read_text())
    assert config["format_version"] == 3


def test_early_stopping_multiclass():
    X_train, y_train = make_blobs(400, seed=0)
    X_valid, y_valid = make_blobs(200, seed=1)
    model = RepLeafClassifier(
        n_estimators=200, num_leaves=8, early_stopping_rounds=5, random_state=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
    history = model.evals_result_["valid_0"]["multi_logloss"]
    assert model.best_iteration_ is not None
    # best_iteration_ counts rounds; one history entry per round.
    assert model.best_iteration_ <= len(history)
    assert model.best_score_ == pytest.approx(min(history))
    # Prediction at the best round must beat the final round on the monitor.
    proba = model.predict_proba(X_valid)
    metric = get_metric("multi_logloss")
    codes = np.searchsorted(model.classes_, y_valid)
    assert metric(codes, proba) == pytest.approx(model.best_score_)


def test_accuracy_metric_multiclass():
    X_train, y_train = make_blobs(300, seed=0)
    X_valid, y_valid = make_blobs(150, seed=1)
    model = RepLeafClassifier(
        n_estimators=10, num_leaves=4, eval_metric="accuracy", random_state=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
    history = model.evals_result_["valid_0"]["accuracy"]
    assert len(history) == 10
    assert history[-1] > 0.85


def test_embedded_linear_multiclass():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 4))
    # Class boundaries with within-region linear structure.
    score = X[:, 0] + 0.5 * X[:, 1]
    y = np.digitize(score, [-0.7, 0.7])
    model = RepLeafClassifier(
        n_estimators=20, num_leaves=8, leaf_model="embedded_linear",
        encoder="identity", random_state=0,
    )
    model.fit(X, y)
    assert (model.predict(X) == y).mean() > 0.9


def test_single_class_rejected():
    X = np.zeros((50, 2))
    y = np.zeros(50)
    with pytest.raises(ValueError, match="at least 2 classes"):
        RepLeafClassifier(n_estimators=2).fit(X, y)


def test_eval_set_unknown_label_rejected():
    X, y = make_blobs(200, seed=0)
    model = RepLeafClassifier(n_estimators=2, num_leaves=4)
    Xv, yv = make_blobs(50, seed=1, n_classes=4)
    with pytest.raises(ValueError, match="not seen in training"):
        model.fit(X, y, eval_set=[(Xv, yv)])


def test_summary_mentions_rounds(tmp_path):
    X, y = make_blobs(200, seed=0)
    model = RepLeafClassifier(n_estimators=5, num_leaves=4, random_state=0)
    model.fit(X, y)
    assert "5 rounds x 3 classes" in model.summary()


def test_softmax_objective_statistics():
    obj = MulticlassSoftmax(3)
    y = np.array([0, 1, 2, 1])
    raw = np.array([[2.0, 0.0, -1.0]] * 4)
    grad, hess = obj.grad_hess(y, raw)
    p = _softmax(raw)
    # g = p - onehot, h = p(1-p); rows of g sum to 0 by construction.
    np.testing.assert_allclose(grad.sum(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(grad[0], p[0] - np.array([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(hess, p * (1 - p))
    with pytest.raises(ValueError, match="n_classes >= 3"):
        MulticlassSoftmax(2)
