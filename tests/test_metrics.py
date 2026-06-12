"""Metric correctness tests (cross-checked against scikit-learn)."""

import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    mean_absolute_error,
    roc_auc_score,
)

from repleafgbm.core.metrics import get_metric, make_metric


@pytest.fixture
def scores():
    rng = np.random.default_rng(0)
    y = (rng.random(200) > 0.4).astype(float)
    p = np.clip(0.6 * y + 0.4 * rng.random(200), 1e-6, 1 - 1e-6)
    return y, p


def test_mae_matches_sklearn():
    rng = np.random.default_rng(1)
    y, pred = rng.normal(size=100), rng.normal(size=100)
    assert get_metric("mae")(y, pred) == pytest.approx(mean_absolute_error(y, pred))


def test_logloss_matches_sklearn(scores):
    y, p = scores
    assert get_metric("logloss")(y, p) == pytest.approx(log_loss(y, p))


def test_auc_matches_sklearn(scores):
    y, p = scores
    assert get_metric("auc")(y, p) == pytest.approx(roc_auc_score(y, p))


def test_auc_with_ties_matches_sklearn():
    y = np.array([0, 0, 1, 1, 0, 1, 1, 0], dtype=float)
    p = np.array([0.1, 0.5, 0.5, 0.5, 0.2, 0.9, 0.2, 0.9])  # heavy ties
    assert get_metric("auc")(y, p) == pytest.approx(roc_auc_score(y, p))


def test_auc_single_class_raises():
    with pytest.raises(ValueError, match="single class"):
        get_metric("auc")(np.ones(5), np.linspace(0, 1, 5))


def test_accuracy_matches_sklearn(scores):
    y, p = scores
    assert get_metric("accuracy")(y, p) == pytest.approx(
        accuracy_score(y, (p >= 0.5).astype(float))
    )


def test_metric_directions():
    assert get_metric("rmse").minimize and get_metric("mae").minimize
    assert get_metric("logloss").minimize
    assert not get_metric("auc").minimize and not get_metric("accuracy").minimize


def test_unknown_metric():
    with pytest.raises(ValueError, match="Unknown metric"):
        get_metric("nope")


# --------------------------------------------------------------------- #
# User-supplied callable metrics
# --------------------------------------------------------------------- #
def median_abs_error(y_true, y_pred):
    return float(np.median(np.abs(y_true - y_pred)))


def test_make_metric_wraps_callable():
    rng = np.random.default_rng(2)
    y, pred = rng.normal(size=50), rng.normal(size=50)
    metric = make_metric(median_abs_error)
    assert metric.name == "median_abs_error"
    assert metric.minimize
    assert metric(y, pred) == pytest.approx(np.median(np.abs(y - pred)))


def test_make_metric_name_and_direction():
    metric = make_metric(lambda y, p: 1.0, name="constant_one", minimize=False)
    assert metric.name == "constant_one" and not metric.minimize


def test_make_metric_rejects_non_callable():
    with pytest.raises(TypeError, match="callable"):
        make_metric("rmse")


def test_estimator_accepts_callable_eval_metric(regression_data):
    from repleafgbm import RepLeafRegressor

    Xtr, ytr, Xte, yte = regression_data
    model = RepLeafRegressor(
        n_estimators=8,
        num_leaves=8,
        eval_metric=median_abs_error,
        early_stopping_rounds=3,
        random_state=42,
    )
    model.fit(Xtr, ytr, eval_set=[(Xte, yte)])
    history = model.evals_result_["valid_0"]["median_abs_error"]
    assert len(history) >= 1
    pred = model.predict(Xte)
    assert history[-1] >= 0 and np.isfinite(pred).all()


def test_estimator_accepts_metric_instance_and_maximize(regression_data):
    from repleafgbm import RepLeafRegressor

    Xtr, ytr, Xte, yte = regression_data
    neg_rmse = make_metric(
        lambda y, p: -float(np.sqrt(np.mean((y - p) ** 2))),
        name="neg_rmse",
        minimize=False,
    )
    model = RepLeafRegressor(
        n_estimators=8,
        num_leaves=8,
        eval_metric=neg_rmse,
        early_stopping_rounds=3,
        random_state=42,
    )
    model.fit(Xtr, ytr, eval_set=[(Xte, yte)])
    # The maximize direction is honored: best_score_ is the running maximum.
    history = model.evals_result_["valid_0"]["neg_rmse"]
    if model.best_score_ is not None:
        assert model.best_score_ == pytest.approx(max(history))


def test_callable_metric_model_saves_json_safe(tmp_path, regression_data):
    from repleafgbm import RepLeafRegressor

    Xtr, ytr, _, _ = regression_data
    model = RepLeafRegressor(
        n_estimators=3, num_leaves=8, eval_metric=median_abs_error, random_state=42
    ).fit(Xtr, ytr)
    model.save_model(tmp_path / "model")  # must not choke on the callable
    loaded = RepLeafRegressor.load_model(tmp_path / "model")
    assert loaded.eval_metric == "median_abs_error"
