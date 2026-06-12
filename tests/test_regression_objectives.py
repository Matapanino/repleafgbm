"""Regression objectives beyond squared error: huber, quantile, poisson."""

import json

import numpy as np
import pytest

from repleafgbm import Huber, PoissonRegression, Quantile, RepLeafRegressor
from repleafgbm.core.objectives import get_objective


def make_linear_data(n: int, seed: int, noise: float = 0.3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = 2.0 * X[:, 0] - X[:, 1] + rng.normal(0.0, noise, n)
    return X, y


# --------------------------------------------------------------------- #
# Objective statistics (unit level)
# --------------------------------------------------------------------- #
def test_huber_gradient_is_clipped_residual():
    obj = Huber(delta=1.5)
    y = np.array([0.0, 0.0, 0.0])
    raw = np.array([0.5, 3.0, -3.0])
    grad, hess = obj.grad_hess(y, raw)
    np.testing.assert_allclose(grad, [0.5, 1.5, -1.5])
    np.testing.assert_allclose(hess, 1.0)
    assert obj.init_score(np.array([1.0, 2.0, 100.0])) == 2.0  # median
    with pytest.raises(ValueError, match="delta must be positive"):
        Huber(delta=0.0)


def test_quantile_gradient_signs():
    obj = Quantile(alpha=0.9)
    y = np.array([0.0, 0.0])
    raw = np.array([1.0, -1.0])  # over- and under-prediction
    grad, hess = obj.grad_hess(y, raw)
    np.testing.assert_allclose(grad, [0.1, -0.9])
    np.testing.assert_allclose(hess, 1.0)
    assert obj.init_score(np.arange(101.0)) == pytest.approx(90.0)
    with pytest.raises(ValueError, match="alpha must be in"):
        Quantile(alpha=1.0)


def test_poisson_statistics_and_validation():
    obj = PoissonRegression()
    y = np.array([2.0, 0.0])
    raw = np.array([0.0, 1.0])
    grad, hess = obj.grad_hess(y, raw)
    np.testing.assert_allclose(grad, [1.0 - 2.0, np.e])
    np.testing.assert_allclose(hess, [1.0, np.e])
    np.testing.assert_allclose(obj.transform(raw), [1.0, np.e])
    with pytest.raises(ValueError, match="non-negative"):
        obj.init_score(np.array([-1.0, 2.0]))
    with pytest.raises(ValueError, match="positive target mean"):
        obj.init_score(np.zeros(5))


def test_objective_registry():
    for name in ("huber", "quantile", "poisson"):
        assert get_objective(name).name == name


# --------------------------------------------------------------------- #
# Estimator behavior
# --------------------------------------------------------------------- #
def test_huber_resists_outliers():
    X, y = make_linear_data(600, seed=0, noise=0.1)
    rng = np.random.default_rng(1)
    corrupt = rng.choice(600, size=30, replace=False)
    y_corrupt = y.copy()
    y_corrupt[corrupt] += rng.choice([-50.0, 50.0], size=30)
    X_test, y_test = make_linear_data(300, seed=2, noise=0.1)

    common = dict(n_estimators=60, num_leaves=8, random_state=0)
    rmse = {}
    for objective in (None, "huber"):
        model = RepLeafRegressor(objective=objective, **common)
        model.fit(X, y_corrupt)
        pred = model.predict(X_test)
        rmse[objective] = float(np.sqrt(np.mean((pred - y_test) ** 2)))
    # On clean test data the huber fit must beat the outlier-dragged L2 fit.
    assert rmse["huber"] < rmse[None]


def test_quantile_models_order_and_coverage():
    X, y = make_linear_data(800, seed=0, noise=1.0)
    X_test, y_test = make_linear_data(400, seed=1, noise=1.0)
    common = dict(n_estimators=60, num_leaves=8, random_state=0)
    lo = RepLeafRegressor(objective=Quantile(alpha=0.1), **common).fit(X, y)
    hi = RepLeafRegressor(objective=Quantile(alpha=0.9), **common).fit(X, y)
    pred_lo, pred_hi = lo.predict(X_test), hi.predict(X_test)
    assert (pred_hi > pred_lo).mean() > 0.95
    cover_lo = (y_test >= pred_lo).mean()
    cover_hi = (y_test <= pred_hi).mean()
    assert 0.8 < cover_lo
    assert 0.8 < cover_hi


def test_poisson_fit_predicts_positive_counts():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(700, 3))
    lam = np.exp(0.8 * X[:, 0] + 0.3 * X[:, 1])
    y = rng.poisson(lam).astype(np.float64)
    model = RepLeafRegressor(objective="poisson", n_estimators=60,
                             num_leaves=8, random_state=0)
    model.fit(X, y)
    pred = model.predict(X)
    assert (pred > 0).all()  # exp transform guarantees positivity
    # Must beat the constant-mean baseline comfortably.
    baseline = np.sqrt(np.mean((y - y.mean()) ** 2))
    assert np.sqrt(np.mean((y - pred) ** 2)) < 0.8 * baseline


def test_objective_instance_save_load_roundtrip(tmp_path):
    X, y = make_linear_data(300, seed=0)
    model = RepLeafRegressor(objective=Huber(delta=2.0), n_estimators=10,
                             num_leaves=4, random_state=0)
    model.fit(X, y)
    model.save_model(tmp_path / "m")
    config = json.loads((tmp_path / "m" / "model_config.json").read_text())
    assert config["objective"] == "huber"
    assert config["config"]["objective"] == "huber"  # instance saved by name
    loaded = RepLeafRegressor.load_model(tmp_path / "m")
    np.testing.assert_allclose(loaded.predict(X), model.predict(X), atol=1e-12)


def test_poisson_save_load_keeps_transform(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 2))
    y = rng.poisson(np.exp(0.5 * X[:, 0])).astype(np.float64)
    model = RepLeafRegressor(objective="poisson", n_estimators=10,
                             num_leaves=4, random_state=0)
    model.fit(X, y)
    model.save_model(tmp_path / "m")
    loaded = RepLeafRegressor.load_model(tmp_path / "m")
    np.testing.assert_allclose(loaded.predict(X), model.predict(X), atol=1e-12)
    assert (loaded.predict(X) > 0).all()


def test_classifier_rejects_objective_parameter():
    X = np.random.default_rng(0).normal(size=(60, 2))
    y = (X[:, 0] > 0).astype(int)
    from repleafgbm import RepLeafClassifier

    with pytest.raises(ValueError, match="regression only|RepLeafRegressor only"):
        RepLeafClassifier(objective="huber", n_estimators=2).fit(X, y)


def test_unknown_objective_name_rejected():
    X, y = make_linear_data(100, seed=0)
    with pytest.raises(ValueError, match="Unknown objective"):
        RepLeafRegressor(objective="absolute_error", n_estimators=2).fit(X, y)
