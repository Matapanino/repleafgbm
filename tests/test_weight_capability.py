"""Capability layer: models that cannot reweight rows warn and fall back.

The native estimators always support weights; this covers the
``_supports_sample_weight = False`` path (frozen-route replay and any future
estimator that cannot honor weights) — it must warn and drop the weights, not
raise, and produce the same model as an unweighted fit.
"""

import numpy as np
import pytest

from repleafgbm import RepLeafClassifier, RepLeafRegressor, get_metric


class _NoWeightRegressor(RepLeafRegressor):
    _supports_sample_weight = False


class _NoWeightClassifier(RepLeafClassifier):
    _supports_sample_weight = False


def _reg_data(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(200, 3))
    y = X[:, 0] + rng.normal(0, 0.1, 200)
    return X, y


def _clf_data(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(200, 3))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    return X, y


def test_get_metric_balanced_accuracy_exported():
    metric = get_metric("balanced_accuracy")
    assert metric.name == "balanced_accuracy"
    assert metric.minimize is False


def test_unsupported_regressor_warns_and_falls_back():
    X, y = _reg_data()
    plain = RepLeafRegressor(n_estimators=15, leaf_model="constant", random_state=0).fit(
        X, y
    )
    model = _NoWeightRegressor(n_estimators=15, leaf_model="constant", random_state=0)
    with pytest.warns(UserWarning, match="cannot apply sample_weight"):
        model.fit(X, y, sample_weight=np.where(y > 0, 5.0, 1.0))
    assert np.allclose(model.predict(X), plain.predict(X))


def test_unsupported_classifier_warns_on_class_weight():
    X, y = _clf_data()
    plain = RepLeafClassifier(n_estimators=15, num_leaves=8, random_state=0).fit(X, y)
    model = _NoWeightClassifier(
        n_estimators=15, num_leaves=8, class_weight="balanced", random_state=0
    )
    with pytest.warns(UserWarning, match="cannot apply sample_weight"):
        model.fit(X, y)
    # class_weight dropped -> identical to the unweighted fit.
    assert np.allclose(model.predict_proba(X), plain.predict_proba(X))


def test_supported_model_does_not_warn():
    X, y = _clf_data()
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any UserWarning would fail the test
        RepLeafClassifier(n_estimators=10, num_leaves=8, random_state=0).fit(
            X, y, sample_weight=np.ones(len(y))
        )


def test_router_extraction_warns_but_fits():
    lgb = pytest.importorskip("lightgbm")  # noqa: F841
    from repleafgbm.external.router_extraction import RouterExtractionClassifier

    X, y = _clf_data()
    model = RouterExtractionClassifier(random_state=0)
    with pytest.warns(UserWarning, match="cannot apply sample_weight"):
        model.fit(X, y, sample_weight=np.where(y == 1, 3.0, 1.0))
    assert model.predict(X).shape == (len(y),)
