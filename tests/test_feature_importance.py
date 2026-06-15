"""Feature importance tests (gain / split count)."""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from repleafgbm import RepLeafDataset, RepLeafRegressor


@pytest.fixture
def fitted(regression_data):
    Xtr, ytr, _, _ = regression_data
    model = RepLeafRegressor(
        n_estimators=20, num_leaves=8, min_samples_leaf=10, random_state=42
    )
    model.fit(Xtr, ytr)
    return model


def test_importances_shape_and_normalization(fitted):
    imp = fitted.feature_importances_
    assert imp.shape == (fitted.n_features_in_,)
    assert imp.min() >= 0.0
    assert imp.sum() == pytest.approx(1.0)


def test_informative_features_dominate(fitted):
    """regression_data signal lives in f0 (regime), f1, f2; f3 is noise."""
    imp = fitted.feature_importances_
    assert imp[3] < min(imp[0], imp[1], imp[2])
    assert imp[[0, 1, 2]].sum() > 0.9


def test_split_importance_counts_internal_nodes(fitted):
    splits = fitted.get_feature_importance("split")
    np.testing.assert_allclose(splits, np.round(splits))  # integer counts
    total_internal = sum(
        int((t.feature >= 0).sum()) for t in fitted.booster_.trees_
    )
    assert splits.sum() == total_internal


def test_invalid_importance_type(fitted):
    with pytest.raises(ValueError, match="importance_type"):
        fitted.get_feature_importance("cover")


def test_unfitted_raises():
    with pytest.raises(NotFittedError, match="not fitted"):
        RepLeafRegressor().feature_importances_  # noqa: B018


def test_importance_respects_best_iteration(regression_data):
    Xtr, ytr, Xte, yte = regression_data
    model = RepLeafRegressor(
        n_estimators=200, learning_rate=0.3, num_leaves=8, min_samples_leaf=10,
        early_stopping_rounds=5, random_state=42,
    )
    train = RepLeafDataset(Xtr, ytr)
    model.fit(train, eval_set=[RepLeafDataset(Xte, yte, metadata=train.metadata)])
    assert model.best_iteration_ is not None

    splits = model.get_feature_importance("split")
    manual = sum(
        int((t.feature >= 0).sum())
        for t in model.booster_.trees_[: model.best_iteration_]
    )
    assert splits.sum() == manual


def test_importance_roundtrip(tmp_path, fitted, regression_data):
    fitted.save_model(tmp_path / "m")
    loaded = RepLeafRegressor.load_model(tmp_path / "m")
    np.testing.assert_allclose(
        loaded.feature_importances_, fitted.feature_importances_
    )


def test_router_extraction_importance(regression_data):
    lgb = pytest.importorskip("lightgbm")  # noqa: F841
    from repleafgbm.external import LightGBMExternalModel, RouterExtractionRegressor

    Xtr, ytr, _, _ = regression_data
    model = RouterExtractionRegressor(
        base=LightGBMExternalModel(task="regression", random_state=0,
                                   n_estimators=30, num_leaves=7,
                                   min_child_samples=5),
        min_samples_leaf=10, random_state=42,
    )
    model.fit(Xtr, ytr)
    imp = model.feature_importances_  # gains imported from LightGBM's dump
    assert imp.sum() == pytest.approx(1.0)
    assert imp[3] < imp[:3].max()
