"""sklearn API compatibility and dataset-metadata guard tests."""

import numpy as np
import pandas as pd
import pytest
import sklearn
from sklearn.base import clone
from sklearn.utils.estimator_checks import parametrize_with_checks

from repleafgbm import RepLeafClassifier, RepLeafDataset, RepLeafRegressor

# The check_estimator battery's contents change between scikit-learn releases.
# The modern (dataclass-tags) contract stabilized in 1.6, so the full battery
# is exercised there and later; older installs (the library still supports
# >=1.2) skip it but keep the hand-written compatibility tests below.
_SKLEARN_VERSION = tuple(int(p) for p in sklearn.__version__.split(".")[:2])
_HAS_MODERN_CHECKS = _SKLEARN_VERSION >= (1, 6)

# Tiny-data-friendly config: check_estimator validates behavior on very small
# synthetic datasets, so min_samples_leaf must be small enough to allow splits;
# the production defaults are deliberately more conservative.
_CHECK_CONFIG = dict(
    n_estimators=30, num_leaves=8, learning_rate=0.3, min_samples_leaf=2, random_state=0
)


@pytest.mark.skipif(
    not _HAS_MODERN_CHECKS,
    reason="check_estimator battery targets scikit-learn >= 1.6",
)
@parametrize_with_checks(
    [RepLeafRegressor(**_CHECK_CONFIG), RepLeafClassifier(**_CHECK_CONFIG)]
)
def test_sklearn_check_estimator(estimator, check):
    """Full scikit-learn estimator-compliance battery (Phase 24, v1.0).

    NaN is a supported feature value (routes left), so the estimators set the
    ``allow_nan`` tag and the finiteness checks only require inf-rejection.
    """
    check(estimator)


def test_clone_and_set_params(regression_data):
    Xtr, ytr, Xte, _ = regression_data
    model = RepLeafRegressor(n_estimators=5, num_leaves=4, random_state=42)
    model.fit(Xtr, ytr)

    cloned = clone(model)
    assert cloned.get_params() == model.get_params()
    cloned.set_params(n_estimators=3).fit(Xtr, ytr)
    assert cloned.booster_.n_trees == 3
    # Original is untouched and parameters round-trip through get/set.
    assert model.booster_.n_trees == 5


def test_score_methods(regression_data, classification_data):
    Xtr, ytr, Xte, yte = regression_data
    reg = RepLeafRegressor(n_estimators=15, num_leaves=8, random_state=42).fit(Xtr, ytr)
    assert reg.score(Xte, yte) > 0.5  # R^2 via RegressorMixin

    Xc, yc, Xce, yce = classification_data
    clf = RepLeafClassifier(n_estimators=15, num_leaves=8, random_state=42).fit(Xc, yc)
    assert clf.score(Xce, yce) > 0.8  # accuracy via ClassifierMixin


def test_fitted_attributes(regression_data):
    Xtr, ytr, _, _ = regression_data
    model = RepLeafRegressor(n_estimators=3, random_state=42).fit(Xtr, ytr)
    assert model.n_features_in_ == Xtr.shape[1]
    assert list(model.feature_names_in_) == ["f0", "f1", "f2", "f3"]


def _categorical_frames():
    rng = np.random.default_rng(0)
    df_tr = pd.DataFrame(
        {"c": rng.choice(["a", "b", "c"], size=120), "x": rng.normal(size=120)}
    )
    y_tr = rng.normal(size=120)
    # Validation sample is missing category "a": independently inferred
    # metadata would assign different ordinal codes.
    df_va = pd.DataFrame({"c": rng.choice(["b", "c"], size=40), "x": rng.normal(size=40)})
    y_va = rng.normal(size=40)
    return df_tr, y_tr, df_va, y_va


def test_eval_set_metadata_mismatch_rejected():
    df_tr, y_tr, df_va, y_va = _categorical_frames()
    train = RepLeafDataset(df_tr, y_tr, categorical_features=["c"])
    valid_bad = RepLeafDataset(df_va, y_va, categorical_features=["c"])  # own metadata
    model = RepLeafRegressor(n_estimators=3, random_state=42)
    with pytest.raises(ValueError, match="metadata"):
        model.fit(train, eval_set=[valid_bad])

    # Sharing the training metadata works.
    valid_ok = RepLeafDataset(df_va, y_va, metadata=train.metadata)
    model.fit(train, eval_set=[valid_ok])
    assert "valid_0" in model.evals_result_


def test_predict_dataset_metadata_mismatch_rejected():
    df_tr, y_tr, df_va, _ = _categorical_frames()
    train = RepLeafDataset(df_tr, y_tr, categorical_features=["c"])
    model = RepLeafRegressor(n_estimators=3, random_state=42).fit(train)

    bad = RepLeafDataset(df_va, categorical_features=["c"])
    with pytest.raises(ValueError, match="metadata"):
        model.predict(bad)

    ok = RepLeafDataset(df_va, metadata=train.metadata)
    assert model.predict(ok).shape == (len(df_va),)
    # Plain DataFrames are re-encoded with training metadata automatically.
    assert model.predict(df_va).shape == (len(df_va),)


def test_numeric_ndarray_eval_set_needs_no_explicit_metadata(regression_data):
    """For numerical-only data, independently built datasets have identical
    metadata, so the guard imposes no friction."""
    Xtr, ytr, Xte, yte = regression_data
    model = RepLeafRegressor(n_estimators=3, random_state=42)
    model.fit(RepLeafDataset(Xtr, ytr), eval_set=[RepLeafDataset(Xte, yte)])
    assert "valid_0" in model.evals_result_
