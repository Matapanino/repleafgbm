"""Tests for the CatBoost external_model backend (catboost optional)."""

import numpy as np
import pytest

from repleafgbm import RepLeafDataset, RepLeafRegressor

# Importing the package must work without catboost; only usage requires it.
from repleafgbm.external import (
    CatBoostExternalModel,
    augment_features,
    oof_predictions,
)

catboost = pytest.importorskip("catboost", reason="catboost not installed")


def _fast_base(task="regression"):
    return CatBoostExternalModel(
        task=task, random_state=0, iterations=30, depth=3
    )


def test_fit_predict_score_and_leaves(regression_data):
    Xtr, ytr, Xte, yte = regression_data
    base = _fast_base().fit(Xtr, ytr)

    score = base.predict_score(Xte)
    assert score.shape == (len(yte),)
    baseline = np.sqrt(np.mean((yte - ytr.mean()) ** 2))
    assert np.sqrt(np.mean((yte - score) ** 2)) < baseline

    leaves = base.predict_leaf_indices(Xte)
    assert leaves.shape == (len(yte), base.n_trees_)
    assert leaves.dtype == np.int32
    assert leaves.min() >= 0


def test_binary_task(classification_data):
    Xtr, ytr, Xte, yte = classification_data
    base = _fast_base(task="binary").fit(Xtr, ytr)
    p = base.predict_score(Xte)
    assert ((p >= 0) & (p <= 1)).all()
    assert ((p >= 0.5).astype(int) == yte).mean() > 0.8


def test_repleafdataset_input(regression_data):
    Xtr, ytr, Xte, _ = regression_data
    ds = RepLeafDataset(Xtr, ytr)
    base = _fast_base().fit(ds)
    pred_ds = base.predict_score(RepLeafDataset(Xte, metadata=ds.metadata))
    pred_arr = base.predict_score(Xte)
    np.testing.assert_allclose(pred_ds, pred_arr)


def test_early_stopping_limits_predictions(regression_data):
    Xtr, ytr, Xte, yte = regression_data
    base = CatBoostExternalModel(
        task="regression", random_state=0, iterations=300, depth=3
    )
    base.fit(Xtr, ytr, eval_set=[(Xte, yte)], early_stopping_rounds=5)
    assert base.best_iteration_ is not None
    assert 1 <= base.best_iteration_ <= 300
    leaves = base.predict_leaf_indices(Xte)
    assert leaves.shape[1] == base.best_iteration_
    with pytest.raises(ValueError, match="requires eval_set"):
        _fast_base().fit(Xtr, ytr, early_stopping_rounds=5)


def test_unfitted_and_bad_task():
    with pytest.raises(RuntimeError, match="not fitted"):
        _fast_base().predict_score(np.zeros((2, 4)))
    with pytest.raises(ValueError, match="task"):
        CatBoostExternalModel(task="multiclass")


def test_oof_and_stacked_pipeline(regression_data):
    """OOF base scores -> augmented dataset -> RepLeafRegressor."""
    Xtr, ytr, Xte, yte = regression_data
    oof, models = oof_predictions(_fast_base, Xtr, ytr, n_splits=3, random_state=0)
    assert not np.isnan(oof).any()
    assert len(models) == 3

    base_full = _fast_base().fit(Xtr, ytr)
    df_tr, cats = augment_features(Xtr, base_full, score=oof, n_leaf_features=2)
    df_te, _ = augment_features(Xte, base_full, n_leaf_features=2)

    train = RepLeafDataset(df_tr, ytr, categorical_features=cats)
    model = RepLeafRegressor(n_estimators=20, num_leaves=8, random_state=42)
    model.fit(train)
    pred = model.predict(RepLeafDataset(df_te, metadata=train.metadata))

    baseline = np.sqrt(np.mean((yte - ytr.mean()) ** 2))
    assert np.sqrt(np.mean((yte - pred) ** 2)) < 0.5 * baseline


def test_missing_catboost_message(monkeypatch, regression_data):
    """Without catboost the error must say how to install it."""
    import sys

    Xtr, ytr, _, _ = regression_data
    monkeypatch.setitem(sys.modules, "catboost", None)
    with pytest.raises(ImportError, match="pip install"):
        CatBoostExternalModel().fit(Xtr, ytr)
