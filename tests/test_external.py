"""Tests for the external_model integration (LightGBM optional)."""

import numpy as np
import pytest

from repleafgbm import RepLeafDataset, RepLeafRegressor

# Importing the package must work without lightgbm; only usage requires it.
from repleafgbm.external import (
    LightGBMExternalModel,
    augment_features,
    external_feature_frame,
    oof_predictions,
)

lgb = pytest.importorskip("lightgbm", reason="lightgbm not installed")


@pytest.fixture
def reg_data(regression_data):
    Xtr, ytr, Xte, yte = regression_data
    return Xtr, ytr, Xte, yte


def _fast_base(task="regression"):
    return LightGBMExternalModel(
        task=task, random_state=0, n_estimators=30, num_leaves=7, min_child_samples=5
    )


def test_fit_predict_score_and_leaves(reg_data):
    Xtr, ytr, Xte, yte = reg_data
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


def test_repleafdataset_input(reg_data):
    Xtr, ytr, Xte, _ = reg_data
    ds = RepLeafDataset(Xtr, ytr)
    base = _fast_base().fit(ds)
    pred_ds = base.predict_score(RepLeafDataset(Xte, metadata=ds.metadata))
    pred_arr = base.predict_score(Xte)
    np.testing.assert_allclose(pred_ds, pred_arr)


def test_unfitted_and_bad_task():
    with pytest.raises(RuntimeError, match="not fitted"):
        _fast_base().predict_score(np.zeros((2, 4)))
    with pytest.raises(ValueError, match="task"):
        LightGBMExternalModel(task="multiclass")


def test_oof_predictions_no_leakage_shape(reg_data):
    Xtr, ytr, _, _ = reg_data
    oof, models = oof_predictions(_fast_base, Xtr, ytr, n_splits=3, random_state=0)
    assert oof.shape == ytr.shape
    assert not np.isnan(oof).any()  # every row predicted exactly once
    assert len(models) == 3
    # OOF predictions carry signal but are honest (worse than in-sample).
    in_sample = _fast_base().fit(Xtr, ytr).predict_score(Xtr)
    rmse_oof = np.sqrt(np.mean((ytr - oof) ** 2))
    rmse_in = np.sqrt(np.mean((ytr - in_sample) ** 2))
    assert rmse_in < rmse_oof < np.std(ytr)


def test_oof_stratified(classification_data):
    Xtr, ytr, _, _ = classification_data
    oof, _ = oof_predictions(
        lambda: _fast_base("binary"), Xtr, ytr, n_splits=3, stratify=True
    )
    assert ((oof >= 0) & (oof <= 1)).all()


def test_feature_frames(reg_data):
    Xtr, ytr, Xte, _ = reg_data
    base = _fast_base().fit(Xtr, ytr)

    frame, cats = external_feature_frame(base, Xte, n_leaf_features=3, prefix="lgb")
    assert list(frame.columns) == ["lgb_score", "lgb_leaf_0", "lgb_leaf_1", "lgb_leaf_2"]
    assert cats == ["lgb_leaf_0", "lgb_leaf_1", "lgb_leaf_2"]

    aug, cats = augment_features(Xte, base, n_leaf_features=2)
    assert aug.shape == (len(Xte), Xte.shape[1] + 3)
    assert list(aug.columns[:2]) == ["f0", "f1"]

    # Precomputed (e.g. OOF) scores override the in-sample column.
    custom = np.zeros(len(Xte))
    frame, _ = external_feature_frame(base, Xte, score=custom)
    assert (frame["ext_score"] == 0).all()

    with pytest.raises(ValueError, match="Nothing to emit"):
        external_feature_frame(base, Xte, include_score=False)


def test_stacked_repleaf_pipeline(reg_data):
    """End-to-end: OOF base scores -> augmented dataset -> RepLeafRegressor."""
    Xtr, ytr, Xte, yte = reg_data
    oof, _ = oof_predictions(_fast_base, Xtr, ytr, n_splits=3, random_state=0)
    base_full = _fast_base().fit(Xtr, ytr)

    df_tr, cats = augment_features(Xtr, base_full, score=oof)
    df_te, _ = augment_features(Xte, base_full)  # in-sample score is fine for test

    train = RepLeafDataset(df_tr, ytr, categorical_features=cats)
    model = RepLeafRegressor(n_estimators=20, num_leaves=8, random_state=42)
    model.fit(train)
    pred = model.predict(RepLeafDataset(df_te, metadata=train.metadata))

    baseline = np.sqrt(np.mean((yte - ytr.mean()) ** 2))
    assert np.sqrt(np.mean((yte - pred) ** 2)) < 0.5 * baseline


def test_missing_lightgbm_message(monkeypatch, reg_data):
    """Without lightgbm the error must say how to install it."""
    import sys

    Xtr, ytr, _, _ = reg_data
    monkeypatch.setitem(sys.modules, "lightgbm", None)
    with pytest.raises(ImportError, match="pip install"):
        LightGBMExternalModel().fit(Xtr, ytr)
