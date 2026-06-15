"""End-to-end regression tests."""

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from repleafgbm import RepLeafDataset, RepLeafRegressor


def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


@pytest.mark.parametrize("leaf_model", ["constant", "embedded_linear", "raw_linear"])
def test_fit_predict_beats_mean_baseline(regression_data, leaf_model):
    Xtr, ytr, Xte, yte = regression_data
    model = RepLeafRegressor(
        n_estimators=30,
        learning_rate=0.2,
        num_leaves=8,
        min_samples_leaf=10,
        leaf_model=leaf_model,
        encoder="identity",
        random_state=42,
    )
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    baseline = _rmse(np.full_like(yte, ytr.mean()), yte)
    assert _rmse(pred, yte) < 0.5 * baseline


def test_embedded_linear_beats_constant_on_locally_linear_data(regression_data):
    """With few shallow trees, constant leaves cannot capture within-region
    linear structure but embedded-linear leaves can."""
    Xtr, ytr, Xte, yte = regression_data
    common = dict(
        n_estimators=10,
        learning_rate=0.5,
        num_leaves=4,
        min_samples_leaf=10,
        encoder="identity",
        random_state=42,
    )
    constant = RepLeafRegressor(leaf_model="constant", **common).fit(Xtr, ytr)
    linear = RepLeafRegressor(leaf_model="embedded_linear", **common).fit(Xtr, ytr)
    assert _rmse(linear.predict(Xte), yte) < _rmse(constant.predict(Xte), yte)


def test_plr_encoder_pipeline(regression_data):
    Xtr, ytr, Xte, yte = regression_data
    model = RepLeafRegressor(
        n_estimators=20,
        num_leaves=8,
        leaf_model="embedded_linear",
        encoder="plr",
        encoder_params={"n_bins": 8},
        max_leaf_emb_dim=12,  # forces random projection (4 features * 8 bins = 32)
        random_state=42,
    )
    model.fit(Xtr, ytr)
    assert model.encoder_.output_dim == 12
    baseline = _rmse(np.full_like(yte, ytr.mean()), yte)
    assert _rmse(model.predict(Xte), yte) < baseline


def test_dataset_api_and_eval_set(regression_data):
    Xtr, ytr, Xte, yte = regression_data
    train = RepLeafDataset(Xtr, ytr)
    model = RepLeafRegressor(n_estimators=15, num_leaves=8, random_state=42)
    valid = RepLeafDataset(Xte, yte, metadata=train.metadata)
    model.fit(train, eval_set=[valid])
    assert "valid_0" in model.evals_result_
    history = model.evals_result_["valid_0"]["rmse"]
    assert len(history) == 15
    assert history[-1] < history[0]  # validation error decreased


def test_determinism(regression_data):
    Xtr, ytr, Xte, _ = regression_data
    kwargs = dict(n_estimators=10, num_leaves=8, encoder="plr", max_leaf_emb_dim=8,
                  leaf_model="embedded_linear", random_state=42)
    p1 = RepLeafRegressor(**kwargs).fit(Xtr, ytr).predict(Xte)
    p2 = RepLeafRegressor(**kwargs).fit(Xtr, ytr).predict(Xte)
    np.testing.assert_allclose(p1, p2)


def test_pandas_with_categorical_feature():
    rng = np.random.default_rng(5)
    n = 300
    cat = rng.choice(["low", "mid", "high"], size=n)
    shift = np.select([cat == "low", cat == "mid"], [-2.0, 0.0], default=2.0)
    x = rng.normal(size=n)
    y = shift + x + rng.normal(0, 0.1, n)
    df = pd.DataFrame({"group": cat, "x": x})

    model = RepLeafRegressor(
        n_estimators=20, num_leaves=8, min_samples_leaf=10, random_state=42
    )
    model.fit(RepLeafDataset(df, y, categorical_features=["group"]))
    pred = model.predict(df)
    assert _rmse(pred, y) < 0.5 * np.std(y)


def test_min_samples_leaf_respected(regression_data):
    Xtr, ytr, _, _ = regression_data
    model = RepLeafRegressor(
        n_estimators=3, num_leaves=64, min_samples_leaf=50, random_state=42
    )
    model.fit(Xtr, ytr)
    for tree in model.booster_.trees_:
        leaf_idx = tree.apply(np.asarray(Xtr, dtype=np.float64))
        counts = np.bincount(leaf_idx)
        assert counts[counts > 0].min() >= 50


def test_experiment_driven_defaults():
    """Defaults set from experiments/results/plr_projection_gap.md — change
    them only with new experimental evidence."""
    from repleafgbm.encoders import SimplePLREncoder

    assert SimplePLREncoder().n_bins == 4
    assert RepLeafRegressor().max_leaf_emb_dim == 64


def test_projection_engagement_warns(regression_data):
    Xtr, ytr, _, _ = regression_data
    model = RepLeafRegressor(
        n_estimators=2,
        num_leaves=4,
        leaf_model="embedded_linear",
        encoder="plr",
        encoder_params={"n_bins": 8},
        max_leaf_emb_dim=4,  # 4 features * 8 bins = 32 > 4 -> projection
        random_state=42,
    )
    with pytest.warns(UserWarning, match="max_leaf_emb_dim"):
        model.fit(Xtr, ytr)
    assert model.encoder_.output_dim == 4


def test_freeze_encoder_false_rejected(regression_data):
    Xtr, ytr, _, _ = regression_data
    model = RepLeafRegressor(freeze_encoder=False)
    with pytest.raises(NotImplementedError, match="freeze_encoder"):
        model.fit(Xtr, ytr)


def test_predict_before_fit_raises():
    with pytest.raises(NotFittedError, match="not fitted"):
        RepLeafRegressor().predict(np.zeros((2, 2)))
