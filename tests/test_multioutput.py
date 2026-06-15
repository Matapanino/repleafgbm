"""Multi-output regression: shared-routing vector leaves."""

import json

import numpy as np
import pandas as pd
import pytest

from repleafgbm import RepLeafRegressor
from repleafgbm.core.multioutput import MultiOutputBooster


def make_multioutput(n: int, seed: int):
    """Two correlated regression targets over shared features."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    y0 = X[:, 0] + 0.5 * X[:, 1] ** 2 - X[:, 2]
    y1 = -X[:, 2] + X[:, 0] * X[:, 3] + 0.3 * X[:, 1]
    Y = np.column_stack([y0, y1]) + 0.05 * rng.normal(size=(n, 2))
    return X, Y


@pytest.mark.parametrize("leaf_model", ["constant", "embedded_linear"])
def test_fit_predict_shape(leaf_model):
    X, Y = make_multioutput(400, seed=0)
    model = RepLeafRegressor(
        n_estimators=30, leaf_model=leaf_model, num_leaves=8, random_state=42
    )
    model.fit(X[:300], Y[:300])
    assert model.n_outputs_ == 2
    assert isinstance(model.booster_, MultiOutputBooster)
    # One shared tree per round (unlike multiclass).
    assert len(model.booster_.trees_) == 30
    pred = model.predict(X[300:])
    assert pred.shape == (100, 2)


def test_shared_routing_competitive_with_independent_fits():
    """A shared-routing vector model should be in the ballpark of fitting each
    output independently (correlated targets, shared structure)."""
    X, Y = make_multioutput(500, seed=1)
    Xtr, Ytr, Xte, Yte = X[:400], Y[:400], X[400:], Y[400:]

    multi = RepLeafRegressor(
        n_estimators=60, leaf_model="embedded_linear", num_leaves=8, random_state=0
    ).fit(Xtr, Ytr)
    multi_rmse = np.sqrt(np.mean((multi.predict(Xte) - Yte) ** 2))

    indep_preds = []
    for k in range(Y.shape[1]):
        m = RepLeafRegressor(
            n_estimators=60, leaf_model="embedded_linear", num_leaves=8, random_state=0
        ).fit(Xtr, Ytr[:, k])
        indep_preds.append(m.predict(Xte))
    indep_rmse = np.sqrt(np.mean((np.column_stack(indep_preds) - Yte) ** 2))

    # Shared routing trades some flexibility for fewer trees; require it within
    # 25% of independent per-output fits on this shared-structure data.
    assert multi_rmse <= 1.25 * indep_rmse


def test_save_load_roundtrip(tmp_path):
    X, Y = make_multioutput(300, seed=2)
    model = RepLeafRegressor(
        n_estimators=20, leaf_model="embedded_linear", num_leaves=8, random_state=0
    ).fit(X, Y)
    before = model.predict(X)

    model.save_model(tmp_path)
    config = json.loads((tmp_path / "model_config.json").read_text())
    assert config["format_version"] == 6
    ensemble = json.loads((tmp_path / "tree_ensemble.json").read_text())
    assert ensemble["n_outputs"] == 2

    reloaded = RepLeafRegressor.load_model(tmp_path)
    assert np.allclose(reloaded.predict(X), before)


def test_eval_set_and_early_stopping():
    X, Y = make_multioutput(500, seed=3)
    model = RepLeafRegressor(
        n_estimators=200,
        leaf_model="embedded_linear",
        num_leaves=8,
        early_stopping_rounds=10,
        random_state=0,
    )
    model.fit(X[:400], Y[:400], eval_set=[(X[400:], Y[400:])])
    assert model.best_iteration_ is not None
    assert 0 < model.best_iteration_ <= 200
    assert len(model.evals_result_["valid_0"]["rmse"]) >= model.best_iteration_


def test_categorical_multioutput():
    """Categorical features route via ordered thresholds in multi-output mode."""
    rng = np.random.default_rng(4)
    n = 400
    cat = rng.integers(0, 5, n)
    num = rng.normal(size=n)
    X = pd.DataFrame({"cat": pd.Categorical(cat), "num": num})
    Y = np.column_stack([cat + num, -cat + 0.5 * num]) + 0.05 * rng.normal(size=(n, 2))
    model = RepLeafRegressor(
        n_estimators=30, leaf_model="constant", num_leaves=8, random_state=0
    )
    model.fit(X.iloc[:300], Y[:300])
    pred = model.predict(X.iloc[300:])
    assert pred.shape == (100, 2)


def test_single_output_unchanged():
    """1-D y keeps the scalar Booster (no behavior change)."""
    X, Y = make_multioutput(300, seed=5)
    model = RepLeafRegressor(n_estimators=20, num_leaves=8, random_state=0)
    model.fit(X, Y[:, 0])
    assert model.n_outputs_ == 1
    assert not isinstance(model.booster_, MultiOutputBooster)
    assert model.predict(X).ndim == 1


def test_multioutput_rejects_nondefault_objective():
    X, Y = make_multioutput(200, seed=6)
    model = RepLeafRegressor(n_estimators=10, objective="huber", random_state=0)
    with pytest.raises(ValueError, match="squared error only"):
        model.fit(X, Y)
