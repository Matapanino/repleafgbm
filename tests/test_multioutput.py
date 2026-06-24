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


def test_backend_multioutput_defaults_match_reference():
    """The BaseSplitBackend multi-output defaults reproduce the historical host
    stack + free-function scan exactly, so NumPy/Rust multi-output is unchanged.

    Locks the refactor that moved the splitter's per-output ``np.stack`` and the
    direct ``find_best_split_multioutput`` call behind the backend so a device
    backend (CUDA) can override them. Bitwise here (all NumPy).
    """
    from repleafgbm.backends.numpy_backend import (
        NumPySplitBackend,
    )
    from repleafgbm.backends.numpy_backend import (
        find_best_split_multioutput as ref_scan,
    )

    rng = np.random.default_rng(100)
    n, F, n_bins_max, K = 500, 5, 17, 3
    binned = rng.integers(0, 16, size=(n, F)).astype(np.uint16)
    binned[rng.random((n, F)) < 0.05] = 16  # missing bin
    rows = np.sort(rng.choice(n, size=300, replace=False)).astype(np.int64)
    grad = rng.normal(size=(n, K))
    hess = np.abs(rng.normal(size=(n, K))) + 0.1
    n_bins_pf = np.full(F, 16, dtype=np.int64)

    b = NumPySplitBackend()
    hist = b.build_histograms_multioutput(binned, rows, grad, hess, n_bins_max)
    expected = np.stack(
        [b.build_histograms(binned, rows, grad[:, k], hess[:, k], n_bins_max)
         for k in range(K)],
        axis=-1,
    )
    np.testing.assert_array_equal(hist, expected)
    assert hist.shape == (F, n_bins_max, 3, K)

    s_default = b.find_best_split_multioutput(hist, n_bins_pf, 20, 1.0)
    s_ref = ref_scan(hist, n_bins_pf, 20, 1.0)
    assert (s_default is None) == (s_ref is None)
    assert (s_default.feature, s_default.bin) == (s_ref.feature, s_ref.bin)
    assert s_default.gain == s_ref.gain  # bitwise: same host code path
    assert (s_default.n_left, s_default.n_right) == (s_ref.n_left, s_ref.n_right)


def test_single_output_unchanged():
    """1-D y keeps the scalar Booster (no behavior change)."""
    X, Y = make_multioutput(300, seed=5)
    model = RepLeafRegressor(n_estimators=20, num_leaves=8, random_state=0)
    model.fit(X, Y[:, 0])
    assert model.n_outputs_ == 1
    assert not isinstance(model.booster_, MultiOutputBooster)
    assert model.predict(X).ndim == 1


def test_multioutput_rejects_nonconstant_hessian_objective():
    """Constant-Hessian losses (squared/huber/quantile) are allowed; poisson
    (non-constant Hessian) is rejected for multi-output."""
    X, Y = make_multioutput(200, seed=6)
    model = RepLeafRegressor(n_estimators=10, objective="poisson", random_state=0)
    with pytest.raises(ValueError, match="constant-Hessian"):
        model.fit(np.abs(X), np.abs(Y))


@pytest.mark.parametrize("objective", ["huber", "quantile"])
@pytest.mark.parametrize("leaf_model", ["constant", "embedded_linear"])
def test_multioutput_robust_objectives_fit_predict(objective, leaf_model):
    """Multi-output huber/quantile fit, predict the right shape, are
    deterministic, and the booster carries the vector objective."""
    X, Y = make_multioutput(400, seed=7)
    kw = dict(
        n_estimators=25, leaf_model=leaf_model, num_leaves=8,
        objective=objective, random_state=0,
    )
    m1 = RepLeafRegressor(**kw).fit(X[:300], Y[:300])
    p1 = m1.predict(X[300:])
    assert p1.shape == (100, 2)
    assert m1.booster_.objective.name == f"multioutput_{objective}"
    # determinism: same seed => bitwise-identical predictions
    p2 = RepLeafRegressor(**kw).fit(X[:300], Y[:300]).predict(X[300:])
    assert np.array_equal(p1, p2)


@pytest.mark.parametrize("objective", ["huber", "quantile"])
def test_multioutput_robust_save_load_roundtrip(tmp_path, objective):
    X, Y = make_multioutput(300, seed=8)
    model = RepLeafRegressor(
        n_estimators=20, leaf_model="embedded_linear", num_leaves=8,
        objective=objective, random_state=0,
    ).fit(X, Y)
    before = model.predict(X)

    model.save_model(tmp_path)
    config = json.loads((tmp_path / "model_config.json").read_text())
    assert config["format_version"] == 6
    assert config["objective"] == f"multioutput_{objective}"

    reloaded = RepLeafRegressor.load_model(tmp_path)
    # Identity transform => predictions round-trip exactly.
    assert np.array_equal(reloaded.predict(X), before)
    assert reloaded.booster_.objective.name == f"multioutput_{objective}"


def test_multioutput_huber_robust_to_outliers():
    """On outlier-contaminated multi-output targets, huber tracks the clean
    signal better than squared error (the point of the robust loss)."""
    rng = np.random.default_rng(9)
    n = 600
    X = rng.normal(size=(n, 5))
    clean = np.column_stack([X[:, 0] * 2 + X[:, 1], -X[:, 2] + 0.5 * X[:, 3]])
    Y = clean + 0.1 * rng.normal(size=(n, 2))
    # Heavy-tailed contamination on the training targets only.
    contam = Y.copy()
    mask = rng.random(n) < 0.08
    contam[mask] += rng.normal(scale=30, size=contam[mask].shape)

    Xtr, Xte = X[:450], X[450:]
    clean_te = clean[450:]
    kw = dict(n_estimators=60, num_leaves=8, random_state=0)
    se = RepLeafRegressor(objective=None, **kw).fit(Xtr, contam[:450])
    hub = RepLeafRegressor(objective="huber", **kw).fit(Xtr, contam[:450])
    se_err = np.sqrt(np.mean((se.predict(Xte) - clean_te) ** 2))
    hub_err = np.sqrt(np.mean((hub.predict(Xte) - clean_te) ** 2))
    assert hub_err < se_err
