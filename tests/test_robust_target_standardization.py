"""Robust-objective target standardization (docs/proposals/robust-target-standardization.md).

Robust regression objectives (huber/quantile) are fit in per-output standardized
target space so a fixed delta=1 / unit quantile step is scale-consistent (~1 sigma)
across target scales — eliminating the large-scale clean-fit underfit — while
squared_error and classification are unchanged (identity transform). The booster
carries (target_loc_, target_scale_) and un-standardizes predictions/eval;
serialization persists them (format v7, bump-on-use, backward-compatible).
"""

from __future__ import annotations

import json

import numpy as np

from repleafgbm import RepLeafClassifier, RepLeafRegressor


def _large_scale_reg(n=400, scale=200.0, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    y = scale * (2.0 * X[:, 0] - X[:, 1] + 0.3 * rng.normal(size=n))  # std ~ hundreds
    return X[: n * 3 // 4], X[n * 3 // 4:], y[: n * 3 // 4], y[n * 3 // 4:]


def _rmse(m, X, y):
    return float(np.sqrt(np.mean((y - m.predict(X)) ** 2)))


def test_huber_large_scale_clean_fit_matches_squared():
    """The bug: fixed delta=1 underfits large-scale targets (~15x worse than
    squared on clean data). The fix makes huber clean-fit ~ squared."""
    Xtr, Xte, ytr, yte = _large_scale_reg(scale=200.0)
    sq = RepLeafRegressor(n_estimators=100, random_state=0).fit(Xtr, ytr)
    hb = RepLeafRegressor(n_estimators=100, objective="huber", random_state=0).fit(Xtr, ytr)
    assert _rmse(hb, Xte, yte) < 1.5 * _rmse(sq, Xte, yte)
    # the transform was actually applied (non-identity scale ~ target sigma)
    assert hb.booster_.target_scale_ > 10.0


def test_quantile_large_scale_clean_fit_reasonable():
    Xtr, Xte, ytr, yte = _large_scale_reg(scale=200.0)
    sq = RepLeafRegressor(n_estimators=100, random_state=0).fit(Xtr, ytr)
    q = RepLeafRegressor(n_estimators=100, objective="quantile", random_state=0).fit(Xtr, ytr)
    assert _rmse(q, Xte, yte) < 2.0 * _rmse(sq, Xte, yte)


def test_multioutput_huber_heterogeneous_scales():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 5))
    Y = np.column_stack([300.0 * X[:, 0], 0.5 * X[:, 1]])  # very different scales
    Y = Y + rng.normal(size=Y.shape) * np.array([30.0, 0.05])
    Xtr, Xte, Ytr, Yte = X[:300], X[300:], Y[:300], Y[300:]
    hb = RepLeafRegressor(n_estimators=100, objective="huber", random_state=0).fit(Xtr, Ytr)
    sq = RepLeafRegressor(n_estimators=100, random_state=0).fit(Xtr, Ytr)
    assert hb.predict(Xte).shape == Yte.shape
    assert np.asarray(hb.booster_.target_scale_).shape == (2,)  # per-output
    assert _rmse(hb, Xte, Yte) < 1.5 * _rmse(sq, Xte, Yte)


def test_squared_error_is_identity_and_format_v3(tmp_path):
    Xtr, _Xte, ytr, _yte = _large_scale_reg(scale=200.0)
    m = RepLeafRegressor(n_estimators=20, random_state=0).fit(Xtr, ytr)
    assert m.booster_.target_loc_ == 0.0 and m.booster_.target_scale_ == 1.0
    p = tmp_path / "sq"
    m.save_model(str(p))
    assert json.loads((p / "model_config.json").read_text())["format_version"] == 3
    assert "target_loc" not in json.loads((p / "tree_ensemble.json").read_text())


def test_robust_save_load_roundtrip_format_v7(tmp_path):
    Xtr, Xte, ytr, _yte = _large_scale_reg(scale=200.0)
    m = RepLeafRegressor(n_estimators=30, objective="huber", random_state=0).fit(Xtr, ytr)
    pred_before = m.predict(Xte)
    p = tmp_path / "hb"
    m.save_model(str(p))
    assert json.loads((p / "model_config.json").read_text())["format_version"] == 7
    ens = json.loads((p / "tree_ensemble.json").read_text())
    assert "target_loc" in ens and "target_scale" in ens
    loaded = RepLeafRegressor.load_model(str(p))
    np.testing.assert_allclose(loaded.predict(Xte), pred_before, rtol=1e-10)


def test_eval_metric_reported_on_raw_scale():
    """A robust model's eval metric must be on the raw target scale (rmse ~ the
    target magnitude), not in standardized units (~O(1))."""
    Xtr, Xva, ytr, yva = _large_scale_reg(scale=200.0)
    m = RepLeafRegressor(n_estimators=20, objective="huber", random_state=0).fit(
        Xtr, ytr, eval_set=[(Xva, yva)])
    last = next(iter(next(iter(m.evals_result_.values())).values()))[-1]
    assert last > 5.0  # raw-scale rmse (target sigma ~ hundreds); std units would be ~1


def test_classifier_unaffected(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    y = (X[:, 0] + 0.5 * rng.normal(size=200) > 0).astype(int)
    m = RepLeafClassifier(n_estimators=20, random_state=0).fit(X, y)
    assert m.booster_.target_loc_ == 0.0 and m.booster_.target_scale_ == 1.0
    m.save_model(str(tmp_path / "clf"))
    fmt = json.loads((tmp_path / "clf" / "model_config.json").read_text())["format_version"]
    assert fmt != 7  # binary classifier never standardizes the target


def test_pre_v7_model_loads_with_identity_transform(tmp_path):
    """A model directory without the target keys (pre-v7 / non-robust) loads with
    the identity transform without error."""
    Xtr, _Xte, ytr, _yte = _large_scale_reg(scale=200.0)
    m = RepLeafRegressor(n_estimators=10, objective="huber", random_state=0).fit(Xtr, ytr)
    p = tmp_path / "old"
    m.save_model(str(p))
    ens_path = p / "tree_ensemble.json"
    ens = json.loads(ens_path.read_text())
    del ens["target_loc"], ens["target_scale"]  # simulate a pre-v7 directory
    ens_path.write_text(json.dumps(ens))
    cfg_path = p / "model_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["format_version"] = 6
    cfg_path.write_text(json.dumps(cfg))
    loaded = RepLeafRegressor.load_model(str(p))  # must not raise
    assert np.asarray(loaded.booster_.target_loc_).item() == 0.0
    assert np.asarray(loaded.booster_.target_scale_).item() == 1.0


def test_degenerate_constant_target_floors_scale():
    """A constant target -> MAD 0 -> scale floored to 1.0 (no divide-by-zero)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 4))
    y = np.full(100, 5.0)
    m = RepLeafRegressor(n_estimators=10, objective="huber", random_state=0).fit(X, y)
    assert m.booster_.target_scale_ == 1.0
    assert np.all(np.isfinite(m.predict(X)))


def test_constant_fallback_small_leaves_robust():
    """embedded_linear + huber on tiny leaves must fall back to a constant leaf
    (finite predictions) in standardized space."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    y = 100.0 * X[:, 0] + rng.normal(size=40)
    m = RepLeafRegressor(
        n_estimators=10, leaf_model="embedded_linear", encoder="identity",
        num_leaves=16, min_samples_leaf=5, objective="huber", random_state=0).fit(X, y)
    assert np.all(np.isfinite(m.predict(X)))


def test_poisson_is_not_standardized():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 4))
    y = rng.poisson(3.0, 120).astype(float)
    m = RepLeafRegressor(n_estimators=10, objective="poisson", random_state=0).fit(X, y)
    assert m.booster_.target_loc_ == 0.0 and m.booster_.target_scale_ == 1.0


def test_parameterized_robust_instances_are_detected():
    """Instance objectives (not just the string form) trigger standardization."""
    from repleafgbm.core.objectives import Huber, Quantile
    Xtr, _Xte, ytr, _yte = _large_scale_reg(scale=200.0)
    mh = RepLeafRegressor(
        n_estimators=10, objective=Huber(delta=2.0), random_state=0).fit(Xtr, ytr)
    mq = RepLeafRegressor(
        n_estimators=10, objective=Quantile(alpha=0.9), random_state=0).fit(Xtr, ytr)
    assert mh.booster_.target_scale_ > 10.0
    assert mq.booster_.target_scale_ > 10.0
