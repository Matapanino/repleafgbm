"""Unit tests for benchmarks/hpo.py (same-budget Optuna search).

Kept tiny and fast: a 2-trial quick search on a small synthetic set must return
a constructor-ready config that refits and predicts. ``optuna`` (and the GBM
libs) are optional, so each tune test skips cleanly when its dependency is absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import hpo  # noqa: E402


def _regression(n=120, d=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = 2.0 * X[:, 0] - X[:, 1] + 0.1 * rng.normal(size=n)
    return X[: n // 2], y[: n // 2], X[n // 2:], y[n // 2:]


def _binary(n=160, d=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    return X[: n // 2], y[: n // 2], X[n // 2:], y[n // 2:]


def test_tune_repleaf_regression_returns_buildable_config():
    pytest.importorskip("optuna")
    Xtr, ytr, Xva, yva = _regression()
    res = hpo.tune("repleaf", Xtr, ytr, Xva, yva, "regression",
                   n_trials=2, seed=0, quick=True)
    assert res.family == "repleaf"
    assert res.n_trials == 2
    assert np.isfinite(res.value)
    assert res.params["leaf_model"] in {"constant", "embedded_linear", "adaptive"}
    # The winning config refits and predicts.
    model = hpo.build_model("repleaf", res.params, "regression", seed=0)
    model.fit(Xtr, ytr)
    assert model.predict(Xva).shape == (Xva.shape[0],)


def test_tune_lightgbm_binary_returns_buildable_config():
    pytest.importorskip("optuna")
    pytest.importorskip("lightgbm")
    Xtr, ytr, Xva, yva = _binary()
    res = hpo.tune("lightgbm", Xtr, ytr, Xva, yva, "binary",
                   n_trials=2, seed=0, quick=True)
    assert np.isfinite(res.value) and res.value >= 0.0  # logloss
    model = hpo.build_model("lightgbm", res.params, "binary", seed=0)
    model.fit(Xtr, ytr)
    assert model.predict_proba(Xva).shape[0] == Xva.shape[0]


def test_build_model_histgb_without_optuna():
    # build_model needs no optuna; covers the sklearn family path directly.
    Xtr, ytr, Xva, yva = _regression()
    params = {"max_iter": 30, "learning_rate": 0.1, "max_leaf_nodes": 31,
              "min_samples_leaf": 20, "l2_regularization": 0.0}
    model = hpo.build_model("hist_gradient_boosting", params, "regression", seed=0)
    model.fit(Xtr, ytr)
    assert model.predict(Xva).shape == (Xva.shape[0],)


def test_unknown_family_raises():
    with pytest.raises(ValueError):
        hpo.build_model("bogus", {}, "regression", seed=0)
