"""Same-budget hyperparameter search for the fair leaderboard.

Every model family is tuned with **the same number of Optuna trials** on the
**same** ``(train, valid)`` split and seed, then evaluated once on the held-out
test set by the caller. This removes the "tuned-vs-default" bias that made the
old benchmarks (LightGBM hardcoded at ``n_estimators=600``, RepLeaf at ``400``)
uninformative: a win or loss now reflects the model, not the tuning effort.

Fairness notes:

* Identical ``n_trials`` is the budget definition (the user's choice). Equal trial
  count is **not** equal wall-clock — that caveat is surfaced in the report.
* All families fit on the *same ordinal-encoded feature matrix* (the leaderboard
  passes arrays), so inputs are identical across models.
* Learned (torch) RepLeaf encoders are **excluded** from the budgeted search by
  default — per-trial supervised pretraining would blow the budget; they stay a
  separate opt-in study.

Lives under ``benchmarks/`` only; ``optuna`` is a ``[bench]`` extra, imported
lazily inside :func:`tune` so the module imports without it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import log_loss

FAMILIES = ("repleaf", "lightgbm", "xgboost", "catboost", "hist_gradient_boosting")


@dataclass
class HPOResult:
    """Outcome of one budgeted search: constructor-ready ``params`` + val score."""

    family: str
    params: dict[str, Any]
    value: float
    n_trials: int


# --------------------------------------------------------------------------- #
# Primary metric (lower is better) — RMSE for regression, logloss otherwise.
# --------------------------------------------------------------------------- #
def _rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def _primary(y_true, model, X, task: str, classes) -> float:
    if task == "regression":
        return _rmse(y_true, model.predict(X))
    proba = model.predict_proba(X)
    return float(log_loss(y_true, proba, labels=classes))


# --------------------------------------------------------------------------- #
# Per-family search spaces. Each returns a **constructor-ready** params dict
# (keys match the estimator's __init__), so build_model can ``**params`` it.
# --------------------------------------------------------------------------- #
def _n_estimators(trial, quick: bool, name: str = "n_estimators") -> int:
    lo, hi = (20, 80) if quick else (100, 800)
    return trial.suggest_int(name, lo, hi, log=True)


def _space_repleaf(trial, task: str, quick: bool) -> dict[str, Any]:
    leaf_model = trial.suggest_categorical(
        "leaf_model", ["constant", "embedded_linear", "adaptive"]
    )
    params: dict[str, Any] = {
        "n_estimators": _n_estimators(trial, quick),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127, log=True),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 60, log=True),
        "l2_leaf": trial.suggest_float("l2_leaf", 1e-2, 10.0, log=True),
        "leaf_model": leaf_model,
    }
    if leaf_model == "constant":
        params["encoder"] = "identity"
    else:
        encoder = trial.suggest_categorical("encoder", ["identity", "plr"])
        params["encoder"] = encoder
        params["max_leaf_emb_dim"] = 256 if encoder == "plr" else 64
    return params


def _space_lightgbm(trial, task: str, quick: bool) -> dict[str, Any]:
    return {
        "n_estimators": _n_estimators(trial, quick),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 60, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    }


def _space_xgboost(trial, task: str, quick: bool) -> dict[str, Any]:
    return {
        "n_estimators": _n_estimators(trial, quick),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def _space_catboost(trial, task: str, quick: bool) -> dict[str, Any]:
    return {
        "iterations": _n_estimators(trial, quick, name="iterations"),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
    }


def _space_histgb(trial, task: str, quick: bool) -> dict[str, Any]:
    return {
        "max_iter": _n_estimators(trial, quick, name="max_iter"),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127, log=True),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 60, log=True),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
    }


SEARCH_SPACES: dict[str, Callable[[Any, str, bool], dict[str, Any]]] = {
    "repleaf": _space_repleaf,
    "lightgbm": _space_lightgbm,
    "xgboost": _space_xgboost,
    "catboost": _space_catboost,
    "hist_gradient_boosting": _space_histgb,
}


# --------------------------------------------------------------------------- #
# Model construction from a constructor-ready params dict.
# --------------------------------------------------------------------------- #
def build_model(family: str, params: dict[str, Any], task: str, seed: int):
    """Instantiate an unfitted estimator for ``family`` from tuned ``params``."""
    p = dict(params)
    if family == "repleaf":
        from repleafgbm import RepLeafClassifier, RepLeafRegressor

        cls = RepLeafRegressor if task == "regression" else RepLeafClassifier
        return cls(random_state=seed, **p)
    if family == "lightgbm":
        import lightgbm as lgb

        cls = lgb.LGBMRegressor if task == "regression" else lgb.LGBMClassifier
        return cls(random_state=seed, verbose=-1, **p)
    if family == "xgboost":
        import xgboost as xgb

        cls = xgb.XGBRegressor if task == "regression" else xgb.XGBClassifier
        return cls(random_state=seed, tree_method="hist", verbosity=0, **p)
    if family == "catboost":
        import catboost as cb

        cls = cb.CatBoostRegressor if task == "regression" else cb.CatBoostClassifier
        return cls(random_seed=seed, verbose=False, **p)
    if family == "hist_gradient_boosting":
        from sklearn.ensemble import (
            HistGradientBoostingClassifier,
            HistGradientBoostingRegressor,
        )

        cls = (HistGradientBoostingRegressor if task == "regression"
               else HistGradientBoostingClassifier)
        return cls(random_state=seed, **p)
    raise ValueError(f"unknown model family: {family!r}")


# --------------------------------------------------------------------------- #
# The budgeted search.
# --------------------------------------------------------------------------- #
def tune(
    family: str,
    Xtr, ytr, Xva, yva,
    task: str,
    n_trials: int,
    seed: int,
    quick: bool = False,
    classes=None,
) -> HPOResult:
    """Run ``n_trials`` of Optuna TPE for one family; return the best config.

    The objective fits a candidate on ``(Xtr, ytr)`` and scores the primary
    metric on ``(Xva, yva)`` (lower is better). The full constructor-ready param
    dict for each trial is stashed as a user attribute, so the returned
    :class:`HPOResult` carries everything :func:`build_model` needs to refit the
    winner for the test evaluation.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if family not in SEARCH_SPACES:
        raise ValueError(f"unknown model family: {family!r}")
    if classes is None and task != "regression":
        classes = np.unique(np.concatenate([np.asarray(ytr), np.asarray(yva)]))

    space = SEARCH_SPACES[family]

    def objective(trial):
        params = space(trial, task, quick)
        trial.set_user_attr("params", params)
        model = build_model(family, params, task, seed)
        try:
            model.fit(Xtr, ytr)
        except Exception as exc:  # a bad corner of the space prunes, not crashes
            raise optuna.TrialPruned() from exc
        return _primary(yva, model, Xva, task, classes)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_trial
    return HPOResult(
        family=family,
        params=best.user_attrs["params"],
        value=float(best.value),
        n_trials=n_trials,
    )
