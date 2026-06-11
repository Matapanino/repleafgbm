"""Shared helpers for the synthetic benchmark scripts.

These benchmarks exist to *track* RepLeafGBM across development, not to claim
victories. Datasets are synthetic and small; numbers are indicative only.

External GBMs (LightGBM / XGBoost / CatBoost) are optional: they are included
when importable and silently skipped otherwise.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class BenchResult:
    name: str
    fit_seconds: float
    predict_seconds: float
    metrics: dict[str, float]


def time_model(
    name: str,
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    metric_fns: dict[str, Callable[[np.ndarray, np.ndarray], float]],
    predict_fn: Callable[[Any, np.ndarray], np.ndarray],
) -> BenchResult:
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred = predict_fn(model, X_test)
    pred_s = time.perf_counter() - t0

    metrics = {mname: fn(y_test, pred) for mname, fn in metric_fns.items()}
    return BenchResult(name, fit_s, pred_s, metrics)


def print_table(results: list[BenchResult]) -> None:
    metric_names = list(results[0].metrics)
    header = f"{'model':36s} {'fit[s]':>8s} {'pred[s]':>8s}" + "".join(
        f" {m:>10s}" for m in metric_names
    )
    print(header)
    print("-" * len(header))
    for r in results:
        row = f"{r.name:36s} {r.fit_seconds:8.2f} {r.predict_seconds:8.3f}"
        row += "".join(f" {r.metrics[m]:10.4f}" for m in metric_names)
        print(row)


def make_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--n-train", type=int, default=10_000)
    p.add_argument("--n-test", type=int, default=5_000)
    p.add_argument("--n-features", type=int, default=20)
    p.add_argument("--n-estimators", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--quick", action="store_true", help="small/fast settings for smoke runs"
    )
    return p


def apply_quick(args: argparse.Namespace) -> argparse.Namespace:
    if args.quick:
        args.n_train, args.n_test, args.n_estimators = 2_000, 1_000, 30
    return args


def synthetic_tabular(n_rows: int, n_features: int, rng: np.random.Generator):
    """Tabular signal mixing what trees and what embeddings are good at:

    discontinuous regimes on raw features, smooth nonlinearities, a linear
    backbone, and an interaction term. Returns (X, signal) without noise.
    """
    X = rng.normal(size=(n_rows, n_features))
    signal = (
        3.0 * (X[:, 0] > 0.5)                  # discontinuity (router)
        - 2.0 * (X[:, 1] < -0.5)               # discontinuity (router)
        + 2.0 * X[:, 2]                        # linear backbone
        + np.sin(2.0 * X[:, 3])                # smooth nonlinearity (encoder)
        + 0.5 * X[:, 4] ** 2                   # smooth nonlinearity (encoder)
        + 1.5 * X[:, 5] * (X[:, 0] > 0.5)      # regime-dependent slope
    )
    return X, signal


def external_gbm_models(task: str, n_estimators: int, seed: int) -> list[tuple[str, Any]]:
    """LightGBM/XGBoost/CatBoost entries when installed; [] otherwise."""
    models: list[tuple[str, Any]] = []
    try:
        import lightgbm as lgb

        cls = lgb.LGBMRegressor if task == "regression" else lgb.LGBMClassifier
        models.append(
            ("lightgbm", cls(n_estimators=n_estimators, random_state=seed, verbose=-1))
        )
    except ImportError:
        pass
    try:
        import xgboost as xgb

        cls = xgb.XGBRegressor if task == "regression" else xgb.XGBClassifier
        models.append(("xgboost", cls(n_estimators=n_estimators, random_state=seed)))
    except ImportError:
        pass
    try:
        import catboost as cb

        cls = cb.CatBoostRegressor if task == "regression" else cb.CatBoostClassifier
        models.append(
            ("catboost", cls(iterations=n_estimators, random_seed=seed, verbose=False))
        )
    except ImportError:
        pass
    return models
