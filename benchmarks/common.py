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
from datetime import date
from pathlib import Path
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


def write_markdown_report(name: str, title: str, preamble: list[str],
                          results: list[BenchResult]) -> Path:
    """Persist a benchmark table as ``experiments/results/<date>-<name>.md``.

    The synthetic benchmarks used to print to stdout only; this gives them a
    dated, committable report like the heavier suites, reusing the ``metrics``
    dict on :class:`BenchResult` for the columns.
    """
    metric_names = list(results[0].metrics)
    header = "| model | fit[s] | pred[s] | " + " | ".join(metric_names) + " |"
    sep = "|---|---|---|" + "---|" * len(metric_names)
    lines = [f"# {title}", "", *preamble, "", header, sep]
    for r in results:
        cells = " | ".join(f"{r.metrics[m]:.4f}" for m in metric_names)
        lines.append(
            f"| {r.name} | {r.fit_seconds:.2f} | {r.predict_seconds:.3f} | {cells} |"
        )
    out_path = (Path(__file__).resolve().parents[1] / "experiments" / "results"
                / f"{date.today().isoformat()}-{name}.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def make_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--n-train", type=int, default=10_000)
    p.add_argument("--n-test", type=int, default=5_000)
    p.add_argument("--n-features", type=int, default=20)
    p.add_argument("--n-estimators", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--contamination", type=float, default=0.0,
        help="fraction of TRAIN targets to replace with heavy-tailed outliers "
             "(regression only; shows where robust objectives help)",
    )
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


def multioutput_signal(
    n_rows: int, n_features: int, n_outputs: int, rng: np.random.Generator
):
    """``n_outputs`` correlated targets sharing one ``X`` (shared tree routing).

    Each output mixes a shared router discontinuity with an output-specific
    smooth term, so multi-output leaves (one routing, vector leaf) have a real
    advantage over independent per-output fits. Returns ``(X, signal)`` with
    ``signal`` of shape ``(n_rows, n_outputs)`` and no noise added.
    """
    X, base = synthetic_tabular(n_rows, n_features, rng)
    router = 3.0 * (X[:, 3] > 0.5)
    signal = np.column_stack([
        base + 2.0 * np.sin((k + 2) * X[:, 0]) + 1.5 * X[:, 1] + router
        for k in range(n_outputs)
    ])
    return X, signal


def contaminate(y: np.ndarray, frac: float, scale: float, rng: np.random.Generator):
    """Replace a ``frac`` of rows with heavy-tailed outliers (training only).

    ``scale`` multiplies each column's own std so contamination is comparable
    across outputs. ``y`` may be 1-D or 2-D ``(n, K)``; a copy is returned.
    """
    if frac <= 0.0:
        return y.copy()
    out = np.array(y, dtype=np.float64, copy=True)
    y2 = out if out.ndim == 2 else out[:, None]
    mask = rng.random(y2.shape[0]) < frac
    sd = y2.std(axis=0, keepdims=True)
    y2[mask] += rng.normal(0.0, 1.0, (int(mask.sum()), y2.shape[1])) * scale * sd
    return out


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
