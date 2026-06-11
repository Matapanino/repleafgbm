"""Experiment: native router vs extracted LightGBM router (ADR 0002, M4 v2).

Fair-comparison revision of the Phase 3 experiment: the LightGBM base now
uses native early stopping on the same validation set the native models use,
and the replay stage early-stops as well. Adds a binary classification
section (RouterExtractionClassifier).

Compared per dataset:
  lightgbm alone (es) | native RepLeaf (constant / embedded_linear, es) |
  RouterExtraction (constant — sanity — / embedded_linear identity /
  embedded_linear plr, replay es).

Run from the repository root:
    python3 experiments/router_extraction.py
Results are written to experiments/results/router_extraction.md.
Requires lightgbm.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.datasets import make_friedman1

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from common import synthetic_tabular  # noqa: E402

from repleafgbm import RepLeafClassifier, RepLeafDataset, RepLeafRegressor  # noqa: E402
from repleafgbm.external import (  # noqa: E402
    LightGBMExternalModel,
    RouterExtractionClassifier,
    RouterExtractionRegressor,
)

N_FEATURES = 10
BASE_PARAMS = dict(n_estimators=500, learning_rate=0.05, num_leaves=31)
ES_ROUNDS = 25


@dataclass
class RunResult:
    label: str
    metric: list[float]
    n_trees: list[int]


def make_dataset(name: str, n_rows: int, seed: int):
    rng = np.random.default_rng(seed)
    if name == "piecewise_linear":
        X, signal = synthetic_tabular(n_rows, N_FEATURES, rng)
        return X, signal + rng.normal(0.0, 0.3, n_rows), "regression"
    if name == "friedman1":
        X, y = make_friedman1(n_samples=n_rows, n_features=N_FEATURES,
                              noise=1.0, random_state=seed)
        return X, y, "regression"
    if name == "binary_piecewise":
        X, signal = synthetic_tabular(n_rows, N_FEATURES, rng)
        logit = signal - np.median(signal)
        y = (logit + rng.normal(0.0, 1.0, n_rows) > 0).astype(float)
        return X, y, "binary"
    raise ValueError(name)


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def logloss(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def run_dataset(name: str, seeds: list[int], n_train: int, n_valid: int, n_test: int):
    results: dict[str, RunResult] = {}

    def record(label: str, value: float, n_trees: int):
        results.setdefault(label, RunResult(label, [], []))
        results[label].metric.append(value)
        results[label].n_trees.append(n_trees)

    for seed in seeds:
        X, y, task = make_dataset(name, n_train + n_valid + n_test, seed)
        Xtr, ytr = X[:n_train], y[:n_train]
        Xva = X[n_train:n_train + n_valid]
        yva = y[n_train:n_train + n_valid]
        Xte, yte = X[n_train + n_valid:], y[n_train + n_valid:]
        score = rmse if task == "regression" else logloss

        # Shared early-stopped LightGBM base (also "lightgbm alone").
        base = LightGBMExternalModel(task=task, random_state=seed, **BASE_PARAMS)
        base.fit(Xtr, ytr, eval_set=[(Xva, yva)], early_stopping_rounds=ES_ROUNDS)
        record("lightgbm alone (es)", score(yte, base.predict_score(Xte)), base.n_trees_)

        # Native models with early stopping on the same validation set.
        native_cls = RepLeafRegressor if task == "regression" else RepLeafClassifier
        for leaf_model in ("constant", "embedded_linear"):
            model = native_cls(
                n_estimators=300, learning_rate=0.1, num_leaves=16,
                min_samples_leaf=20, leaf_model=leaf_model, encoder="identity",
                early_stopping_rounds=ES_ROUNDS, random_state=seed,
            )
            train = RepLeafDataset(Xtr, ytr)
            valid = RepLeafDataset(Xva, yva, metadata=train.metadata)
            model.fit(train, eval_set=[valid])
            pred = (model.predict(Xte) if task == "regression"
                    else model.predict_proba(Xte)[:, 1])
            record(f"native {leaf_model} (es)", score(yte, pred),
                   model.best_iteration_ or model.booster_.n_trees)

        # Router extraction on the shared base, replay early stopping.
        routerx_cls = (RouterExtractionRegressor if task == "regression"
                       else RouterExtractionClassifier)
        configs = [
            ("routerx constant (sanity)", dict(leaf_model="constant")),
            ("routerx embedded_linear identity",
             dict(leaf_model="embedded_linear", encoder="identity")),
            ("routerx embedded_linear plr",
             dict(leaf_model="embedded_linear", encoder="plr")),
        ]
        for label, kwargs in configs:
            model = routerx_cls(
                base=base, min_samples_leaf=20,
                early_stopping_rounds=ES_ROUNDS, random_state=seed, **kwargs,
            )
            train = RepLeafDataset(Xtr, ytr)
            valid = RepLeafDataset(Xva, yva, metadata=train.metadata)
            model.fit(train, eval_set=[valid])
            pred = (model.predict(Xte) if task == "regression"
                    else model.predict_proba(Xte)[:, 1])
            record(label, score(yte, pred),
                   model.best_iteration_ or model.booster_.n_trees)

    return sorted(results.values(), key=lambda r: np.mean(r.metric))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--n-train", type=int, default=4000)
    parser.add_argument("--n-valid", type=int, default=1500)
    parser.add_argument("--n-test", type=int, default=4000)
    args = parser.parse_args()
    seeds = list(range(args.seeds))

    out_lines = [
        "# Experiment: native router vs extracted LightGBM router (fair, v2)",
        "",
        "Auto-generated by `experiments/router_extraction.py`. "
        "See the Analysis section at the bottom for conclusions.",
        "",
        f"Settings: n_train={args.n_train}, n_valid={args.n_valid}, "
        f"n_test={args.n_test}, n_features={N_FEATURES}, seeds={seeds}. "
        f"Base/lightgbm: {BASE_PARAMS} with early stopping ({ES_ROUNDS}) on the "
        "shared validation set. Native: 300 trees, lr=0.1, num_leaves=16, same "
        "early stopping. Router extraction: replay early stopping on the same "
        "validation set. All leaf refits: l2_leaf=1.0, min_samples_leaf=20. "
        "Metric: RMSE for regression, logloss for binary.",
    ]

    for dataset_name in ("piecewise_linear", "friedman1", "binary_piecewise"):
        print(f"=== dataset: {dataset_name} ===")
        ordered = run_dataset(dataset_name, seeds, args.n_train, args.n_valid,
                              args.n_test)
        for r in ordered:
            print(f"  {r.label:36s} metric={np.mean(r.metric):.4f} "
                  f"trees={np.mean(r.n_trees):.0f}")
        out_lines += [
            "",
            f"## Dataset: {dataset_name}",
            "",
            "| config | test metric (mean ± std) | trees used |",
            "|---|---|---|",
            *[
                f"| {r.label} | {np.mean(r.metric):.4f} ± {np.std(r.metric):.4f} "
                f"| {np.mean(r.n_trees):.0f} |"
                for r in ordered
            ],
        ]

    out_path = Path(__file__).resolve().parent / "results" / "router_extraction.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines) + "\n")
    print(f"\nreport written to {out_path}")


if __name__ == "__main__":
    main()
