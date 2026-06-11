"""Experiment: why does PLR + random projection underperform?

Phase 0.5 benchmarking (docs/audit_v0.md) found that
``leaf_model="embedded_linear", encoder="plr"`` with the default random
projection to ``max_leaf_emb_dim=32`` was clearly *worse* than plain identity
embeddings. This experiment separates the candidate causes:

* the projection itself (axis mixing destroying per-feature structure),
* the PLR basis (clipped components cannot extrapolate; high dimension
  triggers constant fallbacks in small leaves),
* under/over-regularization (``l2_leaf``).

Grid: encoder in {identity, plr(n_bins=4/8/16)} x projection {8, 32, off}
x l2_leaf {0.3, 3.0}, with a constant-leaf baseline, on two regression
datasets, multiple seeds, with early stopping on a validation set.

Run from the repository root (about 5-10 minutes):
    python3 experiments/plr_projection_gap.py
Results are written to experiments/results/plr_projection_gap.md.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.datasets import make_friedman1

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from common import synthetic_tabular  # noqa: E402

from repleafgbm import RepLeafDataset, RepLeafRegressor  # noqa: E402

N_FEATURES = 10
NO_PROJECTION = 1_000_000  # max_leaf_emb_dim large enough to disable projection


@dataclass
class RunResult:
    label: str
    emb_dim: int | None
    rmse: list[float]
    best_iter: list[int]
    linear_frac: list[float]
    fit_seconds: list[float]


def make_dataset(name: str, n_rows: int, seed: int):
    rng = np.random.default_rng(seed)
    if name == "piecewise_linear":
        X, signal = synthetic_tabular(n_rows, N_FEATURES, rng)
        y = signal + rng.normal(0.0, 0.3, n_rows)
    elif name == "friedman1":
        X, y = make_friedman1(n_samples=n_rows, n_features=N_FEATURES,
                              noise=1.0, random_state=seed)
    else:
        raise ValueError(name)
    return X, y


def linear_leaf_fraction(booster) -> float:
    """Fraction of leaves (up to best_iteration) that kept a linear model."""
    n = booster.best_iteration_ or booster.n_trees
    total = linear = 0
    for lv in booster.leaf_values_[:n]:
        total += len(lv.bias)
        if lv.weights.shape[1] > 0:
            linear += int(np.any(lv.weights != 0.0, axis=1).sum())
    return linear / max(total, 1)


def run_config(label: str, params: dict, dataset_name: str, seeds: list[int],
               n_train: int, n_valid: int, n_test: int) -> RunResult:
    res = RunResult(label, None, [], [], [], [])
    for seed in seeds:
        X, y = make_dataset(dataset_name, n_train + n_valid + n_test, seed)
        Xtr, ytr = X[:n_train], y[:n_train]
        Xva, yva = X[n_train:n_train + n_valid], y[n_train:n_train + n_valid]
        Xte, yte = X[n_train + n_valid:], y[n_train + n_valid:]

        model = RepLeafRegressor(
            n_estimators=300,
            learning_rate=0.1,
            num_leaves=16,
            min_samples_leaf=20,
            max_bins=256,
            early_stopping_rounds=25,
            random_state=seed,
            **params,
        )
        train = RepLeafDataset(Xtr, ytr)
        valid = RepLeafDataset(Xva, yva, metadata=train.metadata)
        t0 = time.perf_counter()
        model.fit(train, eval_set=[valid])
        res.fit_seconds.append(time.perf_counter() - t0)

        pred = model.predict(Xte)
        res.rmse.append(float(np.sqrt(np.mean((pred - yte) ** 2))))
        res.best_iter.append(model.best_iteration_ or model.booster_.n_trees)
        res.linear_frac.append(
            linear_leaf_fraction(model.booster_) if model.encoder_ is not None else 0.0
        )
        res.emb_dim = model.encoder_.output_dim if model.encoder_ is not None else 0
    return res


def build_grid() -> list[tuple[str, dict]]:
    grid: list[tuple[str, dict]] = [
        ("constant", {"leaf_model": "constant"}),
    ]
    for l2 in (0.3, 3.0):
        grid.append((
            f"identity l2={l2}",
            {"leaf_model": "embedded_linear", "encoder": "identity",
             "max_leaf_emb_dim": NO_PROJECTION, "l2_leaf": l2},
        ))
    for n_bins in (4, 8, 16):
        for proj in (8, 32, None):
            for l2 in (0.3, 3.0):
                label = f"plr bins={n_bins} proj={proj or 'off'} l2={l2}"
                grid.append((
                    label,
                    {"leaf_model": "embedded_linear", "encoder": "plr",
                     "encoder_params": {"n_bins": n_bins},
                     "max_leaf_emb_dim": proj or NO_PROJECTION, "l2_leaf": l2},
                ))
    return grid


def fmt_row(r: RunResult) -> str:
    rmse_m, rmse_s = np.mean(r.rmse), np.std(r.rmse)
    return (
        f"| {r.label} | {r.emb_dim} | {rmse_m:.4f} ± {rmse_s:.4f} "
        f"| {np.mean(r.best_iter):.0f} | {np.mean(r.linear_frac):.2f} "
        f"| {np.mean(r.fit_seconds):.1f} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--n-train", type=int, default=4000)
    parser.add_argument("--n-valid", type=int, default=1500)
    parser.add_argument("--n-test", type=int, default=4000)
    args = parser.parse_args()
    seeds = list(range(args.seeds))

    grid = build_grid()
    out_lines = [
        "# Experiment: PLR + random projection gap",
        "",
        "Auto-generated by `experiments/plr_projection_gap.py`. "
        "See the Analysis section at the bottom for conclusions.",
        "",
        f"Settings: n_train={args.n_train}, n_valid={args.n_valid}, "
        f"n_test={args.n_test}, n_features={N_FEATURES}, seeds={seeds}, "
        "n_estimators=300, lr=0.1, num_leaves=16, min_samples_leaf=20, "
        "early_stopping_rounds=25, eval_metric=rmse.",
        "",
        "Columns: emb_dim = embedding dimension after optional projection; "
        "linear% = fraction of leaves that kept a linear model (rest fell "
        "back to constants); best_it = early-stopped iteration.",
    ]

    for dataset_name in ("piecewise_linear", "friedman1"):
        print(f"=== dataset: {dataset_name} ===")
        results: list[RunResult] = []
        for label, params in grid:
            r = run_config(label, params, dataset_name, seeds,
                           args.n_train, args.n_valid, args.n_test)
            results.append(r)
            print(f"  {label:34s} rmse={np.mean(r.rmse):.4f} "
                  f"best_it={np.mean(r.best_iter):.0f} "
                  f"linear%={np.mean(r.linear_frac):.2f}")
        results.sort(key=lambda r: np.mean(r.rmse))
        out_lines += [
            "",
            f"## Dataset: {dataset_name}",
            "",
            "| config | emb_dim | test RMSE (mean ± std) | best_it | linear% | fit[s] |",
            "|---|---|---|---|---|---|",
            *[fmt_row(r) for r in results],
        ]

    out_path = Path(__file__).resolve().parent / "results" / "plr_projection_gap.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines) + "\n")
    print(f"\nreport written to {out_path}")


if __name__ == "__main__":
    main()
