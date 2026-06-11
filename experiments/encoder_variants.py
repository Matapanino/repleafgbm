"""Experiment: encoder variants for the embedded-linear leaf model.

Follow-up to experiments/results/plr_projection_gap.md, which showed that
plain PLR (clipped piecewise-linear basis) cannot extrapolate and loses to
identity embeddings on friedman1. Candidates tested here:

* ``plr`` with an appended per-feature linear term (extrapolation fix),
* ``periodic`` — PBLD-style frozen sinusoidal features + linear term
  (RealMLP-inspired; random frequencies instead of learned ones),
* the previous ``plr`` (no linear) / ``identity`` / ``constant`` baselines.

Run from the repository root (a few minutes):
    python3 experiments/encoder_variants.py
Results are written to experiments/results/encoder_variants.md.
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


@dataclass
class RunResult:
    label: str
    emb_dim: int | None
    rmse: list[float]
    best_iter: list[int]
    fit_seconds: list[float]


def make_dataset(name: str, n_rows: int, seed: int):
    rng = np.random.default_rng(seed)
    if name == "piecewise_linear":
        X, signal = synthetic_tabular(n_rows, N_FEATURES, rng)
        y = signal + rng.normal(0.0, 0.3, n_rows)
    elif name == "friedman1":
        X, y = make_friedman1(n_samples=n_rows, n_features=N_FEATURES,
                              noise=1.0, random_state=seed)
    elif name == "periodic_mix":
        # Oscillation + regime jump + linear backbone: the periodic encoder's
        # home turf, with enough tree-friendly structure to stay realistic.
        X = rng.normal(size=(n_rows, N_FEATURES))
        y = (
            2.0 * np.sin(3.0 * X[:, 0])
            + 3.0 * (X[:, 1] > 0.5)
            + 1.5 * X[:, 2]
            + rng.normal(0.0, 0.3, n_rows)
        )
    else:
        raise ValueError(name)
    return X, y


def build_grid() -> list[tuple[str, dict]]:
    emb = {"leaf_model": "embedded_linear", "max_leaf_emb_dim": 1_000_000}
    return [
        ("constant", {"leaf_model": "constant"}),
        ("identity", {**emb, "encoder": "identity"}),
        ("plr4 (no linear)", {**emb, "encoder": "plr",
                              "encoder_params": {"n_bins": 4, "add_linear": False}}),
        ("plr4 + linear", {**emb, "encoder": "plr",
                           "encoder_params": {"n_bins": 4, "add_linear": True}}),
        ("plr8 + linear", {**emb, "encoder": "plr",
                           "encoder_params": {"n_bins": 8, "add_linear": True}}),
        ("periodic k=4 (no linear)", {**emb, "encoder": "periodic",
                                      "encoder_params": {"n_frequencies": 4,
                                                         "add_linear": False}}),
        ("periodic k=4 + linear", {**emb, "encoder": "periodic",
                                   "encoder_params": {"n_frequencies": 4}}),
        ("periodic k=8 + linear", {**emb, "encoder": "periodic",
                                   "encoder_params": {"n_frequencies": 8}}),
    ]


def run_config(label: str, params: dict, dataset_name: str, seeds: list[int],
               n_train: int, n_valid: int, n_test: int) -> RunResult:
    res = RunResult(label, None, [], [], [])
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
            l2_leaf=1.0,
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
        res.emb_dim = model.encoder_.output_dim if model.encoder_ is not None else 0
    return res


def fmt_row(r: RunResult) -> str:
    return (
        f"| {r.label} | {r.emb_dim} | {np.mean(r.rmse):.4f} ± {np.std(r.rmse):.4f} "
        f"| {np.mean(r.best_iter):.0f} | {np.mean(r.fit_seconds):.1f} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--n-train", type=int, default=4000)
    parser.add_argument("--n-valid", type=int, default=1500)
    parser.add_argument("--n-test", type=int, default=4000)
    args = parser.parse_args()
    seeds = list(range(args.seeds))

    out_lines = [
        "# Experiment: encoder variants (PLR linear term, PBLD-style periodic)",
        "",
        "Auto-generated by `experiments/encoder_variants.py`. "
        "See the Analysis section at the bottom for conclusions.",
        "",
        f"Settings: n_train={args.n_train}, n_valid={args.n_valid}, "
        f"n_test={args.n_test}, n_features={N_FEATURES}, seeds={seeds}, "
        "n_estimators=300, lr=0.1, num_leaves=16, min_samples_leaf=20, "
        "l2_leaf=1.0, early_stopping_rounds=25, no projection.",
    ]

    for dataset_name in ("piecewise_linear", "friedman1", "periodic_mix"):
        print(f"=== dataset: {dataset_name} ===")
        results: list[RunResult] = []
        for label, params in build_grid():
            r = run_config(label, params, dataset_name, seeds,
                           args.n_train, args.n_valid, args.n_test)
            results.append(r)
            print(f"  {label:28s} rmse={np.mean(r.rmse):.4f} "
                  f"best_it={np.mean(r.best_iter):.0f}")
        results.sort(key=lambda r: np.mean(r.rmse))
        out_lines += [
            "",
            f"## Dataset: {dataset_name}",
            "",
            "| config | emb_dim | test RMSE (mean ± std) | best_it | fit[s] |",
            "|---|---|---|---|---|",
            *[fmt_row(r) for r in results],
        ]

    out_path = Path(__file__).resolve().parent / "results" / "encoder_variants.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines) + "\n")
    print(f"\nreport written to {out_path}")


if __name__ == "__main__":
    main()
