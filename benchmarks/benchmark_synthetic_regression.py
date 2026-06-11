"""Synthetic regression benchmark.

Run from the repository root:
    python3 benchmarks/benchmark_synthetic_regression.py [--quick]
"""

from __future__ import annotations

import numpy as np
from common import (
    apply_quick,
    external_gbm_models,
    make_parser,
    print_table,
    synthetic_tabular,
    time_model,
)
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)

from repleafgbm import RepLeafRegressor


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def repleaf(leaf_model: str, encoder: str, n_estimators: int, seed: int) -> RepLeafRegressor:
    # Encoder settings (n_bins, max_leaf_emb_dim) are library defaults on
    # purpose: the benchmark tracks what users get out of the box.
    return RepLeafRegressor(
        n_estimators=n_estimators,
        learning_rate=0.1,
        num_leaves=31,
        min_samples_leaf=20,
        leaf_model=leaf_model,
        encoder=encoder,
        random_state=seed,
    )


def main() -> None:
    args = apply_quick(make_parser(__doc__).parse_args())
    rng = np.random.default_rng(args.seed)

    X, signal = synthetic_tabular(args.n_train + args.n_test, args.n_features, rng)
    y = signal + rng.normal(0.0, 0.3, len(signal))
    Xtr, Xte = X[: args.n_train], X[args.n_train :]
    ytr, yte = y[: args.n_train], y[args.n_train :]

    models: list[tuple[str, object]] = [
        ("sklearn GradientBoosting", GradientBoostingRegressor(
            n_estimators=args.n_estimators, random_state=args.seed)),
        ("sklearn HistGradientBoosting", HistGradientBoostingRegressor(
            max_iter=args.n_estimators, random_state=args.seed)),
        ("sklearn RandomForest", RandomForestRegressor(
            n_estimators=args.n_estimators, random_state=args.seed, n_jobs=-1)),
        ("RepLeaf constant", repleaf("constant", "identity", args.n_estimators, args.seed)),
        ("RepLeaf embedded_linear identity",
         repleaf("embedded_linear", "identity", args.n_estimators, args.seed)),
        ("RepLeaf embedded_linear plr",
         repleaf("embedded_linear", "plr", args.n_estimators, args.seed)),
    ]
    models += external_gbm_models("regression", args.n_estimators, args.seed)

    results = [
        time_model(name, m, Xtr, ytr, Xte, yte, {"rmse": rmse}, lambda m, X: m.predict(X))
        for name, m in models
    ]
    print(
        f"\nsynthetic regression: n_train={args.n_train} n_test={args.n_test} "
        f"n_features={args.n_features} n_estimators={args.n_estimators}\n"
    )
    print_table(results)


if __name__ == "__main__":
    main()
