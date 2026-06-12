"""Experiment: interaction-aware encoders (Phase 16).

Phases 14/14b closed the per-feature learned-encoder line: regularized or
not, ``torch_periodic`` / ``torch_plr`` lose to ``identity`` on every real
dataset because they extract per-feature structure the router already
exploits. The surviving hypothesis is *cross-feature* structure: a leaf's
linear model can only use interactions if the representation carries them.
This experiment tests the first two interaction-aware encoders:

* ``cross`` — deterministic control: standardized features + the 16
  pairwise products most correlated with the initial Newton residual.
* ``torch_mlp`` — learned version: a small MLP (64 hidden, 16 outputs +
  linear passthrough) pretrained on the residual with the Phase 14b
  regularization, then frozen.

Conditions mirror Phases 6-14b: real datasets (california / house_sales /
diamonds / adult) under the standard harness, plus an ``interaction_mix``
synthetic home turf whose target is dominated by a product term that
axis-aligned routing plus per-feature-linear leaves cannot represent
cheaply.

Run from the repository root (torch required, lightgbm not):
    python3 experiments/encoder_interactions.py [--seeds K] [--max-rows N]
Results are written to experiments/results/encoder_interactions.md.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from benchmark_real_data import clean_features, load_dataset, logloss, rmse  # noqa: E402

from repleafgbm import RepLeafClassifier, RepLeafDataset, RepLeafRegressor  # noqa: E402

ES_ROUNDS = 25


@dataclass
class Row:
    label: str
    test: list[float] = field(default_factory=list)
    train: list[float] = field(default_factory=list)
    epochs: list[int] = field(default_factory=list)
    fit_s: list[float] = field(default_factory=list)


def build_grid() -> list[tuple[str, dict]]:
    emb = {"leaf_model": "embedded_linear", "max_leaf_emb_dim": 256}
    return [
        ("identity (reference)", {**emb, "encoder": "identity"}),
        ("cross (16 pairs)", {**emb, "encoder": "cross"}),
        ("torch_mlp", {**emb, "encoder": "torch_mlp"}),
    ]


def make_interaction_mix(n_rows: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic home turf: the dominant term is a feature product."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_rows, 10))
    y = (
        2.0 * X[:, 0] * X[:, 1]
        + 1.5 * X[:, 2]
        + 3.0 * (X[:, 3] > 0.5)
        + rng.normal(0.0, 0.3, n_rows)
    )
    return X, y


def run_dataset(name: str, max_rows: int, seeds: list[int]) -> tuple[dict, str, int]:
    if name == "interaction_mix":
        task = "regression"
        cats: list[str] = []
        # Mirrors the periodic_mix home-turf condition (Phase 13):
        # 4000/1500/4000 sequential split, num_leaves=16, l2_leaf=1.0.
        model_kwargs = dict(n_estimators=300, num_leaves=16, l2_leaf=1.0)
    else:
        X_all, y_all, task = load_dataset(name)
        X_all, cats = clean_features(X_all)
        model_kwargs = dict(n_estimators=400, num_leaves=31)
    score = rmse if task == "regression" else logloss
    rows: dict[str, Row] = {}
    n_used = 0

    for seed in seeds:
        if name == "interaction_mix":
            X_all, y_all = make_interaction_mix(9_500, seed)
            n_used = len(X_all)
            i_tr, i_va = np.arange(4_000), np.arange(4_000, 5_500)
            i_te = np.arange(5_500, 9_500)
            Xtr, Xva, Xte = X_all[i_tr], X_all[i_va], X_all[i_te]
        else:
            rng = np.random.default_rng(seed)
            idx = rng.permutation(len(X_all))[: min(max_rows, len(X_all))]
            n_used = len(idx)
            n_tr, n_va = int(n_used * 0.55), int(n_used * 0.20)
            i_tr, i_va, i_te = (idx[:n_tr], idx[n_tr:n_tr + n_va],
                                idx[n_tr + n_va:])
            Xtr, Xva, Xte = (X_all.iloc[i] for i in (i_tr, i_va, i_te))
        ytr, yva, yte = y_all[i_tr], y_all[i_va], y_all[i_te]

        train_ds = RepLeafDataset(Xtr, ytr, categorical_features=cats)
        valid_ds = RepLeafDataset(Xva, yva, metadata=train_ds.metadata)
        test_ds = RepLeafDataset(Xte, yte, metadata=train_ds.metadata)

        native_cls = RepLeafRegressor if task == "regression" else RepLeafClassifier
        predict = ((lambda m, X: m.predict(X)) if task == "regression"
                   else (lambda m, X: m.predict_proba(X)[:, 1]))
        for label, kwargs in build_grid():
            model = native_cls(
                learning_rate=0.1, min_samples_leaf=20,
                early_stopping_rounds=ES_ROUNDS,
                random_state=seed, **model_kwargs, **kwargs,
            )
            t0 = time.perf_counter()
            model.fit(train_ds, eval_set=[valid_ds])
            fit_s = time.perf_counter() - t0
            r = rows.setdefault(label, Row(label))
            r.test.append(score(yte, predict(model, test_ds)))
            r.train.append(score(ytr, predict(model, train_ds)))
            r.fit_s.append(fit_s)
            epochs = getattr(model.encoder_, "pretrain_epochs_used_", None)
            if epochs is not None:
                r.epochs.append(epochs)
            print(f"  seed={seed} {label:24s} {score.__name__}="
                  f"{r.test[-1]:.4f} (train {r.train[-1]:.4f})"
                  + (f" epochs={epochs}" if epochs is not None else ""),
                  flush=True)
    return rows, task, n_used


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-rows", type=int, default=15_000)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--datasets", nargs="*", default=[
        "california", "house_sales", "diamonds", "adult", "interaction_mix",
    ])
    args = parser.parse_args()
    seeds = list(range(args.seeds))

    out_lines = [
        "# Experiment: interaction-aware encoders (Phase 16)",
        "",
        "Auto-generated by `experiments/encoder_interactions.py`. "
        "See the Analysis section at the bottom for conclusions.",
        "",
        f"Settings: max_rows={args.max_rows} (55/20/25 train/valid/test), "
        f"seeds={seeds}, n_estimators=400, lr=0.1, num_leaves=31, "
        f"min_samples_leaf=20, early stopping {ES_ROUNDS} rounds, "
        "max_leaf_emb_dim=256. interaction_mix uses the home-turf condition "
        "(4000/1500/4000 split, n_estimators=300, num_leaves=16, "
        "l2_leaf=1.0). 'cross' = standardized features + 16 residual-"
        "correlated pairwise products; 'torch_mlp' = 64-hidden/16-output "
        "MLP + linear passthrough, pretrained on the initial residual with "
        "Phase 14b regularization. 'epochs' is the mean number of "
        "pretraining epochs actually run (budget 30).",
    ]

    for name in args.datasets:
        print(f"=== dataset: {name} ===", flush=True)
        rows, task, n_used = run_dataset(name, args.max_rows, seeds)
        metric = "rmse" if task == "regression" else "logloss"
        ordered = sorted(rows.values(), key=lambda r: np.mean(r.test))
        out_lines += [
            "",
            f"## {name} ({task}, n={n_used}, metric: {metric})",
            "",
            "| config | test (mean ± std) | train | gap | epochs | fit[s] |",
            "|---|---|---|---|---|---|",
        ]
        for r in ordered:
            epochs = f"{np.mean(r.epochs):.0f}" if r.epochs else "—"
            out_lines.append(
                f"| {r.label} | {np.mean(r.test):.4f} ± {np.std(r.test):.4f} "
                f"| {np.mean(r.train):.4f} "
                f"| {np.mean(r.test) - np.mean(r.train):+.4f} "
                f"| {epochs} | {np.mean(r.fit_s):.1f} |")

    out_path = Path(__file__).resolve().parent / "results" / "encoder_interactions.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines) + "\n")
    print(f"\nreport written to {out_path}", flush=True)


if __name__ == "__main__":
    main()
