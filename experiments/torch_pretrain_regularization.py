"""Experiment: pretraining regularization for learned encoders (Phase 14b).

Phase 14 (experiments/results/real_data_validation.md) found that the
learned encoders overfit on all four real datasets with a uniform signature
(lowest train metric, highest test metric) and recorded one plausible fix
before blaming the architecture: regularize the supervised pretraining loop
with validation early stopping and weight decay. Those knobs now exist on
``torch_periodic`` / ``torch_plr`` (``weight_decay``, ``val_fraction``,
``patience``; conservative defaults on). This experiment measures whether
they change the Phase 14 picture:

* real datasets (california / house_sales / diamonds / adult), identical
  harness settings to Phases 6-14: does regularization close the gap to
  ``identity``?
* synthetic ``periodic_mix`` (the Phase 13 home-turf win): does
  regularization give back the win that motivated learned encoders?

Arms per encoder: ``regularized`` (the new defaults) vs ``unregularized``
(weight_decay=0, patience=0 — the Phase 14 condition re-measured under the
same code path), plus the ``identity`` reference. The mean number of
pretraining epochs actually run is reported as the early-stopping
diagnostic.

Run from the repository root (lightgbm not required, torch required;
~30-60 minutes):
    python3 experiments/torch_pretrain_regularization.py [--seeds K] [--max-rows N]
Results are written to experiments/results/torch_pretrain_regularization.md.
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
REG_OFF = {"weight_decay": 0.0, "val_fraction": 0.0, "patience": 0}


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
        ("torch_periodic regularized", {**emb, "encoder": "torch_periodic"}),
        ("torch_periodic unregularized", {**emb, "encoder": "torch_periodic",
                                          "encoder_params": dict(REG_OFF)}),
        ("torch_plr regularized", {**emb, "encoder": "torch_plr"}),
        ("torch_plr unregularized", {**emb, "encoder": "torch_plr",
                                     "encoder_params": dict(REG_OFF)}),
    ]


def make_periodic_mix(n_rows: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    # Same generator as experiments/encoder_variants.py (Phase 13 home turf).
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_rows, 10))
    y = (
        2.0 * np.sin(3.0 * X[:, 0])
        + 3.0 * (X[:, 1] > 0.5)
        + 1.5 * X[:, 2]
        + rng.normal(0.0, 0.3, n_rows)
    )
    return X, y


def run_dataset(name: str, max_rows: int, seeds: list[int]) -> tuple[dict, str, int]:
    if name == "periodic_mix":
        task = "regression"
        cats: list[str] = []
        # Phase 13 home-turf condition (experiments/encoder_variants.py):
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
        if name == "periodic_mix":
            X_all, y_all = make_periodic_mix(9_500, seed)
            n_used = len(X_all)
            i_tr, i_va = np.arange(4_000), np.arange(4_000, 5_500)
            i_te = np.arange(5_500, 9_500)
        else:
            rng = np.random.default_rng(seed)
            idx = rng.permutation(len(X_all))[: min(max_rows, len(X_all))]
            n_used = len(idx)
            n_tr, n_va = int(n_used * 0.55), int(n_used * 0.20)
            i_tr, i_va, i_te = (idx[:n_tr], idx[n_tr:n_tr + n_va],
                                idx[n_tr + n_va:])
        if name == "periodic_mix":
            Xtr, Xva, Xte = X_all[i_tr], X_all[i_va], X_all[i_te]
        else:
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
            print(f"  seed={seed} {label:32s} {score.__name__}="
                  f"{r.test[-1]:.4f} (train {r.train[-1]:.4f})"
                  + (f" epochs={epochs}" if epochs is not None else ""),
                  flush=True)
    return rows, task, n_used


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-rows", type=int, default=15_000)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--datasets", nargs="*", default=[
        "california", "house_sales", "diamonds", "adult", "periodic_mix",
    ])
    args = parser.parse_args()
    seeds = list(range(args.seeds))

    out_lines = [
        "# Experiment: pretraining regularization for learned encoders (Phase 14b)",
        "",
        "Auto-generated by `experiments/torch_pretrain_regularization.py`. "
        "See the Analysis section at the bottom for conclusions.",
        "",
        f"Settings: max_rows={args.max_rows} (55/20/25 train/valid/test), "
        f"seeds={seeds}, n_estimators=400, lr=0.1, num_leaves=31, "
        f"min_samples_leaf=20, early stopping {ES_ROUNDS} rounds, "
        "max_leaf_emb_dim=256. periodic_mix instead uses the Phase 13 "
        "condition: 4000/1500/4000 split, n_estimators=300, num_leaves=16, "
        "l2_leaf=1.0. 'regularized' = new encoder defaults "
        "(weight_decay=1e-3, val_fraction=0.15, patience=5); "
        "'unregularized' = the Phase 14 condition (weight_decay=0, "
        "patience=0) re-measured under the same code path. 'epochs' is the "
        "mean number of pretraining epochs actually run (budget 30).",
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

    out_path = Path(__file__).resolve().parent / "results" / "torch_pretrain_regularization.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines) + "\n")
    print(f"\nreport written to {out_path}", flush=True)


if __name__ == "__main__":
    main()
