"""Experiment: (n, K) vector pretraining target for multiclass / multi-output.

Until now learned encoders pretrained on a *scalar* Newton residual, so for 3+
classes and multi-output regression ``_pretrain_target`` returned ``None`` and
the projection fit **unsupervised**. The synthetic trainable-embeddings sweep
(experiments/results/<date>-trainable-embeddings.md) showed that this was the
only reproducible loss signal: on multiclass ``torch_periodic == periodic``
(no benefit) because the projection never learned.

The encoders now pretrain on the full ``(n, K)`` negative-gradient residual
(``onehot - softmax(F0)`` for multiclass, ``Y - mean`` for multi-output;
docs/math.md). This experiment quantifies the limitation's removal by an
explicit **before / after** comparison on identical seeds and splits:

* ``before`` arms use experiment-local subclasses that override
  ``_pretrain_target`` to return ``None`` (the pre-change unsupervised path);
* ``after`` arms are the stock estimators (vector pretraining on).

Frozen baselines (``identity`` / ``periodic`` / ``plr``) and, for multiclass,
the external GBMs are included as references. Real OpenML multiclass datasets
(``wine``, ``vehicle``) plus a synthetic multi-output regression target are
covered; seeds default to 5 and every cell is reported as mean ± std.

Run from the repository root (torch required; a few minutes):
    OMP_NUM_THREADS=1 python3 experiments/vector_target_pretraining.py [--seeds K]
Results are written to experiments/results/<date>-vector-target-pretraining.md.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from common import external_gbm_models, synthetic_tabular  # noqa: E402
from openml_suite import clean_features, load_dataset, r2, rmse  # noqa: E402
from sklearn.metrics import accuracy_score, log_loss  # noqa: E402

from repleafgbm import RepLeafClassifier, RepLeafRegressor  # noqa: E402

# Real OpenML multiclass datasets (name, data_id) and the synthetic multi-output.
MULTICLASS = [("wine", 187), ("vehicle", 54)]
LEARNED = ("torch_periodic", "torch_plr", "torch_periodic_plr")


class _NoVectorPretrainClassifier(RepLeafClassifier):
    """Pre-change behavior: multiclass encoders fit unsupervised (the 'before'
    arm). Binary keeps the scalar residual, exactly like the stock estimator."""

    def _pretrain_target(self, dataset, sample_weight=None):
        if self.n_classes_ > 2:
            return None
        return super()._pretrain_target(dataset, sample_weight=sample_weight)


class _NoVectorPretrainRegressor(RepLeafRegressor):
    """Pre-change behavior: multi-output encoders fit unsupervised."""

    def _pretrain_target(self, dataset, sample_weight=None):
        if getattr(self, "n_outputs_", 1) > 1:
            return None
        return super()._pretrain_target(dataset, sample_weight=sample_weight)


@dataclass
class Row:
    label: str
    primary: list[float] = field(default_factory=list)
    secondary: list[float] = field(default_factory=list)
    fit_s: list[float] = field(default_factory=list)


def make_multioutput(n_rows: int, n_outputs: int, seed: int):
    """K periodic-plus-router regression targets sharing X: a router-friendly
    discontinuity, a linear backbone, and a per-output sinusoid (different
    frequency per output) so both raw routing and a learned periodic
    representation carry signal."""
    rng = np.random.default_rng(seed)
    X, _ = synthetic_tabular(n_rows, 8, rng)
    router = 3.0 * (X[:, 3] > 0.5)
    cols = [
        2.0 * np.sin((k + 2) * X[:, 0]) + 1.5 * X[:, 1] + router
        + rng.normal(0.0, 0.3, n_rows)
        for k in range(n_outputs)
    ]
    return X, np.column_stack(cols)


def repleaf_grid(task: str) -> list[tuple[str, type, dict]]:
    emb = {"leaf_model": "embedded_linear", "max_leaf_emb_dim": 256}
    after = RepLeafClassifier if task == "multiclass" else RepLeafRegressor
    before = (_NoVectorPretrainClassifier if task == "multiclass"
              else _NoVectorPretrainRegressor)
    grid: list[tuple[str, type, dict]] = [
        ("RepLeaf identity", after, {**emb, "encoder": "identity"}),
        ("RepLeaf periodic (frozen)", after, {**emb, "encoder": "periodic"}),
        ("RepLeaf plr (frozen)", after, {**emb, "encoder": "plr"}),
    ]
    for enc in LEARNED:
        grid.append((f"RepLeaf {enc} (before: unsupervised)", before,
                     {**emb, "encoder": enc}))
        grid.append((f"RepLeaf {enc} (after: vector pretrain)", after,
                     {**emb, "encoder": enc}))
    return grid


def _score_multiclass(model, Xte, yte, n_classes):
    proba = model.predict_proba(Xte)
    primary = float(log_loss(yte, proba, labels=np.arange(n_classes)))
    secondary = float(accuracy_score(yte, proba.argmax(1)))
    return primary, secondary


def _score_multioutput(model, Xte, Yte):
    pred = model.predict(Xte)
    primary = float(np.mean([rmse(Yte[:, k], pred[:, k]) for k in range(Yte.shape[1])]))
    secondary = float(np.mean([r2(Yte[:, k], pred[:, k]) for k in range(Yte.shape[1])]))
    return primary, secondary


def run_multiclass(name: str, data_id: int, seeds: list[int], args):
    X_all, y_all, _ = load_dataset(name, data_id, "multiclass")
    X_all, _cats = clean_features(X_all)  # wine/vehicle are all numeric -> []
    X_all = X_all.to_numpy(np.float64)
    n_classes = int(np.unique(y_all).size)
    rows: dict[str, Row] = {}
    n_used = len(X_all)
    for seed in seeds:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n_used)
        cut = int(n_used * 0.6)
        i_tr, i_te = idx[:cut], idx[cut:]
        Xtr, Xte, ytr, yte = X_all[i_tr], X_all[i_te], y_all[i_tr], y_all[i_te]

        for label, ncls, kwargs in repleaf_grid("multiclass"):
            ep = ({"encoder_params": {"n_epochs": args.epochs}}
                  if kwargs["encoder"].startswith("torch") else {})
            model = ncls(n_estimators=args.n_estimators, num_leaves=16,
                         min_samples_leaf=10, learning_rate=0.1,
                         random_state=seed, **kwargs, **ep)
            t0 = time.perf_counter()
            model.fit(Xtr, ytr)
            fit_s = time.perf_counter() - t0
            primary, secondary = _score_multiclass(model, Xte, yte, n_classes)
            r = rows.setdefault(label, Row(label))
            r.primary.append(primary)
            r.secondary.append(secondary)
            r.fit_s.append(fit_s)
            print(f"  seed={seed} {label:46s} mlogloss={primary:.4f} "
                  f"acc={secondary:.4f}", flush=True)

        for label, model in external_gbm_models("multiclass", args.n_estimators, seed):
            t0 = time.perf_counter()
            model.fit(Xtr, ytr)
            fit_s = time.perf_counter() - t0
            primary, secondary = _score_multiclass(model, Xte, yte, n_classes)
            r = rows.setdefault(label, Row(label))
            r.primary.append(primary)
            r.secondary.append(secondary)
            r.fit_s.append(fit_s)
            print(f"  seed={seed} {label:46s} mlogloss={primary:.4f} "
                  f"acc={secondary:.4f}", flush=True)
    return rows, "mlogloss", "accuracy", n_used, n_classes


def run_multioutput(seeds: list[int], args):
    rows: dict[str, Row] = {}
    n_used = args.n_rows
    for seed in seeds:
        X, Y = make_multioutput(args.n_rows, args.n_outputs, seed)
        cut = int(n_used * 0.6)
        Xtr, Xte, Ytr, Yte = X[:cut], X[cut:], Y[:cut], Y[cut:]
        for label, ncls, kwargs in repleaf_grid("multioutput"):
            ep = ({"encoder_params": {"n_epochs": args.epochs}}
                  if kwargs["encoder"].startswith("torch") else {})
            model = ncls(n_estimators=args.n_estimators, num_leaves=16,
                         min_samples_leaf=10, learning_rate=0.1,
                         random_state=seed, **kwargs, **ep)
            t0 = time.perf_counter()
            model.fit(Xtr, Ytr)
            fit_s = time.perf_counter() - t0
            primary, secondary = _score_multioutput(model, Xte, Yte)
            r = rows.setdefault(label, Row(label))
            r.primary.append(primary)
            r.secondary.append(secondary)
            r.fit_s.append(fit_s)
            print(f"  seed={seed} {label:46s} rmse={primary:.4f} "
                  f"r2={secondary:.4f}", flush=True)
    return rows, "rmse", "r2", n_used, args.n_outputs


def emit_table(out_lines: list[str], title: str, rows: dict[str, Row],
               primary_name: str, secondary_name: str) -> None:
    ordered = sorted(rows.values(), key=lambda r: np.mean(r.primary))
    out_lines += [
        "",
        f"## {title}",
        "",
        f"| model | {primary_name} (mean ± std) | {secondary_name} | fit[s] |",
        "|---|---|---|---|",
    ]
    for r in ordered:
        out_lines.append(
            f"| {r.label} | {np.mean(r.primary):.4f} ± {np.std(r.primary):.4f} "
            f"| {np.mean(r.secondary):.4f} ± {np.std(r.secondary):.4f} "
            f"| {np.mean(r.fit_s):.1f} |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--n-rows", type=int, default=3_000,
                        help="rows for the synthetic multi-output target")
    parser.add_argument("--n-outputs", type=int, default=3,
                        help="outputs for the synthetic multi-output target")
    parser.add_argument("--datasets", nargs="*",
                        default=["wine", "vehicle", "multioutput"])
    args = parser.parse_args()
    seeds = list(range(args.seeds))

    out_lines = [
        "# Experiment: (n, K) vector pretraining target (multiclass / multi-output)",
        "",
        "Auto-generated by `experiments/vector_target_pretraining.py`. "
        "`before` = `_pretrain_target` returns None (the pre-change unsupervised "
        "path); `after` = stock estimator (vector pretraining on). The two rows "
        "isolate the vector-target change on identical seeds/splits; everything "
        "else is held fixed.",
        "",
        f"Settings: seeds={seeds}, n_estimators={args.n_estimators}, lr=0.1, "
        f"num_leaves=16, min_samples_leaf=10, leaf_model=embedded_linear, "
        f"max_leaf_emb_dim=256, torch_epochs={args.epochs} (encoder defaults "
        "otherwise). Multiclass: 60/40 random train/test on real OpenML data, "
        "metric multi_logloss (lower is better). Multi-output: synthetic "
        f"router+periodic target, n_rows={args.n_rows}, n_outputs="
        f"{args.n_outputs}, metric mean per-output RMSE.",
    ]

    dmap = dict(MULTICLASS)
    for name in args.datasets:
        print(f"=== dataset: {name} ===", flush=True)
        if name == "multioutput":
            rows, pm, sm, n_used, k = run_multioutput(seeds, args)
            emit_table(out_lines,
                       f"multioutput (synthetic, n={n_used}, outputs={k}, "
                       f"metric: {pm})", rows, pm, sm)
        else:
            rows, pm, sm, n_used, k = run_multiclass(name, dmap[name], seeds, args)
            emit_table(out_lines,
                       f"{name} (multiclass, n={n_used}, classes={k}, "
                       f"metric: {pm})", rows, pm, sm)

    out_path = (Path(__file__).resolve().parent / "results"
                / f"{date.today().isoformat()}-vector-target-pretraining.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines) + "\n")
    print(f"\nreport written to {out_path}", flush=True)


if __name__ == "__main__":
    main()
