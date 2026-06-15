"""OpenML benchmark suite (Phase 25).

A reproducible, multi-dataset comparison of RepLeafGBM against the standard
gradient-boosting libraries (LightGBM, XGBoost, CatBoost) and scikit-learn's
HistGradientBoosting, across a curated set of OpenML tabular datasets. Unlike
``benchmark_real_data.py`` (a decision-oriented deep dive on four datasets),
this suite is a breadth-first **leaderboard snapshot** that substantiates the
README's accuracy claims with numbers anyone can regenerate.

Design choices for fairness and reproducibility:

* Every model is trained on the **same ordinal-encoded feature matrix** that
  RepLeafGBM uses (``RepLeafDataset.get_raw_features()``), so differences are
  in the model, not in categorical preprocessing. NaN is passed through (all
  the libraries here handle it).
* Fixed seed, a 60/20/20 train/valid/test split (**stratified by class** for
  classification, random for regression), and early stopping on the validation
  split for every model that supports it.
* Datasets are downloaded once and cached by scikit-learn in
  ``~/scikit_learn_data``. By default a failed download or a missing optional
  library is skipped rather than aborting the run; pass ``--strict`` for a
  release-grade run that **fails** if any required external GBM is missing or
  any model errors, so published numbers can't quietly omit a competitor.
* The report embeds a reproducibility manifest (package versions, OpenML
  dataset version anchors, seeds, split policy).

Run from the repository root (needs ``PYTHONPATH=src`` or an editable install;
LightGBM/XGBoost/CatBoost are optional ``[bench]`` extras)::

    PYTHONPATH=src python3 benchmarks/openml_suite.py [--quick] [--seeds K]
    PYTHONPATH=src python3 benchmarks/openml_suite.py --strict   # release run

Results are written to ``experiments/results/openml_benchmark.md``.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

# macOS framework Python often lacks system CA certs for urllib; certifi's
# bundle makes the OpenML downloads work. No effect when already set.
try:
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except ImportError:  # pragma: no cover
    pass

import numpy as np
from sklearn.datasets import fetch_california_housing, fetch_openml
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.preprocessing import LabelEncoder

from repleafgbm import RepLeafClassifier, RepLeafDataset, RepLeafRegressor

ES_ROUNDS = 25
MAX_CATEGORIES = 100

#: External GBMs that a release-grade (``--strict``) run must be able to train.
REQUIRED_GBMS = ("lightgbm", "xgboost", "catboost")

# name -> (openml data_id or None for a sklearn builtin, task)
DATASETS: list[tuple[str, int | None, str]] = [
    ("california", None, "regression"),
    ("house_sales", 42731, "regression"),
    ("diamonds", 42225, "regression"),
    ("wine_quality", 287, "regression"),
    ("credit_g", 31, "binary"),
    ("phoneme", 1489, "binary"),
    ("adult", 1590, "binary"),
    ("wine", 187, "multiclass"),
    ("vehicle", 54, "multiclass"),
]
QUICK_DATASETS = {"california", "diamonds", "credit_g", "phoneme", "wine"}


@dataclass
class Row:
    label: str
    primary: list[float] = field(default_factory=list)  # rmse or logloss
    secondary: list[float] = field(default_factory=list)  # r2/auc/accuracy
    fit_s: list[float] = field(default_factory=list)


def _fetch(data_id: int):
    try:
        return fetch_openml(data_id=data_id, as_frame=True, parser="auto")
    except TypeError:  # older sklearn without parser=
        return fetch_openml(data_id=data_id, as_frame=True)


def load_dataset(name: str, data_id: int | None, task: str):
    """Return (X DataFrame, y ndarray, task). Classification y is label-encoded
    to {0..k-1} so every library accepts it uniformly."""
    if name == "california":
        d = fetch_california_housing(as_frame=True)
        return d.data, d.target.to_numpy(np.float64), task
    d = _fetch(data_id)
    X = d.data
    if task == "regression":
        y = d.target.to_numpy(np.float64)
        if name in ("house_sales", "diamonds"):  # heavy-tailed price -> log
            y = np.log1p(y)
        return X, y, task
    # classification: normalize labels then encode
    labels = d.target.astype(str).str.strip(" '\"")
    y = LabelEncoder().fit_transform(labels)
    return X, y, task


def clean_features(X):
    """Cast object->category, drop id-like and over-wide categorical columns."""
    X = X.copy()
    drop = [c for c in X.columns if c.lower() in ("id",)]
    for c in X.columns:
        if str(X[c].dtype) in ("object", "category"):
            X[c] = X[c].astype("category")
            if X[c].cat.categories.size > MAX_CATEGORIES:
                drop.append(c)
    X = X.drop(columns=drop)
    cats = [c for c in X.columns if str(X[c].dtype) == "category"]
    return X, cats


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def r2(y, p):
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def _external_models(task: str, seed: int, strict: bool = False):
    """(label, builder) pairs for installed external GBMs. builder(n) -> model.

    In ``strict`` mode a missing required library is a hard error instead of a
    silent skip, so a release-grade run cannot quietly publish numbers with a
    competitor absent."""
    out = []
    missing = []
    try:
        import lightgbm as lgb

        def lgb_model(task=task):
            cls = lgb.LGBMRegressor if task == "regression" else lgb.LGBMClassifier
            return cls(n_estimators=600, learning_rate=0.05, num_leaves=31,
                       random_state=seed, verbose=-1)

        out.append(("lightgbm", lgb_model))
    except ImportError:
        missing.append("lightgbm")
    try:
        import xgboost as xgb

        def xgb_model(task=task):
            cls = xgb.XGBRegressor if task == "regression" else xgb.XGBClassifier
            return cls(n_estimators=600, learning_rate=0.05, max_depth=6,
                       early_stopping_rounds=ES_ROUNDS, random_state=seed,
                       tree_method="hist")

        out.append(("xgboost", xgb_model))
    except ImportError:
        missing.append("xgboost")
    try:
        import catboost as cb

        def cb_model(task=task):
            cls = cb.CatBoostRegressor if task == "regression" else cb.CatBoostClassifier
            return cls(iterations=600, learning_rate=0.05, depth=6,
                       random_seed=seed, verbose=False)

        out.append(("catboost", cb_model))
    except ImportError:
        missing.append("catboost")
    if strict and missing:
        raise RuntimeError(
            f"--strict run requires all external GBMs {list(REQUIRED_GBMS)} but "
            f"{missing} are not installed; `pip install \"repleafgbm[bench]\"`."
        )
    return out


def _fit_external(label, build, task, Xtr, ytr, Xva, yva):
    """Fit one external GBM with early stopping; return (proba_or_pred, fit_s)."""
    model = build()
    t0 = time.perf_counter()
    if label == "lightgbm":
        import lightgbm as lgb

        model.fit(Xtr, ytr, eval_set=[(Xva, yva)],
                  callbacks=[lgb.early_stopping(ES_ROUNDS, verbose=False)])
    elif label == "xgboost":
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    elif label == "catboost":
        model.fit(Xtr, ytr, eval_set=(Xva, yva),
                  early_stopping_rounds=ES_ROUNDS, use_best_model=True)
    fit_s = time.perf_counter() - t0
    return model, fit_s


def _score(task, model, X, y, classes):
    if task == "regression":
        return model.predict(X)
    return model.predict_proba(X)


def _split_indices(idx, y_sub, task, rng):
    """60/20/20 split of the (already random-subsampled) absolute indices
    ``idx``. Classification is **stratified** by class so every split keeps each
    label's proportion — important for the small/imbalanced datasets and the
    AUC/logloss metrics. Regression keeps the plain random split (``idx`` is
    already permuted)."""
    if task == "regression":
        n = len(idx)
        n_tr, n_va = int(n * 0.60), int(n * 0.20)
        return idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]
    tr, va, te = [], [], []
    for c in np.unique(y_sub):
        pos = np.where(y_sub == c)[0]
        rng.shuffle(pos)
        n_tr, n_va = int(len(pos) * 0.60), int(len(pos) * 0.20)
        tr.extend(idx[pos[:n_tr]])
        va.extend(idx[pos[n_tr:n_tr + n_va]])
        te.extend(idx[pos[n_tr + n_va:]])
    return (np.array(tr, dtype=int), np.array(va, dtype=int),
            np.array(te, dtype=int))


def run_dataset(name, data_id, task, max_rows, seeds, strict=False):
    X_all, y_all, task = load_dataset(name, data_id, task)
    X_all, cats = clean_features(X_all)
    classes = np.unique(y_all) if task != "regression" else None
    rows: dict[str, Row] = {}

    def record(label, primary, secondary, fit_s):
        r = rows.setdefault(label, Row(label))
        r.primary.append(primary)
        r.secondary.append(secondary)
        r.fit_s.append(fit_s)

    def metrics(y, pred_or_proba):
        if task == "regression":
            return rmse(y, pred_or_proba), r2(y, pred_or_proba)
        proba = pred_or_proba
        ll = float(log_loss(y, proba, labels=classes))
        pred = classes[np.argmax(proba, axis=1)]
        acc = float(accuracy_score(y, pred))
        if task == "binary":
            return ll, float(roc_auc_score(y, proba[:, 1]))
        return ll, acc

    for seed in seeds:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(X_all))[: min(max_rows, len(X_all))]
        i_tr, i_va, i_te = _split_indices(idx, y_all[idx], task, rng)
        Xtr, Xva, Xte = (X_all.iloc[i] for i in (i_tr, i_va, i_te))
        ytr, yva, yte = y_all[i_tr], y_all[i_va], y_all[i_te]

        train_ds = RepLeafDataset(Xtr, ytr, categorical_features=cats)
        valid_ds = RepLeafDataset(Xva, yva, metadata=train_ds.metadata)
        test_ds = RepLeafDataset(Xte, yte, metadata=train_ds.metadata)
        Xtr_e, Xva_e, Xte_e = (
            d.get_raw_features() for d in (train_ds, valid_ds, test_ds)
        )

        # --- RepLeafGBM variants (fit on the dataset objects) --------------
        native_cls = RepLeafRegressor if task == "regression" else RepLeafClassifier
        for label, kwargs in (
            ("RepLeaf constant", dict(leaf_model="constant")),
            ("RepLeaf embedded_linear", dict(leaf_model="embedded_linear",
                                             encoder="identity")),
        ):
            try:
                model = native_cls(
                    n_estimators=400, learning_rate=0.1, num_leaves=31,
                    min_samples_leaf=20, early_stopping_rounds=ES_ROUNDS,
                    random_state=seed, **kwargs,
                )
                t0 = time.perf_counter()
                model.fit(train_ds, eval_set=[valid_ds])
                fit_s = time.perf_counter() - t0
                pred = (model.predict(test_ds) if task == "regression"
                        else model.predict_proba(test_ds))
                p, s = metrics(yte, pred)
                record(label, p, s, fit_s)
            except Exception as exc:  # pragma: no cover - robustness
                if strict:
                    raise
                print(f"  [skip] {label} on {name}: {type(exc).__name__}: {exc}")

        # --- sklearn HistGradientBoosting (internal early stopping) --------
        try:
            hgb_cls = (HistGradientBoostingRegressor if task == "regression"
                       else HistGradientBoostingClassifier)
            hgb = hgb_cls(max_iter=600, learning_rate=0.05, early_stopping=True,
                          n_iter_no_change=ES_ROUNDS, validation_fraction=0.2,
                          random_state=seed)
            t0 = time.perf_counter()
            hgb.fit(Xtr_e, ytr)
            fit_s = time.perf_counter() - t0
            pred = _score(task, hgb, Xte_e, yte, classes)
            p, s = metrics(yte, pred)
            record("hist_gradient_boosting", p, s, fit_s)
        except Exception as exc:  # pragma: no cover
            if strict:
                raise
            print(f"  [skip] hist_gradient_boosting on {name}: {exc}")

        # --- external GBMs -------------------------------------------------
        for label, build in _external_models(task, seed, strict=strict):
            try:
                model, fit_s = _fit_external(label, build, task, Xtr_e, ytr,
                                             Xva_e, yva)
                pred = _score(task, model, Xte_e, yte, classes)
                p, s = metrics(yte, pred)
                record(label, p, s, fit_s)
            except Exception as exc:  # pragma: no cover
                if strict:
                    raise
                print(f"  [skip] {label} on {name}: {type(exc).__name__}: {exc}")

    return rows, task, len(idx), len(cats)


def _version_manifest(selected, seeds, args) -> list[str]:
    """Reproducibility block: exact package versions, dataset version anchors
    (the OpenML data_id pins a specific dataset version), seeds, and split
    policy. Embedded in the report so a published leaderboard can be tied to
    the environment that produced it."""
    def ver(dist):
        try:
            return importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            return "(not installed)"

    pkgs = ["numpy", "pandas", "scipy", "scikit-learn", "repleafgbm",
            "lightgbm", "xgboost", "catboost"]
    lines = [
        "## Reproducibility manifest",
        "",
        f"- Python: {platform.python_version()} ({sys.platform})",
        "- Packages: " + ", ".join(f"{p}={ver(p)}" for p in pkgs),
        f"- Seeds: {seeds}",
        f"- max_rows: {args.max_rows}; early stopping: {ES_ROUNDS} rounds",
        "- Split: 60/20/20 — stratified by class for classification, random "
        "for regression",
        f"- strict mode: {bool(args.strict)}",
        "- Datasets (version anchor = OpenML data_id):",
    ]
    for name, data_id, task in selected:
        src = "sklearn builtin" if data_id is None else f"openml data_id={data_id}"
        lines.append(f"  - {name} ({task}, {src})")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-rows", type=int, default=6000)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--quick", action="store_true",
                        help="fewer datasets and rows for a smoke run")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument(
        "--strict", action="store_true",
        help="release mode: fail (don't skip) if a required external GBM "
             f"{list(REQUIRED_GBMS)} is missing or any model errors",
    )
    args = parser.parse_args()
    seeds = list(range(args.seeds))
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    selected = DATASETS
    if args.quick:
        args.max_rows = min(args.max_rows, 2000)
        selected = [d for d in DATASETS if d[0] in QUICK_DATASETS]
    if args.datasets:
        selected = [d for d in DATASETS if d[0] in set(args.datasets)]

    out: list[str] = [
        "# OpenML benchmark suite (Phase 25)",
        "",
        "Auto-generated by `benchmarks/openml_suite.py`. Every model trains on "
        "the same ordinal-encoded feature matrix (RepLeafGBM's), fixed seed(s), "
        "a 60/20/20 split, and early stopping on the validation split. Primary "
        "metric: **RMSE** (regression, log1p target for house_sales/diamonds) "
        "or **logloss** (classification); secondary: R² / AUC / accuracy. "
        "Lower primary is better.",
        "",
        f"Settings: max_rows={args.max_rows}, seeds={seeds}, "
        f"early_stopping={ES_ROUNDS} rounds.",
        "",
        *_version_manifest(selected, seeds, args),
    ]
    # ranking accumulators per task family
    reg_ranks: dict[str, list[float]] = {}
    clf_ranks: dict[str, list[float]] = {}

    for name, data_id, task in selected:
        print(f"=== {name} ({task}) ===", flush=True)
        try:
            rows, task, n_used, n_cats = run_dataset(
                name, data_id, task, args.max_rows, seeds, strict=args.strict)
        except Exception as exc:
            if args.strict:
                raise
            print(f"  [skip dataset] {name}: {type(exc).__name__}: {exc}")
            out += ["", f"## {name} — skipped ({type(exc).__name__})"]
            continue
        ordered = sorted(rows.values(), key=lambda r: np.mean(r.primary))
        pmetric = "rmse" if task == "regression" else "logloss"
        smetric = {"regression": "r2", "binary": "auc",
                   "multiclass": "accuracy"}[task]
        for rank, r in enumerate(ordered):
            print(f"  {r.label:26s} {pmetric}={np.mean(r.primary):.4f} "
                  f"{smetric}={np.mean(r.secondary):.4f}")
            acc = reg_ranks if task == "regression" else clf_ranks
            acc.setdefault(r.label, []).append(rank + 1)
        out += [
            "",
            f"## {name} ({task}, n={n_used}, categorical: {n_cats})",
            "",
            f"| model | {pmetric} | {smetric} | fit[s] |",
            "|---|---|---|---|",
        ]
        for r in ordered:
            out.append(
                f"| {r.label} | {np.mean(r.primary):.4f} | "
                f"{np.mean(r.secondary):.4f} | {np.mean(r.fit_s):.1f} |"
            )

    # aggregate mean-rank tables
    for title, acc in (("Regression", reg_ranks), ("Classification", clf_ranks)):
        if not acc:
            continue
        out += ["", f"## Aggregate mean rank — {title} (lower is better)", "",
                "| model | mean rank | datasets |", "|---|---|---|"]
        for label, ranks in sorted(acc.items(), key=lambda kv: np.mean(kv[1])):
            out.append(f"| {label} | {np.mean(ranks):.2f} | {len(ranks)} |")

    out_path = (Path(__file__).resolve().parents[1] / "experiments" / "results"
                / "openml_benchmark.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out) + "\n")
    print(f"\nreport written to {out_path}")


if __name__ == "__main__":
    main()
