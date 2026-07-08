"""Real-data validation harness (Phase 6).

Compares RepLeafGBM variants, RouterExtraction, LightGBM, and sklearn
HistGradientBoosting on standard OpenML/sklearn datasets, all with early
stopping on a shared validation split. Decision-oriented, not a leaderboard:

* (a) where do representation-conditioned leaves help on real data?
* (b) when they lose, is it missing regularization knobs (watch the
  train/test gap) or categorical handling (watch the "lightgbm native-cat"
  vs "lightgbm encoded" delta — both are recorded)?

Datasets (downloaded once, cached by scikit-learn in ~/scikit_learn_data):
  california (regression, numeric) | house_sales (regression, mixed) |
  diamonds (regression, 3 cats) | adult (binary, 8 cats).

Run from the repository root (lightgbm required):
    python3 benchmarks/benchmark_real_data.py [--max-rows N] [--seeds K]
Results are written to experiments/results/real_data_validation.md.
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

# macOS framework Python often lacks system CA certs for urllib; point it at
# certifi's bundle so the OpenML downloads work. No effect when already set.
try:
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except ImportError:  # pragma: no cover
    pass

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, fetch_openml
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import roc_auc_score

from repleafgbm import RepLeafClassifier, RepLeafDataset, RepLeafRegressor
from repleafgbm.external import (
    LightGBMExternalModel,
    RouterExtractionClassifier,
    RouterExtractionRegressor,
)

ES_ROUNDS = 25
MAX_CATEGORIES = 100  # drop id/date-like categorical columns above this


@dataclass
class Row:
    label: str
    test: list[float] = field(default_factory=list)
    train: list[float] = field(default_factory=list)
    auc: list[float] = field(default_factory=list)
    fit_s: list[float] = field(default_factory=list)


def _fetch(data_id: int):
    try:
        return fetch_openml(data_id=data_id, as_frame=True, parser="auto")
    except TypeError:  # older sklearn without parser=
        return fetch_openml(data_id=data_id, as_frame=True)


def load_dataset(name: str) -> tuple[pd.DataFrame, np.ndarray, str]:
    if name == "california":
        d = fetch_california_housing(as_frame=True)
        return d.data, d.target.to_numpy(np.float64), "regression"
    if name == "house_sales":
        d = _fetch(42731)
        X, y = d.data, d.target.to_numpy(np.float64)
        return X, np.log1p(y), "regression"  # heavy-tailed price -> log scale
    if name == "diamonds":
        d = _fetch(42225)
        return d.data, np.log1p(d.target.to_numpy(np.float64)), "regression"
    if name == "adult":
        d = _fetch(1590)
        # The pandas OpenML parser keeps ARFF quoting/whitespace in labels
        # (e.g. "' >50K'"); normalize before mapping to {0, 1}.
        labels = d.target.astype(str).str.strip(" '\"")
        y = (labels == ">50K").astype(np.float64).to_numpy()
        assert 0.0 < y.mean() < 1.0, "adult label mapping produced one class"
        return d.data, y, "binary"
    raise ValueError(name)


def clean_features(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
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


def logloss(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def run_dataset(name: str, max_rows: int, seeds: list[int],
                robust: bool = False,
                gate_margin_sweep: bool = False,
                grow_policy_sweep: bool = False) -> tuple[dict, str, int, int]:
    X_all, y_all, task = load_dataset(name)
    X_all, cats = clean_features(X_all)
    score = rmse if task == "regression" else logloss
    rows: dict[str, Row] = {}

    def record(label: str, model, predict, Xtr_in, ytr, Xte_in, yte, fit_s):
        p_te, p_tr = predict(model, Xte_in), predict(model, Xtr_in)
        r = rows.setdefault(label, Row(label))
        r.test.append(score(yte, p_te))
        r.train.append(score(ytr, p_tr))
        if task == "binary":
            r.auc.append(float(roc_auc_score(yte, p_te)))
        r.fit_s.append(fit_s)

    for seed in seeds:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(X_all))[: min(max_rows, len(X_all))]
        n = len(idx)
        n_tr, n_va = int(n * 0.55), int(n * 0.20)
        i_tr, i_va, i_te = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]
        Xtr, Xva, Xte = (X_all.iloc[i] for i in (i_tr, i_va, i_te))
        ytr, yva, yte = y_all[i_tr], y_all[i_va], y_all[i_te]

        train_ds = RepLeafDataset(Xtr, ytr, categorical_features=cats)
        valid_ds = RepLeafDataset(Xva, yva, metadata=train_ds.metadata)
        test_ds = RepLeafDataset(Xte, yte, metadata=train_ds.metadata)
        Xtr_e, Xva_e, Xte_e = (
            d.get_raw_features() for d in (train_ds, valid_ds, test_ds)
        )

        # --- sklearn HistGradientBoosting (internal early stopping) -------
        hgb_cls = (HistGradientBoostingRegressor if task == "regression"
                   else HistGradientBoostingClassifier)
        hgb = hgb_cls(max_iter=500, early_stopping=True, n_iter_no_change=ES_ROUNDS,
                      validation_fraction=0.15, random_state=seed)
        t0 = time.perf_counter()
        hgb.fit(Xtr_e, ytr)
        hgb_pred = ((lambda m, X: m.predict(X)) if task == "regression"
                    else (lambda m, X: m.predict_proba(X)[:, 1]))
        record("hist_gradient_boosting (encoded)", hgb, hgb_pred,
               Xtr_e, ytr, Xte_e, yte, time.perf_counter() - t0)

        # --- LightGBM on the same encoded matrix (shared with routerx) ----
        lgb_enc = LightGBMExternalModel(task=task, random_state=seed,
                                        n_estimators=500, learning_rate=0.05,
                                        num_leaves=31)
        t0 = time.perf_counter()
        lgb_enc.fit(Xtr_e, ytr, eval_set=[(Xva_e, yva)],
                    early_stopping_rounds=ES_ROUNDS)
        record("lightgbm (encoded, es)", lgb_enc,
               lambda m, X: m.predict_score(X),
               Xtr_e, ytr, Xte_e, yte, time.perf_counter() - t0)

        # --- LightGBM with native categorical handling ---------------------
        if cats:
            import lightgbm as lgb

            nat_cls = lgb.LGBMRegressor if task == "regression" else lgb.LGBMClassifier
            nat = nat_cls(n_estimators=500, learning_rate=0.05, num_leaves=31,
                          random_state=seed, verbose=-1)
            t0 = time.perf_counter()
            nat.fit(Xtr, ytr, eval_set=[(Xva, yva)],
                    callbacks=[lgb.early_stopping(ES_ROUNDS, verbose=False)])
            nat_pred = ((lambda m, X: m.predict(X)) if task == "regression"
                        else (lambda m, X: m.predict_proba(X)[:, 1]))
            record("lightgbm (native cat, es)", nat, nat_pred,
                   Xtr, ytr, Xte, yte, time.perf_counter() - t0)

        # --- RepLeaf variants ----------------------------------------------
        native_cls = RepLeafRegressor if task == "regression" else RepLeafClassifier
        rep_pred = ((lambda m, X: m.predict(X)) if task == "regression"
                    else (lambda m, X: m.predict_proba(X)[:, 1]))
        repleaf_configs = [
            ("RepLeaf constant (es)", dict(leaf_model="constant")),
            ("RepLeaf embedded_linear identity (es)",
             dict(leaf_model="embedded_linear", encoder="identity")),
            # Per-leaf adaptive constant<->embedded_linear gate (weighted-LOO)
            # at the default margin, plus the in-sample baseline arm that drops
            # the leverage correction (docs/proposals/adaptive-leaf-model.md).
            ("RepLeaf adaptive identity (es)",
             dict(leaf_model="adaptive", encoder="identity")),
            ("RepLeaf adaptive_insample identity (es)",
             dict(leaf_model="adaptive", encoder="identity", leaf_gate="insample")),
            # max_leaf_emb_dim raised to keep PLR unprojected — projection
            # degrades accuracy (experiments/results/plr_projection_gap.md).
            ("RepLeaf embedded_linear plr (es)",
             dict(leaf_model="embedded_linear", encoder="plr",
                  max_leaf_emb_dim=256)),
        ]
        try:  # learned encoders (Phase 14+) when the torch extra is present
            import torch  # noqa: F401

            repleaf_configs += [
                ("RepLeaf embedded torch_periodic (es)",
                 dict(leaf_model="embedded_linear", encoder="torch_periodic",
                      max_leaf_emb_dim=256)),
                ("RepLeaf embedded torch_plr (es)",
                 dict(leaf_model="embedded_linear", encoder="torch_plr",
                      max_leaf_emb_dim=256)),
                # v1.3.0: full rtdl PeriodicEmbeddings + interaction-aware MLP.
                ("RepLeaf embedded torch_periodic_plr (es)",
                 dict(leaf_model="embedded_linear", encoder="torch_periodic_plr",
                      max_leaf_emb_dim=256)),
                ("RepLeaf embedded torch_mlp (es)",
                 dict(leaf_model="embedded_linear", encoder="torch_mlp",
                      max_leaf_emb_dim=256)),
            ]
        except ImportError:
            pass
        # Robust regression objectives (constant leaf) — opt-in; most useful when
        # the target has outliers, recorded here as a reference on real data.
        if robust and task == "regression":
            repleaf_configs += [
                ("RepLeaf constant huber (es)",
                 dict(leaf_model="constant", objective="huber")),
                ("RepLeaf constant quantile(0.5) (es)",
                 dict(leaf_model="constant", objective="quantile")),
            ]
        if gate_margin_sweep:  # sensitivity around the 0.01 default
            repleaf_configs += [
                (f"RepLeaf adaptive m={m} identity (es)",
                 dict(leaf_model="adaptive", encoder="identity", leaf_gate_margin=m))
                for m in (0.0, 0.05)
            ]
        if grow_policy_sweep:
            # Capacity-matched grow_policy arms (ADR 0006): the stock arms
            # above are the leaf-wise free shape. Symmetric ignores num_leaves
            # (complete 2**5 tree); depthwise keeps the stock 31-leaf budget —
            # at num_leaves=32 it is identical to the capped leaf-wise shape.
            # All four datasets here are scalar tasks, so symmetric never
            # raises. Full sweep: experiments/grow_policy_real_data.py.
            for lm in ("constant", "adaptive"):
                repleaf_configs += [
                    (f"RepLeaf {lm} leafwise capped d5 (es)",
                     dict(leaf_model=lm, encoder="identity",
                          grow_policy="leafwise", num_leaves=32, max_depth=5)),
                    (f"RepLeaf {lm} depthwise d5 (es)",
                     dict(leaf_model=lm, encoder="identity",
                          grow_policy="depthwise", num_leaves=31, max_depth=5)),
                    (f"RepLeaf {lm} symmetric d5 (es)",
                     dict(leaf_model=lm, encoder="identity",
                          grow_policy="symmetric", num_leaves=32, max_depth=5)),
                ]
        for label, kwargs in repleaf_configs:
            # Arm kwargs override the shared defaults (the grow-policy arms set
            # their own num_leaves/max_depth capacity match).
            params = dict(n_estimators=400, learning_rate=0.1, num_leaves=31,
                          min_samples_leaf=20, early_stopping_rounds=ES_ROUNDS)
            params.update(kwargs)
            model = native_cls(random_state=seed, **params)
            t0 = time.perf_counter()
            model.fit(train_ds, eval_set=[valid_ds])
            record(label, model, rep_pred, train_ds, ytr, test_ds, yte,
                   time.perf_counter() - t0)

        # --- Router extraction on the shared encoded base -------------------
        routerx_cls = (RouterExtractionRegressor if task == "regression"
                       else RouterExtractionClassifier)
        model = routerx_cls(base=lgb_enc, leaf_model="embedded_linear",
                            encoder="identity", min_samples_leaf=20,
                            early_stopping_rounds=ES_ROUNDS, random_state=seed)
        t0 = time.perf_counter()
        model.fit(train_ds, eval_set=[valid_ds])
        record("routerx embedded_linear identity (es)", model, rep_pred,
               train_ds, ytr, test_ds, yte, time.perf_counter() - t0)

    return rows, task, len(idx), len(cats)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-rows", type=int, default=15_000)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--datasets", nargs="*", default=[
        "california", "house_sales", "diamonds", "adult",
    ])
    parser.add_argument("--robust", action="store_true",
                        help="also fit constant-leaf huber/quantile objectives on "
                             "the regression datasets (reference numbers)")
    parser.add_argument("--gate-margin-sweep", action="store_true",
                        help="also fit adaptive leaves at leaf_gate_margin in "
                             "{0.0, 0.05} (sensitivity around the 0.01 default)")
    parser.add_argument("--grow-policy-sweep", action="store_true",
                        help="also fit capacity-matched grow_policy arms "
                             "(leafwise capped / depthwise / symmetric at "
                             "max_depth=5; ADR 0006)")
    parser.add_argument("--out", type=str, default=None,
                        help="output markdown path; default is the canonical "
                             "experiments/results/real_data_validation.md (pass a "
                             "dated path to avoid clobbering its curated analysis)")
    args = parser.parse_args()
    seeds = list(range(args.seeds))
    warnings.filterwarnings("ignore", category=FutureWarning)
    # Benign LightGBM notices when eval_set reuses categorical params.
    warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

    out_lines = [
        "# Real-data validation (Phase 6)",
        "",
        "Auto-generated by `benchmarks/benchmark_real_data.py`. "
        "See the Analysis section at the bottom for conclusions.",
        "",
        f"Settings: max_rows={args.max_rows} (55/20/25 train/valid/test), "
        f"seeds={seeds}, early stopping {ES_ROUNDS} rounds everywhere "
        "(HistGB uses its internal validation split). Regression targets for "
        "house_sales/diamonds are log1p-transformed; metric: RMSE "
        "(log scale where noted) or logloss. 'encoded' = the ordinal-encoded "
        "float matrix RepLeafGBM uses; 'native cat' = LightGBM's own "
        "categorical handling on the raw frame.",
    ]

    for name in args.datasets:
        print(f"=== dataset: {name} ===", flush=True)
        rows, task, n_used, n_cats = run_dataset(name, args.max_rows, seeds,
                                                  robust=args.robust,
                                                  gate_margin_sweep=args.gate_margin_sweep,
                                                  grow_policy_sweep=args.grow_policy_sweep)
        ordered = sorted(rows.values(), key=lambda r: np.mean(r.test))
        metric_name = "rmse" if task == "regression" else "logloss"
        for r in ordered:
            print(f"  {r.label:42s} {metric_name}={np.mean(r.test):.4f} "
                  f"(train {np.mean(r.train):.4f})")
        header = "| config | test (mean ± std) | train | gap |"
        sep = "|---|---|---|---|"
        if task == "binary":
            header += " auc |"
            sep += "---|"
        header += " fit[s] |"
        sep += "---|"
        out_lines += [
            "",
            f"## {name} ({task}, n={n_used}, categorical features: {n_cats}, "
            f"metric: {metric_name})",
            "",
            header,
            sep,
        ]
        for r in ordered:
            line = (f"| {r.label} | {np.mean(r.test):.4f} ± {np.std(r.test):.4f} "
                    f"| {np.mean(r.train):.4f} "
                    f"| {np.mean(r.test) - np.mean(r.train):+.4f} |")
            if task == "binary":
                line += f" {np.mean(r.auc):.4f} |"
            line += f" {np.mean(r.fit_s):.1f} |"
            out_lines.append(line)

    out_path = (Path(args.out) if args.out else
                Path(__file__).resolve().parents[1] / "experiments" / "results"
                / "real_data_validation.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines) + "\n")
    print(f"\nreport written to {out_path}")


if __name__ == "__main__":
    main()
