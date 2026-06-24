"""Prediction-traversal benchmark: routing (`Tree.apply`) vs leaf-eval.

Post-PR #30 the next CPU pressure point is prediction traversal. ``predict``
loops over every tree calling :meth:`Tree.apply` (a NumPy level-synchronous
router), and multiclass stores ``n_rounds * n_classes`` trees, so its predict
cost scales linearly in class count (docs/gpu_audit.md "Compiled Prediction
Path"). This harness decomposes a fitted model's ``predict`` into

    routing   = sum over predicting trees of  Tree.apply(X_raw)
    leaf_eval = sum over predicting trees of  LeafValues.predict(leaf_idx, Z)

so ``routing_share = routing / predict`` is the *ceiling* a future Rust
``apply_forest`` could remove. It is measurement only: it reuses the public
:meth:`Tree.apply` / :meth:`LeafValues.predict` on the fitted estimator and
changes no core behavior. The split is backend-independent (routing is pure
NumPy; leaf-eval uses the native ``predict_linear`` fast path regardless of
``split_backend``; only *fit* differs), so ``--backend rust`` only speeds up the
harness's own fits, not the numbers it reports.

It sweeps ``n_rows x n_estimators x task x leaf_model`` (plus one
categorical/missing worst-case that forces the per-level ``np.unique``
categorical loop in :meth:`Tree.apply`), writes one JSONL row per case plus a
regenerated ``summary.md``, and returns the rows from :func:`main` (like
``benchmarks/gpu_profile.py``). It reuses the data/estimator/env helpers from
:mod:`benchmarks.gpu_profile` and the synthetic signal from
:mod:`benchmarks.common` rather than reimplementing them.

    OMP_NUM_THREADS=1 python -m benchmarks.predict_profile --quick
    OMP_NUM_THREADS=1 RAYON_NUM_THREADS=8 python -m benchmarks.predict_profile \\
        --backend rust --size medium --sweep-trees 50 200 \\
        --sweep-rows 10000 50000 200000 \\
        --out artifacts/predict_bench/std/cases.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Allow ``python benchmarks/predict_profile.py`` (not just ``-m``) by making the
# repo root + src importable, mirroring the gpu_profile bootstrap.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from benchmarks.common import synthetic_tabular  # noqa: E402
from benchmarks.gpu_profile import (  # noqa: E402
    _SIZES,
    _peak_rss_bytes,
    _quality,
    build_data,
    build_estimator,
    collect_env,
)

from repleafgbm.data import RepLeafDataset  # noqa: E402


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #
def _best_seconds(fn, repeats: int) -> float:
    """Best (min) wall time over ``repeats`` runs after one warmup call."""
    fn()  # warmup: first-call native/import overhead excluded from the timing
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


# --------------------------------------------------------------------------- #
# Decomposition (measurement only — reuses public Tree.apply / LeafValues.predict)
# --------------------------------------------------------------------------- #
def _predicting(booster: Any) -> tuple[list, list, int | None]:
    """The trees/leaf-values prediction actually uses, and n_classes (or None).

    Scalar boosters count ``best_iteration_`` in trees; the multiclass booster
    counts it in rounds and stores ``n_rounds * n_classes`` trees round-major.
    """
    n_classes = getattr(booster, "n_classes", None)
    if n_classes is not None:
        n = (booster.best_iteration_ or booster.n_rounds) * n_classes
    else:
        n = booster.best_iteration_ or len(booster.trees_)
    return booster.trees_[:n], booster.leaf_values_[:n], n_classes


def _decompose(model: Any, X_test: Any, repeats: int) -> dict[str, Any]:
    """Split ``model``'s predict on ``X_test`` into routing vs leaf-eval.

    Mirrors ``sklearn._predict_raw`` for inputs (raw features + cached
    embeddings), then times :meth:`Tree.apply` and :meth:`LeafValues.predict`
    separately and validates the decomposition against ``booster.predict_raw``.
    """
    booster = model.booster_
    ds = RepLeafDataset(X_test, metadata=model.metadata_)
    X_raw = ds.get_raw_features()
    Z = ds.get_embeddings(model.encoder_) if model.encoder_ is not None else None
    trees, lvs, n_classes = _predicting(booster)
    n_rows = X_raw.shape[0]

    # Materialize leaf ids / outputs once for leaf-eval input + parity recon.
    leaf_idx = [t.apply(X_raw) for t in trees]
    leaf_out = [lv.predict(li, Z) for li, lv in zip(leaf_idx, lvs)]

    routing_s = _best_seconds(lambda: [t.apply(X_raw) for t in trees], repeats)
    leaf_eval_s = _best_seconds(
        lambda: [lv.predict(li, Z) for li, lv in zip(leaf_idx, lvs)], repeats
    )
    predict_s = _best_seconds(lambda: booster.predict_raw(X_raw, Z), repeats)

    # Reconstruct the additive score from the timed pieces using the task's
    # column rule (mirrors core/prediction.py) and check it against the real
    # path: confirms the benchmark traverses exactly the predicting trees.
    lr = booster.params.learning_rate
    if n_classes is not None:
        recon = np.tile(np.asarray(booster.init_score_, dtype=np.float64), (n_rows, 1))
        for i, ev in enumerate(leaf_out):
            recon[:, i % n_classes] += lr * ev
    else:
        recon = float(booster.init_score_) + lr * sum(leaf_out)
    ref = booster.predict_raw(X_raw, Z)
    parity = float(np.max(np.abs(np.asarray(recon) - np.asarray(ref))))

    return {
        "n_rows": int(n_rows),
        "n_trees": len(trees),
        "n_classes": int(n_classes) if n_classes is not None else 1,
        "routing_seconds": routing_s,
        "leaf_eval_seconds": leaf_eval_s,
        "predict_seconds": predict_s,
        "overhead_seconds": predict_s - routing_s - leaf_eval_s,
        "routing_share": routing_s / predict_s if predict_s > 0 else 0.0,
        "parity_max_abs_diff": parity,
    }


# --------------------------------------------------------------------------- #
# Estimator + data
# --------------------------------------------------------------------------- #
def _estimator(task: str, leaf_model: str, n_estimators: int, backend: str,
               args: argparse.Namespace) -> Any:
    """Public estimator for one refit config (reuses gpu_profile.build_estimator)."""
    est_args = argparse.Namespace(
        n_estimators=n_estimators,
        num_leaves=args.num_leaves,
        max_bins=args.max_bins,
        leaf_model=leaf_model,
        encoder=args.encoder,
        max_leaf_emb_dim=args.max_leaf_emb_dim,
        seed=args.seed,
    )
    return build_estimator(task, est_args, backend)


def build_categorical_data(n_train: int, n_test: int, n_features: int, n_cat: int,
                           seed: int):
    """Regression frame whose first ``n_cat`` columns are categorical + missing.

    The synthetic signal uses the first six raw columns, so bucketizing them into
    category labels (with 5% NaN) forces the tree to grow ``left_categories``
    subset splits — exercising the per-level ``np.unique`` categorical loop in
    :meth:`Tree.apply`, the worst-case routing path. Returns pandas frames so the
    public categorical input path (auto-detected ``category`` dtype) is used.
    """
    import pandas as pd

    rng = np.random.default_rng(seed + 1)
    n = n_train + n_test
    X, signal = synthetic_tabular(n, n_features, rng)
    y = signal + rng.normal(scale=0.1, size=n)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    for i in range(min(n_cat, n_features)):
        col = f"f{i}"
        vals = df[col].to_numpy()
        edges = np.quantile(vals, np.linspace(0.0, 1.0, 12)[1:-1])
        labels = np.array([f"c{c}" for c in np.digitize(vals, edges)], dtype=object)
        labels[rng.random(n) < 0.05] = None  # missing -> NaN category
        df[col] = pd.Series(labels, index=df.index, dtype="category")
    return (
        df.iloc[:n_train].reset_index(drop=True), y[:n_train],
        df.iloc[n_train:].reset_index(drop=True), y[n_train:],
    )


def _quality_once(task: str, model: Any, X_test: Any, y_test: np.ndarray,
                  n_classes: int, cap: int = 50_000) -> dict[str, float]:
    """Quality on a capped test slice (a non-degeneracy sanity, not a sweep axis)."""
    m = min(len(X_test), cap)
    Xc = X_test.iloc[:m] if hasattr(X_test, "iloc") else X_test[:m]
    return _quality(task, model, Xc, y_test[:m], n_classes)


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def _row(args: argparse.Namespace, task: str, leaf_model: str, n_est: int,
         dec: dict[str, Any], quality: dict[str, float]) -> dict[str, Any]:
    cls = dec["n_classes"]
    cls_tag = f"_c{cls}" if task.startswith("multiclass") else ""
    case_id = (f"{task}{cls_tag}_{leaf_model}_{n_est}est_{dec['n_trees']}t_"
               f"{dec['n_rows']}r_{args.backend}")
    return {
        "case_id": case_id,
        "task": task,
        "backend": args.backend,
        "n_classes": cls,
        "n_estimators": n_est,
        "n_trees": dec["n_trees"],
        "n_rows": dec["n_rows"],
        "leaf_model": leaf_model,
        "encoder": args.encoder,
        "routing_seconds": dec["routing_seconds"],
        "leaf_eval_seconds": dec["leaf_eval_seconds"],
        "predict_seconds": dec["predict_seconds"],
        "overhead_seconds": dec["overhead_seconds"],
        "routing_share": dec["routing_share"],
        "parity_max_abs_diff": dec["parity_max_abs_diff"],
        "quality": quality,
        "peak_rss_bytes": _peak_rss_bytes(),
        "env": collect_env(args.backend),
    }


def write_rows(out_path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the whole matrix fresh (overwrite), so reruns never duplicate rows."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def write_summary(out_path: Path) -> Path:
    """(Re)render a markdown decomposition table beside the JSONL."""
    rows = [json.loads(line) for line in out_path.read_text().splitlines() if line]
    lines = [
        "# Prediction-traversal benchmark summary",
        "",
        f"Auto-generated by `benchmarks/predict_profile.py` from `{out_path.name}`.",
        "",
        "| case_id | rows | trees | routing[s] | leaf_eval[s] | predict[s] | "
        "routing% | overhead[s] | parity | quality |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        q = ", ".join(f"{k}={v:.4g}" for k, v in r.get("quality", {}).items())
        lines.append(
            f"| {r['case_id']} | {r['n_rows']} | {r['n_trees']} | "
            f"{r['routing_seconds']:.4f} | {r['leaf_eval_seconds']:.4f} | "
            f"{r['predict_seconds']:.4f} | {100 * r['routing_share']:.1f} | "
            f"{r['overhead_seconds']:.4f} | {r['parity_max_abs_diff']:.1e} | {q} |"
        )
    summary = out_path.parent / "summary.md"
    summary.write_text("\n".join(lines) + "\n")
    return summary


def _print_case(r: dict[str, Any]) -> None:
    q = ", ".join(f"{k}={v:.4g}" for k, v in r["quality"].items())
    print(
        f"[{r['case_id']}] routing={r['routing_seconds']:.4f}s "
        f"leaf_eval={r['leaf_eval_seconds']:.4f}s predict={r['predict_seconds']:.4f}s "
        f"routing%={100 * r['routing_share']:.1f}  "
        f"parity={r['parity_max_abs_diff']:.1e}  {q}"
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="RepLeafGBM prediction-traversal benchmark (routing vs leaf-eval)"
    )
    p.add_argument("--size", choices=sorted(_SIZES), default="medium",
                   help="dataset preset (train rows + feature width)")
    p.add_argument("--backend", choices=["numpy", "rust", "cuda"], default="numpy",
                   help="split backend for the harness fits only; predict numbers "
                        "are backend-independent")
    p.add_argument("--tasks", nargs="+",
                   choices=["regression", "binary", "multiclass"],
                   default=["regression", "binary", "multiclass"])
    p.add_argument("--n-classes", type=int, default=5, help="multiclass class count")
    p.add_argument("--leaf-models", nargs="+",
                   choices=["constant", "embedded_linear", "raw_linear"],
                   default=["constant", "embedded_linear"])
    p.add_argument("--sweep-trees", type=int, nargs="+", default=[50, 200],
                   help="n_estimators per refit (= rounds for multiclass; total "
                        "predicting trees = rounds x n_classes)")
    p.add_argument("--sweep-rows", type=int, nargs="+",
                   default=[10_000, 50_000, 200_000],
                   help="predict-only inner sweep (test rows); no refit")
    p.add_argument("--encoder", default="identity")
    p.add_argument("--num-leaves", type=int, default=31)
    p.add_argument("--max-bins", type=int, default=256)
    p.add_argument("--max-leaf-emb-dim", type=int, default=64)
    p.add_argument("--repeats", type=int, default=5,
                   help="best-of-N wall-time repeats per timed region")
    p.add_argument("--no-categorical", dest="categorical", action="store_false",
                   help="skip the categorical/missing worst-case routing case")
    p.add_argument("--n-cat", type=int, default=8,
                   help="categorical columns in the categorical case")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true",
                   help="tiny matrix for CI/smoke (seconds)")
    p.add_argument("--n-train", type=int, default=None,
                   help="override training rows (default: --size preset)")
    p.add_argument("--n-features", type=int, default=None,
                   help="override feature count (default: --size preset)")
    p.add_argument("--out", type=Path,
                   default=Path("artifacts/predict_bench/cases.jsonl"),
                   help="JSONL output path (the whole matrix is rewritten)")
    return p


def _apply_quick(args: argparse.Namespace) -> None:
    """Collapse the matrix to a seconds-long CI smoke run (covers every shape)."""
    if not args.quick:
        return
    args.size = "small"
    args.n_train = 1_500
    args.n_features = 8
    args.tasks = ["regression", "binary", "multiclass"]
    args.leaf_models = ["constant", "embedded_linear"]
    args.sweep_trees = [20]
    args.sweep_rows = [1_500]
    args.n_classes = 3
    args.repeats = 2
    args.n_cat = 4


def _resolve_dims(args: argparse.Namespace) -> tuple[int, int]:
    preset_train, _, preset_features = _SIZES[args.size]
    n_train = args.n_train if args.n_train is not None else preset_train
    n_features = args.n_features if args.n_features is not None else preset_features
    return n_train, n_features


def main(argv: list[str] | None = None) -> list[dict[str, Any]]:
    args = build_parser().parse_args(argv)
    _apply_quick(args)
    n_train, n_features = _resolve_dims(args)
    max_rows = max(args.sweep_rows)
    rows: list[dict[str, Any]] = []

    # Numeric matrix: fit once per (task, leaf_model, n_estimators), then sweep
    # test rows on the predict side (no refit).
    for task in args.tasks:
        k = args.n_classes if task == "multiclass" else (2 if task == "binary" else 1)
        Xtr, ytr, Xte, yte = build_data(
            task, n_train, max_rows, n_features, k, args.seed
        )
        for leaf_model in args.leaf_models:
            for n_est in args.sweep_trees:
                model = _estimator(task, leaf_model, n_est, args.backend, args)
                model.fit(Xtr, ytr)
                quality = _quality_once(task, model, Xte, yte, k)
                for n_rows in args.sweep_rows:
                    dec = _decompose(model, Xte[:n_rows], args.repeats)
                    rows.append(_row(args, task, leaf_model, n_est, dec, quality))

    # Categorical/missing worst-case routing (regression, embedded_linear).
    if args.categorical:
        try:
            Xtr, ytr, Xte, yte = build_categorical_data(
                n_train, max_rows, n_features, args.n_cat, args.seed
            )
        except ImportError:  # pragma: no cover - pandas optional
            print("  (skipping categorical case: pandas not importable)")
        else:
            n_est = args.sweep_trees[-1]
            model = _estimator("regression", "embedded_linear", n_est,
                               args.backend, args)
            model.fit(Xtr, ytr)
            quality = _quality_once("regression", model, Xte, yte, 1)
            for n_rows in args.sweep_rows:
                dec = _decompose(model, Xte.iloc[:n_rows], args.repeats)
                rows.append(
                    _row(args, "regression_cat", "embedded_linear", n_est, dec, quality)
                )

    write_rows(args.out, rows)
    for r in rows:
        _print_case(r)
    summary = write_summary(args.out)
    print(f"  -> {args.out}  (summary: {summary})")
    return rows


if __name__ == "__main__":
    main()
