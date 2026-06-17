"""Trainable-embeddings benchmark: compare encoder families on regression,
binary, and multiclass synthetic tasks.

It contrasts the fixed encoders (identity / plr / periodic / cross) with the
pretrained-then-frozen learned encoders (torch_periodic / torch_plr /
torch_periodic_plr / torch_mlp) and, when installed, external GBMs — so the
new ``torch_periodic_plr`` (full rtdl PeriodicEmbeddings) can be placed
alongside its peers honestly.

Run from the repository root:

    python3 benchmarks/trainable_embeddings.py --quick        # CPU smoke
    python3 benchmarks/trainable_embeddings.py --seeds 3      # fuller local run

The heavy full run is meant to be offloaded to a Colab VM via
``scripts/colab_trainable_embeddings.{sh,py}`` — not because pretraining is
GPU-accelerated (it runs on CPU, fit-time only) but for scale, isolation, and a
torch-equipped, reproducible environment. Artifacts are written under
``artifacts/trainable_embeddings/<date>/`` (metrics.jsonl, summary.md, env.json)
and a human summary at ``experiments/results/<date>-trainable-embeddings.md``.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import date
from pathlib import Path

import numpy as np
from common import external_gbm_models, synthetic_tabular
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    log_loss,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from repleafgbm import RepLeafClassifier, RepLeafRegressor

ROOT = Path(__file__).resolve().parents[1]

# (label, estimator kwargs). torch_* are gated on torch being importable.
FIXED_SPECS = [
    ("constant", {"leaf_model": "constant"}),
    ("raw_linear", {"leaf_model": "raw_linear"}),
    ("emb+identity", {"leaf_model": "embedded_linear", "encoder": "identity"}),
    ("emb+plr", {"leaf_model": "embedded_linear", "encoder": "plr"}),
    ("emb+periodic", {"leaf_model": "embedded_linear", "encoder": "periodic"}),
    ("emb+cross", {"leaf_model": "embedded_linear", "encoder": "cross"}),
]
TORCH_SPECS = [
    ("emb+torch_periodic", "torch_periodic"),
    ("emb+torch_plr", "torch_plr"),
    ("emb+torch_periodic_plr", "torch_periodic_plr"),
    ("emb+torch_mlp", "torch_mlp"),
]


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def make_tasks(n_train: int, n_test: int, n_features: int, seed: int):
    """Three synthetic tasks sharing one signal generator (common.py)."""
    rng = np.random.default_rng(seed)
    X, sig = synthetic_tabular(n_train + n_test, n_features, rng)
    Xtr, Xte = X[:n_train], X[n_train:]
    # regression: signal + noise
    y_reg = sig + rng.normal(0.0, 0.3, size=sig.shape[0])
    # binary: above-median signal
    y_bin = (sig > np.median(sig)).astype(int)
    # multiclass: tertiles of the signal
    edges = np.quantile(sig, [1 / 3, 2 / 3])
    y_mc = np.digitize(sig, edges)
    return {
        "regression": (Xtr, y_reg[:n_train], Xte, y_reg[n_train:]),
        "binary": (Xtr, y_bin[:n_train], Xte, y_bin[n_train:]),
        "multiclass": (Xtr, y_mc[:n_train], Xte, y_mc[n_train:]),
    }


def build_model(task: str, spec: dict, n_estimators: int, epochs: int, seed: int):
    kwargs = dict(spec)
    encoder = kwargs.get("encoder")
    common = dict(
        n_estimators=n_estimators, num_leaves=16, min_samples_leaf=20,
        learning_rate=0.1, max_leaf_emb_dim=64, random_state=seed,
    )
    if encoder is not None and encoder.startswith("torch_"):
        kwargs["encoder_params"] = {"n_epochs": epochs, "random_state": seed}
    cls = RepLeafRegressor if task == "regression" else RepLeafClassifier
    return cls(**common, **kwargs)


def evaluate(task: str, model, Xte, yte) -> dict[str, float]:
    if task == "regression":
        pred = model.predict(Xte)
        rmse = float(np.sqrt(mean_squared_error(yte, pred)))
        return {"rmse": rmse, "r2": float(r2_score(yte, pred))}
    proba = model.predict_proba(Xte)
    pred = model.predict(Xte)
    acc = float(accuracy_score(yte, pred))
    if task == "binary":
        return {
            "logloss": float(log_loss(yte, proba[:, 1], labels=[0, 1])),
            "auc": float(roc_auc_score(yte, proba[:, 1])),
            "accuracy": acc,
            "balanced_accuracy": float(balanced_accuracy_score(yte, pred)),
        }
    return {
        "multi_logloss": float(log_loss(yte, proba, labels=sorted(set(yte)))),
        "accuracy": acc,
    }


def diagnostics(model) -> dict:
    enc = getattr(model, "encoder_", None)
    out: dict = {}
    if enc is not None:
        try:
            out["output_dim"] = int(enc.output_dim)
        except Exception:  # noqa: BLE001 - diagnostics must never break a run
            pass
        ep = getattr(enc, "pretrain_epochs_used_", None)
        if ep is not None:
            out["pretrain_epochs"] = int(ep)
    return out


def run(args) -> list[dict]:
    specs = list(FIXED_SPECS)
    if _torch_available():
        specs += [(label, {"leaf_model": "embedded_linear", "encoder": enc})
                  for label, enc in TORCH_SPECS]
    else:
        print("torch not installed -> skipping torch_* encoders")

    rows: list[dict] = []
    for seed in range(args.seeds):
        tasks = make_tasks(args.n_train, args.n_test, args.n_features, args.seed + seed)
        for task, (Xtr, ytr, Xte, yte) in tasks.items():
            for label, spec in specs:
                model = build_model(task, spec, args.n_estimators, args.epochs, seed)
                t0 = time.perf_counter()
                model.fit(Xtr, ytr)
                fit_s = time.perf_counter() - t0
                rec = {
                    "task": task, "model": label, "seed": seed,
                    "fit_seconds": round(fit_s, 4),
                    **evaluate(task, model, Xte, yte), **diagnostics(model),
                }
                rows.append(rec)
                print(f"[{task:11s}] {label:24s} seed={seed} {fit_s:6.2f}s "
                      + " ".join(f"{k}={v:.4f}" for k, v in rec.items()
                                 if isinstance(v, float) and k != "fit_seconds"))
            for label, model in external_gbm_models(
                task, args.n_estimators, args.seed + seed,
            ):
                t0 = time.perf_counter()
                model.fit(Xtr, ytr)
                fit_s = time.perf_counter() - t0
                rows.append({
                    "task": task, "model": label, "seed": seed,
                    "fit_seconds": round(fit_s, 4),
                    **evaluate(task, model, Xte, yte),
                })
    return rows


def _agg_by_model(rows: list[dict], task: str) -> list[dict]:
    """Per-model aggregate for a task: each metric becomes a ``{mean, std, n}``
    dict so the summary can report seed spread, not just the mean."""
    by: dict[str, list[dict]] = {}
    for r in rows:
        if r["task"] == task:
            by.setdefault(r["model"], []).append(r)
    out = []
    for model, recs in by.items():
        keys = {k for rec in recs for k, v in rec.items()
                if isinstance(v, float) or k in ("output_dim", "pretrain_epochs")}
        agg: dict = {"model": model}
        for k in sorted(keys):
            vals = [rec[k] for rec in recs if k in rec]
            if vals:
                agg[k] = {"mean": float(np.mean(vals)),
                          "std": float(np.std(vals)), "n": len(vals)}
        out.append(agg)
    return out


def write_summary(rows: list[dict], out_dir: Path, report_path: Path, args) -> None:
    lines = [
        "# Trainable-embeddings benchmark",
        "",
        f"- Date: {date.today().isoformat()}",
        f"- Settings: seeds={args.seeds}, n_train={args.n_train}, "
        f"n_test={args.n_test}, n_features={args.n_features}, "
        f"n_estimators={args.n_estimators}, torch_epochs={args.epochs}, "
        f"quick={args.quick}",
        "- Synthetic signal mixes router-friendly discontinuities with "
        "encoder-friendly smooth/periodic structure (benchmarks/common.py).",
        "- Pretraining of torch_* encoders is CPU, fit-time only; numbers are "
        "indicative, not a leaderboard claim.",
        "",
    ]
    for task in ("regression", "binary", "multiclass"):
        agg = _agg_by_model(rows, task)
        if not agg:
            continue
        metric_cols = [k for k in sorted(agg[0]) if k != "model"]
        # stable, readable column order
        metric_cols = sorted(metric_cols, key=lambda c: (c == "fit_seconds", c))
        lines += [f"## {task} (mean ± std over {args.seeds} seed(s))", "",
                  "| model | " + " | ".join(metric_cols) + " |",
                  "| --- | " + " | ".join("---" for _ in metric_cols) + " |"]
        for row in agg:
            cells = []
            for c in metric_cols:
                cells.append(f"{row[c]['mean']:.4f} ± {row[c]['std']:.4f}"
                             if c in row else "")
            lines.append(f"| {row['model']} | " + " | ".join(cells) + " |")
        lines.append("")
    text = "\n".join(lines)
    (out_dir / "summary.md").write_text(text)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text)


def write_env(out_dir: Path) -> None:
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("numpy", "sklearn", "torch", "lightgbm", "xgboost", "catboost",
                "repleafgbm"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            versions[mod] = None
    (out_dir / "env.json").write_text(json.dumps(versions, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--n-train", type=int, default=6_000)
    p.add_argument("--n-test", type=int, default=3_000)
    p.add_argument("--n-features", type=int, default=8)
    p.add_argument("--n-estimators", type=int, default=100)
    p.add_argument("--epochs", type=int, default=20, help="torch pretraining epochs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--quick", action="store_true", help="small/fast smoke settings")
    args = p.parse_args()
    if args.quick:
        args.n_train, args.n_test = 2_000, 1_000
        args.n_estimators, args.epochs = 30, 5

    rows = run(args)

    out_dir = Path(args.out_dir) if args.out_dir else (
        ROOT / "artifacts" / "trainable_embeddings" / date.today().isoformat()
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "metrics.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    report = ROOT / "experiments" / "results" / (
        f"{date.today().isoformat()}-trainable-embeddings.md"
    )
    write_summary(rows, out_dir, report, args)
    write_env(out_dir)
    print(f"\nwrote {out_dir}/metrics.jsonl, summary.md, env.json")
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
