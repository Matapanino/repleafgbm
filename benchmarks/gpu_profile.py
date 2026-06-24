"""GPU / native split-backend benchmark + profiling harness.

Runs one RepLeafGBM fit/predict case and appends a JSONL row matching the schema
in ``benchmarks/README_gpu.md`` (timings, quality, peak memory, per-fit transfer
volume, environment). It is the *measurement* harness the GPU acceleration
roadmap (``docs/gpu_roadmap.md`` Phase 0) calls for: it gathers the evidence that
justifies later kernel work (e.g. caching grad/hess on the device), without
changing any kernel, default, or public API.

Backends:

* ``--backend numpy`` / ``--backend rust`` run anywhere (CPU). ``transfer_bytes``
  is empty for them — only the CUDA backend tracks device transfers.
* ``--backend cuda`` requires CuPy + an NVIDIA GPU (run it on the Colab loop,
  ``scripts/colab_gpu_test.sh``). It reports the per-fit H2D/D2H byte counts read
  back from the fitted booster's split backend.

The harness reuses the synthetic signal, argparse base, and quick-mode helper
from :mod:`benchmarks.common`; it does not reimplement them.

Examples::

    python -m benchmarks.gpu_profile --task regression --size small \\
        --backend numpy --out artifacts/gpu_bench/dev/cases.jsonl
    python -m benchmarks.gpu_profile --task multiclass --n-classes 5 \\
        --size medium --backend cuda --out artifacts/gpu_bench/dev/cases.jsonl
    python -m benchmarks.gpu_profile --task multioutput --n-outputs 5 \\
        --size large --backend cuda --out artifacts/gpu_bench/dev/cases.jsonl

``phase_seconds`` is populated by the internal phase profiler: the harness sets
``REPLEAFGBM_PROFILE=1`` around the timed fit/predict and reads the breakdown
(preprocessing, encoder, binning, histogram, split scan, partition, leaf fit,
eval, predict) off the fitted estimator's ``phase_seconds_`` attribute
(``repleafgbm.core.profiling``). The profiler is off by default and never part
of the public API.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Allow ``python benchmarks/gpu_profile.py`` (not just ``-m``) by making the repo
# root importable for the sibling ``benchmarks`` package and ``src`` layout.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from benchmarks.common import (  # noqa: E402
    apply_quick,
    make_parser,
    multioutput_signal,
    synthetic_tabular,
)

# Size presets: (n_train, n_test, n_features). Mirrors docs/gpu_roadmap.md.
_SIZES: dict[str, tuple[int, int, int]] = {
    "small": (20_000, 10_000, 30),
    "medium": (100_000, 50_000, 100),
    "large": (500_000, 100_000, 200),
    "stress": (1_000_000, 200_000, 200),
}

# CUDA adaptive-scan threshold override env var. Set around the timed fit below;
# the cuda backend reads it once at construction (_resolve_scan_min_cells).
# numpy/rust backends ignore it (they have no adaptive GPU scan).
_SCAN_ENV = "REPLEAFGBM_CUDA_SCAN_MIN_CELLS"


def _parse_threshold(s: str) -> int:
    """Parse a scan-threshold CLI value: an int, or 'very_large' (force host)."""
    return 1_000_000_000 if s == "very_large" else int(s)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def build_data(
    task: str, n_train: int, n_test: int, n_features: int, n_classes: int,
    seed: int, *, n_outputs: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic (X_train, y_train, X_test, y_test) for the requested task.

    Regression uses the shared :func:`synthetic_tabular` signal plus light noise;
    binary thresholds the signal at its median; multiclass quantile-bins the
    signal into ``n_classes`` balanced classes; multioutput stacks ``n_outputs``
    correlated targets sharing one ``X`` (:func:`multioutput_signal`), so
    ``y`` is ``(n, n_outputs)`` and routes through the shared-routing booster.
    """
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    if task == "multioutput":
        X, signal = multioutput_signal(n, n_features, n_outputs, rng)
        y = signal + rng.normal(scale=0.1, size=signal.shape)  # (n, n_outputs)
        return X[:n_train], y[:n_train], X[n_train:], y[n_train:]
    X, signal = synthetic_tabular(n, n_features, rng)
    if task == "regression":
        y = signal + rng.normal(scale=0.1, size=n)
    elif task == "binary":
        y = (signal > np.median(signal)).astype(np.int64)
    elif task == "multiclass":
        edges = np.quantile(signal, np.linspace(0, 1, n_classes + 1)[1:-1])
        y = np.digitize(signal, edges).astype(np.int64)
    else:  # pragma: no cover - guarded by argparse choices
        raise ValueError(f"unknown task {task!r}")
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


# --------------------------------------------------------------------------- #
# Estimator
# --------------------------------------------------------------------------- #
def build_estimator(task: str, args: argparse.Namespace, backend: str) -> Any:
    """Construct the public estimator for ``task`` with the swept knobs."""
    from repleafgbm.classifier import RepLeafClassifier
    from repleafgbm.regressor import RepLeafRegressor

    common = dict(
        n_estimators=args.n_estimators,
        num_leaves=args.num_leaves,
        max_bins=args.max_bins,
        leaf_model=args.leaf_model,
        encoder=args.encoder,
        max_leaf_emb_dim=args.max_leaf_emb_dim,
        split_backend=backend,
        random_state=args.seed,
    )
    # Opt-in GPU encoder pretraining (v1.5.0): only the learned torch encoders
    # accept a ``device``; fixed encoders (identity/plr/...) would reject it.
    device = getattr(args, "device", None)
    if device and str(args.encoder).startswith("torch"):
        common["encoder_params"] = {"device": device}
    # Opt-in float32 wide-embedding leaf-fit (forwarded only when requested, so
    # the harness still works against an estimator that predates the param).
    precision = getattr(args, "leaf_fit_precision", None)
    if precision:
        common["leaf_fit_precision"] = precision
    # Multi-output uses the regressor too: a 2-D y auto-routes to the
    # shared-routing MultiOutputBooster inside the regressor wrapper.
    if task in ("regression", "multioutput"):
        return RepLeafRegressor(**common)
    return RepLeafClassifier(**common)


def _get_transfer_stats(model: Any) -> dict[str, int]:
    """Per-fit transfer counters from the fitted booster's split backend.

    Returns ``{}`` for non-CUDA backends (only the CUDA backend exposes
    ``get_transfer_stats``) or if the handle is unavailable. The backend is built
    fresh per fit, so its cumulative counters equal this fit's totals.
    """
    backend = getattr(getattr(model, "booster_", None), "split_backend_", None)
    getter = getattr(backend, "get_transfer_stats", None)
    return dict(getter()) if callable(getter) else {}


# --------------------------------------------------------------------------- #
# Quality + resources
# --------------------------------------------------------------------------- #
def _quality(task: str, model: Any, X_test: np.ndarray, y_test: np.ndarray,
             n_classes: int) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, log_loss, r2_score, roc_auc_score

    if task in ("regression", "multioutput"):
        # Multi-output predict/y_test are (n, K); rmse/mae average over all
        # elements and r2_score uniform-averages over outputs.
        pred = model.predict(X_test)
        err = pred - y_test
        return {
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mae": float(np.mean(np.abs(err))),
            "r2": float(r2_score(y_test, pred)),
        }
    proba = model.predict_proba(X_test)
    labels = model.predict(X_test)
    if task == "binary":
        return {
            "logloss": float(log_loss(y_test, proba[:, 1], labels=[0, 1])),
            "auc": float(roc_auc_score(y_test, proba[:, 1])),
            "accuracy": float(accuracy_score(y_test, labels)),
        }
    return {
        "multi_logloss": float(log_loss(y_test, proba, labels=np.arange(n_classes))),
        "accuracy": float(accuracy_score(y_test, labels)),
    }


def _peak_rss_bytes() -> int | None:
    """Process peak resident set size (high-water mark), or None if unavailable.

    Uses stdlib ``resource`` (no extra dependency); ``ru_maxrss`` is bytes on
    macOS and KiB on Linux. This is a cumulative process high-water mark, not a
    per-case delta — a coarse memory ceiling, adequate for ranking cases.
    """
    try:
        import resource
    except ImportError:  # pragma: no cover - non-unix
        return None
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss if sys.platform == "darwin" else rss * 1024)


def _peak_gpu_bytes(backend: str) -> int | None:
    if backend != "cuda":
        return None
    try:
        import cupy
    except ImportError:  # pragma: no cover - off-GPU
        return None
    # The default pool's high-water (total_bytes) is a stable proxy for peak GPU
    # allocation across the fit.
    return int(cupy.get_default_memory_pool().total_bytes())


def _parity_max_abs_diff(task: str, model: Any, args: argparse.Namespace,
                         X_train, y_train, X_test) -> float:
    """Max abs prediction diff vs a numpy-backend twin (sanity vs ~1e-6)."""
    twin = build_estimator(task, args, "numpy").fit(X_train, y_train)
    if task in ("regression", "multioutput"):
        a, b = model.predict(X_test), twin.predict(X_test)
    else:
        a, b = model.predict_proba(X_test), twin.predict_proba(X_test)
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:  # pragma: no cover - git missing / not a repo
        return None


def _version(dist: str) -> str | None:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


# Thread-pool env vars that gate BLAS/OpenMP parallelism. The wide-embedding
# leaf-fit phase is BLAS-bound, so a float64-vs-float32 delta is only
# attributable when the thread config is logged alongside it (see
# docs/perf-notes/harness-log.md).
_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
)


def _blas_vendor() -> str | None:
    """Best-effort NumPy BLAS backend name (openblas/mkl/accelerate/...).

    Dependency-free: tries the NumPy 1.25+ ``__config__.show(mode='dicts')`` API
    first, then the legacy ``*_opt_info`` dicts. Returns ``None`` if unknown.
    """
    cfg = getattr(np, "__config__", None)
    if cfg is None:
        return None
    show = getattr(cfg, "show", None)
    if callable(show):
        try:
            built = show(mode="dicts").get("Build Dependencies", {})
            name = built.get("blas", {}).get("name")
            if name:
                return str(name)
        except (TypeError, AttributeError, KeyError):
            pass
    for attr in ("blas_opt_info", "blas_ilp64_opt_info", "openblas_info"):
        info = getattr(cfg, attr, None)
        if isinstance(info, dict) and info.get("libraries"):
            return str(info["libraries"][0])
    return None


def collect_env(backend: str) -> dict[str, Any]:
    sha = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain")
    env: dict[str, Any] = {
        "git_sha": sha,
        "git_dirty": bool(dirty) if dirty is not None else None,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "threads": {v: os.environ.get(v) for v in _THREAD_ENV_VARS},
        "blas": _blas_vendor(),
        "packages": {
            p: _version(p)
            for p in ("numpy", "cupy", "scikit-learn", "repleafgbm",
                      "repleafgbm-native")
        },
    }
    if backend == "cuda":
        try:
            import cupy

            props = cupy.cuda.runtime.getDeviceProperties(0)
            name = props["name"]
            env["gpu"] = name.decode() if isinstance(name, bytes) else name
        except Exception:  # pragma: no cover - off-GPU
            env["gpu"] = None
    return env


# --------------------------------------------------------------------------- #
# Case runner
# --------------------------------------------------------------------------- #
def run_case(args: argparse.Namespace) -> dict[str, Any]:
    """Run one case and return its JSONL row dict (does not write it)."""
    task, backend = args.task, args.backend
    n_classes = args.n_classes if task == "multiclass" else (
        2 if task == "binary" else 1
    )
    n_outputs = args.n_outputs if task == "multioutput" else 1
    X_train, y_train, X_test, y_test = build_data(
        task, args.n_train, args.n_test, args.n_features, n_classes, args.seed,
        n_outputs=n_outputs
    )
    model = build_estimator(task, args, backend)

    # Enable the internal phase profiler for this measurement run (restored
    # afterwards). The per-phase breakdown is read off the fitted estimator's
    # ``phase_seconds_`` attribute; profiling is off outside this window. The
    # CUDA scan-threshold override (if any) is set the same way: the backend
    # reads it once at construction during the fit, so it must be live here.
    scan = getattr(args, "cuda_scan_min_cells", None)
    prev_env = {k: os.environ.get(k) for k in ("REPLEAFGBM_PROFILE", _SCAN_ENV)}
    os.environ["REPLEAFGBM_PROFILE"] = "1"
    if scan is not None:
        os.environ[_SCAN_ENV] = str(scan)
    try:
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        model.predict(X_test)
        predict_seconds = time.perf_counter() - t0
    finally:
        for key, val in prev_env.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val
    phase_seconds = dict(getattr(model, "phase_seconds_", {}))

    cls_tag = f"_c{n_classes}" if task == "multiclass" else ""
    out_tag = f"_k{n_outputs}" if task == "multioutput" else ""
    scan_tag = f"_scan{scan}" if scan is not None else ""
    row: dict[str, Any] = {
        "case_id":
            f"{task}{cls_tag}{out_tag}_{args.n_features}f_bins{args.max_bins}"
            f"{scan_tag}_{backend}",
        "task": task,
        "backend": backend,
        "n_classes": n_classes,
        "n_outputs": n_outputs,
        "n_train": args.n_train,
        "n_test": args.n_test,
        "n_features": args.n_features,
        "max_bins": args.max_bins,
        "num_leaves": args.num_leaves,
        "leaf_model": args.leaf_model,
        "encoder": args.encoder,
        "device": getattr(args, "device", None),
        "cuda_scan_min_cells": scan,
        "n_estimators": args.n_estimators,
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "quality": _quality(task, model, X_test, y_test, n_classes),
        "peak_rss_bytes": _peak_rss_bytes(),
        "peak_gpu_bytes": _peak_gpu_bytes(backend),
        "phase_seconds": phase_seconds,
        "transfer_bytes": _get_transfer_stats(model),
        "env": collect_env(backend),
    }
    if args.parity and backend != "numpy":
        row["parity_max_abs_diff"] = _parity_max_abs_diff(
            task, model, args, X_train, y_train, X_test
        )
    return row


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def write_row(out_path: Path, row: dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a") as fh:
        fh.write(json.dumps(row) + "\n")


def write_summary(out_path: Path) -> Path:
    """(Re)render a markdown table from every JSONL row beside the cases file."""
    rows = [json.loads(line) for line in out_path.read_text().splitlines() if line]
    lines = [
        "# GPU benchmark summary",
        "",
        f"Auto-generated by `benchmarks/gpu_profile.py` from `{out_path.name}`.",
        "",
        "| case_id | backend | fit[s] | pred[s] | quality | peak_gpu | "
        "grad/hess H2D | transfer total | phases[s] |",
        "|---|---|---:|---:|---|---:|---:|---:|---|",
    ]
    for r in rows:
        q = ", ".join(f"{k}={v:.4g}" for k, v in r.get("quality", {}).items())
        tb = r.get("transfer_bytes") or {}
        gh = tb.get("gradhess_h2d_bytes", 0)
        total = sum(v for k, v in tb.items() if k.endswith("_bytes"))
        peak_gpu = r.get("peak_gpu_bytes")
        ps = r.get("phase_seconds") or {}
        phases = ", ".join(
            f"{k}={v:.3g}" for k, v in sorted(ps.items(), key=lambda kv: -kv[1])
        ) or "-"
        lines.append(
            f"| {r['case_id']} | {r['backend']} | {r['fit_seconds']:.3f} | "
            f"{r['predict_seconds']:.3f} | {q} | "
            f"{peak_gpu if peak_gpu is not None else '-'} | {gh} | {total} | "
            f"{phases} |"
        )
    summary = out_path.parent / "summary.md"
    summary.write_text("\n".join(lines) + "\n")
    return summary


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = make_parser("RepLeafGBM GPU/native benchmark + profiling harness")
    p.add_argument("--task",
                   choices=["regression", "binary", "multiclass", "multioutput"],
                   default="regression")
    p.add_argument("--size", choices=sorted(_SIZES),
                   help="dataset preset; overrides --n-train/--n-test/--n-features")
    p.add_argument("--backend", choices=["numpy", "rust", "cuda"], default="numpy")
    p.add_argument("--n-classes", type=int, default=3, help="multiclass only")
    p.add_argument("--n-outputs", type=int, default=3, help="multioutput only")
    p.add_argument("--leaf-model", default="embedded_linear",
                   choices=["constant", "embedded_linear", "raw_linear"])
    p.add_argument("--encoder", default="identity",
                   help="encoder name (identity/plr/periodic/torch_periodic_plr/...)")
    p.add_argument("--device", default=None, choices=["cpu", "cuda", "auto"],
                   help="device for learned-encoder pretraining (torch encoders "
                        "only; v1.5.0). transform/serialization stay NumPy")
    p.add_argument("--max-bins", type=int, default=256)
    p.add_argument("--num-leaves", type=int, default=31)
    p.add_argument("--max-leaf-emb-dim", type=int, default=64)
    p.add_argument("--leaf-fit-precision", default=None,
                   choices=["float64", "float32_gram"],
                   help="opt-in wide-embedding (emb>64) leaf-fit Gram precision; "
                        "default None leaves the estimator default (float64)")
    p.add_argument(
        "--cuda-scan-min-cells", type=_parse_threshold, default=None, metavar="N",
        help="override the CUDA adaptive split-scan threshold for this run via "
             f"{_SCAN_ENV} (cuda backend only; an int, or 'very_large' to force "
             "the host path). Default: backend default (32768).",
    )
    p.add_argument(
        "--scan-min-cells-sweep", type=_parse_threshold, nargs="+", default=None,
        metavar="N",
        help="run one case per threshold (e.g. 0 8192 32768 131072 very_large); "
             "writes one JSONL row each. Overrides --cuda-scan-min-cells.",
    )
    p.add_argument("--parity", action="store_true",
                   help="also fit a numpy twin and record parity_max_abs_diff")
    p.add_argument("--out", type=Path, default=Path("artifacts/gpu_bench/cases.jsonl"),
                   help="JSONL output path (rows are appended)")
    return p


def _emit_case(args: argparse.Namespace) -> dict[str, Any]:
    """Run one case, append its JSONL row, print a one-line summary, return it."""
    row = run_case(args)
    write_row(args.out, row)
    q = ", ".join(f"{k}={v:.4g}" for k, v in row["quality"].items())
    print(f"[{row['case_id']}] fit={row['fit_seconds']:.3f}s "
          f"predict={row['predict_seconds']:.3f}s  {q}")
    if row.get("transfer_bytes"):
        print(f"  transfer_bytes: {row['transfer_bytes']}")
    return row


def main(argv: list[str] | None = None) -> dict[str, Any] | list[dict[str, Any]]:
    args = apply_quick(build_parser().parse_args(argv))
    if args.size:
        args.n_train, args.n_test, args.n_features = _SIZES[args.size]
        if args.quick:  # quick shrinks rows but keep the preset's feature width
            args.n_train, args.n_test = 2_000, 1_000

    # Threshold sweep: one case (one JSONL row) per threshold. Only the cuda
    # backend reads the value; numpy/rust ignore the knob but still record the
    # requested threshold per row so sweep rows stay distinguishable.
    if args.scan_min_cells_sweep:
        rows = [
            _emit_case(_with_scan(args, thresh))
            for thresh in args.scan_min_cells_sweep
        ]
        summary = write_summary(args.out)
        print(f"  -> {args.out}  (summary: {summary})")
        return rows

    row = _emit_case(args)
    summary = write_summary(args.out)
    print(f"  -> {args.out}  (summary: {summary})")
    return row


def _with_scan(args: argparse.Namespace, thresh: int) -> argparse.Namespace:
    """Set the per-case scan threshold on ``args`` (mutates and returns it)."""
    args.cuda_scan_min_cells = thresh
    return args


if __name__ == "__main__":
    main()
