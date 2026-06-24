"""Overnight perf-optimization loop orchestrator (thin wrapper).

This does **not** reimplement measurement. It reuses
:func:`benchmarks.gpu_profile.run_case` (one fit/predict case → row dict with
``phase_seconds`` + ``transfer_bytes``) and adds only the three things the
overnight loop needs on top of a single sample:

1. **Repeats + robust aggregation** — each case is run ``--reps`` times (the
   first rep is a discarded warmup); we report the **median** and a spread
   (min/max + relative spread) of ``fit_seconds`` / ``predict_seconds`` and the
   per-phase medians, so accept/reject decisions are noise-aware
   (see ``docs/perf-notes`` acceptance criteria: +3% median over ≥5 reps).
2. **Interleaved A/B** — ``--mode ab`` alternates two variants per rep
   (the ``scripts/colab_scan_ab.py`` pattern) to fight thermal drift, and reports
   the paired median delta + a signal flag (variant wins ≥ reps-1 AND
   |Δ| > 1σ). Variants differ only by backend and/or environment overrides, so
   this stays a single process — code-version A/B is done across commits.
3. **Rolling result file with harness provenance** — aggregated rows are written
   to ``benchmarks/results/latest.jsonl`` carrying ``harness_version``,
   ``run_id``, ``n_reps`` and ``median_*`` fields **on top of** the existing
   ``gpu_profile`` schema (see ``benchmarks/results/schema.md``). The dated
   per-GPU archive under ``artifacts/gpu_bench/`` is unchanged — no schema fork.

The CUDA backend needs CuPy + an NVIDIA GPU (run via ``scripts/colab_gpu_test.sh``);
``numpy``/``rust`` run anywhere. ``--quick`` is the local dry-run
(tiny rows, 3 reps, numpy+rust only) used to validate the plumbing.

Examples::

    python -m benchmarks.cuda_overnight_loop --quick
    python -m benchmarks.cuda_overnight_loop --mode matrix --reps 5 \\
        --tasks regression multiclass --sizes small medium --backends numpy rust
    python -m benchmarks.cuda_overnight_loop --mode ab --reps 6 \\
        --task multioutput --size large --variant-a cuda --variant-b cuda \\
        --env-b REPLEAFGBM_CUDA_MO_DEVICE_SCAN=0
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from benchmarks import gpu_profile  # noqa: E402

# Bump when the aggregation/schema of THIS orchestrator changes (provenance for
# every row). Product-code changes never bump it; harness-optimizer owns it.
HARNESS_VERSION = "cuda_overnight_loop/0.1.0"
DEFAULT_RESULTS = ROOT / "benchmarks" / "results" / "latest.jsonl"


# --------------------------------------------------------------------------- #
# Args / case construction
# --------------------------------------------------------------------------- #
def _case_args(
    task: str, size: str | None, backend: str, *, quick: bool,
    n_classes: int = 3, n_outputs: int = 3, **overrides: Any
) -> argparse.Namespace:
    """A fully-populated gpu_profile Namespace for one case (no I/O)."""
    argv = ["--task", task, "--backend", backend,
            "--n-classes", str(n_classes), "--n-outputs", str(n_outputs)]
    if size:
        argv += ["--size", size]
    args = gpu_profile.build_parser().parse_args(argv)
    # Mirror gpu_profile.main()'s size→shape + quick handling (run_case skips it).
    if args.size:
        args.n_train, args.n_test, args.n_features = gpu_profile._SIZES[args.size]
    if quick:
        args.n_train, args.n_test, args.n_estimators = 2_000, 1_000, 20
    for key, val in overrides.items():
        setattr(args, key, val)
    return args


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _agg(samples: list[float]) -> dict[str, float]:
    """Median + min/max + relative spread ((max-min)/median) of one metric."""
    p50 = statistics.median(samples)
    lo, hi = min(samples), max(samples)
    return {
        "p50": p50, "min": lo, "max": hi,
        "rel_spread": (hi - lo) / p50 if p50 else 0.0,
    }


def _median_phases(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Per-phase median seconds across the measured (non-warmup) reps."""
    keys = set().union(*(r.get("phase_seconds", {}) or {} for r in rows))
    return {
        k: statistics.median([r.get("phase_seconds", {}).get(k, 0.0) for r in rows])
        for k in keys
    }


def measure_case(
    args: argparse.Namespace, reps: int, *, warmup: bool = True
) -> dict[str, Any]:
    """Run one case ``reps`` times (first discarded if warmup) → aggregated row.

    The aggregated row keeps the last rep's descriptive fields (case_id, shape,
    quality, env, transfer_bytes) and replaces single-sample timings with
    ``median_*`` + spread + per-phase medians, plus harness provenance.
    """
    measured: list[dict[str, Any]] = []
    total = reps + (1 if warmup else 0)
    for i in range(total):
        row = gpu_profile.run_case(args)
        if warmup and i == 0:
            continue
        measured.append(row)
    last = measured[-1]
    fit = _agg([r["fit_seconds"] for r in measured])
    pred = _agg([r["predict_seconds"] for r in measured])
    return {
        **{k: last[k] for k in (
            "case_id", "task", "backend", "n_classes", "n_outputs",
            "n_train", "n_test", "n_features", "max_bins", "num_leaves",
            "leaf_model", "encoder", "device", "cuda_scan_min_cells",
            "n_estimators", "quality", "peak_rss_bytes", "peak_gpu_bytes",
            "transfer_bytes", "env",
        )},
        "harness_version": HARNESS_VERSION,
        "n_reps": reps,
        "median_fit_seconds": fit["p50"],
        "median_predict_seconds": pred["p50"],
        "fit_spread": fit,
        "predict_spread": pred,
        "median_phase_seconds": _median_phases(measured),
    }


# --------------------------------------------------------------------------- #
# Interleaved A/B
# --------------------------------------------------------------------------- #
def _set_env(overrides: dict[str, str]) -> dict[str, str | None]:
    prev = {k: os.environ.get(k) for k in overrides}
    os.environ.update(overrides)
    return prev


def _restore_env(prev: dict[str, str | None]) -> None:
    for k, v in prev.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def run_ab(
    args_a: argparse.Namespace, args_b: argparse.Namespace,
    env_a: dict[str, str], env_b: dict[str, str], reps: int,
) -> dict[str, Any]:
    """Interleaved paired A/B (alternating order per rep) → paired-delta verdict.

    A and B differ only by backend and/or env overrides (same data/seed). Returns
    per-variant aggregates plus the paired median Δ (A-B fit seconds), the win
    count for B, and a ``signal`` flag (B wins ≥ reps-1 AND |median Δ| > 1σ).
    """
    fit_a: list[float] = []
    fit_b: list[float] = []
    diffs: list[float] = []
    for rep in range(reps):
        order = [("a", args_a, env_a), ("b", args_b, env_b)]
        if rep % 2:
            order.reverse()
        seconds: dict[str, float] = {}
        for tag, args, env in order:
            prev = _set_env(env)
            try:
                seconds[tag] = gpu_profile.run_case(args)["fit_seconds"]
            finally:
                _restore_env(prev)
        fit_a.append(seconds["a"])
        fit_b.append(seconds["b"])
        diffs.append(seconds["a"] - seconds["b"])
    med_diff = statistics.median(diffs)
    sd = statistics.pstdev(diffs) if len(diffs) > 1 else 0.0
    b_wins = sum(1 for d in diffs if d > 0)  # A slower => B faster
    return {
        "harness_version": HARNESS_VERSION,
        "mode": "ab",
        "reps": reps,
        "fit_a": _agg(fit_a),
        "fit_b": _agg(fit_b),
        "paired_delta_a_minus_b": {"median": med_diff, "stdev": sd},
        "b_win_count": b_wins,
        "b_faster_pct": 100.0 * med_diff / statistics.median(fit_a)
        if fit_a else 0.0,
        "signal": b_wins >= reps - 1 and abs(med_diff) > sd > 0,
    }


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def _write(out: Path, row: dict[str, Any]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as fh:
        fh.write(json.dumps(row) + "\n")


def _available_backends(requested: list[str]) -> list[str]:
    """Drop ``cuda`` if CuPy/GPU is absent (keeps the local dry-run usable)."""
    if "cuda" not in requested:
        return requested
    try:
        import cupy  # noqa: F401
        cupy.cuda.runtime.getDeviceCount()
    except Exception:
        print("[skip] cuda backend unavailable (no CuPy/GPU) — dropping it")
        return [b for b in requested if b != "cuda"]
    return requested


def _parse_env_kv(items: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items or []:
        key, _, val = item.partition("=")
        out[key] = val
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["matrix", "ab"], default="matrix")
    p.add_argument("--reps", type=int, default=5,
                   help="measured reps per case (warmup is extra)")
    p.add_argument("--no-warmup", action="store_true")
    p.add_argument("--out", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--quick", action="store_true",
                   help="tiny rows, 3 reps, numpy+rust only — plumbing dry-run")
    # matrix mode
    p.add_argument("--tasks", nargs="+",
                   default=["regression", "binary", "multiclass", "multioutput"])
    p.add_argument("--sizes", nargs="+", default=["small"])
    p.add_argument("--backends", nargs="+", default=["numpy", "rust"])
    # ab mode
    p.add_argument("--task", default="regression")
    p.add_argument("--size", default="small")
    p.add_argument("--variant-a", default="numpy")
    p.add_argument("--variant-b", default="rust")
    p.add_argument("--env-a", nargs="*", metavar="K=V")
    p.add_argument("--env-b", nargs="*", metavar="K=V")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    warmup = not args.no_warmup
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    if args.quick:
        args.reps = min(args.reps, 3)
        args.backends = [b for b in args.backends if b != "cuda"] or ["numpy"]

    if args.mode == "matrix":
        backends = _available_backends(args.backends)
        sizes = [None] if args.quick else args.sizes
        for task in args.tasks:
            for size in sizes:
                for backend in backends:
                    case = _case_args(task, size, backend, quick=args.quick)
                    t0 = time.perf_counter()
                    row = measure_case(case, args.reps, warmup=warmup)
                    row["run_id"] = run_id
                    _write(args.out, row)
                    print(f"[{row['case_id']}] "
                          f"fit_p50={row['median_fit_seconds']:.4f}s "
                          f"(spread={row['fit_spread']['rel_spread']:.1%}) "
                          f"pred_p50={row['median_predict_seconds']:.4f}s "
                          f"[{time.perf_counter() - t0:.1f}s]")
        print(f"  -> {args.out}")
        return 0

    # ab mode
    backends = _available_backends([args.variant_a, args.variant_b])
    if len(backends) < 2 and "cuda" in (args.variant_a, args.variant_b):
        print("[abort] A/B needs both variants; cuda unavailable locally")
        return 1
    args_a = _case_args(args.task, args.size, args.variant_a, quick=args.quick)
    args_b = _case_args(args.task, args.size, args.variant_b, quick=args.quick)
    verdict = run_ab(args_a, args_b, _parse_env_kv(args.env_a),
                     _parse_env_kv(args.env_b), args.reps)
    verdict.update(run_id=run_id, task=args.task, size=args.size,
                   variant_a=args.variant_a, variant_b=args.variant_b,
                   env_a=_parse_env_kv(args.env_a), env_b=_parse_env_kv(args.env_b))
    _write(args.out, verdict)
    print(f"A/B {args.variant_a} vs {args.variant_b} ({args.task}/{args.size}): "
          f"B faster by {verdict['b_faster_pct']:.1f}% "
          f"(B wins {verdict['b_win_count']}/{args.reps}, "
          f"signal={verdict['signal']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
