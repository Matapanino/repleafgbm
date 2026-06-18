"""Multi-output regression benchmark suite.

The breadth-first counterpart of ``openml_suite.py`` for **multi-output**
regression — the capability added across v1.4.0 (``(n, K)`` vector pretraining)
and v1.5.0 (multi-output Huber/quantile). The synthetic/real and OpenML suites
only cover scalar targets, so multi-output had no committed leaderboard; this is
it. Two studies, both scored as mean per-output RMSE / R²:

* **Study 1 — clean leaderboard.** RepLeafGBM's single-routing vector leaf
  (``RepLeafRegressor`` on 2-D ``y``) vs. independent per-output external GBMs
  (LightGBM/XGBoost/CatBoost wrapped in ``MultiOutputRegressor``) and sklearn
  ``HistGradientBoosting`` per output. Shows RepLeaf multi-output is in the
  ballpark of K independent fits while training one shared router.
* **Study 2 — robustness.** The heavy-tailed-contamination test from
  ``experiments/multioutput_real_and_robust.py``, promoted to the benchmark
  suite: contaminate the *training* targets and score on the clean test target
  (real) or the clean signal (synthetic), comparing the constant-Hessian
  objectives ``squared`` / ``huber`` / ``quantile(0.5)``.

Datasets: ``energy`` (OpenML energy-efficiency, data_id 1472: 8 numeric
features, 2 targets) and ``synthetic`` (``common.multioutput_signal``, K
correlated targets sharing one X). LightGBM/XGBoost/CatBoost are optional
``[bench]`` extras; ``torch`` (optional ``[torch]``) enables the learned-encoder
arm. Nothing here imports torch/lightgbm at module load.

Run from the repository root (network for OpenML; a few minutes)::

    PYTHONPATH=src python3 benchmarks/multioutput_suite.py [--quick] [--seeds K]
    PYTHONPATH=src python3 benchmarks/multioutput_suite.py --strict   # release run

Results are written to ``experiments/results/multioutput_benchmark.md``.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# macOS framework Python often lacks system CA certs for the OpenML download.
try:
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except ImportError:  # pragma: no cover
    pass

import numpy as np

# Allow ``python benchmarks/multioutput_suite.py`` (sibling-module import of
# ``common``) as well as ``python -m benchmarks.multioutput_suite``.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import contaminate, multioutput_signal  # noqa: E402

from repleafgbm import RepLeafRegressor  # noqa: E402

ENERGY_DATA_ID = 1472  # OpenML energy-efficiency (ENB2012): V1..V8 -> y1, y2
LEARNED_ENCODER = "torch_periodic_plr"
REQUIRED_GBMS = ("lightgbm", "xgboost", "catboost")
DATASETS = ("energy", "synthetic")


@dataclass
class Row:
    label: str
    primary: list[float] = field(default_factory=list)  # mean per-output rmse
    secondary: list[float] = field(default_factory=list)  # mean per-output r2
    fit_s: list[float] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_energy() -> tuple[np.ndarray, np.ndarray]:
    """OpenML energy-efficiency -> (X (n, 8), Y (n, 2))."""
    from sklearn.datasets import fetch_openml

    try:
        d = fetch_openml(data_id=ENERGY_DATA_ID, as_frame=True, parser="auto")
    except TypeError:  # older sklearn without parser=
        d = fetch_openml(data_id=ENERGY_DATA_ID, as_frame=True)
    df = d.frame.astype(np.float64)
    Y = df[["y1", "y2"]].to_numpy(np.float64)
    X = df.drop(columns=["y1", "y2"]).to_numpy(np.float64)
    return X, Y


def make_synthetic(n_rows: int, n_features: int, n_outputs: int, seed: int):
    """K correlated targets sharing X; returns (X, Y_noisy, Y_signal) so the
    robustness arm can score against the *clean* signal."""
    rng = np.random.default_rng(seed)
    X, signal = multioutput_signal(n_rows, n_features, n_outputs, rng)
    noisy = signal + rng.normal(0.0, 0.3, signal.shape)
    return X, noisy, signal


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def _rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def _r2(y, p):
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def mean_per_output(Y, P, fn):
    return float(np.mean([fn(Y[:, k], P[:, k]) for k in range(Y.shape[1])]))


def _record(rows, label, P, Yte, fit_s, seed, tag):
    r = rows.setdefault(label, Row(label))
    primary = mean_per_output(Yte, P, _rmse)
    secondary = mean_per_output(Yte, P, _r2)
    r.primary.append(primary)
    r.secondary.append(secondary)
    r.fit_s.append(fit_s)
    print(f"  seed={seed} [{tag}] {label:42s} rmse={primary:.4f} r2={secondary:.4f}",
          flush=True)


# --------------------------------------------------------------------------- #
# Estimators
# --------------------------------------------------------------------------- #
def _repleaf(objective, encoder, args, seed, **extra):
    return RepLeafRegressor(
        n_estimators=args.n_estimators, num_leaves=16, min_samples_leaf=10,
        learning_rate=0.1, leaf_model="embedded_linear", encoder=encoder,
        objective=objective, random_state=seed, **extra,
    )


def _external_multioutput(args, seed, strict):
    """(label, builder) for per-output external GBMs wrapped in
    MultiOutputRegressor (independent fit per target). builder() -> estimator."""
    from sklearn.multioutput import MultiOutputRegressor

    out, missing = [], []
    try:
        from lightgbm import LGBMRegressor

        out.append(("LightGBM (per-output)", lambda: MultiOutputRegressor(
            LGBMRegressor(n_estimators=args.n_estimators, learning_rate=0.1,
                          num_leaves=31, random_state=seed, verbose=-1))))
    except ImportError:
        missing.append("lightgbm")
    try:
        from xgboost import XGBRegressor

        out.append(("XGBoost (per-output)", lambda: MultiOutputRegressor(
            XGBRegressor(n_estimators=args.n_estimators, learning_rate=0.1,
                         max_depth=6, tree_method="hist", random_state=seed))))
    except ImportError:
        missing.append("xgboost")
    try:
        from catboost import CatBoostRegressor

        out.append(("CatBoost (per-output)", lambda: MultiOutputRegressor(
            CatBoostRegressor(iterations=args.n_estimators, learning_rate=0.1,
                              depth=6, random_seed=seed, verbose=False))))
    except ImportError:
        missing.append("catboost")
    if strict and missing:
        raise RuntimeError(
            f"--strict run requires all external GBMs {list(REQUIRED_GBMS)} but "
            f"{missing} are not installed; `pip install \"repleafgbm[bench]\"`."
        )
    return out


# --------------------------------------------------------------------------- #
# Study 1: clean multi-output leaderboard
# --------------------------------------------------------------------------- #
def study_clean(name, seeds, args, strict):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.multioutput import MultiOutputRegressor

    rows: dict[str, Row] = {}
    repleaf_arms = [
        ("RepLeaf identity", "identity", {}),
        ("RepLeaf plr", "plr", {"max_leaf_emb_dim": 256}),
    ]
    try:  # learned-encoder arm (vector pretraining) when torch is present
        import torch  # noqa: F401

        repleaf_arms.append((f"RepLeaf {LEARNED_ENCODER}", LEARNED_ENCODER,
                             {"max_leaf_emb_dim": 256,
                              "encoder_params": {"n_epochs": args.epochs}}))
    except ImportError:
        pass

    for seed in seeds:
        Xtr, Xte, Ytr, Yte, n = _split(name, args, seed)
        for label, encoder, extra in repleaf_arms:
            try:
                model = _repleaf(None, encoder, args, seed, **extra)
                t0 = time.perf_counter()
                model.fit(Xtr, Ytr)
                _record(rows, label, model.predict(Xte), Yte,
                        time.perf_counter() - t0, seed, "clean")
            except Exception as exc:  # pragma: no cover - robustness
                if strict:
                    raise
                print(f"  [skip] {label} on {name}: {type(exc).__name__}: {exc}")
        # sklearn HistGB per output (no native multi-output support).
        try:
            hgb = MultiOutputRegressor(HistGradientBoostingRegressor(
                max_iter=args.n_estimators, learning_rate=0.1, random_state=seed))
            t0 = time.perf_counter()
            hgb.fit(Xtr, Ytr)
            _record(rows, "hist_gradient_boosting (per-output)", hgb.predict(Xte),
                    Yte, time.perf_counter() - t0, seed, "clean")
        except Exception as exc:  # pragma: no cover
            if strict:
                raise
            print(f"  [skip] hist_gradient_boosting on {name}: {exc}")
        # external GBMs per output.
        for label, build in _external_multioutput(args, seed, strict):
            try:
                model = build()
                t0 = time.perf_counter()
                model.fit(Xtr, Ytr)
                _record(rows, label, model.predict(Xte), Yte,
                        time.perf_counter() - t0, seed, "clean")
            except Exception as exc:  # pragma: no cover
                if strict:
                    raise
                print(f"  [skip] {label} on {name}: {type(exc).__name__}: {exc}")
    return rows, n


def _split(name, args, seed):
    """60/40 train/test split; returns (Xtr, Xte, Ytr, Yte, n)."""
    if name == "energy":
        X_all, Y_all = load_energy()
    else:
        X_all, Y_noisy, _ = make_synthetic(args.n_rows, args.n_features,
                                            args.n_outputs, seed)
        Y_all = Y_noisy
    n = len(X_all)
    idx = np.random.default_rng(seed).permutation(n)
    cut = int(n * 0.6)
    i_tr, i_te = idx[:cut], idx[cut:]
    return X_all[i_tr], X_all[i_te], Y_all[i_tr], Y_all[i_te], n


# --------------------------------------------------------------------------- #
# Study 2: robustness under training-target contamination
# --------------------------------------------------------------------------- #
def study_robust(name, seeds, args):
    rows: dict[str, Row] = {}
    objectives = [("squared", None), ("huber", "huber"),
                  ("quantile(0.5)", "quantile")]
    for seed in seeds:
        crng = np.random.default_rng(seed + 9973)
        if name == "energy":
            X_all, Y_all = load_energy()
            n = len(X_all)
            idx = np.random.default_rng(seed).permutation(n)
            cut = int(n * 0.6)
            Xtr, Xte = X_all[idx[:cut]], X_all[idx[cut:]]
            Ytr_clean, Yte = Y_all[idx[:cut]], Y_all[idx[cut:]]
            Ytr = contaminate(Ytr_clean, args.contam_frac, args.contam_scale, crng)
        else:
            X_all, Y_noisy, Y_signal = make_synthetic(
                args.n_rows, args.n_features, args.n_outputs, seed)
            cut = int(args.n_rows * 0.6)
            Xtr, Xte = X_all[:cut], X_all[cut:]
            Ytr = contaminate(Y_noisy[:cut], args.contam_frac, args.contam_scale, crng)
            Yte = Y_signal[cut:]  # score against the clean signal
            n = args.n_rows
        for label, obj in objectives:
            model = _repleaf(obj, "identity", args, seed)
            t0 = time.perf_counter()
            model.fit(Xtr, Ytr)
            _record(rows, f"RepLeaf {label}", model.predict(Xte), Yte,
                    time.perf_counter() - t0, seed, "contam")
    return rows, n


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def emit_table(out, title, rows):
    ordered = sorted(rows.values(), key=lambda r: np.mean(r.primary))
    out += ["", f"## {title}", "",
            "| model | rmse (mean ± std) | r2 | fit[s] |", "|---|---|---|---|"]
    for r in ordered:
        out.append(
            f"| {r.label} | {np.mean(r.primary):.4f} ± {np.std(r.primary):.4f} "
            f"| {np.mean(r.secondary):.4f} | {np.mean(r.fit_s):.1f} |")


def _version_manifest(args, seeds, selected) -> list[str]:
    def ver(dist):
        try:
            return importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            return "(not installed)"

    pkgs = ["numpy", "pandas", "scipy", "scikit-learn", "repleafgbm",
            "torch", "lightgbm", "xgboost", "catboost"]
    return [
        "## Reproducibility manifest",
        "",
        f"- Python: {platform.python_version()} ({sys.platform})",
        "- Packages: " + ", ".join(f"{p}={ver(p)}" for p in pkgs),
        f"- Seeds: {seeds}; n_estimators={args.n_estimators}, lr=0.1, "
        f"num_leaves=16, min_samples_leaf=10, leaf_model=embedded_linear",
        f"- Datasets: {list(selected)} (energy = OpenML data_id "
        f"{ENERGY_DATA_ID}; synthetic n_rows={args.n_rows}, "
        f"n_outputs={args.n_outputs})",
        f"- Split: 60/40 train/test. Contamination: {args.contam_frac:.0%} of "
        f"train rows, +N(0, ({args.contam_scale}·std)²); robust study scores on "
        "clean test (energy) / clean signal (synthetic)",
        f"- strict mode: {bool(args.strict)}",
    ]


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=30, help="torch pretrain")
    parser.add_argument("--n-rows", type=int, default=3_000)
    parser.add_argument("--n-features", type=int, default=8)
    parser.add_argument("--n-outputs", type=int, default=3)
    parser.add_argument("--contam-frac", type=float, default=0.08)
    parser.add_argument("--contam-scale", type=float, default=8.0)
    parser.add_argument("--datasets", nargs="*", default=None,
                        help=f"subset of {list(DATASETS)}")
    parser.add_argument("--quick", action="store_true",
                        help="fast smoke: 2 seeds, fewer trees/epochs/rows")
    parser.add_argument("--strict", action="store_true",
                        help="release mode: fail (don't skip) on missing GBM/errors")
    parser.add_argument("--out", type=Path, default=None,
                        help="report path (default: experiments/results/"
                             "multioutput_benchmark.md)")
    args = parser.parse_args(argv)
    if args.quick:
        args.seeds, args.epochs, args.n_estimators, args.n_rows = 2, 5, 40, 800
    seeds = list(range(args.seeds))
    selected = [d for d in DATASETS if not args.datasets or d in set(args.datasets)]

    out = [
        "# Multi-output regression benchmark suite",
        "",
        "Auto-generated by `benchmarks/multioutput_suite.py`. Study 1 is a clean "
        "leaderboard (RepLeaf single-routing vector leaf vs. independent "
        "per-output external GBMs); Study 2 contaminates the **training** targets "
        "with heavy-tailed outliers and scores on clean targets, comparing the "
        "constant-Hessian objectives. Metric: **mean per-output RMSE** "
        "(secondary: mean per-output R²); lower RMSE is better.",
        "",
        *_version_manifest(args, seeds, selected),
    ]

    for name in selected:
        print(f"=== {name}: Study 1 (clean leaderboard) ===", flush=True)
        rows1, n1 = study_clean(name, seeds, args, args.strict)
        emit_table(out, f"{name} — clean leaderboard (n={n1}, "
                   f"outputs={'2' if name == 'energy' else args.n_outputs})", rows1)
        print(f"=== {name}: Study 2 (robustness) ===", flush=True)
        rows2, n2 = study_robust(name, seeds, args)
        emit_table(out, f"{name} — robustness under contaminated train (n={n2})",
                   rows2)

    out_path = args.out or (Path(__file__).resolve().parents[1] / "experiments"
                            / "results" / "multioutput_benchmark.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out) + "\n")
    print(f"\nreport written to {out_path}", flush=True)
    return out_path


if __name__ == "__main__":
    main()
