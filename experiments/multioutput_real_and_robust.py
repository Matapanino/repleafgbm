"""Experiment: real multi-output validation + robust multi-output objectives.

Two loose ends after v1.4.0 and the Phase 31 multi-output robust-loss work:

1. **Real multi-output validation (E2).** v1.4.0's ``(n, K)`` vector pretraining
   target was only validated on a *synthetic* multi-output target. This runs the
   same **before / after** vector-pretraining comparison on a **real** multi-
   target regression dataset (OpenML energy-efficiency, data_id 1472: 8 numeric
   features, two targets ``y1``/``y2`` = heating/cooling load), alongside frozen
   baselines and a per-output external GBM reference, to check both that vector
   pretraining reproduces on real data and that RepLeaf multi-output is in the
   ballpark of independent per-output GBM fits.

2. **Robust multi-output objectives (A1 / Phase 31).** Multi-output now supports
   Huber/quantile (the constant-Hessian ``h = 1`` family). Under heavy-tailed
   contamination of the *training* targets, ``huber`` / ``quantile(0.5)`` should
   track the clean conditional better than squared error. Shown on both the real
   dataset (scored on uncontaminated test targets) and a synthetic target with a
   known clean signal (scored against the signal).

Canonical reports (experiments/results/openml_benchmark.md and the v1.4.0
vector-target report) are untouched; this writes its own dated report. Defaults
change only via a results-analyst verdict.

Run from the repo root (torch + network for OpenML; a few minutes):
    OMP_NUM_THREADS=1 python3 experiments/multioutput_real_and_robust.py [--seeds K] [--quick]
Report: experiments/results/<date>-multioutput-real-and-robust.md
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

# macOS framework Python often lacks system CA certs for the OpenML download.
try:
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except ImportError:  # pragma: no cover
    pass

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from common import synthetic_tabular  # noqa: E402

from repleafgbm import RepLeafRegressor  # noqa: E402

ENERGY_DATA_ID = 1472  # OpenML energy-efficiency (ENB2012): V1..V8 -> y1, y2
LEARNED_ENCODER = "torch_periodic_plr"


class _NoVectorPretrainRegressor(RepLeafRegressor):
    """Pre-change behavior: multi-output encoders fit unsupervised (the 'before'
    arm), so the projection never sees the vector residual."""

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


def make_synthetic(n_rows: int, n_outputs: int, seed: int):
    """K router-plus-periodic targets sharing X; returns (X, Y_noisy, Y_signal)
    so the robustness arm can score against the *clean* signal."""
    rng = np.random.default_rng(seed)
    X, _ = synthetic_tabular(n_rows, 8, rng)
    router = 3.0 * (X[:, 3] > 0.5)
    signal = np.column_stack([
        2.0 * np.sin((k + 2) * X[:, 0]) + 1.5 * X[:, 1] + router
        for k in range(n_outputs)
    ])
    noisy = signal + rng.normal(0.0, 0.3, signal.shape)
    return X, noisy, signal


def contaminate(Y: np.ndarray, frac: float, scale: float, seed: int) -> np.ndarray:
    """Add heavy-tailed outliers to a fraction of rows (training targets only).
    ``scale`` multiplies each output's own std, so contamination is comparable
    across outputs."""
    rng = np.random.default_rng(seed + 9973)
    out = Y.copy()
    mask = rng.random(Y.shape[0]) < frac
    sd = Y.std(axis=0, keepdims=True)
    out[mask] += rng.normal(0.0, 1.0, (mask.sum(), Y.shape[1])) * scale * sd
    return out


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


def _record(rows, label, primary, secondary, fit_s, tag, p_name, s_name, seed):
    r = rows.setdefault(label, Row(label))
    r.primary.append(primary)
    r.secondary.append(secondary)
    r.fit_s.append(fit_s)
    print(f"  seed={seed} [{tag}] {label:44s} {p_name}={primary:.4f} "
          f"{s_name}={secondary:.4f}", flush=True)


# --------------------------------------------------------------------------- #
# Study 1: real multi-output validation (before/after vector pretraining)
# --------------------------------------------------------------------------- #
def study_real_validation(seeds, args):
    X_all, Y_all = load_energy()
    n = len(X_all)
    rows: dict[str, Row] = {}
    emb = {"leaf_model": "embedded_linear", "max_leaf_emb_dim": 256}
    ep = {"encoder_params": {"n_epochs": args.epochs}}
    grid = [
        ("RepLeaf identity", RepLeafRegressor, {**emb, "encoder": "identity"}, {}),
        ("RepLeaf plr (frozen)", RepLeafRegressor, {**emb, "encoder": "plr"}, {}),
        (f"RepLeaf {LEARNED_ENCODER} (before: unsupervised)",
         _NoVectorPretrainRegressor, {**emb, "encoder": LEARNED_ENCODER}, ep),
        (f"RepLeaf {LEARNED_ENCODER} (after: vector pretrain)",
         RepLeafRegressor, {**emb, "encoder": LEARNED_ENCODER}, ep),
    ]
    for seed in seeds:
        idx = np.random.default_rng(seed).permutation(n)
        cut = int(n * 0.6)
        i_tr, i_te = idx[:cut], idx[cut:]
        Xtr, Xte, Ytr, Yte = X_all[i_tr], X_all[i_te], Y_all[i_tr], Y_all[i_te]
        for label, cls, kwargs, extra in grid:
            model = cls(n_estimators=args.n_estimators, num_leaves=16,
                        min_samples_leaf=10, learning_rate=0.1,
                        random_state=seed, **kwargs, **extra)
            t0 = time.perf_counter()
            model.fit(Xtr, Ytr)
            fit_s = time.perf_counter() - t0
            pred = model.predict(Xte)
            _record(rows, label, mean_per_output(Yte, pred, _rmse),
                    mean_per_output(Yte, pred, _r2), fit_s, "real", "rmse", "r2", seed)
        # External per-output GBM reference (optional).
        for label, primary, secondary, fit_s in _external_reference(
                Xtr, Ytr, Xte, Yte, args.n_estimators, seed):
            _record(rows, label, primary, secondary, fit_s, "real", "rmse", "r2", seed)
    return rows, n


def _external_reference(Xtr, Ytr, Xte, Yte, n_estimators, seed):
    """Independent per-output LightGBM via MultiOutputRegressor, if available."""
    try:
        from lightgbm import LGBMRegressor
        from sklearn.multioutput import MultiOutputRegressor
    except ImportError:
        return []
    base = LGBMRegressor(n_estimators=n_estimators, learning_rate=0.1,
                         num_leaves=31, random_state=seed, verbose=-1)
    model = MultiOutputRegressor(base)
    t0 = time.perf_counter()
    model.fit(Xtr, Ytr)
    fit_s = time.perf_counter() - t0
    pred = model.predict(Xte)
    return [("LightGBM (per-output ref)", mean_per_output(Yte, pred, _rmse),
             mean_per_output(Yte, pred, _r2), fit_s)]


# --------------------------------------------------------------------------- #
# Study 2: robust objectives under training-target contamination
# --------------------------------------------------------------------------- #
def study_robustness(seeds, args):
    rows_real: dict[str, Row] = {}
    rows_syn: dict[str, Row] = {}
    objectives = [("squared", None), ("huber", "huber"), ("quantile(0.5)", "quantile")]

    X_all, Y_all = load_energy()
    n = len(X_all)
    for seed in seeds:
        # --- real: contaminate train targets, score on clean test targets ---
        idx = np.random.default_rng(seed).permutation(n)
        cut = int(n * 0.6)
        i_tr, i_te = idx[:cut], idx[cut:]
        Xtr, Xte, Yte = X_all[i_tr], X_all[i_te], Y_all[i_te]
        Ytr_c = contaminate(Y_all[i_tr], args.contam_frac, args.contam_scale, seed)
        for label, obj in objectives:
            primary, secondary, fit_s = _fit_score_robust(
                Xtr, Ytr_c, Xte, Yte, obj, args, seed)
            _record(rows_real, f"RepLeaf {label}", primary, secondary, fit_s,
                    "real-contam", "rmse", "r2", seed)

        # --- synthetic: contaminate train targets, score against clean signal ---
        Xs, Ys_noisy, Ys_signal = make_synthetic(args.n_rows, args.n_outputs, seed)
        scut = int(args.n_rows * 0.6)
        Xstr, Xste = Xs[:scut], Xs[scut:]
        Ystr_c = contaminate(Ys_noisy[:scut], args.contam_frac, args.contam_scale, seed)
        Yste_signal = Ys_signal[scut:]
        for label, obj in objectives:
            primary, secondary, fit_s = _fit_score_robust(
                Xstr, Ystr_c, Xste, Yste_signal, obj, args, seed)
            _record(rows_syn, f"RepLeaf {label}", primary, secondary, fit_s,
                    "syn-contam", "rmse_signal", "r2_signal", seed)
    return rows_real, rows_syn, n


def _fit_score_robust(Xtr, Ytr, Xte, Yte, objective, args, seed):
    model = RepLeafRegressor(
        n_estimators=args.n_estimators, num_leaves=16, min_samples_leaf=10,
        learning_rate=0.1, leaf_model="embedded_linear", encoder="identity",
        objective=objective, random_state=seed)
    t0 = time.perf_counter()
    model.fit(Xtr, Ytr)
    fit_s = time.perf_counter() - t0
    pred = model.predict(Xte)
    return (mean_per_output(Yte, pred, _rmse),
            mean_per_output(Yte, pred, _r2), fit_s)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def emit_table(out, title, rows, p_name, s_name):
    ordered = sorted(rows.values(), key=lambda r: np.mean(r.primary))
    out += ["", f"## {title}", "",
            f"| model | {p_name} (mean ± std) | {s_name} | fit[s] |",
            "|---|---|---|---|"]
    for r in ordered:
        out.append(
            f"| {r.label} | {np.mean(r.primary):.4f} ± {np.std(r.primary):.4f} "
            f"| {np.mean(r.secondary):.4f} ± {np.std(r.secondary):.4f} "
            f"| {np.mean(r.fit_s):.1f} |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--n-rows", type=int, default=3_000)
    parser.add_argument("--n-outputs", type=int, default=3)
    parser.add_argument("--contam-frac", type=float, default=0.08)
    parser.add_argument("--contam-scale", type=float, default=8.0)
    parser.add_argument("--quick", action="store_true",
                        help="fast smoke: 2 seeds, fewer trees/epochs/rows")
    args = parser.parse_args()
    if args.quick:
        args.seeds, args.epochs, args.n_estimators, args.n_rows = 2, 5, 40, 800
    seeds = list(range(args.seeds))

    out = [
        "# Experiment: real multi-output validation + robust multi-output objectives",
        "",
        "Auto-generated by `experiments/multioutput_real_and_robust.py`. "
        "Study 1 isolates the v1.4.0 `(n, K)` vector-pretraining change on a "
        "**real** dataset (before = unsupervised projection, after = stock). "
        "Study 2 contaminates the **training** targets with heavy-tailed "
        "outliers and scores on uncontaminated targets, comparing the "
        "constant-Hessian objectives.",
        "",
        f"Settings: seeds={seeds}, n_estimators={args.n_estimators}, lr=0.1, "
        f"num_leaves=16, min_samples_leaf=10, leaf_model=embedded_linear, "
        f"torch_epochs={args.epochs}. Real data: OpenML energy-efficiency "
        f"(data_id {ENERGY_DATA_ID}), 60/40 split, mean per-output RMSE. "
        f"Contamination: {args.contam_frac:.0%} of train rows, "
        f"+N(0, ({args.contam_scale}·std)²). Synthetic robustness target: "
        f"n_rows={args.n_rows}, n_outputs={args.n_outputs}, scored vs the clean "
        "signal.",
    ]

    print("=== Study 1: real multi-output validation ===", flush=True)
    rows1, n1 = study_real_validation(seeds, args)
    emit_table(out, f"Study 1 — energy-efficiency (real, n={n1}, outputs=2, "
               "before/after vector pretraining)", rows1, "rmse", "r2")

    print("=== Study 2: robustness under contamination ===", flush=True)
    rows_real, rows_syn, n2 = study_robustness(seeds, args)
    emit_table(out, f"Study 2a — energy-efficiency robustness (real, n={n2}, "
               "outputs=2, contaminated train)", rows_real, "rmse", "r2")
    emit_table(out, f"Study 2b — synthetic robustness (n={args.n_rows}, outputs="
               f"{args.n_outputs}, scored vs clean signal)", rows_syn,
               "rmse_signal", "r2_signal")

    out_path = (Path(__file__).resolve().parent / "results"
                / f"{date.today().isoformat()}-multioutput-real-and-robust.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out) + "\n")
    print(f"\nreport written to {out_path}", flush=True)


if __name__ == "__main__":
    main()
