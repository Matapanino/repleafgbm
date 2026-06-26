"""Robust multi-output objectives under contamination — multi-dataset suite.

Hardens the strongest RepLeafGBM niche win (Phase 31): under heavy-tailed
contamination of the *training* targets, the constant-Hessian robust objectives
(``huber`` / ``quantile(0.5)``) should track the clean conditional far better
than ``squared`` error. The prior study
(``experiments/multioutput_real_and_robust.py``) showed this decisively on a
*single* real dataset (energy-efficiency) at a *single* contamination level. This
generalizes it to:

* **Multiple real multi-target datasets** — energy-efficiency (1472) plus the
  Mulan multi-target sets pinned by literature-scout (wq 41491, jura 41479,
  rf1 41483, scm20d 41486; targets = the trailing columns, validated by the
  documented shape). A synthetic target (scored against its *clean* signal) is
  the offline control.
* **A contamination grid** (``--contam-grid``, default ``0, 4, 8, 16 %``).
* **Significance** — Wilcoxon signed-rank across seeds of huber/quantile vs
  squared, with win/tie/loss under a minimum-relevant difference.

Defaults are not changed here; this writes its own dated report (a
``results-analyst`` verdict gates any default change).

Run from the repo root (network for the real datasets; ``--quick`` is small)::

    OMP_NUM_THREADS=1 PYTHONPATH=src python3 experiments/robust_multioutput_suite.py --quick
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

try:  # macOS framework Python often lacks system CA certs for the OpenML download.
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except ImportError:  # pragma: no cover
    pass

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "benchmarks")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from benchmarks import stats  # noqa: E402
from common import synthetic_tabular  # noqa: E402

from repleafgbm import RepLeafRegressor  # noqa: E402

OBJECTIVES = [("squared", None), ("huber", "huber"), ("quantile(0.5)", "quantile")]


@dataclass(frozen=True)
class MODataset:
    """A real multi-target regression dataset. Targets are the trailing
    ``n_targets`` columns (Mulan convention); ``n_inputs``/``n_targets`` from the
    literature-scout note validate the loaded shape before use."""

    name: str
    data_id: int
    n_inputs: int
    n_targets: int


# energy is special-cased (named y1/y2); the rest use the trailing-columns rule.
REAL_DATASETS = [
    MODataset("energy", 1472, 8, 2),
    MODataset("jura", 41479, 15, 3),
    MODataset("wq", 41491, 16, 14),
    MODataset("rf1", 41483, 64, 8),
    MODataset("scm20d", 41486, 61, 16),
]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def _fetch_frame(data_id: int):
    from sklearn.datasets import fetch_openml

    try:
        d = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
    except TypeError:  # older sklearn without parser=
        d = fetch_openml(data_id=data_id, as_frame=True)
    return d.frame


def load_real(spec: MODataset):
    """Return ``(X (n, p), Y (n, K))`` float arrays, or raise if the shape does
    not match the documented (inputs, targets) — a guard against a wrong target
    layout silently producing garbage."""
    frame = _fetch_frame(spec.data_id)
    if spec.name == "energy":
        df = frame.astype(np.float64)
        Y = df[["y1", "y2"]].to_numpy(np.float64)
        X = df.drop(columns=["y1", "y2"]).to_numpy(np.float64)
        return X, Y
    df = frame.apply(lambda c: c.astype(np.float64))
    expected = spec.n_inputs + spec.n_targets
    if df.shape[1] != expected:
        raise ValueError(
            f"{spec.name}: got {df.shape[1]} columns, expected "
            f"{expected} ({spec.n_inputs} inputs + {spec.n_targets} targets); "
            "target layout unconfirmed — skipping")
    arr = df.to_numpy(np.float64)
    return arr[:, : spec.n_inputs], arr[:, spec.n_inputs:]


def make_synthetic(n_rows: int, n_outputs: int, seed: int):
    """K router+periodic targets sharing X; returns (X, Y_noisy, Y_signal)."""
    rng = np.random.default_rng(seed)
    X, _ = synthetic_tabular(n_rows, 8, rng)
    router = 3.0 * (X[:, 3] > 0.5)
    signal = np.column_stack([
        2.0 * np.sin((k + 2) * X[:, 0]) + 1.5 * X[:, 1] + router
        for k in range(n_outputs)
    ])
    return X, signal + rng.normal(0.0, 0.3, signal.shape), signal


def contaminate(Y: np.ndarray, frac: float, scale: float, seed: int) -> np.ndarray:
    if frac <= 0.0:
        return Y.copy()
    rng = np.random.default_rng(seed + 9973)
    out = Y.copy()
    mask = rng.random(Y.shape[0]) < frac
    sd = Y.std(axis=0, keepdims=True)
    out[mask] += rng.normal(0.0, 1.0, (int(mask.sum()), Y.shape[1])) * scale * sd
    return out


# --------------------------------------------------------------------------- #
# Scoring / fitting
# --------------------------------------------------------------------------- #
def _rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def _mean_per_output(Y, P):
    return float(np.mean([_rmse(Y[:, k], P[:, k]) for k in range(Y.shape[1])]))


def _fit_score(Xtr, Ytr, Xte, Yte_truth, objective, args, seed):
    model = RepLeafRegressor(
        n_estimators=args.n_estimators, num_leaves=16, min_samples_leaf=10,
        learning_rate=0.1, leaf_model="embedded_linear", encoder="identity",
        objective=objective, random_state=seed)
    model.fit(Xtr, Ytr)
    return _mean_per_output(Yte_truth, model.predict(Xte))


def run_dataset(name, loader, seeds, fracs, args):
    """``loader(seed) -> (Xtr, Ytr_clean, Xte, Yte_truth)``; contamination is
    applied to ``Ytr_clean``. Returns ``results[(frac, obj_label)] -> [rmse...]``."""
    results: dict[tuple[float, str], list[float]] = {}
    for seed in seeds:
        Xtr, Ytr_clean, Xte, Yte_truth = loader(seed)
        for frac in fracs:
            Ytr = contaminate(Ytr_clean, frac, args.contam_scale, seed)
            for label, obj in OBJECTIVES:
                rmse = _fit_score(Xtr, Ytr, Xte, Yte_truth, obj, args, seed)
                results.setdefault((frac, label), []).append(rmse)
                print(f"  {name} seed={seed} contam={frac:.0%} {label:13s} "
                      f"rmse={rmse:.4f}", flush=True)
    return results


def _real_loader(spec, args):
    X, Y = load_real(spec)
    n = len(X)

    def loader(seed):
        idx = np.random.default_rng(seed).permutation(n)
        cut = int(n * 0.6)
        i_tr, i_te = idx[:cut], idx[cut:]
        return X[i_tr], Y[i_tr], X[i_te], Y[i_te]  # score on clean test targets

    return loader


def _synth_loader(args):
    def loader(seed):
        X, Y_noisy, Y_signal = make_synthetic(args.n_rows, args.n_outputs, seed)
        cut = int(len(X) * 0.6)
        return X[:cut], Y_noisy[:cut], X[cut:], Y_signal[cut:]  # score vs signal

    return loader


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _emit_dataset(out, title, results, fracs, seeds, alpha, mrd):
    out += ["", f"## {title}", "",
            "| objective | " + " | ".join(f"contam {f:.0%}" for f in fracs) + " |",
            "|---|" + "---|" * len(fracs)]
    for label, _ in OBJECTIVES:
        cells = []
        for f in fracs:
            vals = np.array(results[(f, label)])
            cells.append(f"{vals.mean():.4f} ± {vals.std():.4f}")
        out.append(f"| {label} | " + " | ".join(cells) + " |")

    # Significance vs squared error at each contaminated level.
    contam = [f for f in fracs if f > 0.0]
    if not contam:
        return
    out += ["", f"Robust vs **squared** (Wilcoxon across {len(seeds)} seeds, "
            f"alpha={alpha}, MRD={mrd:.0%}):", "",
            "| contam | arm | median ΔRMSE | Wilcoxon p | win/tie/loss | verdict |",
            "|---|---|---|---|---|---|"]
    for f in contam:
        squared = np.array(results[(f, "squared")])
        for label in ("huber", "quantile(0.5)"):
            arm = np.array(results[(f, label)])
            scores = np.column_stack([squared, arm])  # [squared, arm]
            _, p, md = stats.wilcoxon_pairs(scores, ["squared", label],
                                            baseline="squared")[label]
            w, t, ll = stats.win_tie_loss(arm, squared, mrd=mrd)
            verdict = "sig. better" if (md < 0 and p < alpha) else "not sig."
            out.append(f"| {f:.0%} | {label} | {md:+.4f} | {p:.3g} | "
                       f"{w}/{t}/{ll} | {verdict} |")


def main(argv=None) -> Path:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--n-rows", type=int, default=3_000)
    p.add_argument("--n-outputs", type=int, default=3)
    p.add_argument("--contam-grid", type=float, nargs="*",
                   default=[0.0, 0.04, 0.08, 0.16])
    p.add_argument("--contam-scale", type=float, default=8.0)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--mrd", type=float, default=0.01)
    p.add_argument("--datasets", nargs="*", default=None,
                   help="real datasets to include (default all; 'none' for "
                        "synthetic-only offline run)")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    if args.quick:
        args.seeds, args.n_estimators, args.n_rows = 2, 40, 800
        args.contam_grid = [0.0, 0.08]
        if args.datasets is None:
            args.datasets = ["energy"]  # one real dataset for a fast smoke
    seeds = list(range(args.seeds))
    fracs = list(args.contam_grid)

    out = [
        "# Experiment: robust multi-output objectives under contamination (suite)",
        "",
        "Auto-generated by `experiments/robust_multioutput_suite.py`. Training "
        "targets are contaminated with heavy-tailed outliers; models score on "
        "clean test targets (real) or the clean signal (synthetic). Constant-"
        "Hessian objectives (huber / quantile) are tested against squared error "
        "with Wilcoxon signed-rank significance. **Defaults are not changed "
        "here** (a `results-analyst` report gates that).",
        "",
        f"Settings: seeds={seeds}, n_estimators={args.n_estimators}, lr=0.1, "
        f"num_leaves=16, leaf_model=embedded_linear, contamination grid="
        f"{[f'{f:.0%}' for f in fracs]} at scale {args.contam_scale}×std, "
        f"mean per-output RMSE, alpha={args.alpha}, MRD={args.mrd:.0%}.",
    ]

    wanted = args.datasets
    selected = [] if wanted == ["none"] else [
        s for s in REAL_DATASETS if (wanted is None or s.name in set(wanted))]
    for spec in selected:
        print(f"=== real dataset: {spec.name} ===", flush=True)
        try:
            loader = _real_loader(spec, args)
        except Exception as exc:
            print(f"  [skip] {spec.name}: {type(exc).__name__}: {exc}", flush=True)
            out += ["", f"## {spec.name} — skipped ({type(exc).__name__}: {exc})"]
            continue
        results = run_dataset(spec.name, loader, seeds, fracs, args)
        _emit_dataset(out, f"{spec.name} (real, outputs={spec.n_targets})",
                      results, fracs, seeds, args.alpha, args.mrd)

    print("=== synthetic control ===", flush=True)
    syn = run_dataset("synthetic", _synth_loader(args), seeds, fracs, args)
    _emit_dataset(out, f"synthetic (outputs={args.n_outputs}, scored vs clean "
                  "signal)", syn, fracs, seeds, args.alpha, args.mrd)

    out_path = (Path(args.out) if args.out else
                Path(__file__).resolve().parent / "results"
                / f"{date.today().isoformat()}-robust-multioutput-suite.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out) + "\n")
    print(f"\nreport written to {out_path}", flush=True)
    return out_path


if __name__ == "__main__":
    main()
