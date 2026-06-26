"""Fair general leaderboard (deliverable A of the benchmark overhaul).

Every model family is tuned under the **same Optuna budget** (``benchmarks.hpo``)
on the **same** train/valid split, refit, and scored once on a held-out test set,
across a declarative dataset **suite** (``benchmarks.suites``). Runs are
**resumable** (``benchmarks.ledger``) so a long Colab-CPU run survives the 30s
idle timeout and disconnects, and the aggregate report carries **significance
evidence** (``benchmarks.stats``: Friedman + Nemenyi CD diagram, Wilcoxon vs the
strongest baseline, bootstrap CIs, per-dataset win/tie/loss) so no headline claim
is made without a test.

Honest framing is built in: the preamble states the competitive-but-not-SOTA
expectation, and a model is **bolded** as an improvement only when it is
significant (Wilcoxon ``p < alpha``) **and** beyond the minimum-relevant
difference (MRD). Null/negative results are reported alongside wins.

Run from the repo root (needs ``PYTHONPATH=src`` or an editable install)::

    OMP_NUM_THREADS=1 PYTHONPATH=src python3 benchmarks/leaderboard.py --quick
    OMP_NUM_THREADS=1 PYTHONPATH=src python3 benchmarks/leaderboard.py \\
        --suite grinsztajn_numerical --n-trials 40 --seeds 10   # production (slow)

Lives under ``benchmarks/`` only; never imported by the library (``src/``).
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

# Make the repo root importable so ``benchmarks`` resolves when run as a script
# (python3 benchmarks/leaderboard.py), matching the -m form and sibling runners.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import hpo, stats, suites  # noqa: E402
from benchmarks.ledger import Ledger  # noqa: E402
from benchmarks.openml_suite import r2, rmse  # noqa: E402

SECONDARY = {"regression": "r2", "binary": "auc", "multiclass": "accuracy"}
TASK_ORDER = ("regression", "binary", "multiclass")


def _split_fracs(idx, y_sub, task, rng, train_prop=0.70, val_prop=0.15,
                 train_cap=10_000, eval_cap=50_000):
    """Grinsztajn 2022 split: 70/15/15, train capped at 10k, val/test at 50k.

    Stratified by class for classification (preserves label proportions on the
    small/imbalanced sets); a plain split of the already-permuted ``idx`` for
    regression.
    """
    if task == "regression":
        n = len(idx)
        n_tr, n_va = int(n * train_prop), int(n * val_prop)
        tr, va, te = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]
    else:
        tr, va, te = [], [], []
        for c in np.unique(y_sub):
            pos = np.where(y_sub == c)[0]
            rng.shuffle(pos)
            n_tr, n_va = int(len(pos) * train_prop), int(len(pos) * val_prop)
            tr.extend(idx[pos[:n_tr]])
            va.extend(idx[pos[n_tr:n_tr + n_va]])
            te.extend(idx[pos[n_tr + n_va:]])
        tr, va, te = (np.array(x, dtype=int) for x in (tr, va, te))
        # The per-class indices were concatenated in class order; shuffle before
        # the cap so a capped train set keeps a representative label mix (else the
        # first class fills the cap and skews the train distribution).
        for a in (tr, va, te):
            rng.shuffle(a)
    return tr[:train_cap], va[:eval_cap], te[:eval_cap]


# --------------------------------------------------------------------------- #
# Cell execution: (dataset, model, seed) -> ledger
# --------------------------------------------------------------------------- #
def _evaluate(model, X, y, task: str, classes) -> tuple[float, float]:
    """Return ``(primary, secondary)`` — primary is lower-is-better."""
    if task == "regression":
        pred = model.predict(X)
        return rmse(y, pred), r2(y, pred)
    proba = model.predict_proba(X)
    ll = float(log_loss(y, proba, labels=classes))
    if task == "binary":
        return ll, float(roc_auc_score(y, proba[:, 1]))
    pred = classes[np.argmax(proba, axis=1)]
    return ll, float(accuracy_score(y, pred))


def _prepare(spec: suites.DatasetSpec, seed: int, max_rows: int,
             train_prop: float, val_prop: float):
    """Load, subsample, split (Grinsztajn 70/15/15), and ordinal-encode."""
    from repleafgbm import RepLeafDataset

    X_all, y_all, cats = suites.load(spec, n_rows=max_rows, seed=seed)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X_all))[: min(max_rows, len(X_all))]
    i_tr, i_va, i_te = _split_fracs(idx, y_all[idx], spec.task, rng,
                                    train_prop, val_prop)
    Xtr, Xva, Xte = (X_all.iloc[i] for i in (i_tr, i_va, i_te))
    ytr, yva, yte = y_all[i_tr], y_all[i_va], y_all[i_te]

    train_ds = RepLeafDataset(Xtr, ytr, categorical_features=cats)
    valid_ds = RepLeafDataset(Xva, yva, metadata=train_ds.metadata)
    test_ds = RepLeafDataset(Xte, yte, metadata=train_ds.metadata)
    Xtr_e, Xva_e, Xte_e = (d.get_raw_features() for d in (train_ds, valid_ds, test_ds))
    classes = None if spec.task == "regression" else np.unique(y_all)
    return (Xtr_e, ytr, Xva_e, yva, Xte_e, yte, classes, len(idx))


def run_cell(suite_name, spec, family, seed, args, ledger, prepared) -> None:
    """Tune ``family`` under budget, refit, score on test, append to ledger."""
    Xtr, ytr, Xva, yva, Xte, yte, classes, n_used = prepared
    res = hpo.tune(family, Xtr, ytr, Xva, yva, spec.task, args.n_trials, seed,
                   quick=args.quick, classes=classes)
    model = hpo.build_model(family, res.params, spec.task, seed)
    t0 = time.perf_counter()
    model.fit(Xtr, ytr)
    fit_s = time.perf_counter() - t0
    primary, secondary = _evaluate(model, Xte, yte, spec.task, classes)
    ledger.record(suite_name, spec.name, family, seed, payload={
        "task": spec.task,
        "primary": primary,
        "secondary": secondary,
        "val_value": res.value,
        "fit_seconds": fit_s,
        "n_used": n_used,
        "params": res.params,
        "n_trials": res.n_trials,
    })


def run(args) -> Path:
    suite = suites.get_suite(args.suite)
    datasets = suite.select(args.datasets, quick=args.quick)
    families = list(args.models) if args.models else list(hpo.FAMILIES)
    # --seed-list slices the run into specific seeds (for the keep-alive Colab
    # loop's short, durable per-(dataset, seed) execs); else seeds 0..N-1.
    seeds = (list(args.seed_list) if getattr(args, "seed_list", None)
             else list(range(args.seeds)))

    ledger = Ledger(Path(args.ledger))
    for spec in datasets:
        for seed in seeds:
            pending = [f for f in families
                       if not ledger.done(args.suite, spec.name, f, seed)]
            if not pending:
                continue
            try:
                prepared = _prepare(spec, seed, args.max_rows,
                                    args.train_prop, args.val_prop)
            except Exception as exc:  # pragma: no cover - network/data issues
                if args.strict:
                    raise
                print(f"  [skip dataset] {spec.name} seed={seed}: "
                      f"{type(exc).__name__}: {exc}", flush=True)
                continue
            for family in pending:
                try:
                    run_cell(args.suite, spec, family, seed, args, ledger, prepared)
                    print(f"  [done] {spec.name} {family} seed={seed}", flush=True)
                except ImportError as exc:
                    if args.strict:
                        raise
                    print(f"  [skip] {family} (missing dep): {exc}", flush=True)
                except Exception as exc:  # pragma: no cover - robustness
                    if args.strict:
                        raise
                    print(f"  [skip] {family} on {spec.name} seed={seed}: "
                          f"{type(exc).__name__}: {exc}", flush=True)

    settings = {
        "suite": args.suite, "seeds": seeds, "n_trials": args.n_trials,
        "max_rows": args.max_rows, "alpha": args.alpha, "mrd": args.mrd,
        "quick": bool(args.quick), "train_prop": args.train_prop,
        "val_prop": args.val_prop,
    }
    out_path = (Path(args.out) if args.out else
                Path(__file__).resolve().parents[1] / "experiments" / "results"
                / f"{date.today().isoformat()}-leaderboard.md")
    text = build_report(ledger.records(), settings, ledger.provenance,
                        assets_dir=out_path.parent)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    print(f"\nreport written to {out_path}")
    return out_path


# --------------------------------------------------------------------------- #
# Report generation (pure function over ledger records — unit-testable)
# --------------------------------------------------------------------------- #
def _mean_table(records, task):
    """Per-(dataset, model) mean of primary/secondary/fit over seeds."""
    acc: dict[tuple[str, str], dict[str, list]] = defaultdict(
        lambda: {"primary": [], "secondary": [], "fit": []})
    datasets, models = [], []
    for r in records:
        p = r["payload"]
        if p.get("task") != task:
            continue
        key = (r["dataset"], r["model"])
        acc[key]["primary"].append(p["primary"])
        acc[key]["secondary"].append(p["secondary"])
        acc[key]["fit"].append(p["fit_seconds"])
        if r["dataset"] not in datasets:
            datasets.append(r["dataset"])
        if r["model"] not in models:
            models.append(r["model"])
    cell = {k: {"primary": float(np.mean(v["primary"])),
                "secondary": float(np.mean(v["secondary"])),
                "fit": float(np.mean(v["fit"]))}
            for k, v in acc.items()}
    return datasets, models, cell


def _fmt_p(p: float) -> str:
    if np.isnan(p):
        return "n/a"
    return f"{p:.3g}"


def _task_section(task, records, settings, assets_dir) -> list[str]:
    datasets, models, cell = _mean_table(records, task)
    if not datasets:
        return []
    smetric = SECONDARY[task]
    lines = [f"## {task.capitalize()} ({len(datasets)} datasets)", ""]

    # Per-dataset tables (every model that ran, sorted by primary).
    for ds in datasets:
        present = [(m, cell[(ds, m)]) for m in models if (ds, m) in cell]
        present.sort(key=lambda mc: mc[1]["primary"])
        pmetric = "rmse" if task == "regression" else "logloss"
        lines += [f"### {ds}", "",
                  f"| model | {pmetric} | {smetric} | fit[s] |", "|---|---|---|---|"]
        for m, c in present:
            lines.append(f"| {m} | {c['primary']:.4f} | {c['secondary']:.4f} "
                         f"| {c['fit']:.1f} |")
        lines.append("")

    # Aggregate significance over models complete on every dataset of this task.
    complete = [m for m in models if all((ds, m) in cell for ds in datasets)]
    incomplete = [m for m in models if m not in complete]
    if len(complete) < 3 or len(datasets) < 2:
        lines += ["_Significance aggregation skipped (need >= 3 models complete "
                  f"on >= 2 datasets; have {len(complete)} models, "
                  f"{len(datasets)} datasets)._", ""]
        if incomplete:
            lines += [f"_Models not complete on all datasets (excluded from any "
                      f"aggregate): {', '.join(incomplete)}._", ""]
        return lines

    scores = np.array([[cell[(ds, m)]["primary"] for m in complete]
                       for ds in datasets])
    avg = stats.average_ranks(scores, complete)
    fried_stat, fried_p = stats.friedman_test(scores)
    cd = stats.nemenyi_cd(len(complete), len(datasets), alpha=settings["alpha"])
    # Baseline = strongest *non-RepLeaf* model, so "RepLeaf beats the best GBDT"
    # is expressible (a baseline of the overall-best model can never be beaten).
    non_repleaf = [m for m in complete if m != "repleaf"]
    baseline = (min(non_repleaf, key=avg.get) if non_repleaf
                else min(complete, key=avg.get))
    wilc = stats.wilcoxon_pairs(scores, complete, baseline=baseline)
    base_scale = float(np.mean([cell[(ds, baseline)]["primary"] for ds in datasets]))

    lines += [
        f"### Aggregate — {task}", "",
        f"Friedman chi-square = {fried_stat:.3f}, p = {_fmt_p(fried_p)} "
        f"({'models differ' if fried_p < settings['alpha'] else 'no detected difference'} "
        f"at alpha={settings['alpha']}).", "",
    ]
    cd_png = (assets_dir / f"leaderboard-cd-{task}.png") if assets_dir else None
    lines += [stats.critical_difference_diagram(
        avg, cd, out_path=cd_png, title=f"{task}: avg rank"), ""]

    # Wilcoxon vs the strongest baseline + win/tie/loss, honest bolding.
    lines += [f"Baseline for pairwise tests: **{baseline}** (best average rank). "
              f"A model is **bold** when it beats the baseline with Wilcoxon "
              f"p < {settings['alpha']} **and** by more than the MRD "
              f"({settings['mrd']:.0%} relative).", "",
              "| model | avg rank | Wilcoxon p vs base | median delta | "
              "win/tie/loss | verdict |", "|---|---|---|---|---|---|"]
    for m in sorted(complete, key=avg.get):
        if m == baseline:
            lines.append(f"| {m} (baseline) | {avg[m]:.2f} | - | - | - | - |")
            continue
        stat_, p_, median_delta = wilc[m]
        cand = np.array([cell[(ds, m)]["primary"] for ds in datasets])
        base = np.array([cell[(ds, baseline)]["primary"] for ds in datasets])
        w, t, ll = stats.win_tie_loss(cand, base, mrd=settings["mrd"])
        rel = abs(median_delta) / base_scale if base_scale > 0 else 0.0
        improves = median_delta < 0 and p_ < settings["alpha"] and rel > settings["mrd"]
        regresses = median_delta > 0 and p_ < settings["alpha"] and rel > settings["mrd"]
        name = f"**{m}**" if improves else m
        verdict = ("sig. better" if improves else
                   "sig. worse" if regresses else "not sig.")
        lines.append(f"| {name} | {avg[m]:.2f} | {_fmt_p(p_)} | {median_delta:+.4f} "
                     f"| {w}/{t}/{ll} | {verdict} |")
    lines.append("")
    if incomplete:
        lines += [f"_Excluded from aggregate (incomplete): "
                  f"{', '.join(incomplete)}._", ""]
    return lines


def _manifest(settings, provenance) -> list[str]:
    lines = ["## Reproducibility manifest", ""]
    if provenance:
        pkgs = provenance.get("packages", {})
        lines += [
            f"- run_id: {provenance.get('run_id')}; "
            f"git: {provenance.get('git_sha')} "
            f"(dirty={provenance.get('git_dirty')})",
            f"- python: {provenance.get('python')} on {provenance.get('platform')}",
            f"- OMP_NUM_THREADS: {provenance.get('omp_num_threads')}",
            "- packages: " + ", ".join(f"{k}={v}" for k, v in pkgs.items()),
        ]
    lines += [
        f"- suite: {settings['suite']}; seeds: {settings['seeds']}; "
        f"HPO trials/model: {settings['n_trials']} (identical budget per model); "
        f"max_rows: {settings['max_rows']}",
        f"- split: {settings.get('train_prop', 0.70):.0%}/"
        f"{settings.get('val_prop', 0.15):.0%}/"
        f"{1 - settings.get('train_prop', 0.70) - settings.get('val_prop', 0.15):.0%} "
        "(Grinsztajn; train capped at 10k, stratified for classification); "
        f"alpha={settings['alpha']}; MRD={settings['mrd']:.0%} relative",
        "- Equal trial count is the budget; it is **not** equal wall-clock.",
        "",
    ]
    return lines


def build_report(records, settings, provenance=None, assets_dir=None) -> str:
    """Render the honest leaderboard markdown from completed ledger records."""
    lines = [
        "# Fair leaderboard (same-budget HPO)",
        "",
        "Auto-generated by `benchmarks/leaderboard.py`. Every model is tuned with "
        "an **identical Optuna trial budget** on the same split and seed, then "
        "scored once on held-out test data. This replaces the earlier "
        "tuned-vs-default comparisons.",
        "",
        "**Honest positioning:** under fair tuning RepLeafGBM is expected to be "
        "*competitive but not state-of-the-art on average*; its defensible "
        "support is in niche regimes (see the robust multi-output and "
        "router-extraction studies). No headline is claimed without a "
        "significance test, and null/negative results are reported alongside "
        "wins. **Model defaults are not changed here** — that requires a "
        "`results-analyst` report.",
        "",
    ]
    lines += _manifest(settings, provenance)
    if not records:
        lines += ["_No completed cells in the ledger yet._", ""]
        return "\n".join(lines) + "\n"
    for task in TASK_ORDER:
        lines += _task_section(task, records, settings, assets_dir)
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", default="grinsztajn_num_reg",
                   help="dataset suite name (see benchmarks/suites.py); the "
                        "Grinsztajn suites need network — use 'synthetic' offline")
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--models", nargs="*", default=None,
                   help=f"model families to run (default all: {list(hpo.FAMILIES)})")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--seed-list", type=int, nargs="*", default=None,
                   help="explicit seeds to run (overrides --seeds); used by the "
                        "keep-alive Colab loop to slice into short execs")
    p.add_argument("--n-trials", type=int, default=40)
    p.add_argument("--max-rows", type=int, default=20_000)
    p.add_argument("--train-prop", type=float, default=0.70)
    p.add_argument("--val-prop", type=float, default=0.15)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--mrd", type=float, default=0.01,
                   help="minimum relevant difference (relative) for a real win")
    p.add_argument("--quick", action="store_true",
                   help="smoke settings: seeds=2, n_trials=5, max_rows=2000, "
                        "quick dataset subset")
    p.add_argument("--strict", action="store_true",
                   help="fail instead of skipping on missing dep or model error")
    p.add_argument("--out", default=None)
    p.add_argument("--ledger", default=None)
    args = p.parse_args(argv)
    if args.quick:
        args.seeds = min(args.seeds, 2)
        args.n_trials = min(args.n_trials, 5)
        args.max_rows = min(args.max_rows, 2000)
    if args.models:  # allow comma-separated single arg
        flat = []
        for m in args.models:
            flat.extend(s for s in m.split(",") if s)
        args.models = flat
    if args.ledger is None:
        args.ledger = str(Path(__file__).resolve().parents[1] / "benchmarks"
                          / "results" / f"leaderboard_{args.suite}.jsonl")
    return args


def main(argv=None) -> Path:
    args = _parse(argv)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    return run(args)


if __name__ == "__main__":
    main()
