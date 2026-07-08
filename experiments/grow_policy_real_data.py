"""Experiment: grow_policy on real data — the Phase 33 default-change gate.

The synthetic comparison (experiments/results/2026-06-23-grow-policy-verdict.md)
kept ``leafwise`` as the default but flagged symmetric's wins as a plausible
oblivious-friendly design artifact, and prescribed this real-data follow-up as
the gate before any default change:

* **Datasets:** the 9 legacy OpenML-suite datasets (regression / binary /
  multiclass, several with categorical features) via ``benchmarks.suites`` —
  the union of ``benchmarks/openml_suite.py`` and
  ``benchmarks/benchmark_real_data.py`` loaders.
* **Policies x leaves:** {leafwise, depthwise, symmetric} x {constant,
  adaptive} (identity encoder; adaptive is the strongest RepLeaf arm on real
  data per the standing benchmark verdicts).
* **Capacity-match sensitivity sweep** (the key fix over the synthetic study):
  symmetric's complete ``2**d``-leaf tree is compared against leaf-wise at
  matched *effective capacity*, not matched hyperparameters. At each depth
  level ``d in {5, 6}`` leaf-wise runs both a "free" shape
  (``num_leaves=2**d - 1``, no depth cap — the stock benchmark shape) and a
  "capped" shape (``num_leaves=2**d``, ``max_depth=d``), plus a
  ``min_samples_leaf in {20, 50}`` sweep at ``d=5``. Symmetric gets
  ``num_leaves=2**d`` (it ignores the cap and always completes the level);
  depthwise gets the stock ``2**d - 1`` budget — at ``num_leaves=2**d``
  depthwise is *provably identical* to the capped leaf-wise shape (both grow
  every valid split to depth ``d``; confirmed bitwise on california), so the
  stock budget is where level-order vs gain-order growth can actually differ.
* **Protocol:** >=5 seeds, the 60/20/20 stratified split and early stopping
  shared with ``benchmarks/openml_suite.py``; resumable via
  ``benchmarks.ledger``; significance via ``benchmarks.stats`` (Friedman +
  Wilcoxon + win/tie/loss) and paired per-seed >=1 sigma separation flags.
* **Known limitations woven into the design (ADR 0006):** multiclass is
  covered by every policy (one scalar routing tree per class per round);
  symmetric's scalar-only limit bites only on vector multi-output targets,
  which are absent from this suite (a defensive n/a is recorded if ever hit).
  Symmetric routes categorical features as ordered thresholds (no subset
  splits) — that asymmetry is part of the test and each dataset's categorical
  count is reported.

**Decision rule (stated up front, from the 2026-06-23 verdict):** change the
default only if symmetric or depthwise beats *both* capacity-matched leaf-wise
shapes on a majority of real datasets by >=1 sigma (paired over seeds), across
the capacity regimes; otherwise keep ``leafwise`` and close the roadmap gate.

Run from the repository root::

    OMP_NUM_THREADS=1 PYTHONPATH=src python3 experiments/grow_policy_real_data.py --quick
    OMP_NUM_THREADS=1 PYTHONPATH=src python3 experiments/grow_policy_real_data.py --seeds 5

Shard across processes by dataset (each shard gets its own ledger), then merge::

    ... grow_policy_real_data.py --datasets california diamonds --ledger a.jsonl
    ... grow_policy_real_data.py --report-only --ledger a.jsonl b.jsonl [...]

The report is written to ``experiments/results/<date>-grow-policy-real-data.md``
(``--quick`` appends ``-quick``); the verdict is left to results-analyst.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import stats, suites  # noqa: E402
from benchmarks.ledger import Ledger  # noqa: E402
from benchmarks.openml_suite import ES_ROUNDS, _split_indices, r2, rmse  # noqa: E402
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score  # noqa: E402

from repleafgbm import RepLeafClassifier, RepLeafDataset, RepLeafRegressor  # noqa: E402

SUITE = "grow_policy_real"
LEAF_MODELS = ("constant", "adaptive")
#: --quick covers one dataset per task: credit_g has categorical features
#: (symmetric's ordered-threshold fallback) and wine confirms symmetric covers
#: multiclass (one scalar routing tree per class per round — no vector targets).
QUICK_DATASETS = ("california", "credit_g", "wine")


def arm_grid() -> list[tuple[str, dict]]:
    """``(arm_label, estimator overrides)`` for the capacity-match sweep.

    Symmetric gets ``num_leaves=2**d`` (it ignores the cap; the value is set
    for explicitness). Depthwise gets the stock ``2**d - 1`` budget: at
    ``num_leaves=2**d`` it is deterministically identical to the capped
    leaf-wise shape (both grow every valid split to depth ``d``), so the
    binding cap is where level-order vs gain-order growth can differ.
    """
    out: list[tuple[str, dict]] = []
    for msl in (20, 50):
        out += [
            (f"leafwise_free_nl31_msl{msl}",
             dict(grow_policy="leafwise", num_leaves=31, max_depth=-1,
                  min_samples_leaf=msl)),
            (f"leafwise_capped_nl32_d5_msl{msl}",
             dict(grow_policy="leafwise", num_leaves=32, max_depth=5,
                  min_samples_leaf=msl)),
            (f"depthwise_d5_msl{msl}",
             dict(grow_policy="depthwise", num_leaves=31, max_depth=5,
                  min_samples_leaf=msl)),
            (f"symmetric_d5_msl{msl}",
             dict(grow_policy="symmetric", num_leaves=32, max_depth=5,
                  min_samples_leaf=msl)),
        ]
    out += [
        ("leafwise_free_nl63_msl20",
         dict(grow_policy="leafwise", num_leaves=63, max_depth=-1,
              min_samples_leaf=20)),
        ("leafwise_capped_nl64_d6_msl20",
         dict(grow_policy="leafwise", num_leaves=64, max_depth=6,
              min_samples_leaf=20)),
        ("depthwise_d6_msl20",
         dict(grow_policy="depthwise", num_leaves=63, max_depth=6,
              min_samples_leaf=20)),
        ("symmetric_d6_msl20",
         dict(grow_policy="symmetric", num_leaves=64, max_depth=6,
              min_samples_leaf=20)),
    ]
    return out


#: challenger arm -> the two capacity-matched leaf-wise arms (free, capped).
#: The decision rule requires beating BOTH shapes at the same effective
#: capacity, so a "win" cannot be a leaf-wise-shape artifact.
MATCHED: dict[str, tuple[str, str]] = {
    "depthwise_d5_msl20": ("leafwise_free_nl31_msl20", "leafwise_capped_nl32_d5_msl20"),
    "symmetric_d5_msl20": ("leafwise_free_nl31_msl20", "leafwise_capped_nl32_d5_msl20"),
    "depthwise_d5_msl50": ("leafwise_free_nl31_msl50", "leafwise_capped_nl32_d5_msl50"),
    "symmetric_d5_msl50": ("leafwise_free_nl31_msl50", "leafwise_capped_nl32_d5_msl50"),
    "depthwise_d6_msl20": ("leafwise_free_nl63_msl20", "leafwise_capped_nl64_d6_msl20"),
    "symmetric_d6_msl20": ("leafwise_free_nl63_msl20", "leafwise_capped_nl64_d6_msl20"),
}

#: (regime label, [4 arms in table order]) — the aggregate significance blocks.
REGIMES: list[tuple[str, list[str]]] = [
    ("d5 msl20", ["leafwise_free_nl31_msl20", "leafwise_capped_nl32_d5_msl20",
                  "depthwise_d5_msl20", "symmetric_d5_msl20"]),
    ("d5 msl50", ["leafwise_free_nl31_msl50", "leafwise_capped_nl32_d5_msl50",
                  "depthwise_d5_msl50", "symmetric_d5_msl50"]),
    ("d6 msl20", ["leafwise_free_nl63_msl20", "leafwise_capped_nl64_d6_msl20",
                  "depthwise_d6_msl20", "symmetric_d6_msl20"]),
]


# --------------------------------------------------------------------------- #
# Cell execution
# --------------------------------------------------------------------------- #
def _prepare(name: str, max_rows: int, seed: int):
    """Load + split one dataset with the openml_suite protocol (60/20/20,
    stratified for classification), returning fit-ready dataset objects."""
    spec = suites.find(name)
    X_all, y_all, cats = suites.load(spec)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X_all))[: min(max_rows, len(X_all))]
    i_tr, i_va, i_te = _split_indices(idx, y_all[idx], spec.task, rng)
    Xtr, Xva, Xte = (X_all.iloc[i] for i in (i_tr, i_va, i_te))
    ytr, yva, yte = y_all[i_tr], y_all[i_va], y_all[i_te]

    train_ds = RepLeafDataset(Xtr, ytr, categorical_features=cats)
    valid_ds = RepLeafDataset(Xva, yva, metadata=train_ds.metadata)
    test_ds = RepLeafDataset(Xte, yte, metadata=train_ds.metadata)
    classes = np.unique(y_all) if spec.task != "regression" else None
    return spec.task, (train_ds, valid_ds, test_ds, yte, classes), len(idx), len(cats)


def _fit_eval(task, leaf_model, overrides, seed, data, n_estimators):
    """Fit one arm with early stopping; return (primary, secondary, fit_s,
    best_iteration). Primary is lower-is-better (RMSE / logloss)."""
    train_ds, valid_ds, test_ds, yte, classes = data
    cls = RepLeafRegressor if task == "regression" else RepLeafClassifier
    model = cls(n_estimators=n_estimators, learning_rate=0.1,
                leaf_model=leaf_model, encoder="identity",
                early_stopping_rounds=ES_ROUNDS, random_state=seed, **overrides)
    t0 = time.perf_counter()
    model.fit(train_ds, eval_set=[valid_ds])
    fit_s = time.perf_counter() - t0
    best_it = model.best_iteration_
    if task == "regression":
        pred = model.predict(test_ds)
        return rmse(yte, pred), r2(yte, pred), fit_s, best_it
    proba = model.predict_proba(test_ds)
    ll = float(log_loss(yte, proba, labels=classes))
    if task == "binary":
        return ll, float(roc_auc_score(yte, proba[:, 1])), fit_s, best_it
    pred = classes[np.argmax(proba, axis=1)]
    return ll, float(accuracy_score(yte, pred)), fit_s, best_it


def run(args, ledger: Ledger) -> int:
    """Execute pending (dataset, leaf_model, arm, seed) cells. Returns the
    number of non-NotImplementedError failures (retryable on resume)."""
    arms = arm_grid()
    seeds = (list(args.seed_list) if args.seed_list
             else list(range(args.seeds)))
    errors = 0
    for name in args.datasets:
        for seed in seeds:
            pending = [(lm, label, ov) for lm in LEAF_MODELS
                       for (label, ov) in arms
                       if not ledger.done(SUITE, name, f"{lm}|{label}", seed)]
            if not pending:
                continue
            print(f"=== {name} seed={seed} ({len(pending)} cells) ===", flush=True)
            try:
                task, data, n_used, n_cats = _prepare(name, args.max_rows, seed)
            except Exception as exc:  # network/data issues: retryable
                print(f"  [error dataset] {name} seed={seed}: "
                      f"{type(exc).__name__}: {exc}", flush=True)
                errors += 1
                continue
            base = {"task": task, "n_used": n_used, "n_cats": n_cats}
            for lm, label, ov in pending:
                try:
                    p, s, fit_s, best_it = _fit_eval(
                        task, lm, ov, seed, data, args.n_estimators)
                    payload = dict(base, leaf_model=lm, arm=label, primary=p,
                                   secondary=s, fit_seconds=fit_s,
                                   best_iteration=best_it)
                    print(f"  [done] {lm}|{label}: {p:.4f} ({fit_s:.1f}s)",
                          flush=True)
                except NotImplementedError as exc:
                    # Defensive: symmetric raises on vector multi-output
                    # targets (ADR 0006) — not expected on this suite. Record
                    # the n/a so resume does not retry and the report states
                    # it explicitly.
                    payload = dict(base, leaf_model=lm, arm=label,
                                   skipped=str(exc))
                    print(f"  [n/a] {lm}|{label}: NotImplementedError", flush=True)
                except Exception as exc:  # retryable; not recorded
                    print(f"  [error] {lm}|{label}: {type(exc).__name__}: {exc}",
                          flush=True)
                    errors += 1
                    continue
                ledger.record(SUITE, name, f"{lm}|{label}", seed, payload)
    return errors


# --------------------------------------------------------------------------- #
# Report generation (pure function over ledger records — unit-testable)
# --------------------------------------------------------------------------- #
def _collect(records):
    """records -> (per-cell raw values, dataset meta). Cell key:
    (dataset, leaf_model, arm) -> {"seeds": [..], "primary": [..], ...}."""
    cells: dict[tuple[str, str, str], dict] = defaultdict(
        lambda: {"seeds": [], "primary": [], "secondary": [], "fit": [],
                 "best_it": []})
    skipped: dict[tuple[str, str, str], str] = {}
    meta: dict[str, dict] = {}
    for r in records:
        p = r["payload"]
        ds = r["dataset"]
        meta.setdefault(ds, {"task": p.get("task"), "n_used": p.get("n_used"),
                             "n_cats": p.get("n_cats")})
        key = (ds, p.get("leaf_model"), p.get("arm"))
        if "skipped" in p:
            skipped[key] = p["skipped"]
            continue
        c = cells[key]
        c["seeds"].append(r["seed"])
        c["primary"].append(float(p["primary"]))
        c["secondary"].append(float(p["secondary"]))
        c["fit"].append(float(p["fit_seconds"]))
        c["best_it"].append(p.get("best_iteration"))
    for c in cells.values():  # deterministic seed order for pairing + appendix
        order = np.argsort(c["seeds"])
        for f in ("seeds", "primary", "secondary", "fit", "best_it"):
            c[f] = [c[f][i] for i in order]
    return cells, skipped, meta


def _paired_delta(cells, ds, lm, challenger, baseline):
    """Paired per-seed (challenger - baseline) primary deltas on shared seeds.
    Returns (mean, sigma, n) or None when there is no seed overlap."""
    a, b = cells.get((ds, lm, challenger)), cells.get((ds, lm, baseline))
    if not a or not b:
        return None
    shared = sorted(set(a["seeds"]) & set(b["seeds"]))
    if not shared:
        return None
    av = {s: v for s, v in zip(a["seeds"], a["primary"])}
    bv = {s: v for s, v in zip(b["seeds"], b["primary"])}
    d = np.array([av[s] - bv[s] for s in shared], dtype=float)
    sigma = float(d.std(ddof=1)) if len(d) >= 2 else float("nan")
    return float(d.mean()), sigma, len(d)


def _sep_str(delta) -> str:
    """Render one paired delta as 'mean (x.xσ)'; **bold** at >=1σ separation."""
    if delta is None:
        return "n/a"
    mean, sigma, n = delta
    if n < 2 or not np.isfinite(sigma):
        return f"{mean:+.4f} (n={n})"
    if sigma < 1e-12:
        sep = float("inf") if abs(mean) > 1e-12 else 0.0
    else:
        sep = abs(mean) / sigma
    txt = f"{mean:+.4f} ({sep:.1f}σ)"
    return f"**{txt}**" if sep >= 1.0 else txt


def _is_sep_win(delta) -> bool | None:
    """True/False = >=1σ better/worse than baseline; None = not separated."""
    if delta is None:
        return None
    mean, sigma, n = delta
    if n < 2 or not np.isfinite(sigma):
        return None
    if sigma < 1e-12:
        return None if abs(mean) <= 1e-12 else mean < 0
    if abs(mean) < sigma:
        return None
    return mean < 0


def build_report(records, settings, provenance=None, assets_dir=None) -> str:
    cells, skipped, meta = _collect(records)
    datasets = sorted(meta)
    arms = arm_grid()
    lines = [
        "# grow_policy on real data — Phase 33 default-change gate",
        "",
        "Auto-generated by `experiments/grow_policy_real_data.py`. Compares "
        "`grow_policy in {leafwise, depthwise, symmetric}` x `leaf_model in "
        "{constant, adaptive}` (identity encoder) on the 9 legacy OpenML-suite "
        "datasets under a capacity-match sensitivity sweep (see the module "
        "docstring). Primary metric: RMSE (regression) / logloss "
        "(classification), lower is better; mean ± std over seeds.",
        "",
        "**Decision rule (fixed before the run):** change the default only if "
        "symmetric or depthwise beats **both** capacity-matched leaf-wise "
        "shapes (free + capped) on a **majority of datasets by >=1σ** (paired "
        "per-seed deltas, sample std), across the capacity regimes. The "
        "keep/change/null verdict itself is written by results-analyst, not "
        "this script.",
        "",
    ]
    # Manifest
    lines += ["## Reproducibility manifest", ""]
    if provenance:
        pkgs = provenance.get("packages", {})
        lines += [
            f"- run_id: {provenance.get('run_id')}; git: {provenance.get('git_sha')} "
            f"(dirty={provenance.get('git_dirty')})",
            f"- python: {provenance.get('python')} on {provenance.get('platform')}",
            f"- OMP_NUM_THREADS: {provenance.get('omp_num_threads')}",
            "- packages: " + ", ".join(f"{k}={v}" for k, v in pkgs.items()),
        ]
    lines += [
        f"- settings: seeds={settings.get('seeds')}, "
        f"max_rows={settings.get('max_rows')}, "
        f"n_estimators={settings.get('n_estimators')} (lr=0.1, "
        f"early stopping {ES_ROUNDS} rounds on the validation split), "
        f"split 60/20/20 (stratified for classification), "
        f"quick={settings.get('quick')}",
        "- arms (estimator overrides):",
    ]
    for label, ov in arms:
        lines.append(f"  - `{label}`: {ov}")
    lines.append("")

    # Per-dataset tables
    win_counts: dict[tuple[str, str], dict[str, list[str]]] = defaultdict(
        lambda: {"win": [], "loss": [], "tie": []})
    for ds in datasets:
        m = meta[ds]
        pmetric = "rmse" if m["task"] == "regression" else "logloss"
        smetric = {"regression": "r2", "binary": "auc",
                   "multiclass": "accuracy"}[m["task"]]
        lines += [f"## {ds} ({m['task']}, n={m['n_used']}, "
                  f"categorical: {m['n_cats']})", ""]
        for lm in LEAF_MODELS:
            present = [(label, cells[(ds, lm, label)]) for label, _ in arms
                       if (ds, lm, label) in cells]
            if not present:
                continue
            lines += [f"### leaf_model = {lm}", "",
                      f"| arm | {pmetric} (mean ± std) | {smetric} | fit[s] | "
                      "best_it | Δ vs LW-free | Δ vs LW-capped |",
                      "|---|---|---|---|---|---|---|"]
            for label, c in sorted(present, key=lambda kv: np.mean(kv[1]["primary"])):
                p = np.array(c["primary"])
                its = [i for i in c["best_it"] if i is not None]
                it_txt = f"{np.mean(its):.0f}" if its else "-"
                d_free = d_cap = "-"
                if label in MATCHED:
                    free, capped = MATCHED[label]
                    df = _paired_delta(cells, ds, lm, label, free)
                    dc = _paired_delta(cells, ds, lm, label, capped)
                    d_free, d_cap = _sep_str(df), _sep_str(dc)
                    wf, wc = _is_sep_win(df), _is_sep_win(dc)
                    bucket = ("win" if wf and wc else
                              "loss" if wf is False and wc is False else "tie")
                    win_counts[(lm, label)][bucket].append(ds)
                lines.append(
                    f"| {label} | {p.mean():.4f} ± {p.std(ddof=1) if len(p) > 1 else 0.0:.4f} "
                    f"| {np.mean(c['secondary']):.4f} | {np.mean(c['fit']):.1f} "
                    f"| {it_txt} | {d_free} | {d_cap} |")
            for label, _ in arms:  # explicit n/a rows (symmetric multi-output)
                if (ds, lm, label) in skipped:
                    lines.append(f"| {label} | n/a — NotImplementedError "
                                 "(ADR 0006: symmetric is scalar-only in v0) "
                                 "| - | - | - | - | - |")
            lines.append("")

    # Decision-rule summary
    lines += [
        "## Decision-rule summary (challenger vs both matched leaf-wise shapes)",
        "",
        "A dataset counts as a **win** only when the challenger is >=1σ better "
        "than *both* the free and the capped leaf-wise shape (paired), a "
        "**loss** when >=1σ worse than both; everything else is a tie/mixed.",
        "",
        "| leaf_model | challenger | wins | ties/mixed | losses | datasets |",
        "|---|---|---|---|---|---|",
    ]
    for lm in LEAF_MODELS:
        for label in MATCHED:
            wc = win_counts.get((lm, label))
            if wc is None:
                continue
            n_ds = sum(len(v) for v in wc.values())
            win_names = ", ".join(wc["win"]) or "-"
            loss_names = ", ".join(wc["loss"]) or "-"
            lines.append(
                f"| {lm} | {label} | {len(wc['win'])} ({win_names}) "
                f"| {len(wc['tie'])} | {len(wc['loss'])} ({loss_names}) "
                f"| {n_ds} |")
    lines.append("")

    # Aggregate significance per regime x leaf_model
    for lm in LEAF_MODELS:
        for regime, regime_arms in REGIMES:
            complete = [ds for ds in datasets
                        if all((ds, lm, a) in cells for a in regime_arms)]
            if len(complete) < 2:
                continue
            scores = np.array([[np.mean(cells[(ds, lm, a)]["primary"])
                                for a in regime_arms] for ds in complete])
            lines += [f"## Aggregate — {lm}, regime {regime} "
                      f"({len(complete)} datasets with all 4 arms)", ""]
            excl = [ds for ds in datasets if ds not in complete]
            if excl:
                lines += [f"_Excluded (arms missing for this regime): "
                          f"{', '.join(excl)}._", ""]
            fried_stat, fried_p = stats.friedman_test(scores)
            avg = stats.average_ranks(scores, regime_arms)
            cd = stats.nemenyi_cd(len(regime_arms), len(complete))
            lines += [f"Friedman chi-square = {fried_stat:.3f}, "
                      f"p = {fried_p:.3g}.", ""]
            cd_png = None
            if assets_dir is not None:
                slug = f"{lm}-{regime.replace(' ', '-')}"
                cd_png = Path(assets_dir) / f"grow-policy-real-cd-{slug}.png"
            lines += [stats.critical_difference_diagram(
                avg, cd, out_path=cd_png,
                title=f"{lm} / {regime}: avg rank"), ""]
            baseline = regime_arms[0]  # leafwise_free
            wilc = stats.wilcoxon_pairs(scores, regime_arms, baseline=baseline)
            lines += [f"Wilcoxon vs `{baseline}` (across dataset means; "
                      "negative median delta = challenger better):", "",
                      "| arm | p | median Δ | win/tie/loss (MRD 1%) |",
                      "|---|---|---|---|"]
            for j, a in enumerate(regime_arms):
                if a == baseline:
                    continue
                _, p_, md = wilc[a]
                w, t, ll = stats.win_tie_loss(scores[:, j], scores[:, 0],
                                              mrd=0.01)
                lines.append(f"| {a} | {p_:.3g} | {md:+.4f} | {w}/{t}/{ll} |")
            lines.append("")

    # Raw per-seed appendix (the gate asks for raw values, not just means)
    lines += ["## Appendix — raw per-seed primary values", ""]
    for ds in datasets:
        lines.append(f"### {ds}")
        lines.append("")
        for lm in LEAF_MODELS:
            for label, _ in arms:
                c = cells.get((ds, lm, label))
                if c is None:
                    continue
                vals = ", ".join(f"s{s}={v:.4f}"
                                 for s, v in zip(c["seeds"], c["primary"]))
                lines.append(f"- {lm}|{label}: {vals}")
        lines.append("")

    lines += [
        "## Notes / limitations",
        "",
        "- `symmetric` is numeric/ordered + scalar-only in v0 (ADR 0006). "
        "Multiclass is still covered (one scalar routing tree per class per "
        "round); the scalar-only limit bites only on vector multi-output "
        "targets, absent from this suite. On categorical datasets symmetric "
        "routes categoricals as ordered thresholds while leafwise/depthwise "
        "use gradient-sorted subset splits — a real, documented asymmetry "
        "that is part of this test.",
        "- Equal arm budget = same n_estimators + early stopping on the shared "
        "validation split; `best_it` shows where early stopping actually "
        "landed, so capacity differences are visible rather than assumed.",
        "- This report presents evidence only; the keep/change/null verdict is "
        "results-analyst's.",
        "",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="*",
                   default=[d.name for d in suites.get_suite("legacy").datasets])
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--seed-list", type=int, nargs="*", default=None,
                   help="explicit seeds (overrides --seeds); for sharded runs")
    p.add_argument("--max-rows", type=int, default=6000)
    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument("--quick", action="store_true",
                   help="wiring smoke: 3 datasets (one per task), 2 seeds, "
                        "max_rows=1000, n_estimators=60, separate ledger")
    p.add_argument("--ledger", nargs="+", default=None,
                   help="ledger JSONL path(s); >1 only makes sense with "
                        "--report-only (merges sharded runs)")
    p.add_argument("--report-only", action="store_true",
                   help="skip fitting; rebuild the report from the ledger(s)")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    if args.quick:
        args.datasets = [d for d in args.datasets if d in QUICK_DATASETS]
        args.seeds = min(args.seeds, 2)
        args.max_rows = min(args.max_rows, 1000)
        args.n_estimators = min(args.n_estimators, 60)
    if args.ledger is None:
        stem = "grow_policy_real_data" + ("_quick" if args.quick else "")
        args.ledger = [str(ROOT / "benchmarks" / "results" / f"{stem}.jsonl")]
    if not args.report_only and len(args.ledger) > 1:
        p.error("multiple --ledger paths require --report-only")
    if args.out is None:
        suffix = "-quick" if args.quick else ""
        args.out = str(ROOT / "experiments" / "results" /
                       f"{date.today().isoformat()}-grow-policy-real-data{suffix}.md")
    return args


def main(argv=None) -> Path:
    args = _parse(argv)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    errors = 0
    ledgers = [Ledger(Path(lp)) for lp in args.ledger]
    if not args.report_only:
        errors = run(args, ledgers[0])

    records = [r for led in ledgers for r in led.records()]
    settings = {"seeds": (args.seed_list or list(range(args.seeds))),
                "max_rows": args.max_rows, "n_estimators": args.n_estimators,
                "quick": bool(args.quick)}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_report(records, settings,
                                     provenance=ledgers[0].provenance,
                                     assets_dir=out_path.parent))
    print(f"\nreport written to {out_path}")
    if errors:
        print(f"WARNING: {errors} cell(s) failed and were not recorded; "
              "re-run to retry them.", file=sys.stderr)
        sys.exit(1)
    return out_path


if __name__ == "__main__":
    main()
