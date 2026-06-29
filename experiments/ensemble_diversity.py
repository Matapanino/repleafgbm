"""Does RepLeafGBM add ensemble value despite being mid-pack standalone?

RepLeaf is a consistent mid-pack learner on the Grinsztajn leaderboard, but it is
architecturally distinct from GBDTs (raw-feature routing + representation leaves).
Per the Krogh-Vedelsby ambiguity identity and the AutoGluon/TabArena/Caruana
evidence (docs/research/2026-06-29-diverse-ensemble-stacking.md), a weaker-but-
diverse member can still improve an ensemble if its errors are decorrelated.

This experiment REUSES the leaderboard's per-(dataset, model, seed) tuned params
from the resumable ledger (`benchmarks/results/leaderboard_<suite>.jsonl`):
refit each model once (no new HPO, no `src/` change), cache held-out TEST
predictions, and evaluate **fixed unweighted-average** ensembles (no leakage):

* best single model — selected by **validation** score from the ledger (not test);
* `gbdt_avg`  — unweighted mean of the 4 GBDTs;
* `gbdt+repleaf` — the 4 GBDTs plus RepLeaf;
* `all_avg` — all 5.

Reported per dataset (mean over seeds) and aggregated across datasets:
diversity (inter-GBDT vs RepLeaf-vs-GBDT prediction correlation), the
Krogh-Vedelsby ambiguity harvest (regression), and a Wilcoxon signed-rank test +
bootstrap CI for **gbdt+repleaf vs gbdt_avg** — the clean "RepLeaf adds value"
signal. Refits are cached to `experiments/results/ensemble_cache/<suite>/`, so the
run is resumable and the analysis re-runs without refitting.

    OMP_NUM_THREADS=1 PYTHONPATH=src python3 experiments/ensemble_diversity.py \\
        --suite grinsztajn_num_reg
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from benchmarks import stats  # noqa: E402
from benchmarks.hpo import build_model  # noqa: E402
from benchmarks.leaderboard import _prepare  # noqa: E402
from benchmarks.suites import get_suite  # noqa: E402

GBDTS = ("lightgbm", "xgboost", "catboost", "hist_gradient_boosting")
ALL_MODELS = ("repleaf",) + GBDTS


def _rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def _metric(task, y, pred, classes):
    """Lower-is-better primary. pred: 1-D for regression, (n, K) proba for clf."""
    if task == "regression":
        return _rmse(y, pred)
    return float(log_loss(y, pred, labels=classes))


def _ensemble(task, preds, members):
    """Unweighted average over members (predictions for reg, probabilities clf)."""
    return np.mean([preds[m] for m in members], axis=0)


def _pred_vec(task, pred):
    """A 1-D vector per model for correlation: prediction (reg) or P(class1) (binary)."""
    return pred if task == "regression" else pred[:, 1]


def load_ledger(suite):
    """(dataset, model, seed) -> (params, val_value); plus dataset -> task."""
    path = ROOT / "benchmarks" / "results" / f"leaderboard_{suite}.jsonl"
    params, task_of = {}, {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or '"key"' not in line:
            continue
        o = json.loads(line)
        pl = o["payload"]
        params[(o["dataset"], o["model"], int(o["seed"]))] = (pl["params"], pl["val_value"])
        task_of[o["dataset"]] = pl["task"]
    return params, task_of


def cell_preds(suite, spec, seed, ledger, cache_dir, max_rows, train_prop, val_prop):
    """Refit all 5 models (cached) on this (dataset, seed); return (preds, yte, classes)."""
    cache = cache_dir / f"{spec.name}_{seed}.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        preds = {m: d[f"pred_{m}"] for m in ALL_MODELS}
        classes = None if d["classes"].dtype == object else d["classes"]
        return preds, d["yte"], (None if classes is None or classes.size == 0 else classes)
    Xtr, ytr, _Xva, _yva, Xte, yte, classes, _n = _prepare(
        spec, seed, max_rows, train_prop, val_prop)
    preds = {}
    for m in ALL_MODELS:
        p, _v = ledger[(spec.name, m, seed)]
        model = build_model(m, p, spec.task, seed)
        model.fit(Xtr, ytr)
        preds[m] = (model.predict(Xte) if spec.task == "regression"
                    else model.predict_proba(Xte))
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cache, yte=yte,
             classes=(classes if classes is not None else np.array([])),
             **{f"pred_{m}": preds[m] for m in ALL_MODELS})
    return preds, yte, classes


def main(argv=None) -> Path:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", required=True)
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--max-rows", type=int, default=20_000)
    p.add_argument("--train-prop", type=float, default=0.70)
    p.add_argument("--val-prop", type=float, default=0.15)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--mrd", type=float, default=0.01)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    ledger, task_of = load_ledger(args.suite)
    seeds = sorted({s for (_d, _m, s) in ledger})
    specs = [d for d in get_suite(args.suite).datasets
             if (args.datasets is None or d.name in set(args.datasets))]
    cache_dir = ROOT / "experiments" / "results" / "ensemble_cache" / args.suite

    # per-dataset accumulators (mean over seeds)
    agg = defaultdict(lambda: defaultdict(list))
    task = None
    for spec in specs:
        if spec.task not in ("regression", "binary"):
            continue  # diversity vec assumes reg or binary
        for seed in seeds:
            if not all((spec.name, m, seed) in ledger for m in ALL_MODELS):
                continue
            try:
                preds, yte, classes = cell_preds(
                    args.suite, spec, seed, ledger, cache_dir,
                    args.max_rows, args.train_prop, args.val_prop)
            except Exception as exc:  # pragma: no cover - data/fit issues
                print(f"  [skip] {spec.name} seed={seed}: "
                      f"{type(exc).__name__}: {exc}", flush=True)
                continue
            task = spec.task
            single = {m: _metric(task, yte, preds[m], classes) for m in ALL_MODELS}
            vals = {m: ledger[(spec.name, m, seed)][1] for m in ALL_MODELS}
            best_single = single[min(vals, key=vals.get)]  # val-selected, no leakage

            m_gbdt = _metric(task, yte, _ensemble(task, preds, GBDTS), classes)
            m_gr = _metric(task, yte, _ensemble(task, preds, GBDTS + ("repleaf",)), classes)
            m_all = _metric(task, yte, _ensemble(task, preds, ALL_MODELS), classes)

            vecs = {m: _pred_vec(task, preds[m]) for m in ALL_MODELS}
            inter = np.mean([np.corrcoef(vecs[a], vecs[b])[0, 1]
                             for i, a in enumerate(GBDTS) for b in GBDTS[i + 1:]])
            rep = np.mean([np.corrcoef(vecs["repleaf"], vecs[g])[0, 1] for g in GBDTS])

            a = agg[spec.name]
            a["best_single"].append(best_single)
            a["repleaf"].append(single["repleaf"])
            a["gbdt_avg"].append(m_gbdt)
            a["gbdt+repleaf"].append(m_gr)
            a["all_avg"].append(m_all)
            a["r_interGBDT"].append(float(inter))
            a["r_repleafGBDT"].append(float(rep))
            if task == "regression":  # Krogh-Vedelsby ambiguity harvest
                ens_g = _ensemble(task, preds, GBDTS)
                ens_gr = _ensemble(task, preds, GBDTS + ("repleaf",))
                A_g = np.mean([_rmse(yte, preds[m]) ** 2 for m in GBDTS]) - _rmse(yte, ens_g) ** 2
                A_gr = (np.mean([_rmse(yte, preds[m]) ** 2 for m in GBDTS + ("repleaf",)])
                        - _rmse(yte, ens_gr) ** 2)
                a["dA"].append(float(A_gr - A_g))
            print(f"  {spec.name} seed={seed}: gbdt={m_gbdt:.4f} +repleaf={m_gr:.4f} "
                  f"(r_inter={inter:.3f} r_rep={rep:.3f})", flush=True)

    out = _report(args, task, agg)
    out_path = (Path(args.out) if args.out else ROOT / "experiments" / "results"
                / f"{date.today().isoformat()}-ensemble-diversity-{args.suite}.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out) + "\n")
    print(f"\nreport written to {out_path}", flush=True)
    return out_path


def _report(args, task, agg):
    pm = "rmse" if task == "regression" else "logloss"
    out = [
        f"# Ensemble diversity — {args.suite}",
        "",
        "Auto-generated by `experiments/ensemble_diversity.py`. Does adding "
        "RepLeaf to the tuned-GBDT ensemble help? Tuned params reused from the "
        "leaderboard ledger; unweighted-average ensembles (no leakage); best "
        "single picked by validation score. Lower is better.",
        "",
        f"Datasets: {len(agg)} (suite is single-task). Primary: **{pm}**. "
        f"alpha={args.alpha}, MRD={args.mrd:.0%} relative.",
        "",
        "| dataset | best_single | gbdt_avg | gbdt+repleaf | all_avg | "
        "r(GBDT-GBDT) | r(repleaf-GBDT) |",
        "|---|---|---|---|---|---|---|",
    ]
    names = sorted(agg)
    for d in names:
        a = agg[d]
        out.append(
            f"| {d} | {np.mean(a['best_single']):.4f} | {np.mean(a['gbdt_avg']):.4f} "
            f"| {np.mean(a['gbdt+repleaf']):.4f} | {np.mean(a['all_avg']):.4f} "
            f"| {np.mean(a['r_interGBDT']):.3f} | {np.mean(a['r_repleafGBDT']):.3f} |")

    if len(names) >= 2:
        gbdt = np.array([np.mean(agg[d]["gbdt_avg"]) for d in names])
        gr = np.array([np.mean(agg[d]["gbdt+repleaf"]) for d in names])
        bs = np.array([np.mean(agg[d]["best_single"]) for d in names])
        scores = np.column_stack([gbdt, gr])
        _, pval, md = stats.wilcoxon_pairs(scores, ["gbdt_avg", "gbdt+repleaf"],
                                           baseline="gbdt_avg")["gbdt+repleaf"]
        w, t, ll = stats.win_tie_loss(gr, gbdt, mrd=args.mrd)
        lo, hi = stats.bootstrap_ci(gbdt - gr, seed=0)  # >0 => repleaf helps
        r_inter = np.mean([np.mean(agg[d]["r_interGBDT"]) for d in names])
        r_rep = np.mean([np.mean(agg[d]["r_repleafGBDT"]) for d in names])
        better = md < 0 and pval < args.alpha
        verdict = "RepLeaf adds ensemble value" if better else "not significant"
        decorr = "more decorrelated" if r_rep < r_inter else "not more decorrelated"
        out += [
            "", "## Aggregate", "",
            f"- **gbdt+repleaf vs gbdt_avg**: median delta {md:+.4f} "
            f"({'repleaf helps' if md < 0 else 'no help'}), Wilcoxon p={pval:.3g}, "
            f"win/tie/loss {w}/{t}/{ll}, bootstrap 95% CI of mean improvement "
            f"[{lo:+.4f}, {hi:+.4f}] -> **{verdict}**.",
            f"- **diversity**: mean r(GBDT-GBDT)={r_inter:.3f} vs "
            f"r(repleaf-GBDT)={r_rep:.3f} (RepLeaf {decorr}).",
            f"- best ensemble beats val-selected best single on "
            f"{int(np.sum(np.minimum(gbdt, gr) < bs))}/{len(names)} datasets.",
        ]
        if task == "regression":
            dA = np.mean([np.mean(agg[d]["dA"]) for d in names])
            out.append(f"- **ambiguity harvest** Δ when adding RepLeaf: mean {dA:+.4f} "
                       f"({'positive — diversity exploited' if dA > 0 else 'non-positive'}).")
    return out


if __name__ == "__main__":
    main()
