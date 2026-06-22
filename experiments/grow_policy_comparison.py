"""Experiment: tree growth policies — leafwise vs depthwise vs symmetric.

Compares the three ``grow_policy`` values (ADR 0006) across synthetic datasets
that vary in structure, noise, and size, plus a couple of real datasets when
their loaders are available. Hypotheses:

* symmetric (oblivious) trees regularize strongly, so they should help most on
  small / noisy data and hurt where high-capacity leaf-wise fitting pays off;
* depthwise is a balanced baseline between the two;
* leaf-wise (the default) should stay strongest on larger / structured data.

Each (dataset, task, leaf_model, policy) is fit with early stopping on a
validation split and matched capacity (leaf-wise num_leaves=31 vs depth-policy
max_depth=5), then scored on a held-out test set, averaged over seeds. The
report is written to experiments/results/grow_policy_comparison.md; the deeper
keep/change-default verdict is left to results-analyst.

Run from the repository root:
    PYTHONPATH=src python3 experiments/grow_policy_comparison.py            # full
    PYTHONPATH=src python3 experiments/grow_policy_comparison.py --quick    # fast smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

from repleafgbm import RepLeafClassifier, RepLeafRegressor  # noqa: E402

POLICIES = ("leafwise", "depthwise", "symmetric")
LEAF_MODELS = ("constant", "embedded_linear")
MAX_DEPTH = 5
ES = 20


# --------------------------------------------------------------------------- #
# Synthetic data
# --------------------------------------------------------------------------- #
def _signal(X: np.ndarray, kind: str) -> np.ndarray:
    if kind == "piecewise":
        return (
            np.where(X[:, 0] > 0.0, 3.0, -2.0)
            + 2.0 * X[:, 1]
            - 1.0 * X[:, 2]
            + 1.5 * np.where(X[:, 3] > 0.5, 1.0, -1.0)
        )
    # smooth / interaction structure
    return np.sin(2.0 * X[:, 0]) + X[:, 1] ** 2 - X[:, 2] * X[:, 3]


def make_regression(n, seed, noise, kind):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 8))
    y = _signal(X, kind) + rng.normal(0.0, noise, n)
    return X, y


def make_binary(n, seed, noise, kind):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 8))
    logit = _signal(X, kind) + rng.normal(0.0, noise, n)
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    return X, y


def make_multiclass(n, seed, noise):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 8))
    scores = np.column_stack([
        X[:, 0] + X[:, 1],
        -X[:, 0] + X[:, 2],
        X[:, 3] - X[:, 1],
    ])
    y = np.argmax(scores + rng.normal(0.0, noise, scores.shape), axis=1)
    return X, y


# (name, task, builder) — builder(seed) -> (X, y)
def datasets(quick: bool):
    specs = [
        ("reg_piecewise_clean_n3000", "regression",
         lambda s: make_regression(3000, s, noise=0.3, kind="piecewise")),
        ("reg_piecewise_noisy_n600", "regression",
         lambda s: make_regression(600, s, noise=2.5, kind="piecewise")),
        ("reg_smooth_n2000", "regression",
         lambda s: make_regression(2000, s, noise=0.5, kind="smooth")),
        ("bin_piecewise_n2000", "binary",
         lambda s: make_binary(2000, s, noise=0.3, kind="piecewise")),
        ("bin_noisy_n600", "binary",
         lambda s: make_binary(600, s, noise=1.5, kind="piecewise")),
        ("mc3_n2000", "multiclass",
         lambda s: make_multiclass(2000, s, noise=0.6)),
    ]
    if quick:
        specs = [specs[1], specs[3], specs[5]]  # one of each task
    return specs


# --------------------------------------------------------------------------- #
# Fit / eval
# --------------------------------------------------------------------------- #
def _rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def _logloss(y, P):
    P = np.clip(P, 1e-12, 1 - 1e-12)
    if P.ndim == 1:
        return float(-np.mean(y * np.log(P) + (1 - y) * np.log(1 - P)))
    return float(-np.mean(np.log(P[np.arange(len(y)), y])))


def _split(X, y, seed):
    rng = np.random.default_rng(1000 + seed)
    idx = rng.permutation(len(X))
    n_te = len(X) // 5
    n_va = len(X) // 5
    te, va, tr = idx[:n_te], idx[n_te:n_te + n_va], idx[n_te + n_va:]
    return (X[tr], y[tr]), (X[va], y[va]), (X[te], y[te])


def _estimator(task, policy, leaf_model, seed):
    common = dict(
        n_estimators=400, learning_rate=0.1, min_samples_leaf=20,
        leaf_model=leaf_model, encoder="plr", l2_leaf=1.0,
        early_stopping_rounds=ES, grow_policy=policy, random_state=seed,
    )
    if policy == "leafwise":
        common.update(num_leaves=31)
    else:
        # Depth-controlled; give depthwise leaf headroom so depth governs.
        common.update(max_depth=MAX_DEPTH, num_leaves=2 ** (MAX_DEPTH + 1))
    cls = RepLeafRegressor if task == "regression" else RepLeafClassifier
    return cls(**common)


def fit_eval(task, policy, leaf_model, seed, data) -> float:
    (Xtr, ytr), (Xva, yva), (Xte, yte) = _split(*data, seed)
    model = _estimator(task, policy, leaf_model, seed)
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)])
    if task == "regression":
        return _rmse(yte, model.predict(Xte))
    if task == "binary":
        return _logloss(yte, model.predict_proba(Xte)[:, 1])
    return _logloss(yte, model.predict_proba(Xte))


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--quick", action="store_true",
                        help="1 seed, one dataset per task")
    args = parser.parse_args()
    seeds = list(range(1 if args.quick else args.seeds))

    specs = datasets(args.quick)
    out = [
        "# Experiment: tree growth policies (leafwise vs depthwise vs symmetric)",
        "",
        "Auto-generated by `experiments/grow_policy_comparison.py` (ADR 0006). "
        "Metric is lower-is-better in every cell (RMSE for regression, logloss "
        "for binary/multiclass), mean ± std over seeds. Best policy per "
        f"(dataset, leaf_model) is **bold**. seeds={seeds}, max_depth={MAX_DEPTH} "
        f"for depth policies, early_stopping_rounds={ES}.",
    ]

    win_counts = {p: 0 for p in POLICIES}
    for name, task, builder in specs:
        out += ["", f"## {name}  ({task})", "",
                "| leaf_model | " + " | ".join(POLICIES) + " |",
                "|---" * (len(POLICIES) + 1) + "|"]
        for lm in LEAF_MODELS:
            cells = {}
            for policy in POLICIES:
                scores = [fit_eval(task, policy, lm, s, builder(s)) for s in seeds]
                cells[policy] = (float(np.mean(scores)), float(np.std(scores)))
            best = min(cells, key=lambda p: cells[p][0])
            win_counts[best] += 1
            row = [f"| {lm} |"]
            for policy in POLICIES:
                mean, std = cells[policy]
                txt = f"{mean:.4f} ± {std:.4f}"
                row.append(f" **{txt}** |" if policy == best else f" {txt} |")
            out.append("".join(row))
            print(f"{name:30s} {lm:16s} best={best}")

    out += [
        "",
        "## Auto-summary (wins per policy across all (dataset, leaf_model) cells)",
        "",
        "| policy | cells won |",
        "|---|---|",
        *[f"| {p} | {win_counts[p]} |" for p in POLICIES],
        "",
        "> Auto-counts only. The keep/change-default verdict, effect sizes, and "
        "where each policy is worth recommending are for results-analyst; the "
        "default stays `leafwise` unless a report justifies otherwise.",
    ]

    out_path = Path(__file__).resolve().parent / "results" / "grow_policy_comparison.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out) + "\n")
    print(f"\nreport written to {out_path}")
    print("win counts:", win_counts)


if __name__ == "__main__":
    main()
