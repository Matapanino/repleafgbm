"""Experiment: why do embedded-linear leaves underperform on binary tasks?

Hypothesis (Phase 12): the logistic Hessian h = p(1-p) vanishes in confident
regions, which (a) collapses the *effective* sample size of the h-weighted
leaf ridge fits and (b) makes a shared l2_leaf much stronger in relative
terms than for regression (sum h <= n/4, so l2=1 regularizes >= 4x harder).

Part A — diagnostics: identical X, paired targets (regression y = logit +
noise vs binary y ~ Bernoulli(sigmoid(logit))); instrument leaf fitting and
track, per round bucket, the Kish effective sample size (sum h)^2 / sum h^2,
the linear-leaf fraction, and weight norms.

Part B — remedies, on synthetic binary + adult: l2_leaf grid, a raised
Hessian floor, and leaf-fit-only damped weighting h^alpha (implemented by
transforming (g, h) -> (g h^{alpha-1}, h^alpha), which preserves the Newton
targets t = -g/h while flattening the weights).

Run from the repository root (lightgbm needed for the adult loader path):
    python3 experiments/binary_leaf_gain.py
Results are written to experiments/results/binary_leaf_gain.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import repleafgbm.sklearn as rl_sklearn  # noqa: E402
from repleafgbm import RepLeafClassifier, RepLeafDataset, RepLeafRegressor  # noqa: E402
from repleafgbm.core.leaf_models import EmbeddedLinearLeafModel  # noqa: E402
from repleafgbm.core.objectives import BinaryLogistic  # noqa: E402

ES = 25


def make_paired(n: int, seed: int):
    """Same X; regression and binary targets driven by the same logit with
    strong within-region linear structure (embedded leaves *should* win)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    logit = np.where(X[:, 0] > 0.0, 2.0, -2.0) + 2.0 * X[:, 1] + np.sin(2.0 * X[:, 2])
    y_reg = logit + rng.normal(0.0, 0.3, n)
    y_bin = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(np.float64)
    return X, y_reg, y_bin


def logloss(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


# --------------------------------------------------------------------- #
# Part A: diagnostics
# --------------------------------------------------------------------- #
def run_diagnostics(out: list[str], n_train: int, seed: int) -> None:
    X, y_reg, y_bin = make_paired(n_train, seed)

    records: list[dict] = []
    orig = EmbeddedLinearLeafModel.fit_leaves

    def instrumented(self, leaf_rows, grad, hess, Z):
        lv = orig(self, leaf_rows, grad, hess, Z)
        ess, lin, wn = [], 0, []
        for i, rows in enumerate(leaf_rows):
            h = hess[rows]
            ess.append(float(h.sum() ** 2 / max((h * h).sum(), 1e-300)))
            if np.any(lv.weights[i] != 0.0):
                lin += 1
                wn.append(float(np.linalg.norm(lv.weights[i])))
        records.append({
            "ess": float(np.mean(ess)),
            "linear_frac": lin / max(len(leaf_rows), 1),
            "wnorm": float(np.mean(wn)) if wn else 0.0,
        })
        return lv

    EmbeddedLinearLeafModel.fit_leaves = instrumented
    try:
        results = {}
        for task, y in (("regression", y_reg), ("binary", y_bin)):
            records.clear()
            cls = RepLeafRegressor if task == "regression" else RepLeafClassifier
            model = cls(n_estimators=100, num_leaves=16, min_samples_leaf=20,
                        leaf_model="embedded_linear", encoder="identity",
                        random_state=seed)
            model.fit(X, y)
            results[task] = list(records)
    finally:
        EmbeddedLinearLeafModel.fit_leaves = orig

    out += [
        "",
        "## Part A: diagnostics (paired targets, identical X and routing config)",
        "",
        f"n_train={n_train}, 100 rounds, num_leaves=16, identity encoder, "
        "seed={0}. ESS = Kish effective sample size per leaf, (sum h)^2 / "
        "sum h^2 (equals the row count for regression).".format(seed),
        "",
        "| rounds | reg ESS | bin ESS | reg linear% | bin linear% "
        "| reg mean ||w|| | bin mean ||w|| |",
        "|---|---|---|---|---|---|---|",
    ]
    for lo in range(0, 100, 25):
        hi = lo + 25
        row = [f"| {lo + 1}-{hi} |"]
        for key in ("ess", "linear_frac", "wnorm"):
            for task in ("regression", "binary"):
                vals = [r[key] for r in results[task][lo:hi]]
                row.append(f" {np.mean(vals):.2f} |")
        # reorder: ess pair, linear pair, wnorm pair
        out.append("".join(row))
    print("\n".join(out[-8:]))


# --------------------------------------------------------------------- #
# Part B: remedies
# --------------------------------------------------------------------- #
class DampedEmbeddedLeafModel(EmbeddedLinearLeafModel):
    """Leaf-fit-only damped weighting: weights h^alpha, targets unchanged."""

    def __init__(self, alpha: float, l2: float, min_samples_linear: int) -> None:
        super().__init__(l2=l2, min_samples_linear=min_samples_linear)
        self.alpha = alpha

    def fit_leaves(self, leaf_rows, grad, hess, Z):
        h_safe = np.maximum(hess, 1e-12)
        h_damped = np.power(h_safe, self.alpha)
        return super().fit_leaves(leaf_rows, grad * h_damped / h_safe, h_damped, Z)


def fit_one(label: str, X_parts, y_parts, l2: float, hfloor: float | None,
            alpha: float | None, seed: int, leaf_model: str = "embedded_linear"):
    Xtr, Xva, Xte = X_parts
    ytr, yva, yte = y_parts

    patches = []
    if hfloor is not None:
        orig_gh = BinaryLogistic.grad_hess

        def floored(self, y, raw):
            g, h = orig_gh(self, y, raw)
            return g, np.maximum(h, hfloor)

        BinaryLogistic.grad_hess = floored
        patches.append(("gh", orig_gh))
    if alpha is not None:
        orig_mk = rl_sklearn.make_leaf_model

        def mk(name, l2, min_samples_linear):
            return DampedEmbeddedLeafModel(alpha, l2=l2,
                                           min_samples_linear=min_samples_linear)

        rl_sklearn.make_leaf_model = mk
        patches.append(("mk", orig_mk))
    try:
        train = RepLeafDataset(Xtr, ytr)
        valid = RepLeafDataset(Xva, yva, metadata=train.metadata)
        model = RepLeafClassifier(
            n_estimators=400, learning_rate=0.1, num_leaves=31,
            min_samples_leaf=20, leaf_model=leaf_model, encoder="identity",
            l2_leaf=l2, early_stopping_rounds=ES, random_state=seed,
        )
        model.fit(train, eval_set=[valid])
        test_ds = RepLeafDataset(Xte, metadata=train.metadata)
        return logloss(yte, model.predict_proba(test_ds)[:, 1])
    finally:
        for kind, orig_fn in patches:
            if kind == "gh":
                BinaryLogistic.grad_hess = orig_fn
            else:
                rl_sklearn.make_leaf_model = orig_fn


def load_adult(max_rows: int):
    from benchmark_real_data import clean_features, load_dataset

    X_all, y_all, _ = load_dataset("adult")
    X_all, cats = clean_features(X_all)
    return X_all, y_all, cats


CONFIGS = [
    ("constant (l2=1.0)", dict(leaf_model="constant", l2=1.0, hfloor=None, alpha=None)),
    ("embedded l2=1.0 (default)", dict(l2=1.0, hfloor=None, alpha=None)),
    ("embedded l2=0.25", dict(l2=0.25, hfloor=None, alpha=None)),
    ("embedded l2=0.1", dict(l2=0.1, hfloor=None, alpha=None)),
    ("embedded l2=1.0 hfloor=0.1", dict(l2=1.0, hfloor=0.1, alpha=None)),
    ("embedded l2=0.25 hfloor=0.1", dict(l2=0.25, hfloor=0.1, alpha=None)),
    ("embedded damped a=0.5 l2=1.0", dict(l2=1.0, hfloor=None, alpha=0.5)),
    ("embedded damped a=0.0 l2=1.0", dict(l2=1.0, hfloor=None, alpha=0.0)),
]


def run_remedies(out: list[str], dataset: str, seeds: list[int],
                 n_train: int, n_valid: int, n_test: int) -> None:
    results: dict[str, list[float]] = {label: [] for label, _ in CONFIGS}
    for seed in seeds:
        if dataset == "synthetic":
            X, _, y = make_paired(n_train + n_valid + n_test, seed)
            X_df, cats = X, None
        else:
            X_df, y, cats = load_adult(0)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(X_df))[: n_train + n_valid + n_test]
        take = (lambda i: X_df.iloc[i]) if hasattr(X_df, "iloc") else (lambda i: X_df[i])
        i_tr = idx[:n_train]
        i_va = idx[n_train:n_train + n_valid]
        i_te = idx[n_train + n_valid:]
        parts_X = (take(i_tr), take(i_va), take(i_te))
        parts_y = (y[i_tr], y[i_va], y[i_te])

        for label, cfg in CONFIGS:
            kw = dict(cfg)
            lm = kw.pop("leaf_model", "embedded_linear")
            score = fit_one(label, parts_X, parts_y, seed=seed, leaf_model=lm, **kw)
            results[label].append(score)

    ordered = sorted(results.items(), key=lambda kv: np.mean(kv[1]))
    out += [
        "",
        f"## Part B: remedies on {dataset} "
        f"(logloss, n_train={n_train}, seeds={seeds})",
        "",
        "| config | test logloss (mean ± std) |",
        "|---|---|",
        *[
            f"| {label} | {np.mean(v):.4f} ± {np.std(v):.4f} |"
            for label, v in ordered
        ],
    ]
    print(f"=== {dataset} ===")
    for label, v in ordered:
        print(f"  {label:34s} {np.mean(v):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--n-train", type=int, default=8000)
    parser.add_argument("--n-valid", type=int, default=2500)
    parser.add_argument("--n-test", type=int, default=4000)
    args = parser.parse_args()
    seeds = list(range(args.seeds))

    out = [
        "# Experiment: binary embedded-leaf gain (Phase 12)",
        "",
        "Auto-generated by `experiments/binary_leaf_gain.py`. "
        "See the Analysis section at the bottom for conclusions.",
    ]
    run_diagnostics(out, args.n_train, seed=0)
    run_remedies(out, "synthetic", seeds, args.n_train, args.n_valid, args.n_test)
    run_remedies(out, "adult", seeds, args.n_train, args.n_valid, args.n_test)

    out_path = Path(__file__).resolve().parent / "results" / "binary_leaf_gain.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out) + "\n")
    print(f"\nreport written to {out_path}")


if __name__ == "__main__":
    main()
