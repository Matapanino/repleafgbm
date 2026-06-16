"""Experiment: does class_weight='balanced' help imbalanced multiclass?

Phase 28. RepLeafGBM now supports per-row ``sample_weight`` and a classifier
``class_weight`` (folded into the gradient/Hessian scaling), plus a
``balanced_accuracy`` eval metric. The question this study answers: on
class-imbalanced multiclass data, does ``class_weight='balanced'`` trade overall
accuracy for *balanced* accuracy (mean per-class recall) the way it does for
other GBMs, and is the trade worthwhile?

Design: skewed multiclass datasets (synthetic Gaussian blobs with a geometric
class prior, and sklearn ``make_classification`` with explicit class weights).
For each seed we fit two otherwise-identical classifiers — ``class_weight=None``
and ``class_weight='balanced'`` — and report test accuracy and balanced
accuracy. The default is NOT changed by this study (class_weight stays None);
this validates the feature's effect.

Run from the repository root:
    python3 experiments/imbalanced_multiclass_class_weight.py --seeds 5
Results are written to experiments/results/imbalanced_multiclass_class_weight.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from repleafgbm import RepLeafClassifier


def make_blobs_skewed(n: int, seed: int, n_classes: int = 4, decay: float = 0.45):
    """Gaussian blobs (one center per class) with a geometric class prior."""
    rng = np.random.default_rng(seed)
    angles = 2 * np.pi * np.arange(n_classes) / n_classes
    centers = 4.0 * np.column_stack([np.cos(angles), np.sin(angles)])
    prior = decay ** np.arange(n_classes)
    prior = prior / prior.sum()
    y = rng.choice(n_classes, size=n, p=prior)
    X = centers[y] + rng.normal(0.0, 1.1, (n, 2))
    # A few nuisance features so trees have to work a little.
    X = np.column_stack([X, rng.normal(0.0, 1.0, (n, 4))])
    return X, y, prior


def make_sklearn_imbalanced(n: int, seed: int):
    from sklearn.datasets import make_classification

    weights = [0.7, 0.2, 0.07, 0.03]
    X, y = make_classification(
        n_samples=n,
        n_features=12,
        n_informative=8,
        n_redundant=2,
        n_clusters_per_class=1,
        n_classes=len(weights),
        weights=weights,
        class_sep=0.9,
        flip_y=0.03,
        random_state=seed,
    )
    return X, y, np.asarray(weights)


DATASETS = {
    "blobs_skewed": make_blobs_skewed,
    "sklearn_imbalanced": make_sklearn_imbalanced,
}


def run_dataset(out: list[str], name: str, maker, seeds: list[int],
                n_train: int, n_test: int) -> None:
    rows: dict[str, dict[str, list[float]]] = {
        "none": {"acc": [], "bacc": []},
        "balanced": {"acc": [], "bacc": []},
    }
    priors_seen = None
    for seed in seeds:
        # Generate one dataset and split: some generators (make_classification)
        # randomize the feature→label map per call, so train/test must come
        # from the same draw to share a distribution.
        X, y, prior = maker(n_train + n_test, seed)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(X.shape[0])
        tr, te = perm[:n_train], perm[n_train:n_train + n_test]
        Xtr, ytr = X[tr], y[tr]
        Xte, yte = X[te], y[te]
        priors_seen = prior
        for key, cw in (("none", None), ("balanced", "balanced")):
            model = RepLeafClassifier(
                n_estimators=200,
                learning_rate=0.1,
                num_leaves=15,
                min_samples_leaf=20,
                leaf_model="constant",
                class_weight=cw,
                random_state=seed,
            ).fit(Xtr, ytr)
            pred = model.predict(Xte)
            rows[key]["acc"].append(accuracy_score(yte, pred))
            rows[key]["bacc"].append(balanced_accuracy_score(yte, pred))

    def stat(key: str, metric: str) -> str:
        v = np.asarray(rows[key][metric])
        return f"{v.mean():.4f} ± {v.std():.4f}"

    d_bacc = np.mean(rows["balanced"]["bacc"]) - np.mean(rows["none"]["bacc"])
    d_acc = np.mean(rows["balanced"]["acc"]) - np.mean(rows["none"]["acc"])
    out += [
        "",
        f"## {name}  (class prior ≈ {np.round(priors_seen, 3).tolist()})",
        "",
        f"n_train={n_train}, n_test={n_test}, seeds={seeds}, "
        "constant leaves, 200 rounds.",
        "",
        "| class_weight | accuracy | balanced accuracy |",
        "|---|---|---|",
        f"| None | {stat('none', 'acc')} | {stat('none', 'bacc')} |",
        f"| 'balanced' | {stat('balanced', 'acc')} | {stat('balanced', 'bacc')} |",
        "",
        f"Δ from 'balanced': balanced_acc {d_bacc:+.4f}, accuracy {d_acc:+.4f}.",
    ]
    print(f"=== {name} ===")
    print(f"  none      acc={stat('none','acc')}  bacc={stat('none','bacc')}")
    print(f"  balanced  acc={stat('balanced','acc')}  bacc={stat('balanced','bacc')}")
    print(f"  Δbalanced_acc={d_bacc:+.4f}  Δaccuracy={d_acc:+.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--n-train", type=int, default=4000)
    parser.add_argument("--n-test", type=int, default=4000)
    args = parser.parse_args()
    seeds = list(range(args.seeds))

    out = [
        "# Experiment: class_weight='balanced' on imbalanced multiclass (Phase 28)",
        "",
        "Auto-generated by `experiments/imbalanced_multiclass_class_weight.py`. "
        "Compares `class_weight=None` vs `'balanced'` (otherwise identical "
        "models) on skewed multiclass data. See Analysis at the bottom.",
    ]
    for name, maker in DATASETS.items():
        run_dataset(out, name, maker, seeds, args.n_train, args.n_test)

    out += [
        "",
        "## Analysis",
        "",
        "_Filled in by results-analyst after the run._",
    ]

    out_path = Path(__file__).resolve().parent / "results" / (
        "imbalanced_multiclass_class_weight.md"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out) + "\n")
    print(f"\nreport written to {out_path}")


if __name__ == "__main__":
    main()
