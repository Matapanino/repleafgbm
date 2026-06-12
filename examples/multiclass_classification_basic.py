"""Minimal multiclass classification example.

Targets with three or more classes automatically use softmax boosting with
one tree per class per round; the API is identical to the binary case.

Run from the repository root:
    PYTHONPATH=src python3 examples/multiclass_classification_basic.py
"""

import numpy as np

from repleafgbm import RepLeafClassifier


def make_data(n: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    # Three regimes split by x0, with within-regime structure on x1/x2.
    score = 1.5 * X[:, 0] + np.sin(2.0 * X[:, 1]) - 0.5 * X[:, 2]
    score += rng.normal(0.0, 0.4, n)
    y = np.digitize(score, [-1.0, 1.0])  # labels 0, 1, 2
    return X, np.array(["low", "mid", "high"])[y]


def main() -> None:
    X_train, y_train = make_data(900, seed=0)
    X_test, y_test = make_data(300, seed=1)

    model = RepLeafClassifier(
        n_estimators=60,
        learning_rate=0.1,
        num_leaves=8,
        min_samples_leaf=20,
        leaf_model="embedded_linear",
        encoder="identity",
        random_state=42,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)
    acc = (model.predict(X_test) == y_test).mean()
    codes = np.searchsorted(model.classes_, y_test)
    logloss = -np.mean(
        np.log(np.clip(proba[np.arange(len(y_test)), codes], 1e-12, 1.0))
    )
    print(f"classes: {list(model.classes_)}")
    print(f"test accuracy: {acc:.3f}   test multi-logloss: {logloss:.4f}")


if __name__ == "__main__":
    main()
