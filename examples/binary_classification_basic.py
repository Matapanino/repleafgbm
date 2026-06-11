"""Minimal binary classification example.

Run from the repository root:
    PYTHONPATH=src python3 examples/binary_classification_basic.py
"""

import numpy as np

from repleafgbm import RepLeafClassifier


def make_data(n: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    logit = 2.0 * X[:, 0] + np.sin(2.0 * X[:, 1]) - X[:, 2]
    y = (logit + rng.normal(0.0, 0.5, n) > 0).astype(int)
    return X, y


def main() -> None:
    X_train, y_train = make_data(800, seed=0)
    X_test, y_test = make_data(200, seed=1)

    model = RepLeafClassifier(
        n_estimators=80,
        learning_rate=0.1,
        num_leaves=8,
        min_samples_leaf=20,
        leaf_model="embedded_linear",
        encoder="identity",
        random_state=42,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    acc = (model.predict(X_test) == y_test).mean()
    logloss = -np.mean(
        y_test * np.log(np.clip(proba, 1e-12, 1))
        + (1 - y_test) * np.log(np.clip(1 - proba, 1e-12, 1))
    )
    print(f"test accuracy: {acc:.3f}   test logloss: {logloss:.4f}")


if __name__ == "__main__":
    main()
