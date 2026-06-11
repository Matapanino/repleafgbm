"""Minimal regression example.

Generates data with a discontinuous regime (handled by raw-feature routing)
and smooth linear structure inside each regime (handled by embedded-linear
leaves), then compares constant vs embedded_linear leaf models.

Run from the repository root:
    PYTHONPATH=src python3 examples/regression_basic.py
"""

import numpy as np

from repleafgbm import RepLeafRegressor


def make_data(n: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    # Discontinuous jump on x0, linear structure in x1/x2 inside each regime.
    y = (
        np.where(X[:, 0] > 0.0, 3.0, -2.0)
        + 2.0 * X[:, 1]
        - 1.0 * X[:, 2]
        + rng.normal(0.0, 0.1, n)
    )
    return X, y


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main() -> None:
    X_train, y_train = make_data(800, seed=0)
    X_test, y_test = make_data(200, seed=1)

    for leaf_model in ["constant", "embedded_linear"]:
        model = RepLeafRegressor(
            n_estimators=50,
            learning_rate=0.1,
            num_leaves=8,
            min_samples_leaf=20,
            leaf_model=leaf_model,
            encoder="plr",  # default n_bins=4; 4 features * 4 bins = 16 dims, unprojected
            random_state=42,
        )
        model.fit(X_train, y_train)
        print(f"leaf_model={leaf_model:16s} test RMSE: {rmse(model.predict(X_test), y_test):.4f}")


if __name__ == "__main__":
    main()
