"""Minimal multi-output regression example.

Passing a 2-D ``y`` of shape (n_rows, n_outputs) grows shared-routing vector
leaves: one tree per round whose splits use the raw features (shared across
outputs) and whose leaves emit a vector. ``predict`` returns an
(n_rows, n_outputs) array. Multi-output is squared-error only.

Run from the repository root:
    PYTHONPATH=src python3 examples/multioutput_regression_basic.py
"""

import numpy as np

from repleafgbm import RepLeafRegressor


def make_data(n: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    # Two correlated targets sharing structure on x0/x1/x2.
    y0 = X[:, 0] + 0.5 * X[:, 1] ** 2 - X[:, 2]
    y1 = -X[:, 2] + X[:, 0] * X[:, 3] + 0.3 * X[:, 1]
    Y = np.column_stack([y0, y1]) + 0.05 * rng.normal(size=(n, 2))
    return X, Y


def main() -> None:
    X_train, Y_train = make_data(800, seed=0)
    X_test, Y_test = make_data(300, seed=1)

    model = RepLeafRegressor(
        n_estimators=80,
        learning_rate=0.1,
        num_leaves=8,
        leaf_model="embedded_linear",
        encoder="identity",
        random_state=42,
    )
    model.fit(X_train, Y_train)

    pred = model.predict(X_test)
    per_output_rmse = np.sqrt(np.mean((pred - Y_test) ** 2, axis=0))
    print(f"prediction shape: {pred.shape}")
    print(f"per-output test RMSE: {per_output_rmse.round(4)}")
    print(f"overall test RMSE: {np.sqrt(np.mean((pred - Y_test) ** 2)):.4f}")


if __name__ == "__main__":
    main()
