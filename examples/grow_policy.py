"""Tree growth policies: leafwise (default), depthwise, symmetric.

Trains the same model under each ``grow_policy`` on a small synthetic dataset.
All three route on raw features only and keep representation-conditioned leaves;
they differ in *how* the tree is expanded:

* leafwise  - best-gain-first (LightGBM-style), controlled by num_leaves;
* depthwise - level-order to max_depth (XGBoost-style);
* symmetric - CatBoost-style oblivious trees: one shared (feature, threshold)
  per level, giving a complete 2**depth tree with strong regularization.

Run from the repository root:
    PYTHONPATH=src python3 examples/grow_policy.py
"""

import numpy as np

from repleafgbm import RepLeafRegressor


def make_data(n: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
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

    for policy in ["leafwise", "depthwise", "symmetric"]:
        # depthwise/symmetric are depth-controlled and require max_depth >= 1.
        kwargs = {} if policy == "leafwise" else {"max_depth": 4}
        model = RepLeafRegressor(
            n_estimators=50,
            learning_rate=0.1,
            num_leaves=16,
            min_samples_leaf=20,
            grow_policy=policy,
            leaf_model="embedded_linear",
            encoder="plr",
            random_state=42,
            **kwargs,
        )
        model.fit(X_train, y_train)
        leaves = model.booster_.trees_[0].n_leaves
        print(
            f"grow_policy={policy:10s} test RMSE: "
            f"{rmse(model.predict(X_test), y_test):.4f}  "
            f"(tree 0 leaves: {leaves})"
        )


if __name__ == "__main__":
    main()
