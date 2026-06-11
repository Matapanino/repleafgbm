"""Stacking example: LightGBM as an external base model for RepLeafGBM.

external_model mode (docs/backend_strategy.md): LightGBM is trained
independently; its out-of-fold predictions become a feature that RepLeafGBM
can both route on and feed to its leaf models. OOF scores are essential for
the training rows — in-sample base scores would leak the target into the
second stage.

Run from the repository root (skips cleanly if lightgbm is not installed):
    PYTHONPATH=src python3 examples/stacking_lightgbm.py
"""

import numpy as np


def make_data(n: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 8))
    y = (
        np.where(X[:, 0] > 0.0, 3.0, -2.0)
        + 2.0 * X[:, 1]
        + np.sin(2.0 * X[:, 2])
        + 1.0 * X[:, 3] * (X[:, 0] > 0.0)  # interaction: hard for both alone
        + rng.normal(0.0, 0.3, n)
    )
    return X, y


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main() -> None:
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        print("lightgbm is not installed; skipping this example "
              '(pip install "repleafgbm[external]")')
        return

    from repleafgbm import RepLeafDataset, RepLeafRegressor
    from repleafgbm.external import (
        LightGBMExternalModel,
        augment_features,
        oof_predictions,
    )

    X_train, y_train = make_data(2000, seed=0)
    X_test, y_test = make_data(1000, seed=1)

    def make_base():
        return LightGBMExternalModel(
            task="regression", random_state=42, n_estimators=200, num_leaves=31
        )

    # 1. Base model: OOF scores for train rows, full-model scores for test.
    oof_score, _ = oof_predictions(make_base, X_train, y_train, n_splits=5,
                                   random_state=42)
    base = make_base().fit(X_train, y_train)

    # 2. Augment features and train RepLeafGBM on top.
    df_train, cats = augment_features(X_train, base, score=oof_score, prefix="lgb")
    df_test, _ = augment_features(X_test, base, prefix="lgb")

    def make_repleaf():
        return RepLeafRegressor(
            n_estimators=150, learning_rate=0.1, num_leaves=16,
            min_samples_leaf=20, leaf_model="embedded_linear",
            encoder="identity", random_state=42,
        )

    train = RepLeafDataset(df_train, y_train, categorical_features=cats)
    stacked = make_repleaf()
    stacked.fit(train)
    pred_stacked = stacked.predict(RepLeafDataset(df_test, metadata=train.metadata))

    # 3. Baselines: each model alone.
    plain = make_repleaf().fit(X_train, y_train)

    print(f"lightgbm alone          test RMSE: {rmse(base.predict_score(X_test), y_test):.4f}")
    print(f"RepLeafGBM alone        test RMSE: {rmse(plain.predict(X_test), y_test):.4f}")
    print(f"RepLeafGBM + lgb score  test RMSE: {rmse(pred_stacked, y_test):.4f}")


if __name__ == "__main__":
    main()
