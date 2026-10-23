"""MatGBM leaf-only features with fixed frozen expert blocks.

The first two columns route trees.  A discrete identifier and an external
teacher score are leaf-only: they are unavailable to split search, but two
frozen expert blocks expose them to ``embedded_linear`` leaves.  This is the
recipe that survived the MatGBM experiments (ADR 0012): raw routing plus fixed
blocks, a fold-safe target encoding of the key (plus a teacher score) in the
leaves, and no per-leaf block selection.

Run from the repository root:
    PYTHONPATH=src python3 examples/matgbm_leaf_experts.py
"""

import numpy as np

from repleafgbm import RepLeafClassifier, RepLeafRegressor

TARGET_ENCODING = {
    "name": "te",
    "encoder": "target_encoding",
    "columns": [2],
    "encoder_params": {
        "scales": ["exact", 0],
        "n_folds": 4,
        "alpha": 5.0,
        "random_state": 17,
    },
}
TEACHER = {"name": "teacher", "encoder": "identity", "columns": [3]}


def make_data(n_rows: int, seed: int, *, classification: bool) -> tuple[np.ndarray, np.ndarray]:
    """Create a raw-routing signal plus a leaf-only key and teacher score."""
    rng = np.random.default_rng(seed)
    routing = rng.normal(size=(n_rows, 2))
    identifier = rng.integers(0, 8, size=n_rows).astype(float)
    smooth = rng.normal(size=n_rows)
    teacher = smooth + rng.normal(0.0, 0.4, size=n_rows)
    X = np.column_stack([routing, identifier, teacher])
    score = (
        1.7 * (routing[:, 0] > 0.0)
        - 1.2 * (routing[:, 1] > 0.5)
        + 0.65 * (identifier % 3)
        + 1.4 * smooth
        + rng.normal(0.0, 0.3, size=n_rows)
    )
    if classification:
        return X, (score > np.median(score)).astype(int)
    return X, score


def rmse(y_true: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - prediction) ** 2)))


def logloss(y_true: np.ndarray, probability: np.ndarray) -> float:
    probability = np.clip(probability, 1e-12, 1.0 - 1e-12)
    return float(-np.mean(y_true * np.log(probability) + (1 - y_true) * np.log(1 - probability)))


def fit_models(estimator_cls, X_train: np.ndarray, y_train: np.ndarray):
    """Fit constant leaves and two fixed-block variants with one router."""
    common = {
        "n_estimators": 24,
        "learning_rate": 0.12,
        "num_leaves": 6,
        "min_samples_leaf": 20,
        "leaf_only_features": [2, 3],
        "split_backend": "numpy",
        "random_state": 19,
    }
    constant = estimator_cls(leaf_model="constant", **common).fit(X_train, y_train)
    te_only = estimator_cls(
        leaf_model="embedded_linear", leaf_experts=[TARGET_ENCODING], **common
    ).fit(X_train, y_train)
    te_teacher = estimator_cls(
        leaf_model="embedded_linear", leaf_experts=[TARGET_ENCODING, TEACHER], **common
    ).fit(X_train, y_train)
    return {"constant": constant, "te": te_only, "te+teacher": te_teacher}


def run_regression() -> None:
    X_train, y_train = make_data(480, seed=11, classification=False)
    X_test, y_test = make_data(180, seed=29, classification=False)
    models = fit_models(RepLeafRegressor, X_train, y_train)
    print("regression holdout RMSE")
    for name, model in models.items():
        print(f"  {name:12s} {rmse(y_test, model.predict(X_test)):.4f}")


def run_binary_classification() -> None:
    X_train, y_train = make_data(480, seed=11, classification=True)
    X_test, y_test = make_data(180, seed=29, classification=True)
    models = fit_models(RepLeafClassifier, X_train, y_train)
    print("binary holdout logloss")
    for name, model in models.items():
        probability = model.predict_proba(X_test)[:, 1]
        print(f"  {name:12s} {logloss(y_test, probability):.4f}")


def main() -> None:
    run_regression()
    run_binary_classification()


if __name__ == "__main__":
    main()
