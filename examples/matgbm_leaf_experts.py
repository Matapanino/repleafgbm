"""MatGBM leaf-only features and per-leaf expert selection.

The first two columns route trees.  The smooth value and discrete identifier
are intentionally leaf-only: they are unavailable to split search, but three
frozen experts may use them inside a leaf.

Run from the repository root:
    PYTHONPATH=src python3 examples/matgbm_leaf_experts.py
"""

import numpy as np

from repleafgbm import RepLeafClassifier, RepLeafRegressor

LEAF_EXPERTS = [
    {"name": "identity", "encoder": "identity", "columns": [2]},
    {
        "name": "target_encoding",
        "encoder": "target_encoding",
        "columns": [3],
        "encoder_params": {
            "scales": ["exact", 0],
            "n_folds": 4,
            "alpha": 5.0,
            "random_state": 17,
        },
    },
    {
        "name": "pca",
        "encoder": "pca",
        "columns": [2, 3],
        "encoder_params": {"n_components": 2},
    },
]


def make_data(n_rows: int, seed: int, *, classification: bool) -> tuple[np.ndarray, np.ndarray]:
    """Create a raw-routing signal plus leaf-only smooth and discrete signals."""
    rng = np.random.default_rng(seed)
    routing = rng.normal(size=(n_rows, 2))
    smooth = rng.normal(size=n_rows)
    identifier = rng.integers(0, 8, size=n_rows).astype(float)
    companion = 0.7 * smooth + rng.normal(0.0, 0.45, size=n_rows)
    X = np.column_stack([routing, smooth, identifier, companion])
    score = (
        1.7 * (routing[:, 0] > 0.0)
        - 1.2 * (routing[:, 1] > 0.5)
        + 1.4 * smooth
        + 0.5 * companion
        + 0.65 * (identifier % 3)
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
    """Fit classic, bundled, and per-leaf-selecting variants with one router."""
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
    embedded = estimator_cls(
        leaf_model="embedded_linear", leaf_experts=LEAF_EXPERTS, **common
    ).fit(X_train, y_train)
    select = estimator_cls(
        leaf_model="select",
        leaf_experts=LEAF_EXPERTS,
        leaf_gate_margin=0.01,
        **common,
    ).fit(X_train, y_train)
    return constant, embedded, select


def print_selection(fraction: dict[str, float]) -> None:
    selected = ", ".join(f"{name}={value:.3f}" for name, value in fraction.items())
    print(f"  select expert_selection_fraction_: {selected}")


def run_regression() -> None:
    X_train, y_train = make_data(480, seed=11, classification=False)
    X_test, y_test = make_data(180, seed=29, classification=False)
    constant, embedded, select = fit_models(RepLeafRegressor, X_train, y_train)
    print("regression holdout RMSE")
    for name, model in (("constant", constant), ("embedded_linear", embedded), ("select", select)):
        print(f"  {name:16s} {rmse(y_test, model.predict(X_test)):.4f}")
    print_selection(select.expert_selection_fraction_)


def run_binary_classification() -> None:
    X_train, y_train = make_data(480, seed=11, classification=True)
    X_test, y_test = make_data(180, seed=29, classification=True)
    constant, embedded, select = fit_models(RepLeafClassifier, X_train, y_train)
    print("binary holdout logloss")
    for name, model in (("constant", constant), ("embedded_linear", embedded), ("select", select)):
        probability = model.predict_proba(X_test)[:, 1]
        print(f"  {name:16s} {logloss(y_test, probability):.4f}")
    print_selection(select.expert_selection_fraction_)


def main() -> None:
    run_regression()
    run_binary_classification()


if __name__ == "__main__":
    main()
