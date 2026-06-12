"""RepLeafDataset API example: pandas input, categorical features, eval_set,
and save/load.

Run from the repository root:
    PYTHONPATH=src python3 examples/dataset_api_basic.py
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from repleafgbm import RepLeafDataset, RepLeafRegressor


def make_frame(n: int, seed: int):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "city": rng.choice(["tokyo", "osaka", "nagoya"], size=n),
            "area": rng.uniform(20, 120, size=n),
            "age": rng.uniform(0, 40, size=n),
        }
    )
    city_effect = df["city"].map({"tokyo": 3.0, "osaka": 1.0, "nagoya": 0.0})
    y = city_effect + 0.05 * df["area"] - 0.03 * df["age"] + rng.normal(0, 0.2, n)
    return df, y.to_numpy()


def main() -> None:
    df_train, y_train = make_frame(600, seed=0)
    df_valid, y_valid = make_frame(200, seed=1)

    train_data = RepLeafDataset(df_train, y_train, categorical_features=["city"])
    valid_data = RepLeafDataset(
        df_valid, y_valid, metadata=train_data.metadata  # reuse training encoding
    )

    model = RepLeafRegressor(
        n_estimators=40,
        num_leaves=8,
        min_samples_leaf=20,
        leaf_model="embedded_linear",
        encoder="plr",
        random_state=42,
    )
    model.fit(train_data, eval_set=[valid_data])
    history = model.evals_result_["valid_0"]["rmse"]
    print(f"valid RMSE: first round {history[0]:.4f} -> last round {history[-1]:.4f}")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "repleaf_model"
        model.save_model(path)  # also writes a human-readable summary.txt
        loaded = RepLeafRegressor.load_model(path)
        same = np.allclose(model.predict(df_valid), loaded.predict(df_valid))
        print(f"save/load round-trip predictions identical: {same}")
        print("--- summary.txt ---")
        print((path / "summary.txt").read_text(), end="")

    # Frequency encoding: an opt-in alternative for (high-cardinality)
    # categoricals — the column becomes numerical (its training frequency),
    # so it gets threshold splits and is visible to the encoder.
    freq_train = RepLeafDataset(df_train, y_train, frequency_encoded_features=["city"])
    freq_model = RepLeafRegressor(n_estimators=40, num_leaves=8, random_state=42)
    freq_model.fit(freq_train)
    pred = freq_model.predict(RepLeafDataset(df_valid, metadata=freq_train.metadata))
    rmse = float(np.sqrt(np.mean((pred - y_valid) ** 2)))
    print(f"frequency-encoded city: valid RMSE {rmse:.4f}")


if __name__ == "__main__":
    main()
