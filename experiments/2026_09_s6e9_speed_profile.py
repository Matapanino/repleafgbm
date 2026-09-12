"""Profile the S6E9-sized constant-leaf Rust training path.

Run from the repository root:

    OMP_NUM_THREADS=4 RAYON_NUM_THREADS=4 \
      python3 experiments/2026_09_s6e9_speed_profile.py

The default shape is 130,000 rows and 151 raw columns (133 numerical plus
18 categorical), matching the probe scale. JSON is printed for transcription
into ``experiments/results/2026-09-12-s6e9-speed-profile.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from repleafgbm import RepLeafDataset, RepLeafRegressor


def make_bank(n_rows: int, seed: int) -> tuple[RepLeafDataset, np.ndarray]:
    """Seeded 133-numeric/18-categorical synthetic feature bank."""
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(n_rows, 151))
    frame = pd.DataFrame(values, columns=[f"f{i}" for i in range(151)])
    for index in range(133, 151):
        frame[f"f{index}"] = pd.Categorical(rng.integers(0, 10, size=n_rows))
    y = (
        1.5 * values[:, 0]
        - values[:, 1]
        + 0.5 * values[:, 2] * values[:, 3]
        + 0.2 * (np.asarray(frame["f133"].cat.codes) == 2)
        + rng.normal(scale=0.3, size=n_rows)
    )
    return RepLeafDataset(frame, y), y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=130_000)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()
    os.environ["REPLEAFGBM_PROFILE"] = "1"

    dataset, _ = make_bank(args.rows, args.seed)
    runs = []
    for repeat in range(args.repeats):
        model = RepLeafRegressor(
            n_estimators=args.rounds,
            learning_rate=0.05,
            num_leaves=127,
            min_samples_leaf=20,
            leaf_model="constant",
            l2_leaf=2.0,
            max_bins=255,
            split_backend="rust",
            random_state=args.seed + repeat,
        )
        start = time.perf_counter()
        model.fit(dataset)
        wall = time.perf_counter() - start
        runs.append({"wall": wall, "phases": model.phase_seconds_})

    phase_names = sorted({name for run in runs for name in run["phases"]})
    medians = {
        name: float(np.median([run["phases"].get(name, 0.0) for run in runs]))
        for name in phase_names
    }
    result = {
        "rows": args.rows,
        "columns": 151,
        "numerical_columns": 133,
        "categorical_columns": 18,
        "rounds": args.rounds,
        "repeats": args.repeats,
        "wall_seconds": [run["wall"] for run in runs],
        "median_wall_seconds": float(np.median([run["wall"] for run in runs])),
        "median_phase_seconds": medians,
        "top_three": sorted(medians.items(), key=lambda item: item[1], reverse=True)[:3],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
