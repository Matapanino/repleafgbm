"""Interleaved A/B benchmark for ``SelectLeafModel`` native leaf statistics.

The fixed synthetic workload mirrors M16's acceptance case: binary Newton
statistics, 200k rows, 31 leaves, and three equal-width embedding blocks.
Run before building the new native symbol with ``--mode numpy``; after building
it, run ``--mode ab`` to compare the unchanged NumPy fallback with the optional
Rust fast path.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np

from repleafgbm.core import leaf_models
from repleafgbm.core.leaf_models import SelectLeafModel


def _data(block_dim: int):
    rng = np.random.default_rng(20261023 + block_dim)
    n_rows, n_leaves, n_blocks = 200_000, 31, 3
    emb_dim = n_blocks * block_dim
    Z = rng.normal(size=(n_rows, emb_dim))
    logits = 0.7 * Z[:, 0] - 0.4 * Z[:, block_dim] + rng.normal(size=n_rows)
    prob = 1.0 / (1.0 + np.exp(-logits))
    target = rng.integers(0, 2, size=n_rows)
    grad = np.ascontiguousarray(prob - target, dtype=np.float64)
    hess = np.ascontiguousarray(prob * (1.0 - prob) + 1e-3, dtype=np.float64)
    # Balanced, stable leaf sizes while preserving non-contiguous source rows.
    leaf_id = rng.integers(0, n_leaves, size=n_rows)
    leaf_rows = [np.flatnonzero(leaf_id == i).astype(np.int64) for i in range(n_leaves)]
    blocks = [slice(i * block_dim, (i + 1) * block_dim) for i in range(n_blocks)]
    return np.ascontiguousarray(Z), grad, hess, leaf_rows, blocks


def _one(data, use_native: bool) -> float:
    Z, grad, hess, leaf_rows, blocks = data
    saved = leaf_models._native
    if not use_native:
        leaf_models._native = None
    try:
        model = SelectLeafModel(l2=1.0, min_samples_linear=10, leaf_gate_margin=0.01)
        model.blocks = blocks
        start = time.perf_counter()
        model.fit_leaves(leaf_rows, grad, hess, Z)
        return time.perf_counter() - start
    finally:
        leaf_models._native = saved


def _summary(samples: list[float]) -> dict[str, float | list[float]]:
    median = statistics.median(samples)
    return {
        "median_s": median,
        "min_s": min(samples),
        "max_s": max(samples),
        "spread_s": max(samples) - min(samples),
        "rel_spread": (max(samples) - min(samples)) / median,
        "samples_s": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("numpy", "ab"), default="ab")
    parser.add_argument("--reps", type=int, default=5)
    args = parser.parse_args()
    result = {"rows": 200_000, "leaves": 31, "blocks": 3, "reps": args.reps}
    for block_dim in (8, 16):
        data = _data(block_dim)
        timings: dict[str, list[float]] = {"numpy": [], "native": []}
        # One untimed call removes import/allocation cold-start noise.
        _one(data, use_native=False)
        if args.mode == "ab":
            _one(data, use_native=True)
        for rep in range(args.reps):
            order = ("numpy", "native") if rep % 2 == 0 else ("native", "numpy")
            for arm in order:
                if arm == "native" and args.mode == "numpy":
                    continue
                timings[arm].append(_one(data, use_native=arm == "native"))
        entry = {"numpy": _summary(timings["numpy"])}
        if timings["native"]:
            entry["native"] = _summary(timings["native"])
            entry["speedup"] = entry["numpy"]["median_s"] / entry["native"]["median_s"]
        result[f"block_dim_{block_dim}"] = entry
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
