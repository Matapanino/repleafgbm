"""Deterministic random-state handling."""

from __future__ import annotations

import numpy as np


def check_random_state(random_state: int | np.random.Generator | None) -> np.random.Generator:
    """Return a NumPy Generator from an int seed, an existing Generator, or None.

    RepLeafGBM uses ``np.random.Generator`` everywhere so that results are
    reproducible given the same ``random_state``.
    """
    if random_state is None:
        return np.random.default_rng()
    if isinstance(random_state, np.random.Generator):
        return random_state
    if isinstance(random_state, (int, np.integer)):
        return np.random.default_rng(int(random_state))
    raise TypeError(
        f"random_state must be None, an int, or a numpy Generator, got {type(random_state)!r}"
    )
