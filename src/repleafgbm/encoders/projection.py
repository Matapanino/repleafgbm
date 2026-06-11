"""Random projection wrapper to cap embedding dimensionality.

Per-feature embeddings (e.g. PLR) scale as ``n_features * n_bins``, which can
blow up leaf-model fitting cost and overfitting risk. When an encoder's output
dimension exceeds ``max_leaf_emb_dim``, the model wraps it in this projection
(see docs/dataset_and_memory.md for the memory rationale).
"""

from __future__ import annotations

import numpy as np

from repleafgbm.encoders.base import BaseEncoder
from repleafgbm.utils.random import check_random_state


class RandomProjectionEncoder(BaseEncoder):
    """Compose a base encoder with a fixed Gaussian random projection.

    ``Z = base.transform(X) @ P`` where ``P`` has shape
    ``(base.output_dim, out_dim)`` with entries ``N(0, 1/sqrt(base_dim))``.
    The projection is sampled once at fit time and is deterministic given
    ``random_state``.
    """

    name = "random_projection"

    def __init__(self, base: BaseEncoder, out_dim: int, random_state: int | None = 0) -> None:
        self.base = base
        self.out_dim = out_dim
        self.random_state = random_state
        self.projection_: np.ndarray | None = None

    def fit(self, X_num: np.ndarray) -> RandomProjectionEncoder:
        self.base.fit(X_num)
        base_dim = self.base.output_dim
        if self.out_dim >= base_dim:
            raise ValueError(
                f"out_dim ({self.out_dim}) must be smaller than the base "
                f"encoder dimension ({base_dim}); no projection needed otherwise"
            )
        rng = check_random_state(self.random_state)
        self.projection_ = rng.standard_normal((base_dim, self.out_dim)) / np.sqrt(base_dim)
        return self

    def transform(self, X_num: np.ndarray) -> np.ndarray:
        self._check_fitted("projection_")
        return self.base.transform(X_num) @ self.projection_

    @property
    def output_dim(self) -> int:
        return self.out_dim

    def get_config(self) -> dict:
        return {
            "out_dim": self.out_dim,
            "random_state": self.random_state,
            "base_name": self.base.name,
            "base_config": self.base.get_config(),
        }

    def get_state(self) -> dict[str, np.ndarray]:
        self._check_fitted("projection_")
        state = {"projection": self.projection_}
        for k, v in self.base.get_state().items():
            state[f"base__{k}"] = v
        return state

    def set_state(self, state: dict[str, np.ndarray]) -> None:
        self.projection_ = np.asarray(state["projection"], dtype=np.float64)
        base_state = {
            k.removeprefix("base__"): v for k, v in state.items() if k.startswith("base__")
        }
        self.base.set_state(base_state)
