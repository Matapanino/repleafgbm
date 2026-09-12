"""Composable routing of an encoder to selected numerical columns."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from repleafgbm.encoders.base import BaseEncoder


class ColumnSubsetEncoder(BaseEncoder):
    """Apply ``base`` only to selected numerical-feature columns.

    ``columns`` are zero-based positions in
    :meth:`RepLeafDataset.get_numerical_features`, not positions in the full
    raw matrix. This keeps categorical columns out of the encoder path and
    makes routing stable when raw categorical columns are interleaved.
    """

    name = "column_subset"

    def __init__(self, base: BaseEncoder, columns: Sequence[int]) -> None:
        try:
            values = tuple(columns)
        except TypeError as exc:
            raise ValueError(
                "columns must be a non-empty sequence of integer indices"
            ) from exc
        if not values or any(not isinstance(i, (int, np.integer)) for i in values):
            raise ValueError("columns must be a non-empty sequence of integer indices")
        if len(set(values)) != len(values):
            raise ValueError("columns must not contain duplicate indices")
        self.base = base
        self.columns = tuple(int(i) for i in values)
        self.n_features_in_: int | None = None

    def _select(self, X_num: np.ndarray) -> np.ndarray:
        X_num = np.asarray(X_num, dtype=np.float64)
        if X_num.ndim != 2:
            raise ValueError(f"X_num must be 2-D, got shape {X_num.shape}")
        n_features = X_num.shape[1]
        invalid = [i for i in self.columns if i < 0 or i >= n_features]
        if invalid:
            raise ValueError(
                f"columns contains numerical-feature index {invalid[0]}, but "
                f"X_num has {n_features} columns; indices refer to the "
                "RepLeafDataset numerical-feature block"
            )
        return X_num[:, self.columns]

    def fit(
        self,
        X_num: np.ndarray,
        y: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> ColumnSubsetEncoder:
        X_num = np.asarray(X_num, dtype=np.float64)
        selected = self._select(X_num)
        self.n_features_in_ = X_num.shape[1]
        self.base.fit(selected, y, sample_weight)
        return self

    def transform(self, X_num: np.ndarray) -> np.ndarray:
        self._check_fitted("n_features_in_")
        X_num = np.asarray(X_num, dtype=np.float64)
        if X_num.ndim != 2 or X_num.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} numerical features, got "
                f"{X_num.shape[1] if X_num.ndim == 2 else X_num.shape}"
            )
        return self.base.transform(self._select(X_num))

    @property
    def output_dim(self) -> int:
        self._check_fitted("n_features_in_")
        return self.base.output_dim

    def get_config(self) -> dict:
        return {
            "columns": list(self.columns),
            "base_name": self.base.name,
            "base_config": self.base.get_config(),
        }

    def get_state(self) -> dict[str, np.ndarray]:
        self._check_fitted("n_features_in_")
        state = {"n_features_in": np.asarray(self.n_features_in_, dtype=np.int64)}
        state.update({f"base__{k}": v for k, v in self.base.get_state().items()})
        return state

    def set_state(self, state: dict[str, np.ndarray]) -> None:
        self.n_features_in_ = int(np.asarray(state["n_features_in"]).item())
        self.base.set_state(
            {
                k.removeprefix("base__"): v
                for k, v in state.items()
                if k.startswith("base__")
            }
        )
