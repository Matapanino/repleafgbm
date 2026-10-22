"""Fold-safe target-encoding representations for numerical columns."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from repleafgbm.encoders.base import BaseEncoder
from repleafgbm.utils.random import check_random_state
from repleafgbm.utils.validation import as_sample_weight

Scale = str | int


@dataclass
class _Table:
    """One sorted key table and, for bins, its fitted inner boundaries."""

    keys: np.ndarray
    sums: np.ndarray
    counts: np.ndarray
    edges: np.ndarray


class TargetEncodingEncoder(BaseEncoder):
    """Encode numerical values with fold-safe smoothed target statistics.

    Each selected column is represented at every requested scale: exact
    values, ``round(x, k)`` for an integer scale, or an empirical quantile
    bin for ``"bin:<n>"``. The target mean uses empirical-Bayes smoothing,
    ``(sum + alpha * prior) / (count + alpha)``. Counts are weighted counts
    and optional count features are ``log1p(count)``.

    ``fit_transform_train`` returns seeded K-fold out-of-fold representations
    and then retains full-data tables for ordinary ``transform`` calls. If a
    fold's training complement has zero weight mass, its validation rows use
    the full-data prior (and zero count) instead of failing. If backoff is
    enabled, a mean whose weighted count is below ``min_count`` is replaced by
    the first later (coarser) scale meeting that threshold, or by the prior if
    none does. Its count feature remains the count at the originally requested
    scale.

    The defaults ``alpha=5``, ``n_folds=5``, and ``min_count=20`` are placed
    operating points rather than universal constants. ``alpha=5`` follows the
    S6E9 experiment; its generalization is not yet established.

    Args:
        columns: Numerical-block positions to encode, in output order. ``None``
            selects all input columns.
        scales: Non-empty list containing ``"exact"``, integer rounding
            precisions (negative values allowed), or ``"bin:<n>"`` quantile
            bin counts.
        alpha: Non-negative empirical-Bayes prior strength.
        n_folds: Number of shuffled folds used for training representations.
        add_count: Append ``log1p(weighted_count)`` after each mean.
        backoff: Replace rare means with later scales in ``scales``.
        min_count: Weighted-count threshold used by backoff.
        random_state: Seed accepted by :func:`check_random_state`.
    """

    name = "target_encoding"

    def __init__(
        self,
        columns: list[int] | None = None,
        scales: list[Scale] | None = None,
        alpha: float = 5.0,
        n_folds: int = 5,
        add_count: bool = True,
        backoff: bool = False,
        min_count: int = 20,
        random_state: int | None = 42,
    ) -> None:
        self.columns = None if columns is None else [int(column) for column in columns]
        self.scales = (
            ["exact"]
            if scales is None
            else [int(scale) if isinstance(scale, np.integer) else scale for scale in scales]
        )
        self.alpha = float(alpha)
        self.n_folds = int(n_folds)
        self.add_count = bool(add_count)
        self.backoff = bool(backoff)
        self.min_count = int(min_count)
        self.random_state = random_state
        self._validate_config()
        self.n_features_in_: int | None = None
        self.columns_: list[int] | None = None
        self.prior_: float | None = None
        self.tables_: dict[tuple[int, Scale], _Table] = {}

    def _validate_config(self) -> None:
        if self.columns is not None:
            if not self.columns:
                raise ValueError("columns must not be empty")
            if any(column < 0 for column in self.columns):
                raise ValueError("columns must contain non-negative indices")
            if len(set(self.columns)) != len(self.columns):
                raise ValueError("columns must not contain duplicates")
        if not self.scales:
            raise ValueError("scales must not be empty")
        for scale in self.scales:
            if isinstance(scale, bool):
                raise ValueError(f"Unsupported target-encoding scale {scale!r}")
            if isinstance(scale, (int, np.integer)):
                continue
            if scale == "exact":
                continue
            if isinstance(scale, str) and scale.startswith("bin:"):
                try:
                    n_bins = int(scale.removeprefix("bin:"))
                except ValueError as exc:
                    raise ValueError(f"Invalid target-encoding bin scale {scale!r}") from exc
                if str(n_bins) != scale.removeprefix("bin:") or n_bins < 2:
                    raise ValueError(
                        f"Target-encoding bin count must be an integer >= 2, got {scale!r}"
                    )
                continue
            raise ValueError(f"Unsupported target-encoding scale {scale!r}")
        if not np.isfinite(self.alpha) or self.alpha < 0.0:
            raise ValueError(f"alpha must be finite and >= 0, got {self.alpha}")
        if self.n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {self.n_folds}")
        if self.min_count < 1:
            raise ValueError(f"min_count must be >= 1, got {self.min_count}")

    def _validate_data(
        self,
        X_num: np.ndarray,
        y: np.ndarray | None,
        sample_weight: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = np.asarray(X_num, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError(f"X_num must be a non-empty 2-D array, got shape {X.shape}")
        if np.isinf(X).any():
            raise ValueError("X_num must not contain infinite values")
        if y is None:
            raise ValueError("target_encoding requires y")
        target = np.asarray(y, dtype=np.float64)
        if target.ndim == 2:
            raise NotImplementedError("target_encoding supports scalar target only in v1")
        if target.ndim != 1:
            raise ValueError(f"y must be 1-D, got shape {target.shape}")
        if target.shape[0] != X.shape[0]:
            raise ValueError(f"y has {target.shape[0]} rows but X_num has {X.shape[0]}")
        if not np.isfinite(target).all():
            raise ValueError("y must contain only finite values")
        weight = (
            np.ones(X.shape[0], dtype=np.float64)
            if sample_weight is None
            else as_sample_weight(sample_weight, n_rows=X.shape[0])
        )
        columns = list(range(X.shape[1])) if self.columns is None else self.columns
        invalid = [column for column in columns if column >= X.shape[1]]
        if invalid:
            raise ValueError(
                f"column index {invalid[0]} is invalid for an input with {X.shape[1]} columns"
            )
        return X, target, weight

    @staticmethod
    def _bin_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
        finite = values[~np.isnan(values)]
        if finite.size == 0:
            return np.empty(0, dtype=np.float64)
        quantiles = np.arange(1, n_bins, dtype=np.float64) / n_bins
        return np.unique(np.quantile(finite, quantiles)).astype(np.float64)

    @staticmethod
    def _scale_values(
        values: np.ndarray, scale: Scale, edges: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if scale == "exact":
            return values.copy(), np.empty(0, dtype=np.float64)
        if isinstance(scale, (int, np.integer)):
            return np.round(values, decimals=int(scale)), np.empty(0, dtype=np.float64)
        n_bins = int(scale.removeprefix("bin:"))
        fitted_edges = TargetEncodingEncoder._bin_edges(values, n_bins) if edges is None else edges
        keys = np.searchsorted(fitted_edges, values, side="right").astype(np.float64)
        keys[np.isnan(values)] = np.nan
        return keys, fitted_edges

    @staticmethod
    def _aggregate(keys: np.ndarray, target: np.ndarray, weight: np.ndarray) -> _Table:
        not_nan = ~np.isnan(keys)
        finite_keys, inverse = np.unique(keys[not_nan], return_inverse=True)
        counts = np.bincount(inverse, weights=weight[not_nan], minlength=finite_keys.size).astype(
            np.float64
        )
        sums = np.bincount(
            inverse,
            weights=weight[not_nan] * target[not_nan],
            minlength=finite_keys.size,
        ).astype(np.float64)
        if np.any(~not_nan):
            finite_keys = np.concatenate([finite_keys, np.asarray([np.nan])])
            counts = np.concatenate([counts, [weight[~not_nan].sum()]])
            sums = np.concatenate([sums, [(weight[~not_nan] * target[~not_nan]).sum()]])
        return _Table(
            keys=finite_keys.astype(np.float64),
            sums=sums,
            counts=counts,
            edges=np.empty(0, dtype=np.float64),
        )

    def _build_tables(
        self, X: np.ndarray, target: np.ndarray, weight: np.ndarray, columns: list[int]
    ) -> tuple[float, dict[tuple[int, Scale], _Table]]:
        mass = float(weight.sum())
        if mass <= 0.0:
            raise ValueError("target-encoding fit rows must have positive weight mass")
        prior = float(np.dot(weight, target) / mass)
        tables: dict[tuple[int, Scale], _Table] = {}
        for column in columns:
            values = X[:, column]
            for scale in self.scales:
                keys, edges = self._scale_values(values, scale)
                table = self._aggregate(keys, target, weight)
                table.edges = edges
                tables[(column, scale)] = table
        return prior, tables

    @staticmethod
    def _lookup(table: _Table, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sums = np.zeros(query.shape[0], dtype=np.float64)
        counts = np.zeros(query.shape[0], dtype=np.float64)
        table_nan = np.isnan(table.keys)
        finite_keys = table.keys[~table_nan]
        query_finite = ~np.isnan(query)
        if finite_keys.size and np.any(query_finite):
            values = query[query_finite]
            positions = np.searchsorted(finite_keys, values)
            valid = positions < finite_keys.size
            matched = np.zeros(values.shape[0], dtype=bool)
            matched[valid] = finite_keys[positions[valid]] == values[valid]
            query_positions = np.flatnonzero(query_finite)[matched]
            table_positions = positions[matched]
            sums[query_positions] = table.sums[table_positions]
            counts[query_positions] = table.counts[table_positions]
        if np.any(table_nan):
            missing = np.isnan(query)
            nan_position = int(np.flatnonzero(table_nan)[0])
            sums[missing] = table.sums[nan_position]
            counts[missing] = table.counts[nan_position]
        return sums, counts

    def _smoothed(self, sums: np.ndarray, counts: np.ndarray, prior: float) -> np.ndarray:
        if self.alpha == 0.0:
            result = np.full(counts.shape, prior, dtype=np.float64)
            np.divide(sums, counts, out=result, where=counts > 0.0)
            return result
        return (sums + self.alpha * prior) / (counts + self.alpha)

    def _transform_with(
        self,
        X: np.ndarray,
        prior: float,
        tables: dict[tuple[int, Scale], _Table],
        columns: list[int],
    ) -> np.ndarray:
        blocks: list[np.ndarray] = []
        for column in columns:
            scale_means: list[np.ndarray] = []
            scale_counts: list[np.ndarray] = []
            for scale in self.scales:
                table = tables[(column, scale)]
                keys, _ = self._scale_values(X[:, column], scale, table.edges)
                sums, counts = self._lookup(table, keys)
                scale_means.append(self._smoothed(sums, counts, prior))
                scale_counts.append(counts)
            if self.backoff:
                original = [mean.copy() for mean in scale_means]
                for i in range(len(self.scales)):
                    rare = scale_counts[i] < self.min_count
                    backed = np.full(X.shape[0], prior, dtype=np.float64)
                    unresolved = rare.copy()
                    for j in range(i + 1, len(self.scales)):
                        usable = unresolved & (scale_counts[j] >= self.min_count)
                        backed[usable] = original[j][usable]
                        unresolved[usable] = False
                    scale_means[i][rare] = backed[rare]
            for mean, count in zip(scale_means, scale_counts):
                blocks.append(mean[:, None])
                if self.add_count:
                    blocks.append(np.log1p(count)[:, None])
        return np.concatenate(blocks, axis=1)

    def fit(
        self,
        X_num: np.ndarray,
        y: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> TargetEncodingEncoder:
        X, target, weight = self._validate_data(X_num, y, sample_weight)
        columns = list(range(X.shape[1])) if self.columns is None else list(self.columns)
        prior, tables = self._build_tables(X, target, weight, columns)
        self.n_features_in_ = X.shape[1]
        self.columns_ = columns
        self.prior_ = prior
        self.tables_ = tables
        return self

    def fit_transform_train(
        self,
        X_num: np.ndarray,
        y: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return seeded K-fold OOF values and retain full-data fitted tables.

        A fold whose training complement has no positive weight mass falls
        back to the full-data prior for every mean and zero for every count.
        """
        X, target, weight = self._validate_data(X_num, y, sample_weight)
        if self.n_folds > X.shape[0]:
            raise ValueError(f"n_folds={self.n_folds} cannot exceed n_rows={X.shape[0]}")
        columns = list(range(X.shape[1])) if self.columns is None else list(self.columns)
        selected = X[:, columns]
        fold_columns = list(range(len(columns)))
        permutation = check_random_state(self.random_state).permutation(X.shape[0])
        folds = np.empty(X.shape[0], dtype=np.int64)
        folds[permutation] = np.arange(X.shape[0]) % self.n_folds
        output = np.empty((X.shape[0], self._output_dim(columns)), dtype=np.float64)
        full_prior = float(np.dot(weight, target) / weight.sum())
        for fold in range(self.n_folds):
            validation = folds == fold
            training = ~validation
            if weight[training].sum() <= 0.0:
                fallback = np.empty(self._output_dim(columns), dtype=np.float64)
                fallback[:: 1 + int(self.add_count)] = full_prior
                if self.add_count:
                    fallback[1::2] = 0.0
                output[validation] = fallback
                continue
            prior, tables = self._build_tables(
                selected[training], target[training], weight[training], fold_columns
            )
            output[validation] = self._transform_with(
                selected[validation], prior, tables, fold_columns
            )
        self.fit(X, target, weight)
        return output

    def transform(self, X_num: np.ndarray) -> np.ndarray:
        """Map rows with the full-data tables; unknown keys receive the prior."""
        self._check_fitted("n_features_in_")
        X = np.asarray(X_num, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            got = X.shape[1] if X.ndim == 2 else X.shape
            raise ValueError(f"Expected {self.n_features_in_} numerical features, got {got}")
        if np.isinf(X).any():
            raise ValueError("X_num must not contain infinite values")
        return self._transform_with(X, self.prior_, self.tables_, self.columns_)

    def _output_dim(self, columns: list[int]) -> int:
        return len(columns) * len(self.scales) * (1 + int(self.add_count))

    @property
    def output_dim(self) -> int:
        columns = self.columns_ if self.columns_ is not None else self.columns
        if columns is None:
            self._check_fitted("columns_")
        return self._output_dim(columns)

    def get_config(self) -> dict:
        return {
            "columns": None if self.columns is None else list(self.columns),
            "scales": list(self.scales),
            "alpha": self.alpha,
            "n_folds": self.n_folds,
            "add_count": self.add_count,
            "backoff": self.backoff,
            "min_count": self.min_count,
            "random_state": self.random_state,
        }

    @staticmethod
    def _state_prefix(column: int, scale: Scale) -> str:
        return f"{column}_{scale}"

    def get_state(self) -> dict[str, np.ndarray]:
        self._check_fitted("n_features_in_")
        state = {
            "n_features_in": np.asarray(self.n_features_in_, dtype=np.int64),
            "columns": np.asarray(self.columns_, dtype=np.int64),
            "prior": np.asarray(self.prior_, dtype=np.float64),
        }
        for column in self.columns_:
            for scale in self.scales:
                prefix = self._state_prefix(column, scale)
                table = self.tables_[(column, scale)]
                state[f"{prefix}_keys"] = table.keys
                state[f"{prefix}_sums"] = table.sums
                state[f"{prefix}_counts"] = table.counts
                if isinstance(scale, str) and scale.startswith("bin:"):
                    state[f"{prefix}_edges"] = table.edges
        return state

    def set_state(self, state: dict[str, np.ndarray]) -> None:
        self.n_features_in_ = int(np.asarray(state["n_features_in"]).item())
        self.columns_ = [int(value) for value in np.asarray(state["columns"]).tolist()]
        self.prior_ = float(np.asarray(state["prior"]).item())
        tables: dict[tuple[int, Scale], _Table] = {}
        for column in self.columns_:
            for scale in self.scales:
                prefix = self._state_prefix(column, scale)
                tables[(column, scale)] = _Table(
                    keys=np.asarray(state[f"{prefix}_keys"], dtype=np.float64),
                    sums=np.asarray(state[f"{prefix}_sums"], dtype=np.float64),
                    counts=np.asarray(state[f"{prefix}_counts"], dtype=np.float64),
                    edges=np.asarray(state.get(f"{prefix}_edges", np.empty(0)), dtype=np.float64),
                )
        self.tables_ = tables
