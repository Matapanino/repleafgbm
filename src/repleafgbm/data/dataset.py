"""RepLeafDataset: the data abstraction used by encoders and the booster.

Responsibilities (see docs/dataset_and_memory.md):

* Hold the raw feature matrix (float64, categoricals ordinal-encoded).
* Hold the target and feature metadata.
* Produce numerical-feature views for encoders.
* Lazily compute and cache encoder embeddings.

v0 is fully in-memory, but the API is deliberately shaped so that batch
transforms, GPU transfer, and out-of-core variants can be added behind the
same interface without touching the booster.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from repleafgbm.data.metadata import FeatureMetadata
from repleafgbm.data.preprocessing import encode_features, infer_metadata
from repleafgbm.utils.validation import as_sample_weight, as_target_array


class RepLeafDataset:
    """In-memory dataset for RepLeafGBM training and prediction.

    Args:
        X: Feature matrix. A pandas DataFrame (recommended for categorical
            data) or a 2D NumPy array.
        y: Optional target vector.
        sample_weight: Optional per-row weights, length ``n_rows``. Non-negative
            and finite; they scale each row's gradient/Hessian during boosting
            (and the optimal init score). None means uniform weights.
        categorical_features: Names of categorical columns. For ndarray input
            use "f<index>" names. If None and X is a DataFrame, object /
            category / bool columns are auto-detected as categorical.
        numerical_features: Names of numerical columns. Usually inferred.
        frequency_encoded_features: Columns to encode by training frequency
            instead of an ordinal code. They are numerical downstream
            (threshold splits, encoder input); unseen categories encode to
            0.0, missing values to NaN. See docs/categorical_features.md.
        metadata: Pre-fitted FeatureMetadata. When given (e.g. at prediction
            time), the dataset re-applies the training-time encoding instead
            of inferring a new one.

    Example:
        >>> train_data = RepLeafDataset(X_train, y_train,
        ...                             categorical_features=["cat1", "cat2"])
    """

    def __init__(
        self,
        X: Any,
        y: Any | None = None,
        sample_weight: Any | None = None,
        categorical_features: list[str] | None = None,
        numerical_features: list[str] | None = None,
        frequency_encoded_features: list[str] | None = None,
        leaf_only_features: list[str | int] | None = None,
        metadata: FeatureMetadata | None = None,
    ) -> None:
        if metadata is None:
            metadata = infer_metadata(
                X,
                categorical_features=categorical_features,
                numerical_features=numerical_features,
                frequency_encoded_features=frequency_encoded_features,
                leaf_only_features=leaf_only_features,
            )
        elif leaf_only_features is not None:
            raise ValueError(
                "leaf_only_features must be stored in metadata when metadata is provided"
            )
        self.metadata = metadata
        self._X_raw = encode_features(X, metadata)
        #: Lazily built C-contiguous routing matrix when leaf-only columns are
        #: excluded; ``get_raw_features`` returns ``_X_raw`` itself otherwise.
        self._X_routing: np.ndarray | None = None
        for name in metadata.leaf_only_features:
            values = self._X_raw[:, metadata.feature_names.index(name)]
            if not np.isfinite(values).all():
                raise ValueError(
                    f"leaf-only feature {name!r} must contain only finite float values; "
                    "NaN and inf are not allowed"
                )
        self.y = None if y is None else as_target_array(y, n_rows=self._X_raw.shape[0])
        #: Optional per-row weights (n_rows,) or None for uniform. The booster
        #: scales gradients/Hessians by these (see core/booster.weight_grad_hess).
        self.sample_weight = (
            None
            if sample_weight is None
            else as_sample_weight(sample_weight, n_rows=self._X_raw.shape[0])
        )
        # Embedding cache: at most one encoder's output in v0. The encoder is
        # held by strong reference and compared with `is` — id()-keyed caching
        # would break if a dead encoder's id were reused by a new object.
        self._cached_encoder: Any | None = None
        self._cached_embeddings: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    # Basic properties
    # ------------------------------------------------------------------ #
    @property
    def n_rows(self) -> int:
        return self._X_raw.shape[0]

    @property
    def n_features(self) -> int:
        return self._X_raw.shape[1]

    @property
    def feature_names(self) -> list[str]:
        return self.metadata.feature_names

    # ------------------------------------------------------------------ #
    # Views used by the booster and encoders
    # ------------------------------------------------------------------ #
    def get_raw_features(self) -> np.ndarray:
        """Raw routing matrix: ``_X_raw`` itself unless leaf-only columns are
        excluded, in which case a C-contiguous column subset is built once and
        reused (``Tree.apply`` and the split backends need contiguous input)."""
        if not self.metadata.leaf_only_features:
            return self._X_raw
        if self._X_routing is None:
            self._X_routing = np.ascontiguousarray(
                self._X_raw[:, self.metadata.routing_indices]
            )
        return self._X_routing

    def get_numerical_features(self) -> np.ndarray:
        """Numerical columns only, used as encoder input in v0."""
        idx = self.metadata.numerical_indices
        return self._X_raw[:, idx]

    def get_embeddings(self, encoder: Any) -> np.ndarray:
        """Return encoder embeddings for all rows, with simple caching.

        The encoder must already be fitted. v0 computes the full embedding
        matrix in memory; ``rows`` arguments / batch transforms are future
        work and would slot in here.
        """
        if self._cached_encoder is not encoder:
            self._cached_embeddings = encoder.transform(self.get_numerical_features())
            self._cached_encoder = encoder
        return self._cached_embeddings

    def set_embeddings(self, encoder: Any, embeddings: np.ndarray) -> None:
        """Cache a precomputed training representation for ``encoder``."""
        values = np.asarray(embeddings, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] != self.n_rows:
            raise ValueError(
                "embeddings must be a 2-D matrix with one row per dataset row"
            )
        self._cached_encoder = encoder
        self._cached_embeddings = values

    def clear_embedding_cache(self) -> None:
        self._cached_encoder = None
        self._cached_embeddings = None

    def __repr__(self) -> str:
        return (
            f"RepLeafDataset(n_rows={self.n_rows}, n_features={self.n_features}, "
            f"n_numerical={len(self.metadata.numerical_features)}, "
            f"n_categorical={len(self.metadata.categorical_features)}, "
            f"has_target={self.y is not None})"
        )
