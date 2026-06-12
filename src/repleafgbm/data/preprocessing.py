"""Preprocessing helpers: feature type inference and encoding.

Policy (see docs/categorical_features.md):

* Categorical features are ordinal-encoded into float64 columns; the router
  applies native subset splits to them, so code order does not affect routing.
  ``pandas.Categorical`` columns keep their declared category order (including
  categories not observed at fit time).
* Missing values and categories unseen at fit time become NaN; the tree
  router sends NaN left, so unseen categories degrade gracefully.
* Frequency encoding is an opt-in alternative
  (``frequency_encoded_features``): the column becomes *numerical* (training
  frequency as the value), getting threshold splits and encoder visibility.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from repleafgbm.data.metadata import FeatureMetadata

#: Auto-detected categorical columns wider than this trigger a UserWarning:
#: subset splits silently fall back to ordered thresholds above ``max_bins``
#: categories (256 by default), which is rarely what the user wants.
HIGH_CARDINALITY_WARN_THRESHOLD = 256


def _is_dataframe(X: Any) -> bool:
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover
        return False
    return isinstance(X, pd.DataFrame)


def infer_metadata(
    X: Any,
    categorical_features: list[str] | None = None,
    numerical_features: list[str] | None = None,
    frequency_encoded_features: list[str] | None = None,
) -> FeatureMetadata:
    """Build FeatureMetadata from training input.

    For pandas DataFrames, columns with object/category/bool dtype are treated
    as categorical unless the user says otherwise. For NumPy arrays, all
    columns are numerical unless ``categorical_features`` names them by index
    string (e.g. "3") or the array of names is supplied elsewhere.

    ``frequency_encoded_features`` names columns to encode by their training
    frequency instead of an ordinal code; they are treated as numerical
    afterwards (threshold splits, encoder input).
    """
    frequency_encoded_features = [str(c) for c in (frequency_encoded_features or [])]
    if _is_dataframe(X):
        from pandas.api.types import is_bool_dtype, is_object_dtype, is_string_dtype

        feature_names = [str(c) for c in X.columns]
        if categorical_features is None:
            # is_string_dtype covers pandas >= 3 where plain string columns
            # are "str"/"string" dtype instead of object.
            categorical_features = [
                str(c)
                for c in X.columns
                if (
                    is_object_dtype(X[c])
                    or is_string_dtype(X[c])
                    or is_bool_dtype(X[c])
                    or str(X[c].dtype) == "category"
                )
                and str(c) not in set(frequency_encoded_features)
            ]
        else:
            categorical_features = [str(c) for c in categorical_features]
            missing = [c for c in categorical_features if c not in feature_names]
            if missing:
                raise ValueError(f"categorical_features {missing} not found in columns")
        if numerical_features is None:
            numerical_features = [f for f in feature_names if f not in set(categorical_features)]
        else:
            numerical_features = [str(c) for c in numerical_features]
    else:
        arr = np.asarray(X)
        if arr.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {arr.shape}")
        feature_names = [f"f{i}" for i in range(arr.shape[1])]
        categorical_features = [
            str(c)
            for c in (categorical_features or [])
            if str(c) not in set(frequency_encoded_features)
        ]
        missing = [c for c in categorical_features if c not in feature_names]
        if missing:
            raise ValueError(
                f"categorical_features {missing} not found. For ndarray input, "
                'refer to columns as "f<index>", e.g. "f0".'
            )
        if numerical_features is None:
            numerical_features = [f for f in feature_names if f not in set(categorical_features)]

    missing = [c for c in frequency_encoded_features if c not in feature_names]
    if missing:
        raise ValueError(f"frequency_encoded_features {missing} not found in columns")
    overlap = sorted(set(frequency_encoded_features) & set(categorical_features))
    if overlap:
        raise ValueError(
            f"Features {overlap} appear in both categorical_features and "
            "frequency_encoded_features; choose one encoding per column."
        )
    # Frequency-encoded columns are numerical downstream.
    numerical_features = list(numerical_features) + [
        c for c in frequency_encoded_features if c not in set(numerical_features)
    ]

    # Fit category maps from the training data.
    category_maps: dict[str, list[str]] = {}
    for cat in categorical_features:
        declared = _declared_categories(X, cat)
        if declared is not None:
            # pandas.Categorical: keep the declared order, including
            # categories not observed in this sample.
            category_maps[cat] = declared
        else:
            col = _get_column(X, feature_names, cat)
            values = [str(v) for v in col if not _is_missing(v)]
            category_maps[cat] = sorted(set(values))
        if len(category_maps[cat]) > HIGH_CARDINALITY_WARN_THRESHOLD:
            warnings.warn(
                f"Categorical feature {cat!r} has {len(category_maps[cat])} "
                "categories; subset splits fall back to ordered thresholds "
                "above max_bins categories. Consider "
                "frequency_encoded_features or grouping rare categories.",
                UserWarning,
                stacklevel=2,
            )

    # Fit frequency maps (proportion of training rows per category).
    frequency_maps: dict[str, dict[str, float]] = {}
    n_rows = len(X)
    for feat in frequency_encoded_features:
        col = _get_column(X, feature_names, feat)
        counts: dict[str, int] = {}
        for v in col:
            if not _is_missing(v):
                key = str(v)
                counts[key] = counts.get(key, 0) + 1
        frequency_maps[feat] = {k: c / n_rows for k, c in counts.items()}

    return FeatureMetadata(
        feature_names=feature_names,
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        category_maps=category_maps,
        frequency_maps=frequency_maps,
    )


def encode_features(X: Any, metadata: FeatureMetadata) -> np.ndarray:
    """Encode raw input into a float64 matrix following the fitted metadata.

    Numerical columns are cast to float64. Categorical columns are replaced by
    their ordinal code; missing/unseen categories become NaN.
    """
    if _is_dataframe(X):
        cols = [str(c) for c in X.columns]
        missing = [f for f in metadata.feature_names if f not in cols]
        if missing:
            raise ValueError(f"Input is missing expected columns: {missing}")
    else:
        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.shape[1] != metadata.n_features:
            raise ValueError(
                f"Input has {arr.shape[1]} columns but the model expects "
                f"{metadata.n_features} ({metadata.feature_names})"
            )

    n_rows = len(X)
    out = np.empty((n_rows, metadata.n_features), dtype=np.float64)
    cat_set = set(metadata.categorical_features)
    freq_set = set(metadata.frequency_maps)
    for j, name in enumerate(metadata.feature_names):
        col = _get_column(X, metadata.feature_names, name)
        if name in cat_set:
            mapping = {c: float(i) for i, c in enumerate(metadata.category_maps[name])}
            out[:, j] = [
                np.nan if _is_missing(v) else mapping.get(str(v), np.nan) for v in col
            ]
        elif name in freq_set:
            freq = metadata.frequency_maps[name]
            # Unseen categories get 0.0: zero observed training frequency.
            out[:, j] = [
                np.nan if _is_missing(v) else freq.get(str(v), 0.0) for v in col
            ]
        else:
            try:
                out[:, j] = np.asarray(col, dtype=np.float64)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"Column {name!r} is numerical but contains values that "
                    f"cannot be cast to float ({exc}). Declare it in "
                    "categorical_features (or frequency_encoded_features), "
                    "or clean the column."
                ) from exc
    return out


def _declared_categories(X: Any, name: str) -> list[str] | None:
    """Declared category order for ``pandas.Categorical`` columns, else None."""
    if not _is_dataframe(X):
        return None
    import pandas as pd

    dtype = X[name].dtype
    if isinstance(dtype, pd.CategoricalDtype):
        return [str(c) for c in dtype.categories]
    return None


def _get_column(X: Any, feature_names: list[str], name: str):
    if _is_dataframe(X):
        return X[name].to_numpy()
    arr = np.asarray(X)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr[:, feature_names.index(name)]


def _is_missing(v: Any) -> bool:
    if v is None:
        return True
    try:
        import pandas as pd

        # pd.isna handles NaN, pd.NA (pandas >= 3 string dtype), and NaT.
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False
