"""Preprocessing helpers: feature type inference and ordinal encoding.

v0 policy (see docs/categorical_features.md):

* Categorical features are ordinal-encoded into float64 columns.
* Missing values and categories unseen at fit time become NaN.
* The tree router treats NaN by sending rows to the left child, so unseen
  categories degrade gracefully instead of raising.
* Native categorical splits (LightGBM-style subset splits) are future work.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from repleafgbm.data.metadata import FeatureMetadata


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
) -> FeatureMetadata:
    """Build FeatureMetadata from training input.

    For pandas DataFrames, columns with object/category/bool dtype are treated
    as categorical unless the user says otherwise. For NumPy arrays, all
    columns are numerical unless ``categorical_features`` names them by index
    string (e.g. "3") or the array of names is supplied elsewhere.
    """
    if _is_dataframe(X):
        from pandas.api.types import is_bool_dtype, is_object_dtype, is_string_dtype

        feature_names = [str(c) for c in X.columns]
        if categorical_features is None:
            # is_string_dtype covers pandas >= 3 where plain string columns
            # are "str"/"string" dtype instead of object.
            categorical_features = [
                str(c)
                for c in X.columns
                if is_object_dtype(X[c])
                or is_string_dtype(X[c])
                or is_bool_dtype(X[c])
                or str(X[c].dtype) == "category"
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
        categorical_features = [str(c) for c in (categorical_features or [])]
        missing = [c for c in categorical_features if c not in feature_names]
        if missing:
            raise ValueError(
                f"categorical_features {missing} not found. For ndarray input, "
                'refer to columns as "f<index>", e.g. "f0".'
            )
        if numerical_features is None:
            numerical_features = [f for f in feature_names if f not in set(categorical_features)]

    # Fit category maps from the training data.
    category_maps: dict[str, list[str]] = {}
    for cat in categorical_features:
        col = _get_column(X, feature_names, cat)
        values = [str(v) for v in col if not _is_missing(v)]
        category_maps[cat] = sorted(set(values))

    return FeatureMetadata(
        feature_names=feature_names,
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        category_maps=category_maps,
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
    for j, name in enumerate(metadata.feature_names):
        col = _get_column(X, metadata.feature_names, name)
        if name in cat_set:
            mapping = {c: float(i) for i, c in enumerate(metadata.category_maps[name])}
            out[:, j] = [
                np.nan if _is_missing(v) else mapping.get(str(v), np.nan) for v in col
            ]
        else:
            out[:, j] = np.asarray(col, dtype=np.float64)
    return out


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
