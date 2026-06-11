"""Build stacking features from an external base model.

The produced DataFrame is meant to feed straight into
:class:`~repleafgbm.data.RepLeafDataset`: scores are numerical features
(available to both routing and the encoder), leaf-index columns are
categorical (routing only).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def external_feature_frame(
    base: Any,
    X: Any,
    score: np.ndarray | None = None,
    include_score: bool = True,
    n_leaf_features: int = 0,
    prefix: str = "ext",
) -> tuple[pd.DataFrame, list[str]]:
    """Features derived from a fitted external base model.

    Args:
        base: Fitted model exposing ``predict_score`` and
            ``predict_leaf_indices`` (e.g. LightGBMExternalModel).
        X: Rows to derive features for.
        score: Optional precomputed score column — pass the *OOF* scores for
            training rows to avoid target leakage; when None the base model's
            in-sample ``predict_score(X)`` is used (fine for test rows).
        include_score: Emit the ``{prefix}_score`` numerical column.
        n_leaf_features: Emit leaf-index columns for the first ``n`` trees
            (categorical; routing can isolate individual leaf regions, though
            ordinal code order is arbitrary).
        prefix: Column-name prefix.

    Returns:
        ``(frame, categorical_names)`` — pass ``categorical_names`` to
        ``RepLeafDataset(categorical_features=...)``.
    """
    data: dict[str, np.ndarray] = {}
    categorical: list[str] = []
    if include_score:
        if score is None:
            score = base.predict_score(X)
        data[f"{prefix}_score"] = np.asarray(score, dtype=np.float64)
    if n_leaf_features > 0:
        leaves = base.predict_leaf_indices(X)
        for t in range(min(n_leaf_features, leaves.shape[1])):
            name = f"{prefix}_leaf_{t}"
            data[name] = leaves[:, t]
            categorical.append(name)
    if not data:
        raise ValueError("Nothing to emit: enable include_score or n_leaf_features")
    return pd.DataFrame(data), categorical


def augment_features(
    X: Any,
    base: Any,
    score: np.ndarray | None = None,
    include_score: bool = True,
    n_leaf_features: int = 0,
    prefix: str = "ext",
) -> tuple[pd.DataFrame, list[str]]:
    """Original features plus external-model features, as one DataFrame.

    Same arguments as :func:`external_feature_frame`. ndarray input gets
    ``f0..fN`` column names (matching RepLeafDataset's convention).

    Returns:
        ``(frame, categorical_names)`` where categorical_names covers only
        the added leaf columns; pass your own categorical columns on top.
    """
    if isinstance(X, pd.DataFrame):
        base_df = X.reset_index(drop=True)
    else:
        arr = np.asarray(X)
        base_df = pd.DataFrame(arr, columns=[f"f{i}" for i in range(arr.shape[1])])
    ext_df, categorical = external_feature_frame(
        base, X, score=score, include_score=include_score,
        n_leaf_features=n_leaf_features, prefix=prefix,
    )
    overlap = set(base_df.columns) & set(ext_df.columns)
    if overlap:
        raise ValueError(
            f"Feature name collision with existing columns: {sorted(overlap)}; "
            "pass a different prefix"
        )
    return pd.concat([base_df, ext_df], axis=1), categorical
