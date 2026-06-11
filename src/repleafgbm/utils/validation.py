"""Input validation helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


def as_2d_float_array(X: Any, name: str = "X") -> np.ndarray:
    """Convert array-like input to a 2D float64 ndarray with a clear error message."""
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-dimensional, got shape {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row")
    return arr


def as_1d_float_array(y: Any, n_rows: int | None = None, name: str = "y") -> np.ndarray:
    """Convert target input to a 1D float64 ndarray and optionally check its length."""
    arr = np.asarray(y, dtype=np.float64).ravel()
    if n_rows is not None and arr.shape[0] != n_rows:
        raise ValueError(f"{name} has {arr.shape[0]} rows but X has {n_rows} rows")
    return arr


def as_target_array(y: Any, n_rows: int | None = None, name: str = "y") -> np.ndarray:
    """Convert target input to a 1D ndarray, keeping non-numeric labels intact.

    Numeric targets become float64; string/object class labels are preserved
    so the classifier can map them to {0, 1} itself.
    """
    arr = np.asarray(y).ravel()
    if n_rows is not None and arr.shape[0] != n_rows:
        raise ValueError(f"{name} has {arr.shape[0]} rows but X has {n_rows} rows")
    if np.issubdtype(arr.dtype, np.number) or arr.dtype == bool:
        arr = arr.astype(np.float64)
    return arr
