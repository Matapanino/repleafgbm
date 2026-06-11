"""Encoder interface.

Encoders turn the numerical feature block into a representation matrix
``Z = z_theta(X_num)`` that leaf models consume. v0 encoders are NumPy-based,
deterministic, and frozen after ``fit`` (no parameter updates during
boosting; see docs/design.md for why). PyTorch encoders are planned as an
optional dependency and must implement the same interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseEncoder(ABC):
    """Abstract base class for numerical-feature encoders.

    Lifecycle: construct -> ``fit(X_num)`` once on training data -> ``transform``
    any number of times. Fitted state must be exposed via ``get_state`` /
    ``set_state`` (NumPy arrays only) so that serialization stays
    backend-agnostic.
    """

    #: Registry name, e.g. "identity" or "plr". Set by subclasses.
    name: str = "base"

    @abstractmethod
    def fit(self, X_num: np.ndarray) -> BaseEncoder:
        """Fit encoder statistics on the numerical feature matrix."""

    @abstractmethod
    def transform(self, X_num: np.ndarray) -> np.ndarray:
        """Map (n_rows, n_num_features) to a float64 (n_rows, output_dim) matrix."""

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Dimensionality of the produced representation. Valid after fit."""

    @abstractmethod
    def get_config(self) -> dict:
        """JSON-serializable constructor configuration (no fitted arrays)."""

    @abstractmethod
    def get_state(self) -> dict[str, np.ndarray]:
        """Fitted arrays, suitable for ``np.savez``."""

    @abstractmethod
    def set_state(self, state: dict[str, np.ndarray]) -> None:
        """Restore fitted arrays produced by ``get_state``."""

    def _check_fitted(self, attr: str) -> None:
        if getattr(self, attr, None) is None:
            raise RuntimeError(f"{type(self).__name__} must be fitted before use")
