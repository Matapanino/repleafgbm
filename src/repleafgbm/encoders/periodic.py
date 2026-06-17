"""Frozen periodic numerical embeddings (PBLD-style).

Inspired by the periodic embeddings of "On Embeddings for Numerical Features
in Tabular Deep Learning" (Gorishniy et al., 2022) and the PBLD
(periodic-bias-linear) variant that proved effective in RealMLP
(Holzmueller et al., 2024). RepLeafGBM's v0 constraint is a frozen,
NumPy-only encoder, so frequencies and phases are *sampled once* at fit time
(random-Fourier-feature style) instead of learned:

    z_jk(x) = sin(2 * pi * (f_jk * x_std_j + p_jk))    k = 1..n_frequencies

with f_jk ~ N(0, frequency_scale^2) and p_jk ~ U[0, 1) (the "bias"). The
standardized raw value is appended per feature (the "linear" part), keeping
an unbounded direction so leaf models can extrapolate.
"""

from __future__ import annotations

import numpy as np

from repleafgbm.encoders.base import BaseEncoder
from repleafgbm.utils.random import check_random_state


class PeriodicEncoder(BaseEncoder):
    """Random sinusoidal features per numerical column, plus a linear term.

    **Experimental.** With frozen random frequencies this encoder has not
    beaten ``identity`` on any tested signal — including oscillatory ones —
    because unmatched frequencies act as noise dimensions
    (experiments/results/encoder_variants.md). It exists as a baseline for
    the planned learned-frequency PyTorch encoder.

    Args:
        n_frequencies: Sinusoidal components per feature.
        frequency_scale: Std of the Gaussian the frequencies are drawn from.
            Larger values resolve finer oscillations but risk overfitting.
        add_linear: Append the standardized raw value per feature.
        random_state: Seed for frequency/phase sampling. The sklearn wrapper
            injects the model's ``random_state`` when not set explicitly.

    Output dimension: ``n_features * (n_frequencies + add_linear)``.
    Missing values are mean-imputed in standardized space (x_std = 0).
    """

    name = "periodic"

    def __init__(
        self,
        n_frequencies: int = 4,
        frequency_scale: float = 1.0,
        add_linear: bool = True,
        random_state: int | None = 0,
    ) -> None:
        if n_frequencies < 1:
            raise ValueError(f"n_frequencies must be >= 1, got {n_frequencies}")
        self.n_frequencies = n_frequencies
        self.frequency_scale = frequency_scale
        self.add_linear = add_linear
        self.random_state = random_state
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.frequencies_: np.ndarray | None = None  # (n_features, n_frequencies)
        self.phases_: np.ndarray | None = None  # (n_features, n_frequencies)

    def fit(
        self,
        X_num: np.ndarray,
        y: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> PeriodicEncoder:
        X_num = np.asarray(X_num, dtype=np.float64)
        n_features = X_num.shape[1]
        self.mean_ = np.nan_to_num(np.nanmean(X_num, axis=0), nan=0.0)
        std = np.nan_to_num(np.nanstd(X_num, axis=0), nan=1.0)
        self.scale_ = np.where(std > 0, std, 1.0)
        rng = check_random_state(self.random_state)
        self.frequencies_ = rng.normal(
            0.0, self.frequency_scale, size=(n_features, self.n_frequencies)
        )
        self.phases_ = rng.uniform(0.0, 1.0, size=(n_features, self.n_frequencies))
        return self

    def transform(self, X_num: np.ndarray) -> np.ndarray:
        self._check_fitted("frequencies_")
        X_num = np.asarray(X_num, dtype=np.float64)
        n_rows, n_features = X_num.shape
        if n_features != self.frequencies_.shape[0]:
            raise ValueError(
                f"Expected {self.frequencies_.shape[0]} numerical features, got {n_features}"
            )
        x_std = (np.where(np.isnan(X_num), self.mean_, X_num) - self.mean_) / self.scale_

        d = self.n_frequencies + int(self.add_linear)
        Z = np.empty((n_rows, n_features * d), dtype=np.float64)
        # (n_rows, n_features, n_frequencies) broadcast, then interleave per feature.
        sines = np.sin(
            2.0 * np.pi * (x_std[:, :, None] * self.frequencies_[None, :, :] + self.phases_)
        )
        for j in range(n_features):
            Z[:, j * d : j * d + self.n_frequencies] = sines[:, j, :]
            if self.add_linear:
                Z[:, j * d + self.n_frequencies] = x_std[:, j]
        return Z

    @property
    def output_dim(self) -> int:
        self._check_fitted("frequencies_")
        return int(self.frequencies_.shape[0] * (self.n_frequencies + int(self.add_linear)))

    def get_config(self) -> dict:
        return {
            "n_frequencies": self.n_frequencies,
            "frequency_scale": self.frequency_scale,
            "add_linear": self.add_linear,
            "random_state": self.random_state,
        }

    def get_state(self) -> dict[str, np.ndarray]:
        self._check_fitted("frequencies_")
        return {
            "mean": self.mean_,
            "scale": self.scale_,
            "frequencies": self.frequencies_,
            "phases": self.phases_,
        }

    def set_state(self, state: dict[str, np.ndarray]) -> None:
        self.mean_ = np.asarray(state["mean"], dtype=np.float64)
        self.scale_ = np.asarray(state["scale"], dtype=np.float64)
        self.frequencies_ = np.asarray(state["frequencies"], dtype=np.float64)
        self.phases_ = np.asarray(state["phases"], dtype=np.float64)
