"""Supervised dimensionality reduction for frozen leaf embeddings."""

from __future__ import annotations

import numpy as np

from repleafgbm.encoders.base import BaseEncoder


class TargetCorrelationEncoder(BaseEncoder):
    """Project onto target-correlated directions of the base embedding.

    The reduction is learned once from the initial Newton residual passed to
    the encoder, then frozen for boosting. For vector targets, squared
    correlations are combined with a deterministic SVD. Remaining dimensions
    retain the strongest individual base columns. No global random state is
    used.
    """

    name = "target_correlation"

    def __init__(self, base: BaseEncoder, out_dim: int) -> None:
        if out_dim < 1:
            raise ValueError(f"out_dim must be >= 1, got {out_dim}")
        self.base = base
        self.out_dim = out_dim
        self.projection_: np.ndarray | None = None

    def fit(
        self,
        X_num: np.ndarray,
        y: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> TargetCorrelationEncoder:
        self.base.fit(X_num, y, sample_weight)
        return self.fit_reduction(X_num, y, sample_weight)

    def fit_reduction(
        self,
        X_num: np.ndarray,
        y: np.ndarray | None,
        sample_weight: np.ndarray | None = None,
    ) -> TargetCorrelationEncoder:
        """Fit only the reduction, assuming ``base`` is already fitted."""
        if y is None:
            raise ValueError(
                "target_correlation reduction requires a supervised pretrain target"
            )
        Z = np.asarray(self.base.transform(X_num), dtype=np.float64)
        if self.out_dim >= Z.shape[1]:
            raise ValueError(
                f"out_dim ({self.out_dim}) must be smaller than the base "
                f"encoder dimension ({Z.shape[1]}); no reduction needed otherwise"
            )
        target = np.asarray(y, dtype=np.float64)
        if target.ndim == 1:
            target = target[:, None]
        weight = (
            np.ones(Z.shape[0], dtype=np.float64)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=np.float64)
        )
        weight = weight / max(float(weight.sum()), np.finfo(np.float64).tiny)
        Zc = Z - np.sum(weight[:, None] * Z, axis=0)
        yc = target - np.sum(weight[:, None] * target, axis=0)
        covariance = Zc.T @ (weight[:, None] * yc)
        z_var = np.sum(weight[:, None] * Zc * Zc, axis=0)
        y_var = np.sum(weight[:, None] * yc * yc, axis=0)
        denom = np.maximum(z_var[:, None] * y_var[None, :], 1e-30)
        normalized_covariance = covariance / np.sqrt(denom)
        target_directions, singular, _ = np.linalg.svd(
            normalized_covariance, full_matrices=False
        )
        for i in range(target_directions.shape[1]):
            pivot = int(np.argmax(np.abs(target_directions[:, i])))
            if target_directions[pivot, i] < 0:
                target_directions[:, i] *= -1.0
        n_target = min(self.out_dim, int(np.count_nonzero(singular > 1e-12)))
        columns = [target_directions[:, i] for i in range(n_target)]
        score = np.sum(normalized_covariance * normalized_covariance, axis=1)
        for index in np.argsort(-score, kind="stable"):
            if len(columns) == self.out_dim:
                break
            axis = np.zeros(Z.shape[1], dtype=np.float64)
            axis[index] = 1.0
            columns.append(axis)
        self.projection_ = np.column_stack(columns)
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
            "base_name": self.base.name,
            "base_config": self.base.get_config(),
        }

    def get_state(self) -> dict[str, np.ndarray]:
        self._check_fitted("projection_")
        state = {"projection": self.projection_}
        state.update({f"base__{k}": v for k, v in self.base.get_state().items()})
        return state

    def set_state(self, state: dict[str, np.ndarray]) -> None:
        self.projection_ = np.asarray(state["projection"], dtype=np.float64)
        self.base.set_state(
            {
                k.removeprefix("base__"): v
                for k, v in state.items()
                if k.startswith("base__")
            }
        )
