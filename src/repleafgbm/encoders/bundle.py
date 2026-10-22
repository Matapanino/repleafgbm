"""Named bundles of frozen leaf encoders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from repleafgbm.encoders.base import BaseEncoder
from repleafgbm.encoders.projection import RandomProjectionEncoder
from repleafgbm.encoders.target_reduction import TargetCorrelationEncoder


@dataclass(frozen=True)
class LeafExpert:
    """Public specification for one named leaf representation.

    ``columns`` index the ordered numerical block when integers are used; names
    refer to input columns. ``None`` gives the expert the complete numerical
    block.
    """

    name: str
    encoder: str | BaseEncoder
    columns: list[str | int] | None = None
    encoder_params: dict[str, Any] | None = None


class ExpertBundleEncoder(BaseEncoder):
    """Concatenate named encoders applied to numerical-column subsets."""

    name = "expert_bundle"

    def __init__(
        self,
        expert_names: list[str],
        encoders: list[BaseEncoder],
        column_subsets: list[list[int]],
        max_output_dim: int | None = None,
        reduction: str = "none",
        random_state: int | None = 42,
    ) -> None:
        if not expert_names or len(expert_names) != len(encoders):
            raise ValueError("expert_names and encoders must be non-empty and aligned")
        if len(column_subsets) != len(encoders):
            raise ValueError("column_subsets and encoders must have the same length")
        if any(not name for name in expert_names):
            raise ValueError("expert names must be non-empty strings")
        if len(set(expert_names)) != len(expert_names):
            raise ValueError("expert names must be unique")
        if "constant" in expert_names:
            raise ValueError(
                "expert name 'constant' is reserved for the implicit constant candidate"
            )
        self.expert_names = list(expert_names)
        self.encoders = list(encoders)
        subsets: list[tuple[int, ...]] = []
        for cols in column_subsets:
            if any(not isinstance(i, (int, np.integer)) for i in cols):
                raise ValueError("expert column subsets must contain integer indices")
            if len(set(cols)) != len(cols):
                raise ValueError("expert column subsets must not contain duplicates")
            subsets.append(tuple(int(i) for i in cols))
        self.column_subsets = subsets
        if any(not cols for cols in self.column_subsets):
            raise ValueError("expert column subsets must not be empty")
        self.max_output_dim = max_output_dim
        self.reduction = reduction
        self.random_state = random_state
        self.n_features_in_: int | None = None
        self.input_dims_: list[int] = []
        self.reduction_names_: list[str] = []

    def _select(self, X_num: np.ndarray, columns: tuple[int, ...]) -> np.ndarray:
        X_num = np.asarray(X_num, dtype=np.float64)
        if X_num.ndim != 2:
            raise ValueError(f"X_num must be 2-D, got shape {X_num.shape}")
        invalid = [i for i in columns if i < 0 or i >= X_num.shape[1]]
        if invalid:
            raise ValueError(
                f"expert column index {invalid[0]} is invalid for an ordered "
                f"numerical block with {X_num.shape[1]} columns"
            )
        if columns == tuple(range(X_num.shape[1])):
            return X_num  # every column: no copy
        if columns == tuple(range(columns[0], columns[-1] + 1)):
            return X_num[:, columns[0] : columns[-1] + 1]  # contiguous: a view
        return X_num[:, columns]

    def fit(
        self,
        X_num: np.ndarray,
        y: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> ExpertBundleEncoder:
        self.fit_transform_train(X_num, y=y, sample_weight=sample_weight)
        return self

    def fit_transform_train(
        self,
        X_num: np.ndarray,
        y: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> np.ndarray:
        X_num = np.asarray(X_num, dtype=np.float64)
        self.n_features_in_ = X_num.shape[1]
        blocks: list[np.ndarray] = []
        fitted: list[BaseEncoder] = []
        self.input_dims_ = []
        self.reduction_names_ = []
        for name, encoder, columns in zip(
            self.expert_names, self.encoders, self.column_subsets
        ):
            selected = self._select(X_num, columns)
            train_block = np.asarray(
                encoder.fit_transform_train(selected, y=y, sample_weight=sample_weight),
                dtype=np.float64,
            )
            if train_block.ndim != 2 or train_block.shape[0] != X_num.shape[0]:
                raise ValueError(
                    f"expert {name!r} fit_transform_train must return a 2-D matrix "
                    "with one row per training sample"
                )
            if train_block.shape[1] != encoder.output_dim:
                # Leaf weights are fitted at the training width while predict
                # uses transform(); a mismatch would silently mis-index weights.
                raise ValueError(
                    f"expert {name!r} fit_transform_train returned "
                    f"{train_block.shape[1]} columns but its transform output_dim "
                    f"is {encoder.output_dim}; the training representation must "
                    "have the same width as transform()"
                )
            self.input_dims_.append(train_block.shape[1])
            reduction_name = "none"
            if (
                self.max_output_dim is not None
                and train_block.shape[1] > self.max_output_dim
                and self.reduction != "none"
            ):
                if self.reduction == "random_projection":
                    reduced = RandomProjectionEncoder(
                        encoder,
                        out_dim=self.max_output_dim,
                        random_state=self.random_state,
                    ).fit_reduction(selected)
                elif self.reduction == "target_correlation":
                    reduced = TargetCorrelationEncoder(
                        encoder, out_dim=self.max_output_dim
                    ).fit_reduction(
                        selected, y, sample_weight, base_embeddings=train_block
                    )
                else:
                    raise ValueError(
                        f"leaf_reduction={self.reduction!r} is not supported"
                    )
                encoder = reduced
                train_block = train_block @ reduced.projection_
                reduction_name = self.reduction
            fitted.append(encoder)
            blocks.append(train_block)
            self.reduction_names_.append(reduction_name)
        self.encoders = fitted
        return blocks[0] if len(blocks) == 1 else np.concatenate(blocks, axis=1)

    def transform(self, X_num: np.ndarray) -> np.ndarray:
        self._check_fitted("n_features_in_")
        X_num = np.asarray(X_num, dtype=np.float64)
        if X_num.ndim != 2 or X_num.shape[1] != self.n_features_in_:
            got = X_num.shape[1] if X_num.ndim == 2 else X_num.shape
            raise ValueError(f"Expected {self.n_features_in_} numerical features, got {got}")
        if len(self.encoders) == 1:
            return np.asarray(
                self.encoders[0].transform(self._select(X_num, self.column_subsets[0])),
                dtype=np.float64,
            )
        # Fill the output in place so only one child block is live at a time.
        out = np.empty((X_num.shape[0], self.output_dim), dtype=np.float64)
        for encoder, columns, block in zip(
            self.encoders, self.column_subsets, self.block_slices
        ):
            out[:, block] = encoder.transform(self._select(X_num, columns))
        return out

    @property
    def block_slices(self) -> list[slice]:
        self._check_fitted("n_features_in_")
        slices: list[slice] = []
        start = 0
        for encoder in self.encoders:
            stop = start + encoder.output_dim
            slices.append(slice(start, stop))
            start = stop
        return slices

    @property
    def output_dim(self) -> int:
        return sum(encoder.output_dim for encoder in self.encoders)

    def get_config(self) -> dict:
        return {
            "expert_names": list(self.expert_names),
            "column_subsets": [list(cols) for cols in self.column_subsets],
            "max_output_dim": self.max_output_dim,
            "reduction": self.reduction,
            "random_state": self.random_state,
            "children": [
                {"name": encoder.name, "config": encoder.get_config()}
                for encoder in self.encoders
            ],
        }

    def get_state(self) -> dict[str, np.ndarray]:
        self._check_fitted("n_features_in_")
        state = {"n_features_in": np.asarray(self.n_features_in_, dtype=np.int64)}
        for i, encoder in enumerate(self.encoders):
            state.update({f"child_{i}__{k}": v for k, v in encoder.get_state().items()})
        return state

    def set_state(self, state: dict[str, np.ndarray]) -> None:
        self.n_features_in_ = int(np.asarray(state["n_features_in"]).item())
        for i, encoder in enumerate(self.encoders):
            prefix = f"child_{i}__"
            encoder.set_state(
                {k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)}
            )
