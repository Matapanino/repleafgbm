"""Calibrate winner's curse in ``leaf_model='select'`` on synthetic data.

The experiment plants two four-dimensional leaf-only experts in three regions
that are visible to the raw-feature router. Region A uses T1, region B uses T2,
and region C uses neither. Candidate pools add a permuted copy of T1 and zero
or more independent Gaussian blocks. The script measures per-leaf selection,
holdout accuracy, and three gate families, then emits the Markdown report.

Run from the repository root::

    OMP_NUM_THREADS=1 PYTHONPATH=src python3 \
        experiments/select_null_calibration.py --quick

The full frozen grid omits ``--quick`` and is intended to stay below 30 CPU
minutes. ``--quick`` preserves every K, margin, min-leaf, objective, and rule,
but uses n_rows=2,000, two seeds, and 25 boosting rounds.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, root_mean_squared_error

import repleafgbm.sklearn as sklearn_module
from repleafgbm import RepLeafClassifier, RepLeafRegressor
from repleafgbm.core.leaf_models import (
    LeafValues,
    SelectLeafModel,
    _fit_weighted_ridge,
    _loo_errors,
    _newton_constant,
)

REPORT_PATH = Path("experiments/results/2026-10-22-select-null-calibration.md")
ROUTE_COLUMNS = [f"route_{i}" for i in range(6)]
TRUE_NAMES = ("T1", "T2")
BLOCK_DIM = 4
FALSE_COUNTS = (0, 1, 3, 7)
FIXED_MARGINS = (0.0, 0.01, 0.03, 0.10)
SQRT_CONSTANTS = (0.5, 1.0, 2.0)
MIN_SAMPLES_LEAF = (20, 100)
OBJECTIVES = ("reg", "bin")
SIZE_BINS = (
    (0, 49, "<50"),
    (50, 99, "50-99"),
    (100, 249, "100-249"),
    (250, 499, "250-499"),
    (500, math.inf, "500+"),
)


@dataclass(frozen=True)
class SyntheticData:
    """One seeded train/holdout draw containing the largest candidate pool."""

    train: pd.DataFrame
    test: pd.DataFrame
    y_reg_train: np.ndarray
    y_reg_test: np.ndarray
    y_bin_train: np.ndarray
    y_bin_test: np.ndarray
    region_train: np.ndarray
    region_test: np.ndarray


@dataclass(frozen=True)
class RunSettings:
    """Resolved experiment grid and runtime settings."""

    quick: bool
    n_rows: tuple[int, ...]
    seeds: tuple[int, ...]
    n_estimators: int
    num_leaves: int = 15
    learning_rate: float = 0.1


def block_columns(name: str) -> list[str]:
    """Return the four input-column names belonging to an expert."""

    return [f"{name}_{j}" for j in range(BLOCK_DIM)]


def false_names(n_false: int) -> list[str]:
    """Return the frozen false-block pool for a requested K."""

    if n_false not in FALSE_COUNTS:
        raise ValueError(f"n_false must be one of {FALSE_COUNTS}, got {n_false}")
    return [] if n_false == 0 else ["P1", *[f"N{i}" for i in range(1, n_false)]]


def make_synthetic_data(n_rows: int, seed: int) -> SyntheticData:
    """Generate seeded train and independent holdout samples.

    ``n_rows`` is the training-row count, which controls attainable leaf sizes.
    The holdout has ``max(1,000, n_rows // 4)`` independent rows. P1 uses one
    row permutation for the whole T1 block, preserving within-block covariance.
    """

    train = _make_split(n_rows, seed * 10_007 + 17)
    test = _make_split(max(1_000, n_rows // 4), seed * 10_007 + 29)
    return SyntheticData(
        train=train[0],
        test=test[0],
        y_reg_train=train[1],
        y_reg_test=test[1],
        y_bin_train=train[2],
        y_bin_test=test[2],
        region_train=train[3],
        region_test=test[3],
    )


def _make_split(
    n_rows: int, random_state: int
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(random_state)
    route = rng.normal(size=(n_rows, len(ROUTE_COLUMNS)))
    region = np.where(route[:, 0] < -0.35, "A", np.where(route[:, 0] > 0.35, "B", "C"))
    t1 = rng.normal(size=(n_rows, BLOCK_DIM))
    t2 = rng.normal(size=(n_rows, BLOCK_DIM))
    beta1 = np.array([1.5, -1.0, 0.75, -0.5])
    beta2 = np.array([-1.3, 1.1, 0.6, -0.8])
    local = np.where(
        region == "A",
        t1 @ beta1,
        np.where(region == "B", t2 @ beta2, 0.0),
    )
    region_offset = np.select([region == "A", region == "B"], [-0.8, 0.8], default=0.0)
    y_reg = region_offset + local + rng.normal(scale=0.8, size=n_rows)
    logits = 0.55 * region_offset + 0.85 * local
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
    y_bin = rng.binomial(1, probability, size=n_rows).astype(np.int64)

    frame: dict[str, np.ndarray] = {
        name: route[:, i] for i, name in enumerate(ROUTE_COLUMNS)
    }
    for name, block in (("T1", t1), ("T2", t2)):
        frame.update({column: block[:, j] for j, column in enumerate(block_columns(name))})
    permuted = t1[rng.permutation(n_rows)]
    frame.update(
        {column: permuted[:, j] for j, column in enumerate(block_columns("P1"))}
    )
    for i in range(1, max(FALSE_COUNTS)):
        noise = rng.normal(size=(n_rows, BLOCK_DIM))
        name = f"N{i}"
        frame.update({column: noise[:, j] for j, column in enumerate(block_columns(name))})
    return pd.DataFrame(frame), y_reg, y_bin, region


def expert_specs(n_false: int) -> list[dict[str, Any]]:
    """Build public identity-expert specifications for one candidate pool."""

    names = [*TRUE_NAMES, *false_names(n_false)]
    return [
        {
            "name": name,
            "encoder": "identity",
            "columns": block_columns(name),
            "encoder_params": {"standardize": True},
        }
        for name in names
    ]


def selected_columns(n_false: int) -> list[str]:
    """Return router plus leaf-only columns needed for one K."""

    names = [*TRUE_NAMES, *false_names(n_false)]
    return [*ROUTE_COLUMNS, *[column for name in names for column in block_columns(name)]]


class CalibratedSelectLeafModel(SelectLeafModel):
    """Experiment-only select leaf with a per-leaf or per-tree threshold.

    ``sqrt`` uses ``c * sqrt(log(K + 1) / n_leaf)`` with K equal to the number
    of false candidates. ``perm95`` takes the 95th percentile, across leaves of
    the current tree, of each leaf's maximum false-block LOO improvement. It is
    clipped at zero so a candidate that loses to the constant is never kept.
    """

    def __init__(
        self,
        *,
        calibration: str,
        n_false: int,
        calibration_value: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if calibration not in {"sqrt", "perm95"}:
            raise ValueError(f"unknown calibration rule {calibration!r}")
        self.calibration = calibration
        self.n_false = n_false
        self.calibration_value = calibration_value
        self.threshold_history_: list[list[float]] = []
        self.selection_history_: list[list[int | None]] = []

    def fit_leaves(
        self,
        leaf_rows: list[np.ndarray],
        grad: np.ndarray,
        hess: np.ndarray,
        Z: np.ndarray | None,
    ) -> LeafValues:
        """Fit candidates, apply the configured threshold, and retain audits."""

        if Z is None:
            raise ValueError("CalibratedSelectLeafModel requires embeddings")
        if grad.ndim != 1 or hess.ndim != 1:
            raise NotImplementedError("calibration supports scalar leaves only")
        emb_dim = Z.shape[1]
        blocks = self.blocks or [slice(0, emb_dim)]
        indices = [_slice_indices(block, emb_dim) for block in blocks]
        if len(indices) != 2 + self.n_false:
            raise ValueError("n_false does not match the configured expert blocks")

        candidates: list[list[tuple[int, float, np.ndarray, float]]] = []
        constant_errors: list[float | None] = []
        false_maxima: list[float] = []
        biases = np.empty(len(leaf_rows), dtype=np.float64)
        weights = np.zeros((len(leaf_rows), emb_dim), dtype=np.float64)
        z_min = np.full((len(leaf_rows), emb_dim), -np.inf, dtype=np.float64)
        z_max = np.full((len(leaf_rows), emb_dim), np.inf, dtype=np.float64)

        for leaf, rows in enumerate(leaf_rows):
            gl, hl, zl = grad[rows], hess[rows], Z[rows]
            biases[leaf] = _newton_constant(gl, hl, self.l2)
            leaf_candidates: list[tuple[int, float, np.ndarray, float]] = []
            e_const: float | None = None
            for block_index, columns in enumerate(indices):
                if rows.size < max(self.min_samples_linear, columns.size + 2):
                    continue
                block_z = zl[:, columns]
                block_bias, block_weights = _fit_weighted_ridge(
                    block_z, gl, hl, self.l2
                )
                if block_weights is None:
                    continue
                block_error, current_constant = _loo_errors(
                    block_z,
                    gl,
                    hl,
                    block_bias,
                    block_weights,
                    self.l2,
                    insample=self.leaf_gate == "insample",
                )
                e_const = current_constant if e_const is None else e_const
                improvement = _relative_improvement(block_error, current_constant)
                leaf_candidates.append(
                    (block_index, block_bias, block_weights, improvement)
                )
            candidates.append(leaf_candidates)
            constant_errors.append(e_const)
            false_improvements = [
                item[3] for item in leaf_candidates if item[0] >= len(TRUE_NAMES)
            ]
            if false_improvements:
                false_maxima.append(max(false_improvements))

        tree_threshold = (
            max(0.0, float(np.quantile(false_maxima, 0.95)))
            if self.calibration == "perm95" and false_maxima
            else 0.0
        )
        selections: list[int | None] = []
        thresholds: list[float] = []
        for leaf, (rows, leaf_candidates, e_const) in enumerate(
            zip(leaf_rows, candidates, constant_errors)
        ):
            threshold = self._threshold(rows.size, tree_threshold)
            thresholds.append(threshold)
            best = max(leaf_candidates, key=lambda item: item[3], default=None)
            if best is not None and e_const is not None and best[3] > threshold:
                block_index, block_bias, block_weights, _ = best
                weights[leaf, indices[block_index]] = block_weights
                biases[leaf] = block_bias
                z_leaf = Z[rows]
                z_min[leaf] = z_leaf.min(axis=0)
                z_max[leaf] = z_leaf.max(axis=0)
                selections.append(block_index)
            else:
                selections.append(None)
        self.last_selections_ = selections
        self.selection_history_.append(selections)
        self.threshold_history_.append(thresholds)
        return LeafValues(bias=biases, weights=weights, z_min=z_min, z_max=z_max)

    def _threshold(self, n_leaf: int, tree_threshold: float) -> float:
        if self.calibration == "sqrt":
            return self.calibration_value * math.sqrt(
                math.log(self.n_false + 1) / n_leaf
            )
        return tree_threshold


def _slice_indices(block: slice, emb_dim: int) -> np.ndarray:
    start, stop, step = block.indices(emb_dim)
    if step != 1 or stop <= start:
        raise ValueError("select leaf blocks must be non-empty contiguous slices")
    return np.arange(start, stop, dtype=np.int64)


def _relative_improvement(block_error: float, constant_error: float) -> float:
    if constant_error <= 0.0 or not np.isfinite(constant_error):
        return -math.inf
    return 1.0 - block_error / constant_error


@contextlib.contextmanager
def calibrated_factory(
    calibration: str, n_false: int, calibration_value: float
) -> Iterator[list[CalibratedSelectLeafModel]]:
    """Temporarily replace only construction of ``select`` leaf models."""

    original = sklearn_module.make_leaf_model
    created: list[CalibratedSelectLeafModel] = []

    def factory(name: str, **kwargs: Any):
        if name != "select":
            return original(name, **kwargs)
        model = CalibratedSelectLeafModel(
            calibration=calibration,
            n_false=n_false,
            calibration_value=calibration_value,
            **kwargs,
        )
        created.append(model)
        return model

    sklearn_module.make_leaf_model = factory
    try:
        yield created
    finally:
        sklearn_module.make_leaf_model = original


def size_bin(n_leaf: int) -> str:
    """Map an observed training leaf size to the frozen reporting bins."""

    for lower, upper, label in SIZE_BINS:
        if lower <= n_leaf <= upper:
            return label
    raise AssertionError(f"uncovered leaf size {n_leaf}")


def _model_kwargs(
    settings: RunSettings,
    *,
    min_samples_leaf: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "n_estimators": settings.n_estimators,
        "learning_rate": settings.learning_rate,
        "num_leaves": settings.num_leaves,
        "min_samples_leaf": min_samples_leaf,
        "max_depth": -1,
        "split_backend": "numpy",
        "leaf_reduction": "none",
        "random_state": seed,
    }


def fit_baseline(
    data: SyntheticData,
    settings: RunSettings,
    *,
    objective: str,
    n_false: int,
    min_samples_leaf: int,
    seed: int,
    leaf_model: str,
) -> float:
    """Fit one constant or concatenated embedded-linear holdout baseline."""

    columns = selected_columns(n_false)
    train, test = data.train[columns], data.test[columns]
    y_train, y_test = _targets(data, objective)
    kwargs = _model_kwargs(settings, min_samples_leaf=min_samples_leaf, seed=seed)
    if leaf_model == "constant":
        train, test = data.train[ROUTE_COLUMNS], data.test[ROUTE_COLUMNS]
        model = _estimator(objective, leaf_model="constant", **kwargs)
    elif leaf_model == "embedded_linear":
        model = _estimator(
            objective,
            leaf_model="embedded_linear",
            leaf_experts=expert_specs(n_false),
            leaf_only_features=[column for column in columns if column not in ROUTE_COLUMNS],
            **kwargs,
        )
    else:
        raise ValueError(f"unknown baseline {leaf_model!r}")
    model.fit(train, y_train)
    return _score(model, test, y_test, objective)


def fit_select(
    data: SyntheticData,
    settings: RunSettings,
    *,
    objective: str,
    n_false: int,
    min_samples_leaf: int,
    seed: int,
    rule: str,
    value: float,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Fit one select arm and return model-level and leaf-size diagnostics."""

    columns = selected_columns(n_false)
    train, test = data.train[columns], data.test[columns]
    y_train, y_test = _targets(data, objective)
    kwargs = _model_kwargs(settings, min_samples_leaf=min_samples_leaf, seed=seed)
    model = _estimator(
        objective,
        leaf_model="select",
        leaf_experts=expert_specs(n_false),
        leaf_only_features=[column for column in columns if column not in ROUTE_COLUMNS],
        leaf_gate_margin=value if rule == "fixed" else 0.0,
        **kwargs,
    )
    if rule == "fixed":
        model.fit(train, y_train)
    else:
        with calibrated_factory(rule, n_false, value) as created:
            model.fit(train, y_train)
        if len(created) != 1:
            raise AssertionError("expected exactly one calibrated leaf model")

    metric = _score(model, test, y_test, objective)
    leaf_rows = inspect_leaves(model, train, data.region_train, n_false)
    summary = summarize_leaf_rows(leaf_rows)
    summary["metric"] = metric
    return summary, summarize_size_bins(leaf_rows)


def _targets(data: SyntheticData, objective: str) -> tuple[np.ndarray, np.ndarray]:
    if objective == "reg":
        return data.y_reg_train, data.y_reg_test
    if objective == "bin":
        return data.y_bin_train, data.y_bin_test
    raise ValueError(f"unknown objective {objective!r}")


def _estimator(objective: str, **kwargs: Any):
    return RepLeafRegressor(**kwargs) if objective == "reg" else RepLeafClassifier(**kwargs)


def _score(model: Any, test: pd.DataFrame, y_test: np.ndarray, objective: str) -> float:
    if objective == "reg":
        return float(root_mean_squared_error(y_test, model.predict(test)))
    probability = model.predict_proba(test)[:, 1]
    return float(log_loss(y_test, probability, labels=[0, 1]))


def inspect_leaves(
    model: Any,
    train: pd.DataFrame,
    regions: np.ndarray,
    n_false: int,
) -> list[dict[str, Any]]:
    """Recover selected blocks, observed leaf sizes, and majority regions."""

    routing = train[ROUTE_COLUMNS].to_numpy(dtype=np.float64)
    names = [*TRUE_NAMES, *false_names(n_false)]
    blocks = model.encoder_.block_slices
    rows: list[dict[str, Any]] = []
    for tree_index, (tree, values) in enumerate(
        zip(model.booster_.trees_, model.booster_.leaf_values_)
    ):
        leaf_ids = tree.apply(routing)
        for leaf in range(tree.n_leaves):
            row_indices = np.flatnonzero(leaf_ids == leaf)
            counts = {region: int(np.sum(regions[row_indices] == region)) for region in "ABC"}
            majority = max(counts, key=counts.get)
            selected = "constant"
            for name, block in zip(names, blocks):
                if np.any(values.weights[leaf, block] != 0.0):
                    selected = name
                    break
            rows.append(
                {
                    "tree": tree_index,
                    "leaf": leaf,
                    "n_leaf": row_indices.size,
                    "size_bin": size_bin(row_indices.size),
                    "region": majority,
                    "purity": counts[majority] / max(row_indices.size, 1),
                    "eligible": row_indices.size >= 2 * model.min_samples_leaf,
                    "selected": selected,
                    "is_null": selected not in {*TRUE_NAMES, "constant"},
                }
            )
    return rows


def summarize_leaf_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Compute the four requested per-fit selection rates."""

    total = len(rows)
    null_count = sum(row["is_null"] for row in rows)
    recovery_den = sum(
        row["eligible"] and row["region"] in {"A", "B"} for row in rows
    )
    recovery_num = sum(
        row["eligible"]
        and (
            (row["region"] == "A" and row["selected"] == "T1")
            or (row["region"] == "B" and row["selected"] == "T2")
        )
        for row in rows
    )
    c_den = sum(row["eligible"] and row["region"] == "C" for row in rows)
    c_num = sum(
        row["eligible"]
        and row["region"] == "C"
        and row["selected"] == "constant"
        for row in rows
    )
    return {
        "null_rate": null_count / max(total, 1),
        "recovery_rate": recovery_num / recovery_den if recovery_den else math.nan,
        "c_constant_rate": c_num / c_den if c_den else math.nan,
        "mean_purity": float(np.mean([row["purity"] for row in rows])),
        "eligible_fraction": float(np.mean([row["eligible"] for row in rows])),
        "n_leaves": float(total),
    }


def summarize_size_bins(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute per-fit null selection rates for nonempty leaf-size bins."""

    output: list[dict[str, Any]] = []
    for _, _, label in SIZE_BINS:
        selected = [row for row in rows if row["size_bin"] == label]
        if selected:
            output.append(
                {
                    "size_bin": label,
                    "null_rate": float(np.mean([row["is_null"] for row in selected])),
                    "mean_n_leaf": float(np.mean([row["n_leaf"] for row in selected])),
                    "n_leaves": len(selected),
                }
            )
    return output


def run_experiment(
    settings: RunSettings,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    """Run baselines and all fixed/dynamic calibration candidates."""

    started = time.perf_counter()
    fixed_records: list[dict[str, Any]] = []
    calibration_records: list[dict[str, Any]] = []
    size_records: list[dict[str, Any]] = []
    total_cells = len(settings.n_rows) * len(settings.seeds)
    completed = 0
    for n_rows in settings.n_rows:
        for seed in settings.seeds:
            data = make_synthetic_data(n_rows, seed)
            for objective in OBJECTIVES:
                for min_leaf in MIN_SAMPLES_LEAF:
                    constant_metric = fit_baseline(
                        data,
                        settings,
                        objective=objective,
                        n_false=0,
                        min_samples_leaf=min_leaf,
                        seed=seed,
                        leaf_model="constant",
                    )
                    for n_false in FALSE_COUNTS:
                        embedded_metric = fit_baseline(
                            data,
                            settings,
                            objective=objective,
                            n_false=n_false,
                            min_samples_leaf=min_leaf,
                            seed=seed,
                            leaf_model="embedded_linear",
                        )
                        common = {
                            "n_rows": n_rows,
                            "seed": seed,
                            "objective": objective,
                            "min_samples_leaf": min_leaf,
                            "n_false": n_false,
                            "constant_metric": constant_metric,
                            "embedded_metric": embedded_metric,
                        }
                        for margin in FIXED_MARGINS:
                            summary, bins = fit_select(
                                data,
                                settings,
                                objective=objective,
                                n_false=n_false,
                                min_samples_leaf=min_leaf,
                                seed=seed,
                                rule="fixed",
                                value=margin,
                            )
                            record = {
                                **common,
                                **summary,
                                "family": "fixed",
                                "value": margin,
                                "candidate": f"fixed m={margin:g}",
                            }
                            fixed_records.append(record)
                            calibration_records.append(record)
                            size_records.extend(_annotate_bins(bins, record))
                        for constant in SQRT_CONSTANTS:
                            summary, bins = fit_select(
                                data,
                                settings,
                                objective=objective,
                                n_false=n_false,
                                min_samples_leaf=min_leaf,
                                seed=seed,
                                rule="sqrt",
                                value=constant,
                            )
                            record = {
                                **common,
                                **summary,
                                "family": "sqrt",
                                "value": constant,
                                "candidate": f"sqrt c={constant:g}",
                            }
                            calibration_records.append(record)
                            size_records.extend(_annotate_bins(bins, record))
                        summary, bins = fit_select(
                            data,
                            settings,
                            objective=objective,
                            n_false=n_false,
                            min_samples_leaf=min_leaf,
                            seed=seed,
                            rule="perm95",
                            value=0.95,
                        )
                        record = {
                            **common,
                            **summary,
                            "family": "perm95",
                            "value": 0.95,
                            "candidate": "per-tree null q95",
                        }
                        calibration_records.append(record)
                        size_records.extend(_annotate_bins(bins, record))
            completed += 1
            elapsed = time.perf_counter() - started
            print(
                f"completed data cell {completed}/{total_cells}: "
                f"n_rows={n_rows}, seed={seed}, elapsed={elapsed:.1f}s",
                flush=True,
            )
    return (
        pd.DataFrame(fixed_records),
        pd.DataFrame(calibration_records),
        pd.DataFrame(size_records),
        time.perf_counter() - started,
    )


def _annotate_bins(
    bins: list[dict[str, Any]], record: dict[str, Any]
) -> list[dict[str, Any]]:
    keys = (
        "n_rows",
        "seed",
        "objective",
        "min_samples_leaf",
        "n_false",
        "family",
        "value",
        "candidate",
    )
    return [{**{key: record[key] for key in keys}, **row} for row in bins]


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    """Return mean and sample standard deviation, using zero for one value."""

    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return math.nan, math.nan
    return float(np.mean(array)), float(np.std(array, ddof=1)) if array.size > 1 else 0.0


def format_mean_std(values: Sequence[float], *, percent: bool = False) -> str:
    mean, std = mean_std(values)
    scale = 100.0 if percent else 1.0
    suffix = "%" if percent else ""
    return f"{mean * scale:.2f} ± {std * scale:.2f}{suffix}"


def aggregate_fixed(fixed: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the fixed sweep over seeds at the frozen factor grain."""

    group = ["n_rows", "objective", "min_samples_leaf", "n_false", "value"]
    rows = []
    for key, frame in fixed.groupby(group, sort=True):
        row = dict(zip(group, key))
        for metric in (
            "null_rate",
            "recovery_rate",
            "c_constant_rate",
            "metric",
            "embedded_metric",
            "constant_metric",
        ):
            row[f"{metric}_mean"], row[f"{metric}_std"] = mean_std(frame[metric])
        delta = frame["metric"] - frame["embedded_metric"]
        row["delta_mean"], row["delta_std"] = mean_std(delta)
        rows.append(row)
    return pd.DataFrame(rows)


def candidate_tradeoff(calibration: pd.DataFrame) -> pd.DataFrame:
    """Average every design cell within seed, then report seed variation."""

    per_seed = (
        calibration.assign(delta=lambda frame: frame.metric - frame.embedded_metric)
        .groupby(["candidate", "objective", "seed"], as_index=False)[
            ["null_rate", "recovery_rate", "c_constant_rate", "metric", "delta"]
        ]
        .mean()
    )
    rows = []
    for (candidate, objective), frame in per_seed.groupby(
        ["candidate", "objective"], sort=False
    ):
        row = {"candidate": candidate, "objective": objective}
        for metric in ("null_rate", "recovery_rate", "c_constant_rate", "metric", "delta"):
            row[f"{metric}_mean"], row[f"{metric}_std"] = mean_std(frame[metric])
        rows.append(row)
    return pd.DataFrame(rows)


def judge_candidates(calibration: pd.DataFrame) -> pd.DataFrame:
    """Apply the frozen decision rule to seed-mean design cells.

    The null bound is evaluated only where K>0. A candidate must pass every
    n_rows × objective × min-leaf × K cell, not merely the pooled average.
    """

    cell_columns = [
        "candidate",
        "family",
        "value",
        "n_rows",
        "objective",
        "min_samples_leaf",
        "n_false",
    ]
    cells = calibration.assign(delta=lambda frame: frame.metric - frame.embedded_metric).groupby(
        cell_columns, as_index=False
    )[["null_rate", "recovery_rate", "delta"]].mean()
    rows = []
    for (candidate, family, value), frame in cells.groupby(
        ["candidate", "family", "value"], sort=False
    ):
        with_false = frame[frame.n_false > 0]
        max_null = float(with_false.null_rate.max()) if not with_false.empty else 0.0
        min_recovery = float(frame.recovery_rate.min())
        max_delta = float(frame.delta.max())
        rows.append(
            {
                "candidate": candidate,
                "family": family,
                "value": value,
                "max_null": max_null,
                "min_recovery": min_recovery,
                "max_delta": max_delta,
                "passes": max_null <= 0.05 and min_recovery >= 0.80 and max_delta <= 0.0,
            }
        )
    return pd.DataFrame(rows)


def aggregate_size_curve(size_records: pd.DataFrame, candidate: str) -> pd.DataFrame:
    """Aggregate a K × observed-leaf-size null curve with seed variation."""

    selected = size_records[size_records.candidate == candidate]
    per_seed = (
        selected.groupby(["objective", "n_false", "size_bin", "seed"], as_index=False)
        .apply(
            lambda frame: pd.Series(
                {
                    "null_rate": np.average(frame.null_rate, weights=frame.n_leaves),
                    "mean_n_leaf": np.average(frame.mean_n_leaf, weights=frame.n_leaves),
                    "n_leaves": frame.n_leaves.sum(),
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )
    rows = []
    for (objective, n_false, bin_name), frame in per_seed.groupby(
        ["objective", "n_false", "size_bin"], sort=False
    ):
        mean, std = mean_std(frame.null_rate)
        rows.append(
            {
                "objective": objective,
                "n_false": n_false,
                "size_bin": bin_name,
                "mean_n_leaf": float(frame.mean_n_leaf.mean()),
                "null_mean": mean,
                "null_std": std,
                "n_leaves": int(frame.n_leaves.sum()),
            }
        )
    order = {label: i for i, (_, _, label) in enumerate(SIZE_BINS)}
    output = pd.DataFrame(rows)
    if not output.empty:
        output["_bin_order"] = output.size_bin.map(order)
        output = output.sort_values(["objective", "n_false", "_bin_order"]).drop(
            columns="_bin_order"
        )
    return output


def build_report(
    fixed: pd.DataFrame,
    calibration: pd.DataFrame,
    size_records: pd.DataFrame,
    settings: RunSettings,
    elapsed: float,
) -> str:
    """Render every numeric result from run records into Markdown."""

    fixed_agg = aggregate_fixed(fixed)
    tradeoff = candidate_tradeoff(calibration)
    judgments = judge_candidates(calibration)
    passing = judgments[judgments.passes]
    recommendation = _recommendation(passing)
    curve_candidate = (
        recommendation
        if recommendation in set(size_records.candidate)
        else "fixed m=0.01"
    )
    curve = aggregate_size_curve(size_records, curve_candidate)
    k_effect = _k_effect(fixed)
    accuracy_cost = _accuracy_cost(tradeoff, recommendation)
    mode = "quick" if settings.quick else "full"
    command = (
        "`OMP_NUM_THREADS=1 PYTHONPATH=src python3 "
        "experiments/select_null_calibration.py --quick`"
        if settings.quick
        else "`OMP_NUM_THREADS=1 PYTHONPATH=src python3 "
        "experiments/select_null_calibration.py`"
    )
    lines = [
        "# `leaf_model=\"select\"` null-selection calibration",
        "",
        "## 要約（10行）",
        "",
        f"1. 実行モードは **{mode}**、学習行数は {list(settings.n_rows)}、"
        f"seed は {list(settings.seeds)}。",
        f"2. 固定 margin=0.01 の K=1→7 による null 選択率変化は {k_effect}。",
        "3. null 選択率は実葉サイズ別に集計し、小葉の一部は "
        "`min_samples_linear` により自動的に constant となる。",
        f"4. 凍結判定を全セルで満たす最小候補は **{recommendation}**。",
        "5. 判定は null≤5%、A/B 真ブロック回収≥80%、"
        "各セルの holdout が連結 ridge 以下、の同時条件。",
        f"6. 推奨候補の平均 holdout 差（select−embedded）は {accuracy_cost}（負が select 優位）。",
        "7. per-tree null q95 は各木の各葉における偽ブロック最大 LOO 改善の95分位を使う。",
        "8. `sqrt` 規則の K は偽ブロック数で、"
        "`c*sqrt(log(K+1)/n_leaf)` を実葉サイズごとに適用した。",
        "9. `d` は凍結仕様どおり4固定であり、次元依存はこの実験から識別できない。",
        "10. 結論は合成分布内の較正であり、実データ既定値の変更には追加検証が必要。",
        "",
        "## 実行条件",
        "",
        f"- command: {command}",
        f"- wall time: **{elapsed:.1f} s**",
        f"- n_estimators={settings.n_estimators}, num_leaves={settings.num_leaves}, "
        f"learning_rate={settings.learning_rate}",
        f"- n_rows={list(settings.n_rows)}, seeds={list(settings.seeds)}, d={BLOCK_DIM}",
        f"- K(false)={list(FALSE_COUNTS)}, margin={list(FIXED_MARGINS)}, "
        f"min_samples_leaf={list(MIN_SAMPLES_LEAF)}",
        "- objective: regression (RMSE) / binary logistic (logloss)",
        "- router: raw 6 columns only; all expert columns are `leaf_only_features`",
        "",
        "`n_rows` は学習行数。holdout は独立生成した `max(1,000, n_rows/4)` 行。"
        " quick は K・margin・min-leaf・目的・較正规則を保ち、"
        "n_rows、seed 数、木数だけを削減した。",
        "",
        "## データ生成と集計定義",
        "",
        "`route_0 < -0.35` を A、`route_0 > 0.35` を B、その間を C とした。"
        " A では T1、B では T2 の4次元線形信号を植え、C は局所線形信号なし。"
        " P1 は T1 の行置換、N1... は独立標準ガウス。回帰には Gaussian noise、二値には"
        " planted logit から Bernoulli draw を用いた。",
        "",
        "葉の地域は、その木で当該葉へ route された学習行の多数決。真回収率は A→T1 と B→T2"
        " の合算、C constant 率は多数地域 C の葉を分母とする。両者は select が線形 fit を"
        "許す `n_leaf >= 2*min_samples_leaf` の葉だけを対象とし、null 選択率の分母は全葉。"
        " seed 間 std は sample std（seed 数が1の場合のみ0）で、表の `±` はすべてこの定義。",
        "",
        "## 固定 margin sweep",
        "",
        _fixed_table(fixed_agg),
        "",
        "`Δ vs embedded` は同一 K の全ブロック連結 `embedded_linear` に対する holdout metric 差。"
        " RMSE/logloss とも負が select 優位。",
        "",
        f"## K × n_leaf null curve — {curve_candidate}",
        "",
        _curve_table(curve),
        "",
        "この表は各 bin 内を葉数で重み付けして seed ごとに率を作り、"
        "その後 seed 平均±std を表示する。",
        "",
        "## 較正规則のトレードオフ",
        "",
        _tradeoff_table(tradeoff),
        "",
        "集約値は各 seed 内で n_rows・K・min-leaf を等重み平均してから seed 間 mean±std。"
        " q95 は K=0 では偽ブロックが存在しないため threshold=0 とした。",
        "",
        "## 凍結判定",
        "",
        _judgment_table(judgments),
        "",
        f"**判定: {recommendation}。** 各候補について seed 平均を design cell ごとに計算し、"
        "K>0 セルの最大 null 率、全セルの最小回収率、全セルの最大 holdout 差を使った。"
        " pooled mean だけで悪いセルを隠さない保守的判定である。",
        "",
        "## 前提の誤り・限界",
        "",
        "- Goal は d の関数を求めるが、凍結要因では d=4 しか指定されていない。"
        "したがって d 依存は推定不能。",
        "- K は冒頭では全ブロック数とも読めるが、凍結要因の記述に従い"
        "偽ブロック数 0/1/3/7 と解釈した。総候補数は K+2。",
        "- 葉領域は ground-truth region の多数決であり、境界をまたぐ葉を除外していない。"
        "purity の低い葉は回収率を下げ得る。",
        "- 同じ木の q95 は高々15葉から推定されるため離散的で不安定。"
        "別 calibration split を使う規則ではない。",
        "- boosting 後半の Newton target は初期の planted response と異なるため、"
        "真回収率100%は期待しない。",
        "- quick report は5 seed・20,000行・100木の確証ではなく、"
        "スクリプトの full mode が凍結グリッドを実装する。",
        "",
        "## ライブラリへ入れるべき変更（この発注では未実装）",
        "",
        "1. `leaf_gate_margin` に callable/per-leaf policy を受ける実験 API を設け、"
        "K と n_leaf を policy へ渡す。",
        "2. 固定1%を既定変更する前に、false-candidate 数を考慮した "
        "multiple-selection 補正を組み込む。",
        "3. 各木・各葉の size、選択 expert、LOO constant/best error、"
        "適用 threshold を診断属性として公開する。",
        "4. q95 を本実装するなら、同一候補を選択と較正に併用せず "
        "calibration-only permuted blocks を明示する。",
        "",
        "## 再現性",
        "",
        "数値表・判定・要約中の数値はすべて本スクリプトが run records から生成する。乱数は"
        " `np.random.RandomState(seed-derived)` と estimator の `random_state=seed` に限定した。",
    ]
    return "\n".join(lines) + "\n"


def _recommendation(passing: pd.DataFrame) -> str:
    if passing.empty:
        return "該当なし"
    order = {"fixed": 0, "sqrt": 1, "perm95": 2}
    ranked = passing.assign(_order=passing.family.map(order)).sort_values(["_order", "value"])
    return str(ranked.iloc[0].candidate)


def _k_effect(fixed: pd.DataFrame) -> str:
    frame = fixed[fixed.value == 0.01]
    by_k = frame.groupby("n_false").null_rate.mean()
    if 1 not in by_k or 7 not in by_k:
        return "算出不能"
    return f"{100 * by_k.loc[1]:.2f}% → {100 * by_k.loc[7]:.2f}%"


def _accuracy_cost(tradeoff: pd.DataFrame, recommendation: str) -> str:
    frame = tradeoff[tradeoff.candidate == recommendation]
    if frame.empty:
        return "該当候補なし"
    return ", ".join(
        f"{row.objective} {row.delta_mean:+.4f} ± {row.delta_std:.4f}"
        for row in frame.itertuples()
    )


def _fixed_table(frame: pd.DataFrame) -> str:
    lines = [
        "| n | task | min_leaf | K(false) | margin | null | true recovery | "
        "C constant | holdout | Δ vs embedded |",
        "|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in frame.itertuples():
        lines.append(
            f"| {row.n_rows} | {row.objective} | {row.min_samples_leaf} | {row.n_false} | "
            f"{row.value:g} | {100 * row.null_rate_mean:.2f} ± {100 * row.null_rate_std:.2f}% | "
            f"{100 * row.recovery_rate_mean:.2f} ± {100 * row.recovery_rate_std:.2f}% | "
            f"{100 * row.c_constant_rate_mean:.2f} ± {100 * row.c_constant_rate_std:.2f}% | "
            f"{row.metric_mean:.4f} ± {row.metric_std:.4f} | "
            f"{row.delta_mean:+.4f} ± {row.delta_std:.4f} |"
        )
    return "\n".join(lines)


def _curve_table(frame: pd.DataFrame) -> str:
    lines = [
        "| task | K(false) | n_leaf bin | mean n_leaf | null rate | leaves across seeds |",
        "|:---:|---:|:---:|---:|---:|---:|",
    ]
    for row in frame.itertuples():
        lines.append(
            f"| {row.objective} | {row.n_false} | {row.size_bin} | {row.mean_n_leaf:.1f} | "
            f"{100 * row.null_mean:.2f} ± {100 * row.null_std:.2f}% | {row.n_leaves} |"
        )
    return "\n".join(lines)


def _tradeoff_table(frame: pd.DataFrame) -> str:
    lines = [
        "| candidate | task | null | true recovery | C constant | holdout | Δ vs embedded |",
        "|:---|:---:|---:|---:|---:|---:|---:|",
    ]
    for row in frame.itertuples():
        lines.append(
            f"| {row.candidate} | {row.objective} | "
            f"{100 * row.null_rate_mean:.2f} ± {100 * row.null_rate_std:.2f}% | "
            f"{100 * row.recovery_rate_mean:.2f} ± {100 * row.recovery_rate_std:.2f}% | "
            f"{100 * row.c_constant_rate_mean:.2f} ± {100 * row.c_constant_rate_std:.2f}% | "
            f"{row.metric_mean:.4f} ± {row.metric_std:.4f} | "
            f"{row.delta_mean:+.4f} ± {row.delta_std:.4f} |"
        )
    return "\n".join(lines)


def _judgment_table(frame: pd.DataFrame) -> str:
    lines = [
        "| candidate | max null | min recovery | max Δ vs embedded | pass |",
        "|:---|---:|---:|---:|:---:|",
    ]
    for row in frame.itertuples():
        lines.append(
            f"| {row.candidate} | {100 * row.max_null:.2f}% | "
            f"{100 * row.min_recovery:.2f}% | {row.max_delta:+.4f} | "
            f"{'yes' if row.passes else 'no'} |"
        )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Preserve all rule factors but use n=2k, two seeds, and 25 trees.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORT_PATH,
        help="Markdown report path (default: frozen dated result path).",
    )
    return parser.parse_args(argv)


def resolve_settings(quick: bool) -> RunSettings:
    """Resolve the smoke or frozen full experiment grid."""

    if quick:
        return RunSettings(quick=True, n_rows=(2_000,), seeds=(0, 1), n_estimators=25)
    return RunSettings(
        quick=False,
        n_rows=(2_000, 20_000),
        seeds=(0, 1, 2, 3, 4),
        n_estimators=100,
    )


def main(argv: Sequence[str] | None = None) -> Path:
    """Run the experiment and write the generated Markdown report."""

    args = parse_args(argv)
    settings = resolve_settings(args.quick)
    fixed, calibration, size_records, elapsed = run_experiment(settings)
    report = build_report(fixed, calibration, size_records, settings, elapsed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(f"wrote {args.output} ({elapsed:.1f}s)")
    return args.output


if __name__ == "__main__":
    main()
