"""Tests for the M10 select-null calibration experiment."""

import math

import numpy as np
from experiments.select_null_calibration import (
    BLOCK_DIM,
    FALSE_COUNTS,
    CalibratedSelectLeafModel,
    RunSettings,
    expert_specs,
    false_names,
    fit_select,
    make_synthetic_data,
)


def test_synthetic_data_is_seeded_and_contains_frozen_candidate_pool():
    first = make_synthetic_data(240, seed=7)
    second = make_synthetic_data(240, seed=7)

    assert first.train.equals(second.train)
    np.testing.assert_array_equal(first.y_reg_train, second.y_reg_train)
    np.testing.assert_array_equal(first.y_bin_train, second.y_bin_train)
    np.testing.assert_array_equal(first.region_train, second.region_train)
    assert set(first.region_train) == {"A", "B", "C"}
    assert first.train.shape == (240, 6 + (2 + max(FALSE_COUNTS)) * BLOCK_DIM)
    assert not np.array_equal(
        first.train[[f"T1_{i}" for i in range(BLOCK_DIM)]].to_numpy(),
        first.train[[f"P1_{i}" for i in range(BLOCK_DIM)]].to_numpy(),
    )


def test_false_pool_and_expert_specs_follow_requested_k():
    assert false_names(0) == []
    assert false_names(1) == ["P1"]
    assert false_names(3) == ["P1", "N1", "N2"]
    assert [spec["name"] for spec in expert_specs(7)] == [
        "T1",
        "T2",
        "P1",
        "N1",
        "N2",
        "N3",
        "N4",
        "N5",
        "N6",
    ]


def test_sqrt_threshold_uses_false_k_and_observed_leaf_size():
    selector = CalibratedSelectLeafModel(
        calibration="sqrt",
        n_false=3,
        calibration_value=0.5,
        l2=1.0,
        min_samples_linear=10,
        leaf_gate_margin=0.0,
    )
    assert selector._threshold(100, 0.0) == 0.5 * math.sqrt(math.log(4) / 100)


def test_tiny_end_to_end_select_run_returns_leaf_and_holdout_metrics():
    data = make_synthetic_data(500, seed=3)
    settings = RunSettings(
        quick=True,
        n_rows=(500,),
        seeds=(3,),
        n_estimators=2,
        num_leaves=5,
    )
    summary, bins = fit_select(
        data,
        settings,
        objective="reg",
        n_false=3,
        min_samples_leaf=20,
        seed=3,
        rule="sqrt",
        value=1.0,
    )

    assert np.isfinite(summary["metric"])
    assert 0.0 <= summary["null_rate"] <= 1.0
    assert 0.0 <= summary["eligible_fraction"] <= 1.0
    assert sum(row["n_leaves"] for row in bins) == summary["n_leaves"]
