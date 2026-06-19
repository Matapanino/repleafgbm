"""Tests for the internal per-phase profiler (``repleafgbm.core.profiling``).

The profiler is off by default and enabled only via the ``REPLEAFGBM_PROFILE``
environment variable. Coverage: the accumulator unit behavior, the env gate, the
disabled-default invariant (no ``phase_seconds_``, unchanged predictions), and
that enabling it records the expected fit + predict phases without changing model
output.
"""

from __future__ import annotations

import numpy as np

from repleafgbm.classifier import RepLeafClassifier
from repleafgbm.core.profiling import PhaseProfiler, profiler_from_env, timed
from repleafgbm.regressor import RepLeafRegressor


# --------------------------------------------------------------------------- #
# Unit: PhaseProfiler / timed / profiler_from_env
# --------------------------------------------------------------------------- #
def test_phase_profiler_accumulates():
    prof = PhaseProfiler()
    prof.add("a", 1.0)
    prof.add("a", 0.5)
    prof.add("b", 2.0)
    assert prof.as_dict() == {"a": 1.5, "b": 2.0}
    # as_dict returns a copy: mutating it does not corrupt the profiler.
    prof.as_dict()["a"] = 99.0
    assert prof.as_dict()["a"] == 1.5


def test_phase_context_manager_records_time():
    prof = PhaseProfiler()
    with prof.phase("work"):
        sum(range(10_000))
    assert prof.as_dict()["work"] >= 0.0
    assert "work" in prof.as_dict()


def test_timed_none_is_noop():
    # timed(None, ...) must run the block and record nothing (no clock reads).
    with timed(None, "x"):
        value = 1 + 1
    assert value == 2


def test_timed_records_into_profiler():
    prof = PhaseProfiler()
    with timed(prof, "x"):
        sum(range(1_000))
    assert "x" in prof.as_dict()


def test_profiler_from_env_gate(monkeypatch):
    monkeypatch.delenv("REPLEAFGBM_PROFILE", raising=False)
    assert profiler_from_env() is None
    for truthy in ("1", "true", "yes", "on"):
        monkeypatch.setenv("REPLEAFGBM_PROFILE", truthy)
        assert isinstance(profiler_from_env(), PhaseProfiler)
    monkeypatch.setenv("REPLEAFGBM_PROFILE", "0")
    assert profiler_from_env() is None


# --------------------------------------------------------------------------- #
# Integration: estimators
# --------------------------------------------------------------------------- #
def _reg_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    y = X[:, 0] + 0.1 * rng.normal(size=200)
    return X, y


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("REPLEAFGBM_PROFILE", raising=False)
    X, y = _reg_data()
    model = RepLeafRegressor(n_estimators=5, num_leaves=8).fit(X, y)
    # No phase_seconds_ attribute is created on the default (disabled) path.
    assert not hasattr(model, "phase_seconds_")
    model.predict(X)
    assert not hasattr(model, "phase_seconds_")


def test_enabled_records_phases_without_changing_output(monkeypatch):
    X, y = _reg_data()

    monkeypatch.delenv("REPLEAFGBM_PROFILE", raising=False)
    baseline = RepLeafRegressor(n_estimators=5, num_leaves=8).fit(X, y)
    expected = baseline.predict(X)

    monkeypatch.setenv("REPLEAFGBM_PROFILE", "1")
    profiled = RepLeafRegressor(n_estimators=5, num_leaves=8).fit(X, y)
    out = profiled.predict(X)

    # Enabling profiling must not change the math (same seed + backend).
    np.testing.assert_array_equal(out, expected)

    phases = profiled.phase_seconds_
    # Default leaf_model is embedded_linear, so the encoder phase is present.
    assert {"preprocessing", "encoder", "binning", "histogram", "split_scan",
            "leaf_fit", "eval", "predict"} <= set(phases)
    assert all(isinstance(v, float) and v >= 0.0 for v in phases.values())


def test_multiclass_records_phases(monkeypatch):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(180, 6))
    y = rng.integers(0, 3, size=180)
    monkeypatch.setenv("REPLEAFGBM_PROFILE", "1")
    model = RepLeafClassifier(
        n_estimators=4, num_leaves=8, leaf_model="constant"
    ).fit(X, y)
    model.predict(X)
    phases = model.phase_seconds_
    assert {"preprocessing", "binning", "histogram", "split_scan",
            "leaf_fit", "eval", "predict"} <= set(phases)
