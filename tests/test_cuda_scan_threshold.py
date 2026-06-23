"""CPU-safe unit tests for the CUDA adaptive-scan threshold resolver.

These exercise ``_resolve_scan_min_cells`` and the ``REPLEAFGBM_CUDA_SCAN_MIN_CELLS``
env override **without CuPy or a GPU**: ``cuda_backend`` imports cleanly on a
CPU-only box (CuPy is imported lazily inside ``CudaSplitBackend.__init__``), so the
env plumbing is unit-testable here and runs on CI/macOS. The GPU-gated behavioural
counterparts — which scan path a node actually takes under an override, and split
parity on that path — live in tests/test_cuda_backend.py.

``monkeypatch.setenv`` auto-restores, so the env var never leaks between tests.
"""

from __future__ import annotations

import pytest

from repleafgbm.backends.cuda_backend import (
    _GPU_SCAN_MIN_CELLS,
    _SCAN_MIN_CELLS_ENV,
    _resolve_scan_min_cells,
)


def test_default_is_32768_when_unset(monkeypatch):
    monkeypatch.delenv(_SCAN_MIN_CELLS_ENV, raising=False)
    assert _resolve_scan_min_cells() == _GPU_SCAN_MIN_CELLS == 32768


def test_env_override_parses_int(monkeypatch):
    monkeypatch.setenv(_SCAN_MIN_CELLS_ENV, "8192")
    assert _resolve_scan_min_cells() == 8192


def test_zero_is_kept(monkeypatch):
    # 0 means "no histogram is ever below threshold" → every node scans on-device.
    monkeypatch.setenv(_SCAN_MIN_CELLS_ENV, "0")
    assert _resolve_scan_min_cells() == 0


def test_negative_clamps_to_zero(monkeypatch):
    monkeypatch.setenv(_SCAN_MIN_CELLS_ENV, "-5")
    assert _resolve_scan_min_cells() == 0


def test_blank_uses_default(monkeypatch):
    monkeypatch.setenv(_SCAN_MIN_CELLS_ENV, "   ")
    assert _resolve_scan_min_cells() == _GPU_SCAN_MIN_CELLS


def test_invalid_warns_and_falls_back(monkeypatch):
    monkeypatch.setenv(_SCAN_MIN_CELLS_ENV, "garbage")
    with pytest.warns(RuntimeWarning):
        assert _resolve_scan_min_cells() == _GPU_SCAN_MIN_CELLS
