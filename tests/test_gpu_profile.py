"""CPU-safe smoke test for the GPU benchmark harness (benchmarks/gpu_profile.py).

The CUDA transfer-counter assertions live in tests/test_cuda_backend.py and are
GPU-gated (skipped on CI/macOS). This test runs the harness end-to-end on the
NumPy backend so CI verifies the JSONL schema, the empty deferred fields, and
that non-CUDA backends report no device transfers — without needing a GPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make the repo root importable so ``benchmarks`` resolves regardless of how
# pytest is launched (mirrors the harness's own sys.path bootstrap).
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import gpu_profile  # noqa: E402

_REQUIRED_KEYS = {
    "case_id", "task", "backend", "n_train", "n_test", "n_features", "max_bins",
    "num_leaves", "leaf_model", "encoder", "cuda_scan_min_cells", "fit_seconds",
    "predict_seconds", "quality", "peak_rss_bytes", "peak_gpu_bytes",
    "phase_seconds", "transfer_bytes", "env",
}


def _run(tmp_path, *extra):
    out = tmp_path / "cases.jsonl"
    argv = ["--backend", "numpy", "--quick", "--leaf-model", "constant",
            "--out", str(out), *extra]
    row = gpu_profile.main(argv)
    written = [json.loads(line) for line in out.read_text().splitlines() if line]
    return row, written, out


def test_regression_case_writes_valid_jsonl(tmp_path):
    row, written, out = _run(tmp_path, "--task", "regression")

    assert len(written) == 1
    assert written[0] == row
    assert _REQUIRED_KEYS <= set(row)

    assert row["task"] == "regression"
    assert row["backend"] == "numpy"
    assert row["fit_seconds"] > 0.0
    assert {"rmse", "mae", "r2"} <= set(row["quality"])
    assert all(isinstance(v, float) for v in row["quality"].values())

    # The internal phase profiler populates phase_seconds (the harness enables
    # it around the timed fit/predict); fit + predict phases are present.
    phases = row["phase_seconds"]
    assert {"preprocessing", "binning", "histogram", "split_scan",
            "leaf_fit", "eval", "predict"} <= set(phases)
    assert all(isinstance(v, float) and v >= 0.0 for v in phases.values())
    assert max(phases.values()) > 0.0

    # Non-CUDA transfer fields are empty (not missing).
    assert row["transfer_bytes"] == {}  # only the CUDA backend tracks transfers
    assert row["peak_gpu_bytes"] is None

    # Environment metadata is captured for reproducibility.
    assert row["env"]["python"]
    assert "numpy" in row["env"]["packages"]

    # The summary table is regenerated beside the JSONL.
    assert (out.parent / "summary.md").exists()


def test_binary_case_quality_keys(tmp_path):
    row, _, _ = _run(tmp_path, "--task", "binary")
    assert row["task"] == "binary"
    assert row["n_classes"] == 2
    assert {"logloss", "auc", "accuracy"} <= set(row["quality"])


def test_multiclass_case_quality_keys(tmp_path):
    row, _, _ = _run(tmp_path, "--task", "multiclass", "--n-classes", "4")
    assert row["task"] == "multiclass"
    assert row["n_classes"] == 4
    assert {"multi_logloss", "accuracy"} <= set(row["quality"])


def test_fitted_estimator_pickles_and_drops_backend_handle():
    """The runtime ``split_backend_`` handle must not break pickling: the rust /
    cuda backends wrap an unpicklable native module / CuPy state, so the booster
    drops the handle on pickle (sklearn check_estimator, joblib.dump)."""
    import pickle

    import numpy as np

    from repleafgbm.regressor import RepLeafRegressor

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    y = X[:, 0] + 0.1 * rng.normal(size=200)
    model = RepLeafRegressor(
        n_estimators=5, num_leaves=8, leaf_model="constant", split_backend="auto",
    ).fit(X, y)
    assert model.booster_.split_backend_ is not None  # live handle in-process

    restored = pickle.loads(pickle.dumps(model))
    assert restored.booster_.split_backend_ is None  # dropped on pickle
    np.testing.assert_allclose(restored.predict(X), model.predict(X))


def test_appends_rows(tmp_path):
    out = tmp_path / "cases.jsonl"
    for task in ("regression", "binary"):
        gpu_profile.main(["--backend", "numpy", "--quick", "--leaf-model",
                          "constant", "--task", task, "--out", str(out)])
    written = [json.loads(line) for line in out.read_text().splitlines() if line]
    assert len(written) == 2
    assert {r["task"] for r in written} == {"regression", "binary"}


# --------------------------------------------------------------------------- #
# CUDA scan-threshold plumbing (CPU-safe: numpy backend ignores the knob, but the
# CLI → env → row plumbing is exercised here; the GPU path-switch behaviour is in
# the GPU-gated tests/test_cuda_backend.py).
# --------------------------------------------------------------------------- #
def test_cuda_scan_min_cells_recorded_in_row(tmp_path):
    row, _, _ = _run(tmp_path, "--task", "regression",
                     "--cuda-scan-min-cells", "8192")
    assert row["cuda_scan_min_cells"] == 8192
    assert "scan8192" in row["case_id"]


def test_cuda_scan_min_cells_defaults_to_none(tmp_path):
    row, _, _ = _run(tmp_path, "--task", "regression")
    assert row["cuda_scan_min_cells"] is None
    assert "scan" not in row["case_id"]  # no tag without an override


def test_cuda_scan_min_cells_very_large_token(tmp_path):
    row, _, _ = _run(tmp_path, "--task", "regression",
                     "--cuda-scan-min-cells", "very_large")
    assert row["cuda_scan_min_cells"] == 1_000_000_000


def test_scan_min_cells_sweep_writes_row_per_threshold(tmp_path):
    out = tmp_path / "cases.jsonl"
    rows = gpu_profile.main([
        "--backend", "numpy", "--quick", "--leaf-model", "constant",
        "--task", "regression", "--out", str(out),
        "--scan-min-cells-sweep", "0", "32768", "very_large",
    ])
    # The sweep returns a list (one row per threshold); the no-sweep path still
    # returns a single dict (asserted by the tests above and test_*_writes_*).
    assert isinstance(rows, list)
    assert [r["cuda_scan_min_cells"] for r in rows] == [0, 32768, 1_000_000_000]
    written = [json.loads(line) for line in out.read_text().splitlines() if line]
    assert len(written) == 3
    assert written == rows
    # numpy backend tracks no device transfers, regardless of the threshold knob.
    assert all(r["transfer_bytes"] == {} for r in rows)


def test_scan_env_not_leaked_after_run(tmp_path):
    """run_case sets REPLEAFGBM_CUDA_SCAN_MIN_CELLS only around the fit and
    restores it, so it must be unset again afterwards (no cross-test leak)."""
    import os

    os.environ.pop(gpu_profile._SCAN_ENV, None)
    _run(tmp_path, "--task", "regression", "--cuda-scan-min-cells", "8192")
    assert gpu_profile._SCAN_ENV not in os.environ
