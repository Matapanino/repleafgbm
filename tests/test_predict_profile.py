"""CPU-safe smoke test for the prediction-traversal benchmark.

Runs ``benchmarks/predict_profile.py`` end-to-end on the NumPy backend (the
decomposition is backend-independent) so CI verifies the JSONL schema, the
routing/leaf-eval split, that the decomposition reconstructs the real predict
path (tiny parity), and per-task quality keys — without needing a GPU or the
native extension. Seconds-long ``--quick`` matrix.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make the repo root importable so ``benchmarks`` resolves regardless of how
# pytest is launched (mirrors the harness's own sys.path bootstrap).
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import predict_profile  # noqa: E402

_REQUIRED_KEYS = {
    "case_id", "task", "backend", "n_classes", "n_estimators", "n_trees",
    "n_rows", "leaf_model", "encoder", "routing_seconds", "leaf_eval_seconds",
    "predict_seconds", "overhead_seconds", "routing_share",
    "parity_max_abs_diff", "quality", "peak_rss_bytes", "env",
}


def _run(tmp_path, *extra):
    out = tmp_path / "cases.jsonl"
    rows = predict_profile.main(["--quick", "--backend", "numpy",
                                 "--out", str(out), *extra])
    written = [json.loads(line) for line in out.read_text().splitlines() if line]
    return rows, written, out


def test_quick_matrix_writes_valid_jsonl(tmp_path):
    rows, written, out = _run(tmp_path)

    assert written == rows
    assert len(rows) >= 6  # 3 tasks x 2 leaf models (+ categorical when pandas)

    for r in rows:
        assert _REQUIRED_KEYS <= set(r)
        # routing and the end-to-end predict are real, timed work; leaf-eval can
        # be near-zero for constant leaves (a pure bias gather) so only require
        # it to be non-negative.
        assert r["routing_seconds"] > 0.0
        assert r["predict_seconds"] > 0.0
        assert r["leaf_eval_seconds"] >= 0.0
        assert isinstance(r["overhead_seconds"], float)
        # routing is a subset of predict, so the share is ~<= 1; allow slack
        # because best-of-repeats timing is noisy at the sub-ms quick scale.
        assert 0.0 < r["routing_share"] < 2.0
        # The decomposition reconstructs the real booster.predict_raw output.
        assert r["parity_max_abs_diff"] < 1e-6
        assert all(isinstance(v, float) for v in r["quality"].values())

    # The summary table is regenerated beside the JSONL.
    assert (out.parent / "summary.md").exists()

    # Environment metadata is captured for reproducibility.
    assert rows[0]["env"]["python"]
    assert "numpy" in rows[0]["env"]["packages"]


def test_per_task_quality_keys(tmp_path):
    rows, _, _ = _run(tmp_path)
    by_task = {r["task"]: r["quality"] for r in rows}
    assert {"rmse", "r2"} <= set(by_task["regression"])
    assert {"logloss", "auc"} <= set(by_task["binary"])
    assert {"multi_logloss"} <= set(by_task["multiclass"])


def test_multiclass_predicting_trees_scale_with_classes(tmp_path):
    """Multiclass stores n_rounds x n_classes trees; n_trees reflects that."""
    rows, _, _ = _run(tmp_path)
    mc = next(r for r in rows if r["task"] == "multiclass")
    assert mc["n_classes"] == 3
    assert mc["n_trees"] == mc["n_estimators"] * mc["n_classes"]


def test_categorical_case_present_when_pandas(tmp_path):
    pytest.importorskip("pandas")
    rows, _, _ = _run(tmp_path)
    cat = [r for r in rows if r["task"] == "regression_cat"]
    assert cat  # the worst-case categorical/missing routing case ran
    assert cat[0]["leaf_model"] == "embedded_linear"


def test_rewrites_without_duplicating(tmp_path):
    """The matrix is rewritten fresh each run (not appended)."""
    out = tmp_path / "cases.jsonl"
    argv = ["--quick", "--backend", "numpy", "--no-categorical", "--out", str(out)]
    first = predict_profile.main(argv)
    second = predict_profile.main(argv)
    written = [json.loads(line) for line in out.read_text().splitlines() if line]
    assert len(first) == len(second) == len(written)
