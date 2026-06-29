"""Tests for benchmarks/leaderboard.py.

Two layers: a pure build_report() test (hand-built ledger records, no model
fitting) that exercises the significance aggregation + honest bolding, and a
small offline end-to-end smoke (synthetic suite, 1 trial) that proves the
HPO->test->report->ledger pipeline wires up and resumes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import leaderboard as LB  # noqa: E402

_SETTINGS = {"suite": "t", "seeds": [0], "n_trials": 5, "max_rows": 2000,
             "alpha": 0.05, "mrd": 0.01, "quick": True}


def _rec(dataset, model, primary, task="regression"):
    return {"suite": "t", "dataset": dataset, "model": model, "seed": 0,
            "stage": "eval", "payload": {"task": task, "primary": primary,
                                         "secondary": 0.9, "fit_seconds": 0.1}}


def _ordered_records():
    # 6 datasets, strict ordering repleaf < lightgbm < catboost on every one.
    records = []
    for i in range(6):
        records.append(_rec(f"d{i}", "repleaf", 0.10 + 0.001 * i))
        records.append(_rec(f"d{i}", "lightgbm", 0.20 + 0.001 * i))
        records.append(_rec(f"d{i}", "catboost", 0.30 + 0.001 * i))
    return records


def test_build_report_significance_and_bolding():
    # assets_dir=None -> no PNG; the significance text is matplotlib-independent
    # (the lean .[dev] CI lane has no matplotlib).
    text = LB.build_report(_ordered_records(), _SETTINGS,
                           provenance={"packages": {}}, assets_dir=None)

    assert "Friedman chi-square" in text
    assert "Critical difference" in text
    # Baseline is the strongest non-RepLeaf model (lightgbm), and RepLeaf beats it
    # significantly (6/6, Wilcoxon p=0.03125) -> bolded with a "sig. better" verdict.
    assert "Baseline for pairwise tests: **lightgbm**" in text
    assert "**repleaf**" in text
    assert "sig. better" in text
    assert "6/0/0" in text  # win/tie/loss of repleaf vs lightgbm


def test_cd_diagram_png_written_when_matplotlib_present(tmp_path):
    pytest.importorskip("matplotlib")  # matplotlib is a [bench] extra, not on .[dev]
    text = LB.build_report(_ordered_records(), _SETTINGS, assets_dir=tmp_path)
    # CD filename is suite-qualified (_SETTINGS suite="t") to avoid cross-suite clobber.
    assert (tmp_path / "leaderboard-cd-t-regression.png").exists()
    assert "![CD diagram](leaderboard-cd-t-regression.png)" in text


def test_build_report_skips_aggregation_when_too_small():
    records = [_rec("d0", "repleaf", 0.1), _rec("d0", "lightgbm", 0.2)]
    text = LB.build_report(records, _SETTINGS)
    assert "Significance aggregation skipped" in text


def test_build_report_empty():
    text = LB.build_report([], _SETTINGS)
    assert "No completed cells" in text


def test_leaderboard_end_to_end_and_resume(tmp_path):
    pytest.importorskip("optuna")
    out = tmp_path / "leaderboard.md"
    ledger = tmp_path / "ledger.jsonl"
    argv = ["--suite", "synthetic", "--quick", "--n-trials", "1", "--seeds", "1",
            "--datasets", "synthetic_reg", "synthetic_bin",
            "--models", "repleaf,hist_gradient_boosting",
            "--max-rows", "400", "--out", str(out), "--ledger", str(ledger)]

    returned = LB.main(argv)
    assert returned == out
    text = out.read_text()
    assert text.startswith("# Fair leaderboard")
    assert "Reproducibility manifest" in text
    assert "Regression" in text and "Binary" in text

    from benchmarks.ledger import Ledger
    n_cells = len(Ledger(ledger))
    assert n_cells == 4  # 2 datasets x 2 models x 1 seed

    # Resume: a second run recomputes nothing (all cells already in the ledger).
    LB.main(argv)
    assert len(Ledger(ledger)) == n_cells
