"""Tests for the grow_policy real-data gate experiment + benchmark arms.

Pure report-builder tests (hand-built ledger records — no fitting, no network):
the paired >=1 sigma separation flags, the decision-rule summary counting, the
defensive multi-output n/a row, the aggregate significance block, and the raw
per-seed appendix. Plus a config-level test that the opt-in grow-policy arms in
``benchmarks/openml_suite.py`` carry their own capacity match.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.openml_suite import _repleaf_configs  # noqa: E402


def _load_experiment():
    spec = importlib.util.spec_from_file_location(
        "grow_policy_real_data",
        ROOT / "experiments" / "grow_policy_real_data.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GP = _load_experiment()

_SETTINGS = {"seeds": [0, 1], "max_rows": 1000, "n_estimators": 60,
             "quick": True}


def _rec(dataset, lm, arm, seed, primary, task="regression"):
    return {"suite": GP.SUITE, "dataset": dataset, "model": f"{lm}|{arm}",
            "seed": seed, "stage": "eval",
            "payload": {"task": task, "n_used": 1000, "n_cats": 0,
                        "leaf_model": lm, "arm": arm, "primary": primary,
                        "secondary": 0.9, "fit_seconds": 0.1,
                        "best_iteration": 40}}


def _records():
    """Two regression datasets, both leaf models, all arms, two seeds.

    Constant-leaf symmetric_d5_msl20 is planted 0.5 better than everything on
    d1 (a >=1 sigma win vs both matched leaf-wise shapes) and 0.5 worse on d2
    (a >=1 sigma loss); every other arm sits at 1.0 + a shared seed offset, so
    paired deltas are exactly zero.
    """
    records = []
    for ds in ("d1", "d2"):
        for lm in GP.LEAF_MODELS:
            for arm, _ in GP.arm_grid():
                for seed in (0, 1):
                    val = 1.0 + 0.001 * seed
                    if lm == "constant" and arm == "symmetric_d5_msl20":
                        val += -0.5 if ds == "d1" else 0.5
                    records.append(_rec(ds, lm, arm, seed, val))
    # A skipped cell exercising the defensive NotImplementedError path — a
    # vector multi-output regression dataset, the only place symmetric raises
    # (multiclass is covered via per-class scalar trees).
    records.append({
        "suite": GP.SUITE, "dataset": "mo_reg",
        "model": "constant|symmetric_d5_msl20", "seed": 0, "stage": "eval",
        "payload": {"task": "regression", "n_used": 1000, "n_cats": 0,
                    "leaf_model": "constant", "arm": "symmetric_d5_msl20",
                    "skipped": "symmetric does not support multi-output"}})
    records.append(_rec("mo_reg", "constant", "leafwise_free_nl31_msl20", 0, 1.0))
    return records


def test_paired_delta_and_sep_flags():
    cells, _, _ = GP._collect(_records())
    delta = GP._paired_delta(cells, "d1", "constant", "symmetric_d5_msl20",
                             "leafwise_free_nl31_msl20")
    mean, sigma, n = delta
    assert n == 2
    assert abs(mean - (-0.5)) < 1e-12
    assert sigma < 1e-12  # shared seed offset cancels in the pairing
    assert GP._is_sep_win(delta) is True
    # Identical arms: zero delta -> not separated.
    same = GP._paired_delta(cells, "d1", "adaptive", "depthwise_d5_msl20",
                            "leafwise_free_nl31_msl20")
    assert GP._is_sep_win(same) is None
    assert GP._paired_delta(cells, "d1", "constant", "nonexistent_arm",
                            "leafwise_free_nl31_msl20") is None


def test_build_report_decision_summary_and_appendix():
    report = GP.build_report(_records(), _SETTINGS)
    assert "Decision-rule summary" in report
    # Planted win on d1 and loss on d2 for constant symmetric_d5_msl20.
    row = next(line for line in report.splitlines()
               if line.startswith("| constant | symmetric_d5_msl20 |"))
    assert "1 (d1)" in row and "1 (d2)" in row
    # The symmetric multiclass n/a is stated explicitly, not silently dropped.
    assert "n/a — NotImplementedError" in report
    # Aggregate significance block over the two complete datasets.
    assert "## Aggregate — constant, regime d5 msl20" in report
    assert "Friedman chi-square" in report
    # Raw per-seed values appendix (the gate asks for raw values).
    assert "## Appendix — raw per-seed primary values" in report
    assert "s0=1.0000, s1=1.0010" in report
    # >=1 sigma separations are bolded in the per-dataset tables.
    assert "**-0.5000" in report


def test_report_without_seed_pairs_degrades_gracefully():
    # A single seed: no sigma is computable -> "n=1", never a win/loss flag.
    records = [_rec("d1", "constant", arm, 0, 1.0) for arm, _ in GP.arm_grid()]
    report = GP.build_report(records, dict(_SETTINGS, seeds=[0]))
    assert "(n=1)" in report
    row = next(line for line in report.splitlines()
               if line.startswith("| constant | symmetric_d5_msl20 |"))
    assert "| 0 (-) | 1 | 0 (-) |" in row  # 1 tie/mixed, no wins/losses


def test_openml_suite_grow_policy_arms():
    stock = _repleaf_configs(learned_encoders=False)
    with_arms = _repleaf_configs(learned_encoders=False, grow_policy_arms=True)
    labels = [label for label, _ in with_arms]
    assert [label for label, _ in stock] == labels[:len(stock)]
    for lm in ("constant", "adaptive"):
        for policy in ("leafwise_capped", "depthwise", "symmetric"):
            assert any(label == f"RepLeaf {lm} {policy}_d5" for label in labels)
    # Each arm carries its own capacity match (overrides the shared defaults);
    # depthwise keeps the stock 31-leaf budget (at 32 it is identical to the
    # capped leaf-wise shape).
    for label, kwargs in with_arms[len(stock):]:
        expected_nl = 31 if "depthwise" in label else 32
        assert kwargs["num_leaves"] == expected_nl and kwargs["max_depth"] == 5
