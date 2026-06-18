"""CPU-safe smoke test for the multi-output benchmark suite.

Runs ``benchmarks/multioutput_suite.py`` end-to-end on the **synthetic** dataset
only (no OpenML download) at ``--quick`` settings, so CI verifies the suite wires
up — multi-output RepLeaf fits, the robust objectives, and the report writer —
without needing network or the heavier real dataset. External GBMs are optional
and skipped when absent (no ``--strict``).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable so ``benchmarks`` resolves regardless of how
# pytest is launched (mirrors test_gpu_profile.py's bootstrap).
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import multioutput_suite  # noqa: E402


def test_synthetic_quick_writes_report(tmp_path):
    out = tmp_path / "multioutput_benchmark.md"
    returned = multioutput_suite.main(
        ["--quick", "--datasets", "synthetic", "--out", str(out)]
    )
    assert returned == out
    text = out.read_text()

    # Report structure: title, manifest, both studies present.
    assert text.startswith("# Multi-output regression benchmark suite")
    assert "Reproducibility manifest" in text
    assert "clean leaderboard" in text
    assert "robustness under contaminated train" in text

    # The robust objectives all appear as rows in the contamination study.
    for obj in ("RepLeaf squared", "RepLeaf huber", "RepLeaf quantile(0.5)"):
        assert obj in text


def test_synthetic_signal_shapes():
    """make_synthetic returns aligned (X, noisy, signal) with K outputs."""
    X, noisy, signal = multioutput_suite.make_synthetic(
        n_rows=200, n_features=8, n_outputs=3, seed=0
    )
    assert X.shape == (200, 8)
    assert noisy.shape == signal.shape == (200, 3)
    # Noise perturbs the clean signal but keeps it in the same ballpark.
    assert (noisy != signal).any()
