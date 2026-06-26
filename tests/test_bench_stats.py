"""Unit tests for benchmarks/stats.py (significance utilities).

Seeded and sub-second: known-outcome toy tables for Friedman / Wilcoxon /
win-tie-loss, Nemenyi-CD monotonicity, bootstrap-CI coverage, and that the CD
diagram returns a complete text summary (with a PNG when Matplotlib is present).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make the repo root importable so ``benchmarks`` resolves regardless of how
# pytest is launched (mirrors test_gpu_profile.py's bootstrap).
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import stats as S  # noqa: E402


def test_rank_matrix_and_average_ranks():
    scores = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    ranks = S.rank_matrix(scores)  # lower is better
    assert np.allclose(ranks, [[1, 2, 3], [1, 2, 3]])
    avg = S.average_ranks(scores, ["a", "b", "c"])
    assert avg == {"a": 1.0, "b": 2.0, "c": 3.0}


def test_rank_matrix_handles_ties():
    ranks = S.rank_matrix(np.array([[1.0, 1.0, 2.0]]))
    assert np.allclose(ranks, [[1.5, 1.5, 3.0]])


def test_friedman_detects_consistent_ordering():
    # model 0 always best, model 2 always worst across 5 datasets -> significant.
    scores = np.array([[1.0, 2.0, 3.0]]) + np.arange(5)[:, None] * 0.1
    stat, p = S.friedman_test(scores)
    assert stat > 0
    assert p < 0.05


def test_friedman_guards_shapes():
    with pytest.raises(ValueError):
        S.friedman_test(np.ones((5, 2)))  # < 3 models
    with pytest.raises(ValueError):
        S.friedman_test(np.ones((1, 3)))  # < 2 datasets


def test_nemenyi_cd_positive_and_monotone():
    cd = S.nemenyi_cd(3, 10)
    assert cd > 0
    # more datasets -> tighter CD; more models -> wider CD.
    assert S.nemenyi_cd(3, 40) < cd
    assert S.nemenyi_cd(5, 10) > cd


def test_wilcoxon_pairs_directional():
    # 'other' beats baseline 'b' by exactly 1 (lower better) on every dataset.
    scores = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0],
                       [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
    out = S.wilcoxon_pairs(scores, ["a", "b"], baseline="b")
    stat, p, median_delta = out["a"]
    assert median_delta == -1.0  # 'a' improves on baseline
    assert p <= 0.05


def test_wilcoxon_pairs_identical_is_null():
    scores = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    out = S.wilcoxon_pairs(scores, ["a", "b"], baseline="b")
    stat, p, median_delta = out["a"]
    assert np.isnan(stat)
    assert p == 1.0
    assert median_delta == 0.0


def test_bootstrap_ci_brackets_mean():
    lo, hi = S.bootstrap_ci(np.arange(100.0), n_boot=2000, seed=0)
    assert lo < 49.5 < hi
    assert 0.0 <= lo and hi <= 99.0


def test_win_tie_loss_mrd_band():
    cand = np.array([1.0, 2.0, 3.0001, 4.0])
    base = np.array([2.0, 2.0, 3.0, 3.0])
    assert S.win_tie_loss(cand, base, mrd=0.0) == (1, 1, 2)
    # a 0.1% band turns the 0.0001 gap into a tie.
    assert S.win_tie_loss(cand, base, mrd=0.001) == (1, 2, 1)


def test_cd_cliques_groups_close_models():
    avg = {"a": 1.0, "b": 1.2, "c": 3.0}
    assert S.cd_cliques(avg, cd=0.5) == [["a", "b"]]
    assert S.cd_cliques(avg, cd=2.0) == [["a", "b", "c"]]


def test_cd_diagram_text_fallback_always_complete():
    avg = {"a": 1.0, "b": 1.2, "c": 3.0}
    text = S.critical_difference_diagram(avg, cd=0.5, out_path=None)
    assert "Critical difference" in text
    assert "avg rank" in text
    assert "{a, b}" in text  # the non-significant clique is reported


def test_cd_diagram_writes_png_when_matplotlib_present(tmp_path):
    pytest.importorskip("matplotlib")
    avg = {"a": 1.0, "b": 1.2, "c": 3.0}
    out = tmp_path / "cd.png"
    text = S.critical_difference_diagram(avg, cd=0.5, out_path=out, title="t")
    assert out.exists() and out.stat().st_size > 0
    assert text.startswith("![CD diagram](cd.png)")
