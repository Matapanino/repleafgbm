"""Statistical-significance utilities for the fair benchmark leaderboard.

These helpers turn a *datasets x models* (or *seeds x models*) table of a
**lower-is-better** primary metric (RMSE / logloss) into the significance
evidence the benchmark overhaul prescribes — so no headline claim is made
without a test:

* :func:`friedman_test` — omnibus "are the models different at all?".
* :func:`nemenyi_cd` + :func:`critical_difference_diagram` — Demsar's
  critical-difference visualization over average ranks.
* :func:`wilcoxon_pairs` — pairwise signed-rank tests against a baseline.
* :func:`bootstrap_ci` — bootstrap confidence interval on an aggregate.
* :func:`win_tie_loss` — per-dataset win/tie/loss with a minimum-relevant
  difference (MRD) band, so near-ties are not counted as wins.

This module lives under ``benchmarks/`` and is **never** imported by the library
(``src/``). It uses SciPy (available transitively via scikit-learn; pinned in the
``[bench]`` extra) and, only for the *rendered* CD diagram, optionally Matplotlib
— a text/markdown fallback is always returned when Matplotlib is absent.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy import stats

__all__ = [
    "rank_matrix",
    "average_ranks",
    "friedman_test",
    "nemenyi_cd",
    "cd_cliques",
    "critical_difference_diagram",
    "wilcoxon_pairs",
    "bootstrap_ci",
    "win_tie_loss",
]


def _as_scores(scores) -> np.ndarray:
    arr = np.asarray(scores, dtype=float)
    if arr.ndim != 2:
        raise ValueError("scores must be 2-D (n_datasets, n_models)")
    return arr


def rank_matrix(scores, lower_is_better: bool = True) -> np.ndarray:
    """Per-row (per-dataset) ranks, 1 = best, averaged over ties.

    ``scores`` is ``(n_datasets, n_models)``.
    """
    arr = _as_scores(scores)
    signed = arr if lower_is_better else -arr
    return np.vstack([stats.rankdata(row) for row in signed])


def average_ranks(scores, names, lower_is_better: bool = True) -> dict[str, float]:
    """Mean rank per model across datasets (lower is better)."""
    ranks = rank_matrix(scores, lower_is_better)
    return {name: float(r) for name, r in zip(names, ranks.mean(axis=0))}


def friedman_test(scores) -> tuple[float, float]:
    """Friedman omnibus test across datasets. Returns ``(statistic, p_value)``.

    ``scores`` is ``(n_datasets, n_models)``; needs >= 3 models and >= 2 datasets.
    """
    arr = _as_scores(scores)
    if arr.shape[1] < 3:
        raise ValueError("Friedman test needs >= 3 models")
    if arr.shape[0] < 2:
        raise ValueError("Friedman test needs >= 2 datasets")
    statistic, p = stats.friedmanchisquare(*arr.T)
    return float(statistic), float(p)


def nemenyi_cd(n_models: int, n_datasets: int, alpha: float = 0.05) -> float:
    """Nemenyi critical difference for average ranks at level ``alpha``.

    ``CD = q_alpha * sqrt(k (k + 1) / (6 N))`` where ``q_alpha`` is the
    studentized-range critical value divided by ``sqrt(2)`` (Demsar 2006).
    """
    if n_models < 2:
        raise ValueError("need >= 2 models")
    if n_datasets < 1:
        raise ValueError("need >= 1 dataset")
    q = stats.studentized_range.ppf(1.0 - alpha, n_models, np.inf) / math.sqrt(2.0)
    return float(q * math.sqrt(n_models * (n_models + 1) / (6.0 * n_datasets)))


def cd_cliques(avg_ranks: dict[str, float], cd: float) -> list[list[str]]:
    """Maximal groups of consecutive (rank-sorted) models spanning <= ``cd``.

    Each returned group is a set of models that are **not** significantly
    different from one another (the bars in a CD diagram). Singletons and groups
    contained in a larger group are dropped.
    """
    ordered = sorted(avg_ranks.items(), key=lambda kv: kv[1])
    names = [n for n, _ in ordered]
    ranks = np.array([r for _, r in ordered], dtype=float)
    n = len(names)
    intervals: list[tuple[int, int]] = []
    for i in range(n):
        j = i
        while j + 1 < n and ranks[j + 1] - ranks[i] <= cd + 1e-12:
            j += 1
        intervals.append((i, j))
    maximal = []
    for (i, j) in intervals:
        if j - i < 1:  # need at least a pair to be informative
            continue
        if any(a <= i and j <= b and (a, b) != (i, j) for (a, b) in intervals):
            continue
        maximal.append((i, j))
    seen = sorted(set(maximal))
    return [names[i:j + 1] for (i, j) in seen]


def wilcoxon_pairs(
    scores, names, baseline, lower_is_better: bool = True
) -> dict[str, tuple[float, float, float]]:
    """Wilcoxon signed-rank of ``baseline`` vs every other model across datasets.

    Returns ``{name: (statistic, p_value, median_delta)}`` where ``median_delta``
    is ``median(other - baseline)`` in the lower-is-better orientation (negative
    ⇒ the other model improves on the baseline). ``lower_is_better`` is accepted
    for API symmetry; the sign of ``median_delta`` already encodes direction.
    """
    arr = _as_scores(scores)
    names = list(names)
    bi = names.index(baseline) if isinstance(baseline, str) else int(baseline)
    base = arr[:, bi]
    out: dict[str, tuple[float, float, float]] = {}
    for j, name in enumerate(names):
        if j == bi:
            continue
        other = arr[:, j]
        delta = other - base
        if np.allclose(delta, 0.0):
            out[name] = (float("nan"), 1.0, 0.0)
            continue
        try:
            statistic, p = stats.wilcoxon(other, base)
        except ValueError:  # pragma: no cover - degenerate inputs
            statistic, p = float("nan"), 1.0
        out[name] = (float(statistic), float(p), float(np.median(delta)))
    return out


def bootstrap_ci(
    values, statistic=np.mean, n_boot: int = 10_000, alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for ``statistic`` over a 1-D sample."""
    vals = np.asarray(values, dtype=float)
    if vals.size == 0:
        raise ValueError("empty values")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, vals.size, size=(n_boot, vals.size))
    resampled = vals[idx]
    try:
        boot = np.asarray(statistic(resampled, axis=1), dtype=float)
    except TypeError:  # statistic without an axis kwarg
        boot = np.array([statistic(row) for row in resampled], dtype=float)
    lo, hi = np.quantile(boot, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def win_tie_loss(
    candidate, baseline, mrd: float = 0.0, lower_is_better: bool = True
) -> tuple[int, int, int]:
    """Per-dataset win/tie/loss of ``candidate`` vs ``baseline``.

    ``mrd`` is a *relative* minimum-relevant-difference band: a pair is a tie
    when ``|cand - base| <= mrd * |base|``. Outside the band, the better metric
    wins (lower, unless ``lower_is_better`` is False).
    """
    cand = np.asarray(candidate, dtype=float)
    base = np.asarray(baseline, dtype=float)
    if cand.shape != base.shape:
        raise ValueError("candidate and baseline must have the same shape")
    diff = cand - base
    tie = np.abs(diff) <= mrd * np.abs(base)
    better = diff < 0 if lower_is_better else diff > 0
    win = (~tie) & better
    loss = (~tie) & (~win)
    return int(win.sum()), int(tie.sum()), int(loss.sum())


def critical_difference_diagram(
    avg_ranks: dict[str, float], cd: float, out_path=None, title: str | None = None
) -> str:
    """Return a markdown CD summary; also write a PNG when Matplotlib is present.

    The text summary (rank table + non-significant cliques) is always returned so
    the report is complete even without Matplotlib. When ``out_path`` is given and
    Matplotlib imports, a Demsar diagram is rendered and the returned text is
    prefixed with an image link.
    """
    cliques = cd_cliques(avg_ranks, cd)
    ordered = sorted(avg_ranks.items(), key=lambda kv: kv[1])
    lines = [
        f"Critical difference (Nemenyi, CD = {cd:.3f}); lower average rank = better.",
        "",
        "| place | model | avg rank |",
        "|---|---|---|",
    ]
    for place, (name, r) in enumerate(ordered, 1):
        lines.append(f"| {place} | {name} | {r:.3f} |")
    lines.append("")
    if cliques:
        lines.append("Groups **not** significantly different (avg-rank span <= CD):")
        for c in cliques:
            lines.append(f"- {{{', '.join(c)}}}")
    else:
        lines.append("Every adjacent average-rank gap exceeds CD "
                     "(all models pairwise-separated).")
    text = "\n".join(lines)

    if out_path is not None:
        png = _render_cd_png(ordered, cd, cliques, Path(out_path), title)
        if png is not None:
            text = f"![CD diagram]({Path(png).name})\n\n" + text
    return text


def _render_cd_png(ordered, cd, cliques, out_path: Path, title):
    """Best-effort Demsar CD plot. Returns the path, or None if Matplotlib absent."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - exercised by the no-matplotlib lane
        return None

    names = [n for n, _ in ordered]
    ranks = [r for _, r in ordered]
    k = len(names)
    lo, hi = math.floor(min(ranks)), math.ceil(max(ranks))
    if hi <= lo:
        hi = lo + 1
    rank_of = dict(ordered)

    fig, ax = plt.subplots(figsize=(8.0, 0.5 * k + 1.6))
    ax.set_xlim(lo, hi)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    y_axis = 0.85
    ax.plot([lo, hi], [y_axis, y_axis], "k-", lw=1)
    for t in range(lo, hi + 1):
        ax.plot([t, t], [y_axis, y_axis + 0.02], "k-", lw=1)
        ax.text(t, y_axis + 0.045, str(t), ha="center", va="bottom", fontsize=9)

    half = (k + 1) // 2
    for i, (name, r) in enumerate(ordered):
        if i < half:
            y = y_axis - 0.12 - i * (0.6 / max(half, 1))
            x_text, ha, dx = lo, "right", -0.04
        else:
            y = y_axis - 0.12 - (k - 1 - i) * (0.6 / max(k - half, 1))
            x_text, ha, dx = hi, "left", 0.04
        ax.plot([r, r], [y_axis, y], "k-", lw=1)
        ax.plot([r, x_text], [y, y], "k-", lw=1)
        ax.text(x_text + dx, y, f"{name} ({r:.2f})", ha=ha, va="center", fontsize=9)

    ax.plot([lo, lo + cd], [y_axis + 0.10, y_axis + 0.10], "k-", lw=3)
    ax.text(lo + cd / 2.0, y_axis + 0.13, f"CD = {cd:.2f}", ha="center", fontsize=9)

    for ci, clique in enumerate(cliques):
        rs = [rank_of[n] for n in clique]
        ax.plot([min(rs) - 0.05, max(rs) + 0.05],
                [y_axis - 0.05 - ci * 0.025] * 2, "k-", lw=3, solid_capstyle="round")

    if title:
        ax.set_title(title, fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path
