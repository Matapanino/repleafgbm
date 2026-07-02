"""Per-round evaluation logging for the boosting loops.

With ``verbose=N > 0`` and at least one eval set, every N-th boosting round
prints one line to stdout in the familiar GBM-CLI format::

    [10]	valid_0's rmse: 0.123456

All three boosting loops (scalar :class:`~repleafgbm.core.booster.Booster`,
multiclass, multi-output) share this class so the format stays identical
everywhere. Plain ``print`` is used on purpose: the library imposes no
logging-handler configuration on users, matching LightGBM/XGBoost behavior.
A general callback hook may replace this if one is ever needed.
"""

from __future__ import annotations

__all__ = ["EvalLogger"]

EvalsResult = dict[str, dict[str, list[float]]]


class EvalLogger:
    """Prints eval-set scores every ``period`` boosting rounds.

    ``period <= 0`` (the default everywhere) disables all output. With no
    eval sets there is nothing to report, so the logger also stays silent
    for positive ``period`` — the boosting loops only call it after they
    appended scores to ``evals_result_``.
    """

    def __init__(self, period: int) -> None:
        self.period = int(period)

    @staticmethod
    def _score_line(evals_result: EvalsResult, index: int) -> str:
        """Tab-joined ``name's metric: value`` pairs at round ``index`` (0-based)."""
        return "\t".join(
            f"{name}'s {metric}: {history[index]:.6f}"
            for name, metrics in evals_result.items()
            for metric, history in metrics.items()
            if len(history) > index
        )

    def log_round(self, iteration: int, evals_result: EvalsResult) -> None:
        """Print round ``iteration`` (1-based) when it lands on the period."""
        if self.period <= 0 or iteration % self.period != 0:
            return
        scores = self._score_line(evals_result, iteration - 1)
        if scores:
            print(f"[{iteration}]\t{scores}")

    def log_early_stop(self, best_iteration: int, evals_result: EvalsResult) -> None:
        """Print the best round's scores when early stopping triggers."""
        if self.period <= 0:
            return
        scores = self._score_line(evals_result, best_iteration - 1)
        if scores:
            print(f"Early stopping, best iteration is:\n[{best_iteration}]\t{scores}")
