"""Metric correctness tests (cross-checked against scikit-learn)."""

import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    mean_absolute_error,
    roc_auc_score,
)

from repleafgbm.core.metrics import get_metric


@pytest.fixture
def scores():
    rng = np.random.default_rng(0)
    y = (rng.random(200) > 0.4).astype(float)
    p = np.clip(0.6 * y + 0.4 * rng.random(200), 1e-6, 1 - 1e-6)
    return y, p


def test_mae_matches_sklearn():
    rng = np.random.default_rng(1)
    y, pred = rng.normal(size=100), rng.normal(size=100)
    assert get_metric("mae")(y, pred) == pytest.approx(mean_absolute_error(y, pred))


def test_logloss_matches_sklearn(scores):
    y, p = scores
    assert get_metric("logloss")(y, p) == pytest.approx(log_loss(y, p))


def test_auc_matches_sklearn(scores):
    y, p = scores
    assert get_metric("auc")(y, p) == pytest.approx(roc_auc_score(y, p))


def test_auc_with_ties_matches_sklearn():
    y = np.array([0, 0, 1, 1, 0, 1, 1, 0], dtype=float)
    p = np.array([0.1, 0.5, 0.5, 0.5, 0.2, 0.9, 0.2, 0.9])  # heavy ties
    assert get_metric("auc")(y, p) == pytest.approx(roc_auc_score(y, p))


def test_auc_single_class_raises():
    with pytest.raises(ValueError, match="single class"):
        get_metric("auc")(np.ones(5), np.linspace(0, 1, 5))


def test_accuracy_matches_sklearn(scores):
    y, p = scores
    assert get_metric("accuracy")(y, p) == pytest.approx(
        accuracy_score(y, (p >= 0.5).astype(float))
    )


def test_metric_directions():
    assert get_metric("rmse").minimize and get_metric("mae").minimize
    assert get_metric("logloss").minimize
    assert not get_metric("auc").minimize and not get_metric("accuracy").minimize


def test_unknown_metric():
    with pytest.raises(ValueError, match="Unknown metric"):
        get_metric("nope")
