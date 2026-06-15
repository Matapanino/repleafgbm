"""Label smoothing for binary and multiclass classification."""

import numpy as np
import pytest

from repleafgbm import RepLeafClassifier
from repleafgbm.core.objectives import BinaryLogistic, MulticlassSoftmax


def make_binary(n: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    return X, y


def make_blobs(n: int, seed: int, n_classes: int = 3):
    rng = np.random.default_rng(seed)
    centers = 4.0 * np.column_stack(
        [np.cos(2 * np.pi * np.arange(n_classes) / n_classes),
         np.sin(2 * np.pi * np.arange(n_classes) / n_classes)]
    )
    y = rng.integers(0, n_classes, n)
    X = centers[y] + rng.normal(0.0, 0.8, (n, 2))
    return X, y


def test_eps_zero_matches_unsmoothed():
    X, y = make_binary(400, seed=0)
    base = RepLeafClassifier(n_estimators=30, num_leaves=8, random_state=0).fit(X, y)
    smooth0 = RepLeafClassifier(
        n_estimators=30, num_leaves=8, label_smoothing=0.0, random_state=0
    ).fit(X, y)
    assert np.allclose(base.predict_proba(X), smooth0.predict_proba(X))


def test_binary_smoothing_reduces_confidence():
    X, y = make_binary(400, seed=1)
    sharp = RepLeafClassifier(n_estimators=50, num_leaves=8, random_state=0).fit(X, y)
    soft = RepLeafClassifier(
        n_estimators=50, num_leaves=8, label_smoothing=0.2, random_state=0
    ).fit(X, y)
    # Smoothing pulls the most-confident predictions inward.
    assert soft.predict_proba(X)[:, 1].max() < sharp.predict_proba(X)[:, 1].max()
    assert soft.predict_proba(X)[:, 1].min() > sharp.predict_proba(X)[:, 1].min()


def test_multiclass_smoothing_reduces_confidence():
    X, y = make_blobs(400, seed=0)
    sharp = RepLeafClassifier(n_estimators=40, num_leaves=8, random_state=0).fit(X, y)
    soft = RepLeafClassifier(
        n_estimators=40, num_leaves=8, label_smoothing=0.2, random_state=0
    ).fit(X, y)
    assert soft.predict_proba(X).max() < sharp.predict_proba(X).max()


def test_smoothing_survives_save_load(tmp_path):
    X, y = make_binary(300, seed=2)
    model = RepLeafClassifier(
        n_estimators=20, num_leaves=8, label_smoothing=0.15, random_state=0
    ).fit(X, y)
    before = model.predict_proba(X)
    model.save_model(tmp_path)
    reloaded = RepLeafClassifier.load_model(tmp_path)
    assert reloaded.label_smoothing == 0.15
    assert np.allclose(reloaded.predict_proba(X), before)


@pytest.mark.parametrize("eps", [-0.1, 1.0, 1.5])
def test_invalid_label_smoothing_rejected(eps):
    with pytest.raises(ValueError, match="label_smoothing"):
        BinaryLogistic(label_smoothing=eps)
    with pytest.raises(ValueError, match="label_smoothing"):
        MulticlassSoftmax(3, label_smoothing=eps)


def test_binary_objective_smoothing_targets():
    """g = p - smoothed_target at the init score."""
    y = np.array([0.0, 1.0, 1.0, 0.0])
    obj = BinaryLogistic(label_smoothing=0.2)
    smoothed = y * 0.8 + 0.1
    assert np.allclose(smoothed, [0.1, 0.9, 0.9, 0.1])
    f = np.zeros_like(y)  # p = 0.5
    grad, _ = obj.grad_hess(y, f)
    assert np.allclose(grad, 0.5 - smoothed)
