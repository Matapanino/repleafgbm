"""Sample weighting, class_weight, and the balanced_accuracy metric.

Weighting is implemented by scaling each row's gradient/Hessian (and the init
score) — see ``core.booster.weight_grad_hess``. The split backends and leaf
fitting are untouched, so the principled exact invariant tested here is
*uniform scale invariance*: with ``l2_leaf=0`` a constant positive weight
rescales every leaf's numerator and denominator equally and cancels, leaving
predictions unchanged. (Row duplication is **not** an exact invariant for a
histogram GBM: it shifts the per-feature quantile bin edges and the raw-count
``min_samples_leaf`` constraint, so it is deliberately not asserted.)
"""

import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score

from repleafgbm import RepLeafClassifier, RepLeafRegressor
from repleafgbm.core.booster import weight_grad_hess
from repleafgbm.core.metrics import get_metric
from repleafgbm.core.objectives import (
    BinaryLogistic,
    MulticlassSoftmax,
    Quantile,
    SquaredError,
    _weighted_quantile,
)
from repleafgbm.data import RepLeafDataset


def make_imbalanced(n: int, seed: int, n_classes: int = 3):
    """Well-separated blobs with a skewed class prior (majority class 0)."""
    rng = np.random.default_rng(seed)
    centers = 4.0 * np.column_stack(
        [np.cos(2 * np.pi * np.arange(n_classes) / n_classes),
         np.sin(2 * np.pi * np.arange(n_classes) / n_classes)]
    )
    p = np.array([0.8, 0.15, 0.05])[:n_classes]
    p = p / p.sum()
    y = rng.choice(n_classes, size=n, p=p)
    X = centers[y] + rng.normal(0.0, 0.9, (n, 2))
    return X, y


# --------------------------------------------------------------------------- #
# Core math
# --------------------------------------------------------------------------- #
def test_weight_grad_hess_scales_1d_and_2d():
    g = np.array([1.0, -2.0, 3.0])
    h = np.array([1.0, 1.0, 1.0])
    w = np.array([2.0, 0.5, 4.0])
    gw, hw = weight_grad_hess(g, h, w)
    assert np.allclose(gw, g * w)
    assert np.allclose(hw, h * w)
    # None is a no-op (same arrays returned).
    g2, h2 = weight_grad_hess(g, h, None)
    assert g2 is g and h2 is h
    # 2-D (multiclass / multi-output): weight broadcasts over columns.
    G = np.arange(6, dtype=float).reshape(3, 2)
    H = np.ones((3, 2))
    Gw, Hw = weight_grad_hess(G, H, w)
    assert np.allclose(Gw, G * w[:, None])
    assert np.allclose(Hw, H * w[:, None])


def test_weighted_init_scores():
    rng = np.random.default_rng(0)
    y = rng.normal(size=200)
    w = rng.uniform(0.1, 3.0, size=200)
    # Squared error: weighted mean.
    assert SquaredError().init_score(y, weight=w) == pytest.approx(
        np.dot(w, y) / w.sum()
    )
    # weight=None reproduces the unweighted optimum exactly.
    assert SquaredError().init_score(y) == pytest.approx(float(np.mean(y)))
    # Binary logistic: weighted log-odds.
    yb = (y > 0).astype(float)
    p = np.dot(w, yb) / w.sum()
    assert BinaryLogistic().init_score(yb, weight=w) == pytest.approx(
        np.log(p / (1 - p))
    )
    # Multiclass softmax: weighted class priors.
    yc = rng.integers(0, 3, size=200)
    counts = np.bincount(yc, weights=w, minlength=3)
    expected = np.log(counts / counts.sum())
    assert np.allclose(MulticlassSoftmax(3).init_score(yc, weight=w), expected)


def test_weighted_quantile_conventions():
    rng = np.random.default_rng(2)
    y = rng.normal(size=500)
    # weight=None delegates to np.quantile exactly.
    for q in (0.1, 0.5, 0.9):
        assert _weighted_quantile(y, q, None) == pytest.approx(np.quantile(y, q))
    # The midpoint (Hazen) convention matches np.median at q=0.5 under uniform
    # weights and stays within interpolation-convention distance elsewhere.
    assert _weighted_quantile(y, 0.5, np.ones_like(y)) == pytest.approx(
        np.median(y), abs=1e-9
    )
    for q in (0.1, 0.9):
        assert _weighted_quantile(y, q, np.ones_like(y)) == pytest.approx(
            np.quantile(y, q), abs=0.01
        )
    # Weighting toward the high tail pulls the median up; the result stays
    # within the data range.
    yi = np.array([0.0, 1.0, 2.0, 3.0])
    low = _weighted_quantile(yi, 0.5, np.array([3.0, 3.0, 1.0, 1.0]))
    high = _weighted_quantile(yi, 0.5, np.array([1.0, 1.0, 3.0, 3.0]))
    assert low < high
    assert yi.min() <= low <= yi.max()


def test_quantile_objective_uses_weighted_init():
    rng = np.random.default_rng(3)
    y = rng.normal(size=300)
    w = rng.uniform(0.1, 2.0, size=300)
    obj = Quantile(alpha=0.8)
    assert obj.init_score(y, weight=w) == pytest.approx(_weighted_quantile(y, 0.8, w))


# --------------------------------------------------------------------------- #
# End-to-end invariances
#
# A constant positive weight cancels in every leaf's numerator and denominator
# at l2_leaf=0, so it must leave predictions unchanged. Two choices make the
# end-to-end comparison robust (identical on the NumPy and Rust backends, and
# across platforms) rather than flaky:
#   * scale by a **power of two** (2.0 / 4.0) — multiplying a float64 by a power
#     of two only shifts the exponent, so the histogram sums and split gains
#     are bitwise-scaled and the argmax split cannot flip. (A non-power-of-2
#     scale like 3.0 rounds the scaled gains and can flip a near-tied split,
#     diverging by ~1e-2 on some platforms.)
#   * keep trees shallow (num_leaves=8) so the cancellation is bitwise-exact;
#     much deeper trees leave a deterministic ~1e-8 rounding residual (the same
#     on both backends), which is harmless but no longer literally bitwise.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scale", [2.0, 4.0])
def test_uniform_scale_invariance_regressor(scale):
    rng = np.random.default_rng(4)
    X = rng.normal(size=(300, 4))
    y = X[:, 0] + 0.5 * X[:, 1]
    cfg = dict(n_estimators=15, num_leaves=8, leaf_model="constant", l2_leaf=0.0)
    a = RepLeafRegressor(**cfg).fit(X, y).predict(X)
    b = (
        RepLeafRegressor(**cfg)
        .fit(X, y, sample_weight=np.full(X.shape[0], scale))
        .predict(X)
    )
    np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("scale", [2.0, 4.0])
def test_uniform_scale_invariance_multiclass(scale):
    X, y = make_imbalanced(400, seed=5)
    cfg = dict(n_estimators=12, num_leaves=8, leaf_model="constant", l2_leaf=0.0)
    a = RepLeafClassifier(**cfg).fit(X, y).predict_proba(X)
    b = (
        RepLeafClassifier(**cfg)
        .fit(X, y, sample_weight=np.full(X.shape[0], scale))
        .predict_proba(X)
    )
    np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-9)


def test_sample_weight_is_not_a_noop():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(300, 4))
    y = (X[:, 0] > 0).astype(int)
    w = rng.uniform(0.1, 5.0, size=300)
    a = RepLeafClassifier(n_estimators=10, leaf_model="constant").fit(X, y)
    b = RepLeafClassifier(n_estimators=10, leaf_model="constant").fit(
        X, y, sample_weight=w
    )
    assert not np.allclose(a.predict_proba(X), b.predict_proba(X))


# --------------------------------------------------------------------------- #
# class_weight
# --------------------------------------------------------------------------- #
def test_class_weight_balanced_improves_balanced_accuracy():
    X_tr, y_tr = make_imbalanced(800, seed=7)
    X_te, y_te = make_imbalanced(800, seed=8)
    common = dict(n_estimators=40, num_leaves=8, leaf_model="constant")
    plain = RepLeafClassifier(**common).fit(X_tr, y_tr)
    balanced = RepLeafClassifier(class_weight="balanced", **common).fit(X_tr, y_tr)
    ba_plain = balanced_accuracy_score(y_te, plain.predict(X_te))
    ba_bal = balanced_accuracy_score(y_te, balanced.predict(X_te))
    assert ba_bal > ba_plain


def test_class_weight_dict_with_original_labels():
    X, y = make_imbalanced(400, seed=9)
    labels = np.array(["lo", "mid", "hi"])[y]
    model = RepLeafClassifier(
        n_estimators=8,
        leaf_model="constant",
        class_weight={"lo": 1.0, "mid": 2.0, "hi": 6.0},
    ).fit(X, labels)
    assert set(np.unique(model.predict(X))) <= set(model.classes_)


def test_class_weight_unknown_dict_key_raises():
    X, y = make_imbalanced(200, seed=10)
    model = RepLeafClassifier(
        n_estimators=4, leaf_model="constant", class_weight={0: 1.0, 9: 2.0}
    )
    with pytest.raises(ValueError, match="not one of the training classes"):
        model.fit(X, y)


def test_sample_weight_and_class_weight_combine():
    X, y = make_imbalanced(400, seed=11)
    w = np.linspace(0.5, 2.0, X.shape[0])
    only_w = RepLeafClassifier(n_estimators=8, leaf_model="constant").fit(
        X, y, sample_weight=w
    )
    both = RepLeafClassifier(
        n_estimators=8, leaf_model="constant", class_weight="balanced"
    ).fit(X, y, sample_weight=w)
    assert not np.allclose(only_w.predict_proba(X), both.predict_proba(X))


# --------------------------------------------------------------------------- #
# balanced_accuracy metric
# --------------------------------------------------------------------------- #
def test_balanced_accuracy_metric_matches_sklearn_multiclass():
    X, y = make_imbalanced(500, seed=12)
    model = RepLeafClassifier(n_estimators=20, leaf_model="constant").fit(X, y)
    proba = model.predict_proba(X)
    ours = get_metric("balanced_accuracy")(y, proba)
    assert ours == pytest.approx(balanced_accuracy_score(y, model.predict(X)))
    assert get_metric("balanced_accuracy").minimize is False


def test_balanced_accuracy_metric_matches_sklearn_binary():
    rng = np.random.default_rng(13)
    X = rng.normal(size=(400, 3))
    y = (X[:, 0] + 0.3 * rng.normal(size=400) > 0).astype(int)
    model = RepLeafClassifier(n_estimators=20, leaf_model="constant").fit(X, y)
    proba = model.predict_proba(X)[:, 1]
    ours = get_metric("balanced_accuracy")(y, proba)
    assert ours == pytest.approx(balanced_accuracy_score(y, model.predict(X)))


def test_balanced_accuracy_early_stopping_runs():
    X_tr, y_tr = make_imbalanced(600, seed=14)
    X_va, y_va = make_imbalanced(300, seed=15)
    model = RepLeafClassifier(
        n_estimators=50,
        leaf_model="constant",
        eval_metric="balanced_accuracy",
        early_stopping_rounds=5,
    ).fit(X_tr, y_tr, eval_set=[(X_va, y_va)])
    history = model.evals_result_["valid_0"]["balanced_accuracy"]
    assert len(history) >= 1
    assert model.best_iteration_ is not None


# --------------------------------------------------------------------------- #
# Validation, dataset carrier, and serialization
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "bad",
    [
        np.ones(10),  # wrong length
        np.array([1.0, -1.0, 2.0]),  # negative
        np.array([1.0, np.nan, 2.0]),  # non-finite
        np.zeros(3),  # all-zero: no weight mass
    ],
)
def test_sample_weight_validation_errors(bad):
    X = np.random.default_rng(16).normal(size=(3, 2))
    y = np.array([0, 1, 0])
    with pytest.raises(ValueError):
        RepLeafClassifier(n_estimators=2, leaf_model="constant").fit(
            X, y, sample_weight=bad
        )


def test_individual_zero_weights_allowed():
    """Zeroing some rows (but not all) is valid — it drops them from the fit."""
    rng = np.random.default_rng(19)
    X = rng.normal(size=(200, 3))
    y = X[:, 0]
    w = np.ones(200)
    w[:50] = 0.0
    model = RepLeafRegressor(n_estimators=8, leaf_model="constant").fit(
        X, y, sample_weight=w
    )
    assert model.predict(X).shape == (200,)


def test_sample_weight_via_dataset():
    rng = np.random.default_rng(17)
    X = rng.normal(size=(200, 3))
    y = X[:, 0]
    w = rng.uniform(0.5, 2.0, size=200)
    ds = RepLeafDataset(X, y, sample_weight=w)
    assert ds.sample_weight is not None and ds.sample_weight.shape == (200,)
    model = RepLeafRegressor(n_estimators=8, leaf_model="constant").fit(ds)
    # An explicit fit weight (uniform) overrides the dataset's own weights.
    override = RepLeafRegressor(n_estimators=8, leaf_model="constant").fit(
        RepLeafDataset(X, y, sample_weight=w), sample_weight=np.ones(200)
    )
    assert not np.allclose(model.predict(X), override.predict(X))


def test_class_weight_save_load_roundtrip(tmp_path):
    X, y = make_imbalanced(300, seed=18)
    model = RepLeafClassifier(
        n_estimators=8, leaf_model="constant", class_weight="balanced"
    ).fit(X, y)
    path = tmp_path / "model"
    model.save_model(path)
    loaded = RepLeafClassifier.load_model(path)
    assert loaded.get_params()["class_weight"] == "balanced"
    assert np.allclose(model.predict_proba(X), loaded.predict_proba(X))
