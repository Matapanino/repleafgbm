"""End-to-end binary classification tests."""

import numpy as np

from repleafgbm import RepLeafClassifier


def test_fit_predict_proba(classification_data):
    Xtr, ytr, Xte, yte = classification_data
    model = RepLeafClassifier(
        n_estimators=30,
        num_leaves=8,
        min_samples_leaf=10,
        leaf_model="embedded_linear",
        encoder="identity",
        random_state=42,
    )
    model.fit(Xtr, ytr)

    proba = model.predict_proba(Xte)
    assert proba.shape == (len(yte), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert ((proba >= 0) & (proba <= 1)).all()

    acc = (model.predict(Xte) == yte).mean()
    assert acc > 0.85


def test_string_labels(classification_data):
    Xtr, ytr, Xte, yte = classification_data
    labels = np.array(["neg", "pos"])
    model = RepLeafClassifier(n_estimators=15, num_leaves=8, random_state=42)
    model.fit(Xtr, labels[ytr])
    assert set(model.classes_) == {"neg", "pos"}
    pred = model.predict(Xte)
    assert set(np.unique(pred)) <= {"neg", "pos"}
    assert (pred == labels[yte]).mean() > 0.8


def test_three_classes_use_multiclass_booster():
    # Multiclass targets are no longer rejected: 3+ classes switch to
    # softmax boosting (full coverage in tests/test_multiclass.py).
    X = np.random.default_rng(0).normal(size=(60, 2))
    y = np.arange(60) % 3
    model = RepLeafClassifier(n_estimators=2, num_leaves=4).fit(X, y)
    assert model.n_classes_ == 3
    assert model.predict_proba(X).shape == (60, 3)
