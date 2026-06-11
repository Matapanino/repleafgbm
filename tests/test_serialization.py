"""Save/load round-trip tests."""

import numpy as np
import pytest

from repleafgbm import RepLeafClassifier, RepLeafRegressor


@pytest.mark.parametrize(
    "leaf_model,encoder", [("constant", "identity"), ("embedded_linear", "plr")]
)
def test_regressor_roundtrip(tmp_path, regression_data, leaf_model, encoder):
    Xtr, ytr, Xte, _ = regression_data
    model = RepLeafRegressor(
        n_estimators=10,
        num_leaves=8,
        leaf_model=leaf_model,
        encoder=encoder,
        max_leaf_emb_dim=10,
        random_state=42,
    )
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)

    path = tmp_path / "model"
    model.save_model(path)
    loaded = RepLeafRegressor.load_model(path)
    np.testing.assert_allclose(loaded.predict(Xte), pred)

    # Hyperparameters survive the round-trip.
    assert loaded.get_params()["leaf_model"] == leaf_model
    assert loaded.get_params()["n_estimators"] == 10


def test_classifier_roundtrip(tmp_path, classification_data):
    Xtr, ytr, Xte, _ = classification_data
    model = RepLeafClassifier(n_estimators=10, num_leaves=8, random_state=42)
    model.fit(Xtr, ytr)
    proba = model.predict_proba(Xte)

    path = tmp_path / "model"
    model.save_model(path)
    loaded = RepLeafClassifier.load_model(path)
    np.testing.assert_allclose(loaded.predict_proba(Xte), proba)
    np.testing.assert_array_equal(loaded.classes_, model.classes_)


def test_wrong_class_rejected(tmp_path, regression_data):
    Xtr, ytr, _, _ = regression_data
    model = RepLeafRegressor(n_estimators=3, random_state=42).fit(Xtr, ytr)
    path = tmp_path / "model"
    model.save_model(path)
    with pytest.raises(ValueError, match="RepLeafRegressor"):
        RepLeafClassifier.load_model(path)


def test_missing_directory_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        RepLeafRegressor.load_model(tmp_path / "nope")


def test_unfitted_save_rejected(tmp_path):
    with pytest.raises(RuntimeError, match="not fitted"):
        RepLeafRegressor().save_model(tmp_path / "model")
