"""Save/load round-trip, schema-validation, and format-migration tests."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from repleafgbm import RepLeafClassifier, RepLeafDataset, RepLeafRegressor


def _fitted_model(tmp_path, regression_data, **kwargs):
    Xtr, ytr, Xte, _ = regression_data
    params = dict(n_estimators=5, num_leaves=8, random_state=42)
    params.update(kwargs)
    model = RepLeafRegressor(**params).fit(Xtr, ytr)
    path = tmp_path / "model"
    model.save_model(path)
    return model, path, Xte


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


def test_overwrite_embedded_with_constant_clears_stale_encoder(
    tmp_path, regression_data
):
    """Re-saving a constant model over an embedded one must not leave stale
    encoder files behind (they would be reloaded and corrupt prediction)."""
    Xtr, ytr, Xte, _ = regression_data
    path = tmp_path / "model"

    embedded = RepLeafRegressor(
        n_estimators=10,
        num_leaves=8,
        leaf_model="embedded_linear",
        encoder="plr",
        max_leaf_emb_dim=10,
        random_state=42,
    ).fit(Xtr, ytr)
    embedded.save_model(path)
    assert (path / "encoder_config.json").exists()
    assert (path / "encoder_state.npz").exists()

    constant = RepLeafRegressor(
        n_estimators=10, num_leaves=8, leaf_model="constant", random_state=42
    ).fit(Xtr, ytr)
    constant_pred = constant.predict(Xte)
    constant.save_model(path)  # same directory

    # Stale encoder artifacts from the embedded model are gone.
    assert not (path / "encoder_config.json").exists()
    assert not (path / "encoder_state.npz").exists()

    loaded = RepLeafRegressor.load_model(path)
    assert loaded.encoder_ is None
    np.testing.assert_allclose(loaded.predict(Xte), constant_pred)


def test_overwrite_all_categorical_constant_after_embedded(tmp_path, regression_data):
    """The all-categorical constant case is the one the stale encoder would
    break hardest: an orphaned encoder cannot transform a no-numerical-feature
    dataset, so prediction would fail outright."""
    Xtr, ytr, _, _ = regression_data
    path = tmp_path / "model"

    # First an embedded model on numeric data writes encoder_* files.
    RepLeafRegressor(
        n_estimators=5,
        num_leaves=8,
        leaf_model="embedded_linear",
        encoder="plr",
        max_leaf_emb_dim=10,
        random_state=42,
    ).fit(Xtr, ytr).save_model(path)

    # Then an all-categorical constant model is saved over the same path.
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "city": rng.choice(["tokyo", "osaka", "kyoto"], size=200),
            "size": rng.choice(["s", "m", "l"], size=200),
        }
    )
    y = (df["city"] == "tokyo").to_numpy(float) + rng.normal(0, 0.1, 200)
    train = RepLeafDataset(df, y)
    cat = RepLeafRegressor(
        n_estimators=5, num_leaves=8, leaf_model="constant", random_state=42
    ).fit(train)
    pred = cat.predict(df)
    cat.save_model(path)

    assert not (path / "encoder_config.json").exists()
    loaded = RepLeafRegressor.load_model(path)
    assert loaded.encoder_ is None
    np.testing.assert_allclose(loaded.predict(df), pred)


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
    with pytest.raises(NotFittedError, match="not fitted"):
        RepLeafRegressor().save_model(tmp_path / "model")


# --------------------------------------------------------------------- #
# Schema validation
# --------------------------------------------------------------------- #
def test_missing_required_file_rejected(tmp_path, regression_data):
    _, path, _ = _fitted_model(tmp_path, regression_data)
    (path / "leaf_params.npz").unlink()
    with pytest.raises(FileNotFoundError, match="leaf_params.npz"):
        RepLeafRegressor.load_model(path)


def test_missing_ensemble_key_rejected(tmp_path, regression_data):
    _, path, _ = _fitted_model(tmp_path, regression_data)
    ensemble = json.loads((path / "tree_ensemble.json").read_text())
    del ensemble["trees"]
    (path / "tree_ensemble.json").write_text(json.dumps(ensemble))
    with pytest.raises(ValueError, match="tree_ensemble.json.*trees"):
        RepLeafRegressor.load_model(path)


def test_missing_leaf_arrays_rejected(tmp_path, regression_data):
    _, path, _ = _fitted_model(tmp_path, regression_data)
    with np.load(path / "leaf_params.npz") as data:
        arrays = {k: data[k] for k in data.files if k != "tree_0_bias"}
    np.savez(path / "leaf_params.npz", **arrays)
    with pytest.raises(ValueError, match="tree_0_bias"):
        RepLeafRegressor.load_model(path)


def test_inconsistent_leaf_arrays_rejected(tmp_path, regression_data):
    _, path, _ = _fitted_model(tmp_path, regression_data)
    with np.load(path / "leaf_params.npz") as data:
        arrays = {k: data[k] for k in data.files}
    arrays["tree_0_bias"] = arrays["tree_0_bias"][:-1]  # one leaf short
    np.savez(path / "leaf_params.npz", **arrays)
    with pytest.raises(ValueError, match="tree_0.*leaves"):
        RepLeafRegressor.load_model(path)


def test_missing_encoder_state_rejected(tmp_path, regression_data):
    _, path, _ = _fitted_model(tmp_path, regression_data)
    (path / "encoder_state.npz").unlink()
    with pytest.raises(FileNotFoundError, match="encoder_state.npz"):
        RepLeafRegressor.load_model(path)


# --------------------------------------------------------------------- #
# Format migration
# --------------------------------------------------------------------- #
def test_format_v2_compat(tmp_path, regression_data):
    """v2 directories (no left_categories) load and predict identically."""
    model, path, Xte = _fitted_model(tmp_path, regression_data)
    pred = model.predict(Xte)

    cfg = json.loads((path / "model_config.json").read_text())
    cfg["format_version"] = 2
    (path / "model_config.json").write_text(json.dumps(cfg))
    ensemble = json.loads((path / "tree_ensemble.json").read_text())
    for tree in ensemble["trees"]:
        tree.pop("left_categories", None)
    (path / "tree_ensemble.json").write_text(json.dumps(ensemble))

    loaded = RepLeafRegressor.load_model(path)
    np.testing.assert_allclose(loaded.predict(Xte), pred)


def test_ordinal_models_stay_version_3(tmp_path, regression_data):
    """Models without frequency encoding keep the v3 format for old readers."""
    _, path, _ = _fitted_model(tmp_path, regression_data)
    cfg = json.loads((path / "model_config.json").read_text())
    assert cfg["format_version"] == 3


def test_frequency_encoded_roundtrip_is_version_4(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "num1": rng.normal(size=200),
            "city": rng.choice(["tokyo", "osaka", "kyoto"], size=200),
        }
    )
    y = df["num1"] * 2 + (df["city"] == "tokyo") + rng.normal(0, 0.1, 200)
    train = RepLeafDataset(df, y, frequency_encoded_features=["city"])
    model = RepLeafRegressor(n_estimators=5, num_leaves=8, random_state=42)
    model.fit(train)
    test = RepLeafDataset(df, metadata=train.metadata)
    pred = model.predict(test)

    path = tmp_path / "model"
    model.save_model(path)
    cfg = json.loads((path / "model_config.json").read_text())
    assert cfg["format_version"] == 4

    loaded = RepLeafRegressor.load_model(path)
    assert loaded.metadata_.frequency_maps == train.metadata.frequency_maps
    np.testing.assert_allclose(loaded.predict(df), pred)


# --------------------------------------------------------------------- #
# Model summary
# --------------------------------------------------------------------- #
def test_summary_written_and_readable(tmp_path, regression_data):
    model, path, _ = _fitted_model(tmp_path, regression_data)
    text = (path / "summary.txt").read_text()
    assert text == model.summary() + "\n"
    assert "RepLeafRegressor" in text
    assert "trees: 5 grown" in text
    assert "top features by gain" in text


def test_summary_reports_early_stopping(regression_data):
    Xtr, ytr, Xte, yte = regression_data
    model = RepLeafRegressor(
        n_estimators=50, num_leaves=8, early_stopping_rounds=3, random_state=42
    ).fit(Xtr, ytr, eval_set=[(Xte, yte)])
    if model.best_iteration_ is not None:
        assert "early stopping" in model.summary()


def test_summary_requires_fit():
    with pytest.raises(NotFittedError, match="not fitted"):
        RepLeafRegressor().summary()
