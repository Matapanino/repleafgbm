"""Space encoders removed in 1.14.0 (ADR 0013): actionable errors, survivors intact."""

import importlib.util
import json

import numpy as np
import pytest

from repleafgbm import RepLeafRegressor
from repleafgbm import encoders as encoders_module
from repleafgbm.encoders import _ENCODER_REGISTRY, make_encoder

REMOVED = ("pca", "quantile", "spline", "rff", "row_stats", "prototype")
REMOVED_CLASSES = (
    "PCAEncoder",
    "QuantileEncoder",
    "SplineEncoder",
    "RandomFourierEncoder",
    "RowStatsEncoder",
    "PrototypeDistanceEncoder",
)
REMAINING = {
    "identity",
    "plr",
    "periodic",
    "cross",
    "target_encoding",
    "column_subset",
    "expert_bundle",
    "torch_periodic",
    "torch_plr",
    "torch_periodic_plr",
    "torch_mlp",
}
_REMOVED = r"was removed in repleafgbm 1\.14\.0"
_SMALL = dict(
    n_estimators=2, num_leaves=3, min_samples_leaf=20, split_backend="numpy", random_state=7
)


def _data(seed=5, n=160):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = X[:, 0] - 0.5 * X[:, 1] ** 2 + 0.1 * rng.normal(size=n)
    return X, y


def test_registry_keeps_only_the_remaining_encoders():
    assert set(_ENCODER_REGISTRY) == REMAINING
    for cls_name in REMOVED_CLASSES:
        assert not hasattr(encoders_module, cls_name)
        assert cls_name not in encoders_module.__all__
    for name in REMOVED:
        assert importlib.util.find_spec(f"repleafgbm.encoders.{name}") is None


@pytest.mark.parametrize("name", REMOVED)
def test_removed_encoder_name_raises_actionable_error(name):
    X, y = _data()
    with pytest.raises(ValueError, match=rf"'{name}' {_REMOVED}.*'identity'"):
        make_encoder(name)
    with pytest.raises(ValueError, match=rf"'{name}' {_REMOVED}"):
        RepLeafRegressor(encoder=name, leaf_model="embedded_linear", **_SMALL).fit(X, y)
    with pytest.raises(ValueError, match=rf"'{name}' {_REMOVED}"):
        RepLeafRegressor(
            leaf_model="embedded_linear",
            leaf_experts=[{"name": "space", "encoder": name, "columns": [2, 3]}],
            **_SMALL,
        ).fit(X, y)


def _rename_top(info, name):
    assert info["name"] == "identity"
    info["name"] = name


def _rename_projection_base(info, name):
    assert info["name"] == "random_projection"
    assert info["config"]["base_name"] == "identity"
    info["config"]["base_name"] = name


def _rename_bundle_child(info, name):
    assert info["name"] == "expert_bundle"
    assert info["config"]["children"][1]["name"] == "identity"
    info["config"]["children"][1]["name"] = name


_LAYOUTS = {
    "plain": (dict(encoder="identity", leaf_reduction="none"), _rename_top),
    "projected": (
        dict(encoder="identity", leaf_reduction="random_projection", max_leaf_emb_dim=2),
        _rename_projection_base,
    ),
    "bundle_child": (
        dict(
            leaf_experts=[
                {"name": "a", "encoder": "identity", "columns": [0, 1]},
                {"name": "b", "encoder": "identity", "columns": [2, 3]},
            ],
            leaf_reduction="none",
        ),
        _rename_bundle_child,
    ),
}


@pytest.mark.parametrize("layout", sorted(_LAYOUTS))
@pytest.mark.parametrize("name", REMOVED)
def test_loading_a_model_saved_with_a_removed_encoder_raises_clear_error(
    tmp_path, layout, name
):
    """A 1.13.x model that used a removed encoder must fail with a clear error.

    1.13.x wrote the encoder tree to ``encoder_config.json`` (top-level
    ``name``, a wrapper's ``base_name``, or a bundle child's ``name``). Such a
    directory is emulated by saving an ``identity`` model and rewriting that
    one name; the check keys on the name, before any config or state is used.
    """
    params, rename = _LAYOUTS[layout]
    X, y = _data()
    path = tmp_path / "model"
    RepLeafRegressor(leaf_model="embedded_linear", **params, **_SMALL).fit(X, y).save_model(
        path
    )
    config_path = path / "encoder_config.json"
    info = json.loads(config_path.read_text())
    rename(info, name)
    config_path.write_text(json.dumps(info))

    with pytest.raises(
        ValueError,
        match=rf"The model at .* encoder '{name}', which {_REMOVED}.*repleafgbm 1\.13\.x.*retrain",
    ):
        RepLeafRegressor.load_model(path)


@pytest.mark.parametrize(
    "params",
    [
        dict(encoder="identity"),
        dict(encoder="plr", encoder_params={"n_bins": 4}),
        dict(encoder="periodic", encoder_params={"n_frequencies": 3}),
        dict(encoder="cross"),
        dict(encoder="target_encoding"),
        dict(encoder="identity", leaf_reduction="random_projection", max_leaf_emb_dim=2),
        dict(
            leaf_experts=[
                {"name": "raw", "encoder": "identity", "columns": [0, 1]},
                {"name": "plr", "encoder": "plr", "columns": [2, 3]},
            ]
        ),
    ],
    ids=["identity", "plr", "periodic", "cross", "target_encoding", "projected", "bundle"],
)
def test_remaining_encoders_fit_predict_save_load(tmp_path, params):
    X, y = _data()
    params = {"leaf_reduction": "none", **params}
    model = RepLeafRegressor(leaf_model="embedded_linear", **params, **_SMALL).fit(X, y)
    pred = model.predict(X)
    assert np.isfinite(pred).all()
    assert model.leaf_embedding_dim_ > 0
    model.save_model(tmp_path / "model")
    restored = RepLeafRegressor.load_model(tmp_path / "model")
    np.testing.assert_array_equal(restored.predict(X), pred)
