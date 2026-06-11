"""Directory-based model save/load.

Layout (see docs/serialization.md for the rationale and evolution policy):

    model_dir/
      model_config.json      # format version, model class, hyperparameters
      tree_ensemble.json     # routing trees + init score
      leaf_params.npz        # per-tree leaf biases and weight matrices
      encoder_config.json    # encoder name + constructor config
      encoder_state.npz      # fitted encoder arrays
      feature_metadata.json  # feature names/types/category maps

A directory (not a single JSON) is used because future encoders carry binary
weights (e.g. PyTorch state dicts) that do not belong in JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from repleafgbm.core.booster import Booster, BoosterParams
from repleafgbm.core.leaf_models import LeafValues
from repleafgbm.core.objectives import get_objective
from repleafgbm.core.tree import Tree
from repleafgbm.data.metadata import FeatureMetadata
from repleafgbm.encoders import encoder_from_config
from repleafgbm.encoders.base import BaseEncoder

FORMAT_VERSION = 2
#: Older versions this build can still read. v1 lacks per-node
#: ``missing_left`` (defaulted to True, the convention those trees used).
READABLE_VERSIONS = (1, 2)


def save_model_dir(
    path: str | Path,
    model_class: str,
    config: dict,
    booster: Booster,
    encoder: BaseEncoder | None,
    metadata: FeatureMetadata,
) -> None:
    """Serialize a fitted model into ``path`` (created if missing)."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    model_config = {
        "format_version": FORMAT_VERSION,
        "model_class": model_class,
        "objective": booster.objective.name,
        "config": config,
    }
    _dump_json(path / "model_config.json", model_config)

    ensemble = {
        "init_score": booster.init_score_,
        "learning_rate": booster.params.learning_rate,
        "best_iteration": booster.best_iteration_,
        "best_score": booster.best_score_,
        "trees": [t.to_dict() for t in booster.trees_],
    }
    _dump_json(path / "tree_ensemble.json", ensemble)

    leaf_arrays: dict[str, np.ndarray] = {}
    for i, lv in enumerate(booster.leaf_values_):
        leaf_arrays[f"tree_{i}_bias"] = lv.bias
        leaf_arrays[f"tree_{i}_weights"] = lv.weights
    np.savez(path / "leaf_params.npz", **leaf_arrays)

    if encoder is not None:
        _dump_json(
            path / "encoder_config.json",
            {"name": encoder.name, "config": encoder.get_config()},
        )
        np.savez(path / "encoder_state.npz", **encoder.get_state())

    _dump_json(path / "feature_metadata.json", metadata.to_dict())


def load_model_dir(path: str | Path) -> dict:
    """Deserialize a model directory into its components.

    Returns a dict with keys: model_class, config, objective, booster,
    encoder (or None), metadata. The sklearn-facing classes reassemble
    themselves from these parts.
    """
    path = Path(path)
    if not (path / "model_config.json").exists():
        raise FileNotFoundError(f"{path} does not look like a RepLeafGBM model directory")

    model_config = _load_json(path / "model_config.json")
    version = model_config.get("format_version")
    if version not in READABLE_VERSIONS:
        raise ValueError(
            f"Unsupported model format version {version!r}; this build reads "
            f"versions {READABLE_VERSIONS}"
        )

    config = model_config["config"]
    ensemble = _load_json(path / "tree_ensemble.json")
    defaults = BoosterParams()
    params = BoosterParams(
        n_estimators=config.get("n_estimators", len(ensemble["trees"])),
        # The ensemble's stored rate is authoritative for prediction; model
        # classes without a learning_rate parameter (router_extraction)
        # simply don't carry one in config.
        learning_rate=float(ensemble["learning_rate"]),
        num_leaves=config.get("num_leaves", defaults.num_leaves),
        max_depth=config.get("max_depth", defaults.max_depth),
        min_samples_leaf=config.get("min_samples_leaf", defaults.min_samples_leaf),
        l2_leaf=config.get("l2_leaf", defaults.l2_leaf),
        max_bins=config.get("max_bins", defaults.max_bins),
    )
    booster = Booster(params, get_objective(model_config["objective"]))
    booster.init_score_ = float(ensemble["init_score"])
    booster.best_iteration_ = ensemble.get("best_iteration")
    booster.best_score_ = ensemble.get("best_score")
    booster.trees_ = [Tree.from_dict(d) for d in ensemble["trees"]]

    with np.load(path / "leaf_params.npz") as data:
        booster.leaf_values_ = [
            LeafValues(bias=data[f"tree_{i}_bias"], weights=data[f"tree_{i}_weights"])
            for i in range(len(booster.trees_))
        ]

    encoder = None
    enc_config_path = path / "encoder_config.json"
    if enc_config_path.exists():
        enc_info = _load_json(enc_config_path)
        encoder = encoder_from_config(enc_info["name"], enc_info["config"])
        with np.load(path / "encoder_state.npz") as data:
            encoder.set_state(dict(data))

    metadata = FeatureMetadata.from_dict(_load_json(path / "feature_metadata.json"))

    return {
        "model_class": model_config["model_class"],
        "config": config,
        "objective": model_config["objective"],
        "booster": booster,
        "encoder": encoder,
        "metadata": metadata,
    }


def _dump_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2))


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())
