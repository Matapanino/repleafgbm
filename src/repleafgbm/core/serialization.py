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
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from repleafgbm.core.booster import Booster, BoosterParams
from repleafgbm.core.leaf_models import LeafValues
from repleafgbm.core.multiclass import MulticlassBooster
from repleafgbm.core.multioutput import MultiOutputBooster
from repleafgbm.core.objectives import (
    MulticlassSoftmax,
    get_multioutput_objective,
    get_objective,
)
from repleafgbm.core.tree import Tree
from repleafgbm.data.metadata import FeatureMetadata
from repleafgbm.encoders import encoder_from_config
from repleafgbm.encoders.base import BaseEncoder

FORMAT_VERSION = 6
#: Older versions this build can still read. v1 lacks per-node
#: ``missing_left`` (defaulted to True, the convention those trees used);
#: v2 lacks categorical subset splits (``left_categories``), which v1/v2
#: trees never contained; v4 adds optional ``frequency_maps`` to
#: feature_metadata.json (written only when frequency encoding is used, so
#: ordinal-only models stay readable by v3 builds); v5 adds multiclass
#: ensembles (``n_classes`` + vector ``init_score`` in tree_ensemble.json,
#: written only for multiclass models — binary/regression models keep
#: writing v3/v4); v6 adds shared-routing vector leaves for multi-output
#: regression (``n_outputs`` + vector ``init_score``; per-tree ``bias`` is
#: 2-D and ``weights`` 3-D), written only for multi-output models.
READABLE_VERSIONS = (1, 2, 3, 4, 5, 6)


def save_model_dir(
    path: str | Path,
    model_class: str,
    config: dict,
    booster: Booster | MulticlassBooster | MultiOutputBooster,
    encoder: BaseEncoder | None,
    metadata: FeatureMetadata,
) -> None:
    """Serialize a fitted model into ``path`` (created if missing).

    The directory is written atomically: all files are first written into a
    sibling temporary directory, which then replaces ``path``. This both
    avoids leaving a half-written directory on failure and guarantees that
    stale optional files from a previous save (e.g. ``encoder_*`` left behind
    when re-saving an embedded model as ``constant``) never survive into the
    new model — they would otherwise be reloaded and corrupt prediction.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write into a sibling temp dir (same filesystem ⇒ os.replace is atomic).
    out = Path(tempfile.mkdtemp(prefix=".repleaf_save_", dir=path.parent))
    try:
        multiclass = isinstance(booster, MulticlassBooster)
        multioutput = isinstance(booster, MultiOutputBooster)
        vector_init = multiclass or multioutput
        # Each schema addition bumps the written version only for models that
        # use it, so unaffected models stay readable by older builds:
        # multi-output -> 6, multiclass -> 5, frequency maps -> 4, else -> 3.
        if multioutput:
            version = 6
        elif multiclass:
            version = 5
        elif metadata.frequency_maps:
            version = 4
        else:
            version = 3
        model_config = {
            "format_version": version,
            "model_class": model_class,
            "objective": booster.objective.name,
            "config": config,
        }
        _dump_json(out / "model_config.json", model_config)

        ensemble = {
            "init_score": (
                booster.init_score_.tolist() if vector_init else booster.init_score_
            ),
            "learning_rate": booster.params.learning_rate,
            "best_iteration": booster.best_iteration_,
            "best_score": booster.best_score_,
            "trees": [t.to_dict() for t in booster.trees_],
        }
        if multiclass:
            ensemble["n_classes"] = booster.n_classes
        if multioutput:
            ensemble["n_outputs"] = booster.n_outputs
        _dump_json(out / "tree_ensemble.json", ensemble)

        leaf_arrays: dict[str, np.ndarray] = {}
        for i, lv in enumerate(booster.leaf_values_):
            leaf_arrays[f"tree_{i}_bias"] = lv.bias
            leaf_arrays[f"tree_{i}_weights"] = lv.weights
            if lv.z_min is not None:
                leaf_arrays[f"tree_{i}_zmin"] = lv.z_min
                leaf_arrays[f"tree_{i}_zmax"] = lv.z_max
        np.savez(out / "leaf_params.npz", **leaf_arrays)

        if encoder is not None:
            _dump_json(
                out / "encoder_config.json",
                {"name": encoder.name, "config": encoder.get_config()},
            )
            np.savez(out / "encoder_state.npz", **encoder.get_state())

        _dump_json(out / "feature_metadata.json", metadata.to_dict())

        # Swap the freshly written directory in for the target. os.replace
        # cannot overwrite a non-empty directory, so remove the old one first.
        if path.exists():
            shutil.rmtree(path)
        os.replace(out, path)
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)
        raise


def load_model_dir(path: str | Path) -> dict:
    """Deserialize a model directory into its components.

    Returns a dict with keys: model_class, config, objective, booster,
    encoder (or None), metadata. The sklearn-facing classes reassemble
    themselves from these parts. The directory schema is validated up
    front: missing files, missing keys, or leaf-parameter arrays that do
    not match the trees raise with the offending file named, instead of
    failing deep inside prediction.
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
    _require_keys(model_config, ("model_class", "objective", "config"), "model_config.json")
    _require_files(path, ("tree_ensemble.json", "leaf_params.npz", "feature_metadata.json"))

    config = model_config["config"]
    ensemble = _load_json(path / "tree_ensemble.json")
    _require_keys(ensemble, ("init_score", "learning_rate", "trees"), "tree_ensemble.json")
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
    if "n_outputs" in ensemble:  # multi-output ensemble (format v6)
        # The saved objective name selects the loss (squared_error / huber /
        # quantile); loss parameters are not persisted, but every multi-output
        # loss has an identity transform, so a fitted model predicts
        # identically regardless. Pre-huber/quantile models stored
        # "multioutput_squared_error" and still resolve here.
        mo_objective = get_multioutput_objective(
            model_config["objective"], int(ensemble["n_outputs"])
        )
        booster: Booster | MulticlassBooster | MultiOutputBooster = MultiOutputBooster(
            params, mo_objective
        )
        booster.init_score_ = np.asarray(ensemble["init_score"], dtype=np.float64)
    elif "n_classes" in ensemble:  # multiclass ensemble (format v5)
        objective = MulticlassSoftmax(int(ensemble["n_classes"]))
        booster = MulticlassBooster(params, objective)
        booster.init_score_ = np.asarray(ensemble["init_score"], dtype=np.float64)
    else:
        booster = Booster(params, get_objective(model_config["objective"]))
        booster.init_score_ = float(ensemble["init_score"])
    booster.best_iteration_ = ensemble.get("best_iteration")
    booster.best_score_ = ensemble.get("best_score")
    booster.trees_ = [Tree.from_dict(d) for d in ensemble["trees"]]

    with np.load(path / "leaf_params.npz") as data:
        keys = set(data.files)
        n_trees = len(booster.trees_)
        missing = [
            f"tree_{i}_{part}"
            for i in range(n_trees)
            for part in ("bias", "weights")
            if f"tree_{i}_{part}" not in keys
        ]
        if missing:
            raise ValueError(
                f"leaf_params.npz is missing arrays {missing[:4]} for the "
                f"{n_trees} trees in tree_ensemble.json; the model directory "
                "is incomplete or corrupted"
            )
        booster.leaf_values_ = [
            LeafValues(
                bias=data[f"tree_{i}_bias"],
                weights=data[f"tree_{i}_weights"],
                # Models saved before the extrapolation guard lack bounds;
                # they load with clipping disabled (original behavior).
                z_min=data[f"tree_{i}_zmin"] if f"tree_{i}_zmin" in keys else None,
                z_max=data[f"tree_{i}_zmax"] if f"tree_{i}_zmax" in keys else None,
            )
            for i in range(len(booster.trees_))
        ]
    _validate_leaf_values(booster)

    encoder = None
    enc_config_path = path / "encoder_config.json"
    if enc_config_path.exists():
        if not (path / "encoder_state.npz").exists():
            raise FileNotFoundError(
                "encoder_config.json is present but encoder_state.npz is "
                "missing; the model directory is incomplete or corrupted"
            )
        enc_info = _load_json(enc_config_path)
        _require_keys(enc_info, ("name", "config"), "encoder_config.json")
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


def _require_files(path: Path, names: tuple[str, ...]) -> None:
    missing = [n for n in names if not (path / n).exists()]
    if missing:
        raise FileNotFoundError(
            f"Model directory {path} is missing {missing}; "
            "it is incomplete or corrupted"
        )


def _require_keys(obj: dict, keys: tuple[str, ...], file_name: str) -> None:
    missing = [k for k in keys if k not in obj]
    if missing:
        raise ValueError(
            f"{file_name} is missing required keys {missing}; "
            "the model directory is incomplete or corrupted"
        )


def _validate_leaf_values(booster: Booster | MulticlassBooster | MultiOutputBooster) -> None:
    """Cross-check leaf parameter arrays against the routing trees.

    Scalar leaves have a 1-D ``bias`` and 2-D ``weights``; multi-output vector
    leaves (format v6) have a 2-D ``bias`` (n_leaves, n_outputs) and 3-D
    ``weights`` (n_leaves, emb_dim, n_outputs). The extrapolation bounds are
    always (n_leaves, emb_dim) — shared across outputs — i.e. ``weights``
    without its trailing output axis.
    """
    for i, (tree, lv) in enumerate(zip(booster.trees_, booster.leaf_values_)):
        if not (
            (lv.bias.ndim == 1 and lv.weights.ndim == 2)
            or (lv.bias.ndim == 2 and lv.weights.ndim == 3)
        ):
            raise ValueError(
                f"leaf_params.npz tree_{i}: expected scalar (1-D bias, 2-D "
                f"weights) or vector (2-D bias, 3-D weights) leaves, got shapes "
                f"{lv.bias.shape} and {lv.weights.shape}"
            )
        if lv.bias.shape[0] != tree.n_leaves or lv.weights.shape[0] != tree.n_leaves:
            raise ValueError(
                f"leaf_params.npz tree_{i} has {lv.bias.shape[0]} bias / "
                f"{lv.weights.shape[0]} weight rows but the tree has "
                f"{tree.n_leaves} leaves; the model directory is inconsistent"
            )
        if (lv.z_min is None) != (lv.z_max is None):
            raise ValueError(
                f"leaf_params.npz tree_{i} has only one of zmin/zmax; "
                "the extrapolation-guard bounds must come in pairs"
            )
        # Guard bounds match the weight matrix minus its trailing output axis.
        bound_shape = lv.weights.shape[:2]
        if lv.z_min is not None and (
            lv.z_min.shape != bound_shape or lv.z_max.shape != bound_shape
        ):
            raise ValueError(
                f"leaf_params.npz tree_{i}: zmin/zmax shapes "
                f"{lv.z_min.shape}/{lv.z_max.shape} do not match the leaf "
                f"weight shape {bound_shape}"
            )


def _dump_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2))


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())
