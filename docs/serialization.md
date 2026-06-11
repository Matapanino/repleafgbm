# Model Serialization

## Goals

- Fully reproduce predictions after `save_model` → `load_model`
  (`np.allclose`, tested).
- Survive future encoder types that carry binary weights (PyTorch state
  dicts), which is why the format is a **directory**, not a single JSON file.
- Stay diff-able and inspectable where possible (JSON for structure, npz for
  numeric payloads).

## Directory layout (format_version = 2)

```text
model_dir/
  model_config.json      # format_version, model_class, objective, hyperparameters
  tree_ensemble.json     # init_score, learning_rate, routing trees (flat arrays)
  leaf_params.npz        # tree_{i}_bias (n_leaves,), tree_{i}_weights (n_leaves, d_z)
  encoder_config.json    # encoder registry name + constructor config (absent for constant leaves)
  encoder_state.npz      # fitted encoder arrays (bin edges, means/scales, projection)
  feature_metadata.json  # feature names/types, categorical code maps
```

What each file owns:

- **model_config.json** — everything needed to reconstruct the estimator
  object: class name (`RepLeafRegressor` / `RepLeafClassifier`), objective
  name, all `__init__` hyperparameters, plus subclass extras (e.g. the
  classifier's `classes_`). A `format_version` field is checked on load and
  mismatches are rejected explicitly.
- **tree_ensemble.json** — split features/thresholds, child indices, and
  per-node `missing_left` (v2; True routes NaN left — natively grown trees
  are all-True, extracted LightGBM routes carry learned directions), plus
  the early-stopping state (`best_iteration`, `best_score`; null when early
  stopping was not used). All grown trees are saved; prediction uses the
  first `best_iteration` trees. Thresholds are real values (bin boundaries
  resolved at save time), so prediction does not need the training-time
  binning. The stored `learning_rate` here is authoritative for prediction
  (model classes such as RouterExtractionRegressor have no learning-rate
  hyperparameter of their own).
- **leaf_params.npz** — leaf biases and weight matrices. Constant leaves are
  zero-width weight rows, so one schema covers all leaf models.
- **encoder_config.json / encoder_state.npz** — split between constructor
  config (JSON) and fitted arrays (npz). A projection-wrapped encoder nests
  its base encoder's config and prefixes its state keys with `base__`.
- **feature_metadata.json** — the train-time `FeatureMetadata`; applied to
  prediction inputs so categorical codes and column order always match.

## Versioning policy

- `format_version` increments on any breaking layout change.
- Loaders reject unknown versions rather than guessing.
- Supported read versions: **1 and 2**. v1 directories lack `missing_left`;
  they load with the all-True default, which is exactly the convention those
  trees were trained under (covered by `test_format_v1_compat`).
- Once the library is public: migration code for at least one previous
  version, and a round-trip test per supported version.

## Future extensions (not implemented)

- `encoder_state.pt` for PyTorch encoders (the encoder entry in
  `model_config.json` will record which state file format is used).
- Compiled/compact prediction format (flat buffers) for the native backend.
- Optional single-file archive (`.repleaf` zip of the directory) for
  distribution convenience.
- Storing training history (`evals_result_`) alongside the model.
