# ADR 0010: Deterministic row and per-tree column sampling

**Status:** Accepted (2026-09-12)

## Context

RepLeafGBM could not express the row/column sampling portion of common
LightGBM recipes. Sampling must affect raw-feature routing and leaf fitting
without changing the frozen encoder or introducing global random state.

## Decision

- Add `subsample`, `subsample_freq`, and `colsample_bytree` to both estimators
  and `BoosterParams`, with disabled defaults that preserve prior models.
- A positive `subsample_freq` refreshes a sorted, without-replacement row set
  at that interval. It becomes the tree root and the only input to leaf fits;
  all training rows are routed only when updating the score cache.
- Draw a raw-feature subset for every tree. Slice histograms at the Python
  `Splitter` boundary before NumPy/Rust scans and remap the winning index.
  We do not add a Rust `feature_mask`: host-side selection gives identical
  semantics to old native wheels and avoids an ABI/crate release. The tradeoff
  is that excluded-feature histograms are still constructed.
- Use `check_random_state` exclusively. Retain compact fitted diagnostics for
  sample counts/fingerprints and feature indices, not every row index.
- Keep the directory format version unchanged. Hyperparameters are additive
  estimator config fields and pre-feature saved configs load with defaults.

## Consequences

`subsample=1.0`, `subsample_freq=0`, and `colsample_bytree=1.0` stay on the
historical path bit-for-bit. Sampling is shared by scalar and multi-output
trees; multiclass rounds share one row set while each class tree gets its own
feature subset, matching the per-tree column rule.
