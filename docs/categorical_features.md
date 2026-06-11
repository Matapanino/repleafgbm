# Categorical Feature Policy

RepLeafGBM's research focus is numerical-feature embeddings, but real tabular
data requires a credible categorical story. This document states what v0 does
and where it is going.

## v0 behavior (implemented)

- `RepLeafDataset(X, y, categorical_features=[...], numerical_features=[...])`
  accepts explicit feature lists. For pandas DataFrames, object / category /
  bool dtype columns are auto-detected as categorical when not specified.
  For ndarray input, columns are referred to as `"f<index>"`.
- Categoricals are **ordinal-encoded** to float codes (categories sorted as
  strings; the mapping lives in `FeatureMetadata.category_maps`).
- The router splits on these codes like any numerical feature. This finds
  contiguous code-range partitions — weaker than subset splits, but with
  enough depth trees can isolate individual categories.
- **Missing values and unseen categories → NaN**, and NaN always routes left.
  Prediction on unseen categories therefore degrades gracefully instead of
  raising.
- **Metadata must be shared, and this is enforced.** A `RepLeafDataset` built
  independently from a sample that is missing some category would assign
  different ordinal codes; `fit(eval_set=...)` and `predict` reject datasets
  whose metadata differs from training. Build evaluation/prediction datasets
  with `RepLeafDataset(X, y, metadata=train_data.metadata)`.
- Encoders see **numerical columns only**; categorical information reaches
  the leaf models only through routing. (A leaf reached mostly by one
  category fits that category's local linear model — this is implicit
  category conditioning.)
- `feature_names_in_` / `n_features_in_` are set on fitted models, matching
  sklearn conventions.

## Why ordinal (not one-hot) in v0

One-hot inflates the feature matrix and the histogram count, interacts badly
with `min_samples_leaf`, and buys little once native subset splits exist.
Ordinal is the smallest correct-enough baseline; its known weakness (code
order is arbitrary) is documented and bounded.

## Future directions

1. **Native categorical splits** — subset splits via the LightGBM-style
   gradient-sorting trick (sort categories by `sum_g / sum_h` within the
   node, then scan as if ordinal). This is the main planned upgrade and fits
   the existing histogram machinery.
2. **Target / frequency encoding** — opt-in preprocessing with OOF leakage
   protection (ties into the OOF utilities planned for v0.2).
3. **Category embeddings in the encoder** — learned embedding tables per
   categorical feature feeding the leaf models, once PyTorch encoders land.
   This makes categorical information available to the *smooth* part of the
   model, not only the router.
4. **pandas dtype fidelity** — respect `pandas.Categorical` category order,
   stable dtype inference rules, explicit warnings on high-cardinality
   columns.
5. **CatBoost backend** (backend_strategy.md) for categorical-heavy datasets
   as an external-model diversity source.

## Open questions

- Should ordinal code order be gradient-informed even without full subset
  splits (cheap win, but order then depends on the boosting round)?
- How should unseen categories route once default directions are learned
  (currently: always left)?
- High-cardinality policy: hash bucketing vs. embedding vs. target encoding.
