# Categorical Feature Policy

RepLeafGBM's research focus is numerical-feature embeddings, but real tabular
data requires a credible categorical story. This document states what v0 does
and where it is going.

## Current behavior (implemented)

- `RepLeafDataset(X, y, categorical_features=[...], numerical_features=[...])`
  accepts explicit feature lists. For pandas DataFrames, object / category /
  bool dtype columns are auto-detected as categorical when not specified.
  For ndarray input, columns are referred to as `"f<index>"`.
- Categoricals are **ordinal-encoded** to float codes (the mapping lives in
  `FeatureMetadata.category_maps`). Observed values are sorted as strings —
  except **`pandas.Categorical` columns, whose declared category order is
  preserved** (Phase 15), including categories declared but not observed in
  the training sample, so they keep a stable code at prediction time. A
  column declared numerical that cannot be cast to float raises with the
  column named; auto-detected columns with more than 256 categories raise a
  UserWarning (subset splits silently fall back to ordered thresholds above
  `max_bins` categories).
- **Frequency encoding (opt-in, Phase 15)**:
  `RepLeafDataset(..., frequency_encoded_features=[...])` encodes a column
  by its training-set frequency (proportion of rows) instead of an ordinal
  code. The column is *numerical* downstream — threshold splits, encoder
  visibility — making it the recommended treatment for very-high-cardinality
  columns. Unseen categories encode to 0.0 (zero observed frequency);
  missing values stay NaN. The maps live in `FeatureMetadata.frequency_maps`
  and serialize with the model (format_version 4; ordinal-only models keep
  writing v3).
- **Native categorical subset splits (Phase 8).** Declared categorical
  features get one histogram bin per category; at each node the categories
  are sorted by their smoothed Newton direction `sum_g / (sum_h + cat_smooth)`
  and scanned as prefixes from both ends (the LightGBM gradient-sorting
  trick), yielding a binary *subset* split — code order no longer matters.
  The chosen left-subset is stored per node (`Tree.left_categories`);
  routing tests membership. Features with more than `max_bins` categories
  silently fall back to the ordered-threshold treatment of the codes.
- **High-cardinality guards (Phase 8b)**, exposed as estimator parameters
  with LightGBM semantics and defaults: `cat_smooth=10` (sort-ratio
  smoothing), `min_data_per_group=100` (categories with fewer node rows are
  ineligible for the left subset and implicitly go right), and
  `max_cat_threshold=32` (left-subset size cap; the bidirectional scan lets
  the small side sit on either end of the sorted order).
- **Missing values and unseen categories → NaN**, and NaN always routes left
  in native training (subset splits included: missing joins the left subset
  during search). Categories seen at fit time but not in a node's left
  subset go right. Prediction on unseen categories therefore degrades
  gracefully instead of raising.
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

## Why ordinal storage (not one-hot)

One-hot inflates the feature matrix and the histogram count, and interacts
badly with `min_samples_leaf`. Ordinal codes are only a *storage* format
now: subset splits make the arbitrary code order irrelevant for routing.

## Future directions

1. ~~Native categorical splits~~ — implemented (Phase 8) including the
   high-cardinality guards (Phase 8b: `cat_smooth`, `min_data_per_group`,
   `max_cat_threshold`) and `extract_routes` support for LightGBM `==`
   splits (exact prediction reproduction tested). Remaining refinement:
   guard-value tuning per dataset (current values follow LightGBM defaults).
2. ~~Frequency encoding~~ — implemented (Phase 15,
   `frequency_encoded_features`). **Target encoding** remains future work:
   it needs OOF leakage protection (ties into the `oof_predictions` utility
   from v0.2).
3. **Category embeddings in the encoder** — learned embedding tables per
   categorical feature feeding the leaf models, once PyTorch encoders land.
   This makes categorical information available to the *smooth* part of the
   model, not only the router.
4. ~~pandas dtype fidelity~~ — implemented (Phase 15): `pandas.Categorical`
   declared order respected, clear cast errors for mistyped numerical
   columns, high-cardinality warnings.
5. **CatBoost backend** (backend_strategy.md) for categorical-heavy datasets
   as an external-model diversity source.

## Open questions

- Should ordinal code order be gradient-informed even without full subset
  splits (cheap win, but order then depends on the boosting round)?
- How should unseen categories route once default directions are learned
  (currently: always left)?
- High-cardinality policy: hash bucketing vs. embedding vs. target encoding.
