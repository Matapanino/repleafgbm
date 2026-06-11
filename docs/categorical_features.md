# Categorical Feature Policy

RepLeafGBM's research focus is numerical-feature embeddings, but real tabular
data requires a credible categorical story. This document states what v0 does
and where it is going.

## Current behavior (implemented)

- `RepLeafDataset(X, y, categorical_features=[...], numerical_features=[...])`
  accepts explicit feature lists. For pandas DataFrames, object / category /
  bool dtype columns are auto-detected as categorical when not specified.
  For ndarray input, columns are referred to as `"f<index>"`.
- Categoricals are **ordinal-encoded** to float codes (categories sorted as
  strings; the mapping lives in `FeatureMetadata.category_maps`).
- **Native categorical subset splits (Phase 8).** Declared categorical
  features get one histogram bin per category; at each node the categories
  are sorted by their Newton direction `sum_g / (sum_h + l2)` and scanned as
  prefixes (the LightGBM gradient-sorting trick), yielding an optimal binary
  *subset* split — code order no longer matters. The chosen left-subset is
  stored per node (`Tree.left_categories`); routing tests membership.
  Features with more than `max_bins` categories silently fall back to the
  ordered-threshold treatment of the codes.
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

1. ~~Native categorical splits~~ — implemented (Phase 8, see above), with
   LightGBM-default `cat_smooth=10` on the sort ratio. Remaining
   refinements: `min_data_per_group` / `max_cat_threshold`-style guards
   (high-cardinality features such as adult's still overfit slightly — see
   experiments/results/real_data_validation.md Phase 8), and subset-split
   support in router_extraction (`extract_routes` still rejects `==` splits
   even though the native `Tree` can now represent them).
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
