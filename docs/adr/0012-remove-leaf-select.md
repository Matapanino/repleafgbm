# ADR 0012: Remove `leaf_model="select"` and the `leaf_select_stats` kernel

**Status:** Accepted (2026-10-23). Supersedes the `leaf_model="select"` part of
[ADR 0011](0011-leaf-only-features-and-expert-selection.md); the leaf-only
features, expert bundles, and `fit_transform_train` parts of ADR 0011 stay in
force.

## Context

ADR 0011 (1.13.0) added `leaf_model="select"`: per leaf, constant plus every
expert block were candidates, each block got its own Hessian-weighted ridge
fit and weighted leave-one-out (LOO) score, and the lowest-error block was kept
only if it beat constant by `leaf_gate_margin`. `repleafgbm-native` 0.3.1 added
a fused Rust fast path for it (`leaf_select_stats`). ADR 0011 recorded the
known winner's-curse debt and kept `select` an Experimental research arm until
it beat fixed blocks.

The MatGBM experiments on Kaggle Playground S6E10 tested exactly that. Wave 4
(5 folds, shipped 1.13.0 wheel, reproduction gate against wave 3 passed; the
select and fixed-block arms shared one fixed 600-tree LightGBM route replayed
through `RouterExtractionClassifier`) pre-registered the rule: unless
`select` beats the best fixed block by more than the router-family bar on at
least 4/5 folds, `select` and its Rust kernel are removal candidates.
Mean-fold AUC deltas (bar = seed-repeat floor of the 600-tree router,
0.000127645):

| contrast | mean Δ AUC (positive folds) | outcome |
|---|---:|---|
| KS1: select over the S8 and O blocks − fixed S8+O blocks (`embedded_linear`) | −0.000721639 (0/5) | fixed blocks win |
| KS2: same select − select with the S8 rows permuted | +0.000569533 (5/5) | select does use the signal |

Select used the signal (KS2) but kept constant in 89.3% of leaves and lost
to simply fitting both blocks in every leaf (KS1). Across waves 1–4 it never
beat the best fixed block by more than the bar. The pre-registered verdict was
**reject**, and the project owner chose removal ("2A", 2026-10-23).

Sources: the wave-4 decision audit `matgbm_decision/wave4-decision.md` and its
`SUMMARY-wave4.md` in the S6E10 harness, transcribed in the MatGBM notes
(`ideas.md`, section 「波 4 確定」).

## Decision

- Remove `leaf_model="select"`: `SelectLeafModel`, its select-only helpers
  (`_fit_block_loo`, `_constant_loo_error`, `_LOO_CHUNK_ROWS`), the estimator
  wiring of expert blocks into the leaf model, the scalar-target check, and the
  fitted `expert_selection_counts_` / `expert_selection_fraction_` attributes.
- Remove the Rust `leaf_select_stats` pyo3 function, its registration, and the
  helpers only it used (`DenseLu`, `SelectBlockFit`, `SelectLeafResult`,
  `fit_select_leaf`, `LOO_LEVERAGE_CAP`).
- `leaf_model="select"` now raises a `ValueError` that names the replacement:
  `embedded_linear` with fixed `leaf_experts` blocks, or `adaptive` for the
  per-leaf constant-vs-linear gate. `RouterExtraction*` estimators share the
  same factory and error.
- Loading a model whose saved config has `leaf_model="select"` raises a
  `ValueError` telling the user to load it with repleafgbm 1.13.x or retrain.
  The check keys on the saved `leaf_model`, not on the format version: a
  `select` model is v9 only when it used leaf-only columns or an expert bundle,
  and a single plain encoder writes v3.
- No `DeprecationWarning` cycle: `docs/api_freeze.md` and ADR 0003 list
  `leaf_model="select"` as Experimental (not SemVer-gated), so it is removed in
  the 1.14.0 minor release and listed under `### Removed` in the changelog.
- Unchanged: `leaf_only_features`, `leaf_experts` / `ExpertBundleEncoder`
  (including `block_slices`, which the bundle's own `transform` uses),
  `adaptive` (with `leaf_gate_margin` / `leaf_gate`), `embedded_linear`,
  `constant`, the `target_encoding` expert, the space-encoder registry, and the
  model format (still v9; no new version).

## Consequences

- The MatGBM recipe is fixed blocks: raw routing plus `embedded_linear` over a
  single-key OOF target-encoding block (and a teacher block where an external
  teacher exists). `docs/matgbm/README.md` and
  `examples/matgbm_leaf_experts.py` show it.
- The release that ships this needs `repleafgbm` 1.14.0 and
  `repleafgbm-native` 0.4.0 (a public pyo3 function is removed). This change
  bumps neither.
- Mixed installs keep working. repleafgbm 1.13.x guards the fast path with
  `hasattr(_native, "leaf_select_stats")`, so 1.13.x running on the new native
  build falls back to the NumPy select loop, and 1.14.0 never calls the
  removed function.
- The 1.13.0 calibration report
  (`experiments/results/2026-10-22-select-null-calibration.md`) stays as
  history; its generating script and the select benchmark were removed with
  the feature.

## Alternatives considered

- **Keep `select` as an Experimental arm.** Rejected: four waves without a win
  over fixed blocks, a second Rust kernel to keep in parity, and an open
  winner's-curse debt with no evidence that paying it would change the verdict.
- **Deprecate first (`DeprecationWarning` in 1.14, remove in 2.0).** Rejected:
  ADR 0003's deprecation path applies only to SemVer-covered APIs, and
  `select` was declared Experimental.
- **Load old `select` models as predict-only.** Prediction itself would work
  (the leaf arrays use the ordinary `LeafValues` layout), but the loaded
  estimator would carry a `leaf_model` value that can no longer be refit or
  cloned into a working fit. Rejected in favor of a clear error that names the
  version that can read the model.
- **Remove the space encoders in the same change.** Out of scope: the encoder
  decision is separate and still open.
