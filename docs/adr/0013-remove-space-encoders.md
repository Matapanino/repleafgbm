# ADR 0013: Remove the experimental space encoders

**Status:** Accepted (2026-10-24). Removes the NumPy space-encoder bank added in
1.13.0 (`pca`, `quantile`, `spline`, `rff`, `row_stats`, `prototype`). This
settles the encoder question that [ADR 0012](0012-remove-leaf-select.md) left
open.

## Context

1.13.0 added six CPU-only NumPy encoders as an experimental space bank for
MatGBM `leaf_experts` blocks. Their defaults were chosen, not benchmarked, and
`docs/api_freeze.md` and ADR 0003 listed the registry names as Experimental
(not SemVer-gated).

Wave 4 of the MatGBM experiments on Kaggle Playground S6E10 screened them (the
"KE" encoder screen; 5 folds, shipped 1.13.0 wheel, reproduction gate against
wave 3 passed). Each encoder got one arm: a single `leaf_experts` block fitted
with default parameters on the 17 non-categorical raw columns, with
`embedded_linear` leaves on the same fixed 600-tree LightGBM route (replayed
through `RouterExtractionClassifier`) as the constant-leaf control. The screen
only nominated candidates: an encoder became one only by beating constant
leaves by more than the router-family bar (seed-repeat floor of the 600-tree
router, 0.000127645). Mean-fold AUC deltas:

| encoder | mean Δ AUC vs constant (positive folds) | screen outcome | restack add-one Δ (positive folds) |
|---|---:|---|---:|
| `pca` | +0.000071044 (4/5) | within the bar | −0.000006552 (2/5) |
| `quantile` | +0.000086010 (4/5) | within the bar | −0.000007127 (2/5) |
| `spline` | −0.000363743 (0/5) | worse than constant | −0.000004887 (3/5) |
| `rff` | +0.000027781 (3/5) | within the bar | −0.000005412 (2/5) |
| `row_stats` | −0.000032904 (3/5) | within the bar | −0.000003658 (2/5) |
| `prototype` | +0.000117253 (3/5) | within the bar | −0.000007195 (1/5) |

No encoder became a candidate: `spline` was worse than constant leaves on every
fold, and the other five stayed within the bar. Added one at a time to the
current S6E10 stack, none improved it (every add-one Δ ≤ 0). The project owner
decided that encoders which lose in this screen leave the 1.14.0 registry
("1B", 2026-10-23) and that this covers all six ("1A", 2026-10-24).

Sources: the wave-4 decision audit `matgbm_decision/wave4-decision.md` in the
S6E10 harness, transcribed in the MatGBM notes (`ideas.md`, section
「波 4 確定」).

## Decision

- Remove the six encoder modules, their registry names, and their exports
  (`PCAEncoder`, `QuantileEncoder`, `SplineEncoder`, `RandomFourierEncoder`,
  `RowStatsEncoder`, `PrototypeDistanceEncoder`), together with their tests.
- Naming a removed encoder raises a `ValueError` that says it was removed in
  1.14.0 and lists the remaining encoders. `make_encoder` raises it, so it
  covers `encoder=...`, `leaf_experts` blocks, and every recursive config
  (projection and target-correlation bases, column subsets, bundle children).
- Loading a `save_model` directory whose encoder tree names a removed encoder
  (at the top level, as a wrapper's base, or as a bundle child) raises a
  `ValueError` from `load_model` that names the directory and tells the user to
  load it with repleafgbm 1.13.x or retrain with a remaining encoder. As with
  ADR 0012, the check keys on the saved name, not on the format version: such
  models were written as v3 to v9 depending on their other options.
- No `DeprecationWarning` cycle: the names were Experimental, so they are
  removed in the 1.14.0 minor release and listed under `### Removed` in the
  changelog.
- Unchanged: `identity`, `plr`, `periodic`, `cross`, `target_encoding`, the
  `torch_*` encoders, the `random_projection` / `target_correlation`
  reductions, `column_subset`, `expert_bundle`, the model format (still v9),
  and the `"quantile"` regression objective (a loss, unrelated to the removed
  `quantile` encoder).

## Consequences

- The MatGBM recipe does not change: raw routing plus `embedded_linear` over
  fixed `identity` blocks (the S8 target-encoding block and, where available,
  a teacher block).
- The release that ships this needs `repleafgbm` 1.14.0, the same release as
  ADR 0012 (which also needs `repleafgbm-native` 0.4.0). This change touches no
  native code and bumps no version.
- A pickled or joblib-dumped 1.13.x estimator that holds a removed encoder no
  longer unpickles (`ModuleNotFoundError` for its module); only `save_model`
  directories get the clear error above.
- A model whose `encoder` hyperparameter names a removed encoder but never
  built it (`leaf_model="constant"` writes no encoder files) still loads and
  predicts unchanged. Constant leaves never build the encoder, so its name is
  not validated at fit either; that was already true for any encoder name.

## Alternatives considered

- **Remove only `spline`, the one encoder that lost to constant leaves.**
  Rejected by the owner ("1A"): the other five showed no gain beyond the bar
  and no stack gain, and keeping six unbenchmarked encoders keeps their code
  and tests with no measured benefit.
- **Deprecate first (`DeprecationWarning` in 1.14, remove in 2.0).** Rejected:
  ADR 0003's deprecation path applies only to SemVer-covered APIs.
- **Keep loading old models predict-only.** Rejected for the same reason as in
  ADR 0012: it would keep the removed transforms alive only for loading and
  leave estimators that cannot be refit or cloned into a working fit.
