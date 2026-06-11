# ADR 0002: Router Extraction Mode (design draft)

- Status: **accepted — milestones 1-3 implemented** (2026-06-11, Phase 3);
  milestone 4's experiment lives in experiments/results/router_extraction.md
- Date: 2026-06-11
- Depends on: ADR 0001, docs/backend_strategy.md, the external_model
  utilities shipped in v0.2

## Context

RepLeafGBM's native split finding is correct but young; LightGBM's is
battle-tested. router_extraction reuses an external library's *routing* while
keeping RepLeafGBM's core idea — representation-conditioned leaf models — on
top of the fixed routes:

1. Train a LightGBM ensemble on the raw features.
2. Freeze its tree structures ("routes").
3. Refit every leaf as a RepLeaf-style model (constant / embedded linear
   over `z_theta(x)`).

This isolates the research question "do representation-conditioned leaves
help?" from the quality of our own splitter, and gives a strong-router
variant for benchmarks.

## The boosting-consistency problem (the crux)

Leaf values cannot be refit independently per tree: tree t+1 was grown
against residuals produced by tree t's *original* leaf values. Two sound
options:

- **(A) Sequential replay (chosen).** Keep LightGBM's tree order and
  learning rate. For t = 1..T: route rows with frozen tree t, fit RepLeaf
  leaf models on the current Newton targets (same `-g/h`, weights `h`
  machinery as the native booster), update the prediction cache, continue.
  This is exactly the native boosting loop with `TreeGrower.grow` replaced
  by "look up frozen tree t", so `Booster.fit` can host it with a
  `route_provider` seam.
- (B) Joint refit of all leaves (one large ridge over the concatenated
  one-hot leaf-membership × Z design). Statistically appealing, but O(T ·
  num_leaves · d_z) unknowns, memory-heavy, and abandons the stage-wise
  story. Rejected for the first version; worth revisiting as a *post-hoc
  polish* step later.

Note the asymmetry: replay changes residuals relative to what LightGBM saw
while *growing* trees t+1.., so extracted routes become slightly stale as t
grows. This is accepted and must be reported honestly in experiments; it is
the price of not reimplementing split search.

## Structure mapping (LightGBM tree -> native `Tree`)

`Booster.dump_model()` provides per-node `split_feature`, `threshold`,
`decision_type`, `default_left`, children. Mapping to our flat arrays is
mechanical except:

- **Missing-value direction.** LightGBM learns per-node `default_left`;
  our `Tree` hard-codes NaN-left. The native `Tree` needs an optional
  `missing_left: bool` per node (default True). This is a small,
  backward-representable extension — `format_version` bump with a default
  for old files — and is *also* a known native-side roadmap item, so the
  work is shared.
- **Threshold semantics.** LightGBM uses `x <= threshold` for numerical
  splits — same as ours; categorical splits (`decision_type == "=="`) are
  out of scope for the first version (reject models containing them, with a
  clear message).
- Leaf ids: renumber to dense 0..n_leaves-1 per tree, as the grower does.

After mapping, prediction, serialization, and all leaf-model code paths are
the *existing* native ones — one ensemble representation for all modes, as
required by the backend strategy.

## Sketched API

```python
from repleafgbm.external import extract_routes, RouterExtractionRegressor

model = RouterExtractionRegressor(
    base=LightGBMExternalModel(task="regression", num_leaves=31),
    leaf_model="embedded_linear",
    encoder="identity",
    l2_leaf=1.0,
)
model.fit(train_data)          # trains base, extracts routes, replays leaf fits
model.predict(X)               # native prediction over mapped trees
model.save_model(path)         # standard directory format
```

Internally: `extract_routes(lgb_booster) -> list[Tree]` (pure mapping,
testable in isolation) + a `Booster.fit_with_routes(dataset, encoder,
leaf_model, trees)` replay loop.

## Risks / open questions

- Stale-route effect (above): quantify by comparing replayed-leaf constants
  vs LightGBM's own leaf values — they should match closely when
  `leaf_model="constant"`; that equality is the correctness test.
- LightGBM's `learning_rate` interacts with leaf refit scale; replay must
  use the base model's rate, not RepLeafGBM's.
- Linear-leaf overfitting guards (fallback thresholds) must apply per
  routed leaf exactly as in native training.
- Categorical subset splits: ~~postponed~~ supported since Phase 8b —
  LightGBM `==` nodes map onto native `Tree.left_categories` with exact
  prediction reproduction (including NaN routing via `default_left`).

## Milestones

1. ✅ `Tree.missing_left` field + format_version 2 (v1 read-compatible).
2. ✅ `extract_routes` mapping; equality test reproduces LightGBM raw
   predictions to atol 1e-10 including NaN routing
   (`test_extract_routes_reproduces_lightgbm_exactly`). Empirically,
   LightGBM regression folds `boost_from_average` into leaf values, so no
   separate init score is needed.
3. ✅ Replay loop (`Booster.fit_with_routes`) + `RouterExtractionRegressor`.
   Constant-leaf replay with ~zero ridge matches LightGBM predictions
   (`test_replay_constant_matches_lightgbm`). Phase 4 added replay-stage
   early stopping (the generic `_run_boosting` loop serves native growth and
   replay alike) and `RouterExtractionClassifier` (binary logistic replay).
   Note: LightGBM trims an early-stopped booster to its best iteration, so
   `extract_routes` sees exactly the productive route prefix.
4. ✅ Experiment: native router vs extracted router, same encoders/leaves,
   base + replay early stopping on a shared validation set —
   see experiments/results/router_extraction.md.
