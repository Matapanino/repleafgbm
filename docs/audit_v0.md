# Phase 0.5 Audit (v0 stabilization)

> **Historical snapshot.** This document records the state at Phase 0.5
> (plus the Phase 1a update inline) and is intentionally not kept current;
> see docs/roadmap.md for what has shipped since.

Date: 2026-06-11. Scope: full audit of the v0 prototype — correctness, API
consistency, architecture, numerical stability, performance — plus the
benchmark scaffold and minimal OSS tooling.

## Verification performed

- `python -m pytest tests/ -q` — 48 passed (40 pre-existing + 8 added), also
  green with `-W error::RuntimeWarning`.
- All three examples run and produce sane numbers.
- `pip install -e .` + import from outside the repo.
- `ruff check` (rules E, F, W, I, UP) — clean.
- Targeted adversarial checks: id-reuse on the embedding cache, huge-magnitude
  PLR inputs, sklearn `clone`/`set_params`/`score`, mismatched categorical
  metadata between train/eval datasets, perfectly separable logistic training
  (200 rounds, warnings escalated to errors).
- Synthetic benchmarks (quick + full size), see below.

## Implemented features (confirmed working)

Regression + binary classification; constant / embedded_linear / raw_linear
leaves; identity / simplified-PLR encoders with random projection capping;
leaf-wise histogram trees on raw features; frozen-encoder enforcement;
`RepLeafDataset` with pandas/categorical support; per-round eval_set metrics
(incremental, O(T)); directory save/load with exact prediction round-trip;
deterministic training under fixed `random_state`.

## Problems found and FIXED in this phase

1. **PLR encoder produced NaN embeddings for large-magnitude near-constant
   features** (real bug, reproduced). Degenerate quantile edges were separated
   with `+ 1e-12`, which underflows at magnitudes like `1e15`, leaving
   zero-width bins → `0/0 = NaN` in `transform`. Fix: `np.nextafter`-based
   edge separation + overflow-safe division. Test:
   `test_plr_huge_magnitude_constant_feature_stays_finite`.
2. **Silently wrong evaluation/prediction with independently built categorical
   datasets** (real bug, reproduced). A `RepLeafDataset` built from a sample
   missing some category assigns different ordinal codes (e.g. `"b"` → 1.0 in
   train, 0.0 in valid); eval_set and `predict` accepted it silently. Fix:
   metadata-compatibility check on any user-supplied `RepLeafDataset` in
   `fit(eval_set=...)` and `predict`, with an actionable error message
   (share `train_data.metadata`). Numerical-only datasets are unaffected.
   Tests: `test_eval_set_metadata_mismatch_rejected`,
   `test_predict_dataset_metadata_mismatch_rejected`,
   `test_numeric_ndarray_eval_set_needs_no_explicit_metadata`.
3. **Embedding cache keyed by `id(encoder)`** (latent bug, not reproduced but
   deterministic flaw). CPython reuses ids after GC, so a dead encoder's
   cache entry could be served for a new encoder. Fix: strong-reference
   `is`-comparison cache. Test:
   `test_embedding_cache_invalidated_on_encoder_switch`.
4. Minor: lint pass (import order, modern annotations) across the codebase.

## Audited and confirmed OK (no action needed)

- Train/predict routing consistency: bin semantics (`x <= t` left,
  `side="left"` searchsorted) exactly match `Tree.apply`'s `~(x > t)`,
  including NaN-left behavior.
- Logistic objective: numerically stable sigmoid both in training and
  `predict_proba`; Hessian floored at 1e-12; no overflow warnings even on
  separable data at 200 rounds.
- Incremental eval_set scores equal full recomputation (same code path as the
  training F cache; verified by the decreasing-history test).
- Ridge leaf fitting: weighted normal equations with unpenalized intercept;
  singular/non-finite solves fall back to constant; per-leaf fallback
  threshold `max(2*min_samples_leaf, d_z + 2)` enforced (tested).
- sklearn compatibility: `clone`, `get_params`/`set_params`, `score` (both
  mixins), `feature_names_in_`/`n_features_in_`, refit resets classifier
  state. (Full `check_estimator` compliance was *not* attempted — see below.)
- Architecture rules hold: splits read only the binned raw matrix; only
  `LeafValues.predict` consumes Z; `freeze_encoder=False` raises; dataset /
  encoder / booster responsibilities are clean; the split kernel sees only
  ints and float buffers (native-backend-ready).

## Known limitations (documented, deliberate)

- Binary classification only; multiclass raises with a clear message.
- Missing values always route left; no learned default direction.
- Ordinal categorical encoding only (docs/categorical_features.md).
- No early stopping; eval history is recorded but not acted on.
- Histogram thresholds computed once per fit (no per-node re-quantization).
- Simplified PLR (no learned linear layer, no periodic features).
- `min_samples_linear` is hardwired to `2 * min_samples_leaf` in the wrapper.

## Unfixed issues to address later (priority order)

1. **PLR + random projection underperforms** (see benchmark: RMSE 0.519 vs
   0.397 for identity embeddings on a smooth-signal dataset). The projection
   dilutes informative components. Candidates: per-leaf feature selection,
   supervised projection, larger `max_leaf_emb_dim` defaults, or dropping the
   projection in favor of stronger leaf regularization. This is the main
   *research* question for Phase 1.
   **→ Resolved in Phase 1a** (experiments/results/plr_projection_gap.md):
   the projection was the primary cause and excess bins the secondary one;
   defaults changed to `n_bins=4`, `max_leaf_emb_dim=64`, plus a warning when
   projection engages. With the new defaults PLR is competitive with identity
   (and best-in-grid on the piecewise dataset).
2. **Training speed**: pure-Python split loop is ~6–12× slower than LightGBM
   at 10k×20. Biggest costs: per-node per-feature `bincount` without sibling
   histogram subtraction, and per-node fancy-indexed bin-column copies.
   Acceptable for research scale; revisit before any benchmark expansion.
3. **NumPy ndarray predict input for categorical models**: numeric arrays
   containing ordinal codes do not map back through the string category maps
   (categories become NaN silently). DataFrames are the supported path;
   either document loudly or detect-and-raise.
4. **sklearn `check_estimator`** compliance is partial (input validation
   corner cases, 1-feature edge cases). Worth a pass when API stabilizes.
5. Serialized trees are JSON text; fine now, compact format later.

## Performance snapshot (2026-06-11, M-series laptop, single process)

`python3 benchmarks/benchmark_synthetic_regression.py`
(n_train=10000, n_test=5000, n_features=20, n_estimators=100):

| model | fit[s] | pred[s] | rmse |
|---|---|---|---|
| sklearn GradientBoosting | 4.78 | 0.007 | 0.4334 |
| sklearn HistGradientBoosting | 0.75 | 0.009 | 0.4067 |
| sklearn RandomForest | 2.08 | 0.017 | 0.6313 |
| RepLeaf constant | 3.52 | 0.067 | 0.4047 |
| **RepLeaf embedded_linear identity** | 5.92 | 0.131 | **0.3969** |
| RepLeaf embedded_linear plr | 6.81 | 0.150 | 0.5189 |
| lightgbm | 0.55 | 0.009 | 0.4126 |

Takeaways: (a) the core idea shows — embedded-linear leaves beat every
baseline including LightGBM on this signal; (b) the PLR/projection pipeline
is currently a regression and needs study; (c) fit cost is the expected
Python tax, dominated by split search.

**Update (Phase 1a, same date):** after the experiment-driven default change
(PLR `n_bins` 8→4, `max_leaf_emb_dim` 32→64), the same benchmark improved
`RepLeaf embedded_linear plr` from RMSE 0.5189 to **0.4596** (projection to
64 dims still engages here because 20 features × 4 bins = 80; on datasets
with ≤16 numerical features the default now runs unprojected). Analysis in
experiments/results/plr_projection_gap.md.

## Phase 1 priorities (recommendation)

1. **Embedding-quality research loop**: investigate the PLR/projection gap;
   add early stopping + AUC/MAE metrics so experiments are cheap and honest;
   small experiment grid over `max_leaf_emb_dim`, `l2_leaf`, `n_bins`.
2. **LightGBM external_model backend (v0.2 scope)**: leaf-index extraction +
   OOF/stacking utilities — first diversity payoff, low coupling.
3. **Split-search performance pass (still NumPy)**: sibling histogram
   subtraction and pre-transposed bin storage; target ~2–3× on the benchmark
   without touching the backend contract.
