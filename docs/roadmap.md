# RepLeafGBM Roadmap

This roadmap is honest about status: **v0 is the only implemented stage.**
Everything below v0 is a plan, not a promise of API stability.

## v0 — implemented ✅

- Native NumPy prototype (histogram split search, leaf-wise growth)
- Regression (squared error) and binary classification (logistic)
- Constant leaf, embedded linear leaf, raw linear leaf
- Identity / simplified-PLR encoders, random projection to `max_leaf_emb_dim`
- Frozen encoder enforced (`freeze_encoder=True` only)
- `RepLeafDataset` (pandas + categorical ordinal encoding, embedding cache)
- Directory-based save/load
- pytest suite, runnable examples, initial docs

## Phase 0.5 — stabilization ✅ (2026-06-11)

- Full audit (docs/audit_v0.md): 3 bugs found and fixed (PLR NaN at large
  magnitudes, silent categorical-metadata mismatch, id-keyed embedding cache)
- sklearn compat verified (clone / set_params / score) with tests
- Synthetic benchmark scaffold (`benchmarks/`) vs sklearn + optional
  LightGBM/XGBoost/CatBoost; first performance snapshot recorded
- Dev tooling: `scripts/check.sh`, ruff lint config, GitHub Actions CI

## Phase 1a — research infrastructure ✅ (2026-06-11)

- Early stopping (`early_stopping_rounds`, `best_iteration_`/`best_score_`,
  predict at best iteration, serialized)
- Metrics: MAE, AUC (tie-handled rank formulation), accuracy; `eval_metric`
  estimator parameter
- `experiments/` scaffold; PLR/projection gap experiment + report
  (experiments/results/plr_projection_gap.md)
- Experiment-driven default changes: PLR `n_bins` 8→4,
  `max_leaf_emb_dim` 32→64, UserWarning when random projection engages

## Phase 1b — encoder research + performance ✅ (2026-06-11)

- Split search rewritten: vectorized histogram kernels + sibling-histogram
  subtraction behind the same `BaseSplitBackend` boundary (~2x faster fits,
  identical split decisions on the benchmark)
- `plr` encoder: appended per-feature linear term (`add_linear=True`
  default) — fixes extrapolation; best-in-test on piecewise data
- `periodic` encoder (PBLD-style frozen sinusoidal features, RealMLP-
  inspired) — shipped as experimental; frozen random frequencies lost
  everywhere in experiments/results/encoder_variants.md, which is the
  concrete motivation for learned (PyTorch) encoders
- Finding: the remaining friedman1 gap is caused by feature *interactions*,
  unreachable for any per-feature frozen encoder by construction

## v0.1 — robustness

- ~~Early stopping on eval sets~~ done in Phase 1a
- ~~More metrics (MAE, AUC, accuracy)~~ done in Phase 1a; user-supplied
  callable metrics still open
- Improved categorical preprocessing (frequency/target encoding options,
  explicit `pandas.Categorical` dtype support, better dtype inference)
- Better serialization (schema validation, format migration tests,
  human-readable model summary)
- ~~Feature importance (split count / gain)~~ done in Phase 5
  (`feature_importances_` gain-normalized, `get_feature_importance` raw;
  split gains stored per tree node, imported from LightGBM dumps for
  extracted routes)
- ~~Investigate the PLR + random-projection accuracy gap~~ resolved in
  Phase 1a (experiments/results/plr_projection_gap.md): projection was the
  main cause, excess bins the second; defaults updated. Remaining follow-ups:
  smarter dimension reduction (per-leaf selection / supervised projection)
  and PLR extrapolation (unclipped tails or appended raw value)

## v0.2 — LightGBM external_model backend ✅ (2026-06-11, Phase 2)

- `repleafgbm.external.LightGBMExternalModel`: independent base model with
  score and leaf-index extraction (lightgbm as optional `[external]` extra,
  never imported by the native path)
- `oof_predictions`: generic K-fold OOF utility (works for RepLeaf models
  too; stratified option for classification)
- `augment_features` / `external_feature_frame`: stacking feature builders
  feeding straight into `RepLeafDataset`
- `examples/stacking_lightgbm.py`: OOF-score stacking recipe; on the demo
  data the stack beats both LightGBM alone and RepLeafGBM alone
- Still open in this line: a `gbm_backend=...` estimator-level switch was
  deliberately *not* added — composition via utilities keeps the native
  estimator focused (see backend_strategy guardrails)

## v0.3 — more external backends

- XGBoost external_model backend (DMatrix path, custom objectives)
- CatBoost external_model backend (categorical-heavy datasets; deep
  integration is known to be harder and may stay shallow)

## v1 — router_extraction mode ✅ core shipped (2026-06-11, Phase 3)

- ✅ `Tree.missing_left` per node + serialization format v2 (v1 readable)
- ✅ `extract_routes`: LightGBM → native trees, exact prediction
  reproduction (atol 1e-10, NaN routing included)
- ✅ `Booster.fit_with_routes` sequential replay +
  `RouterExtractionRegressor` (regression; eval_set not yet supported)
- ✅ Experiment (experiments/results/router_extraction.md): embedded leaves
  improve LightGBM's own routes by 2-12% RMSE — the cleanest isolation of
  the leaf-model contribution to date
- ✅ Phase 4 (2026-06-11): replay-stage early stopping (eval_set monitored,
  route consumption stops at best iteration); `RouterExtractionClassifier`
  (binary logistic replay); LightGBM base early stopping forwarded for
  unfitted bases; fair-comparison experiment v2 with base + replay early
  stopping and a binary section
- Open: categorical subset splits, joint post-hoc leaf polish (ADR 0002
  option B)

## v1.5 — outputs and objectives

- Multiclass classification (softmax; vector-leaf style output)
- Vector leaves (multi-output regression)
- Improved objectives (Huber, quantile, Poisson), label smoothing

## v2 — native high-performance backend

- Rust or C++ core implementing the `BaseSplitBackend` contract
  (histogram building, split scan, row partitioning)
- Parallel tree construction, careful memory layout
- Compiled predictor

## v3 — GPU and scale

- CUDA histogram building and leaf fitting
- GPU training (`device="cuda"`), multi-GPU (`multi_gpu=True`,
  `distributed_strategy="data_parallel"`)
- Distributed histogram building and leaf assignment
- Out-of-core training; large-scale tabular dataset support
- Distributed / data-parallel encoder computation and pretraining

## Encoder evolution track (cross-cutting, post-v0)

Listed here because v0 deliberately freezes the encoder:

- PyTorch encoders as an optional dependency (periodic embeddings with
  *learned* frequencies — the frozen random version demonstrably does not
  work (encoder_variants.md) — full PLR with learned linear layer,
  RealMLP-style blocks, category embeddings, interaction-aware features)
- Encoder pretraining before boosting (supervised or self-supervised)
- **Alternating optimization** (tree fitting ↔ encoder updates)
- **Stage-wise snapshot encoders** (each tree binds to the encoder version it
  was trained against)
- **Low-rank adapters per boosting stage**
- **Residual encoder refinement**
- **Encoder updates with prediction-cache invalidation**

None of these are implemented; all of them interact with the stage-wise
additive assumption analyzed in docs/math.md and must be designed against it.

## OSS quality track

- ~~CI (lint + tests)~~, ~~contribution guide~~ done (Phases 0.5 / 5);
  git history initialized in Phase 5; publication steps tracked in
  docs/publishing_checklist.md
- Issue templates, SECURITY.md, PyPI release (deferred until API stability)
- Benchmark suite (OpenML/tabular benchmarks) under `benchmarks/`
- Versioned docs; API reference generation
