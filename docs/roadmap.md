# RepLeafGBM Roadmap

This roadmap is honest about status. As of the **1.0.0** release the capability
tiers **v0 through v1.5 are implemented**, plus the v2 native-backend core; the
public API is stable under [SemVer](docs/adr/0003-api-stability.md). Remaining
v2 polish and v3 (GPU/scale) are plans, not promises.

> The `v0/v1/v1.5/v2/v3` labels below are **capability tiers**, not package
> versions. Package versioning is independent and starts its stable line at
> 1.0.0 (see ADR 0003).

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

## Phase 6 — real-data validation ✅ (2026-06-11)

- `benchmarks/benchmark_real_data.py`: california / house_sales / diamonds /
  adult vs LightGBM (encoded + native-cat) and HistGradientBoosting, all
  early-stopped; report in experiments/results/real_data_validation.md
- Findings: native router competitive with LightGBM (≤2.5% on shared
  features); embedded leaves add nothing as shipped and blow up on diamonds
  via **leaf-linear extrapolation** (predictions 4x outside the target
  range on z-outlier rows; excluding the worst 1% of rows they would be
  best-in-table at 0.0844 vs LightGBM-native-cat 0.0948)
- Measured native-categorical headroom: +0.3% to +2.5%
- Decided Phase 7 priority: (1) leaf-linear extrapolation guards
  (per-leaf z clipping), (2) native categorical splits, (3) capacity knobs

## Phase 7 — leaf-linear extrapolation guard ✅ (2026-06-11)

- Per-leaf embedding clip bounds (`z_min`/`z_max` stored at fit, Z clipped
  at predict): outside its training support a linear leaf extrapolates as a
  constant. Training trajectory unchanged; serialization is additive
  (pre-guard models load with clipping off)
- Real-data rerun: diamonds failure resolved (0.276 → 0.0953); with the
  guard, embedded leaves beat constant leaves on 3/3 regression datasets
  and beat LightGBM-on-shared-features on all three (router-extracted
  variant ties LightGBM-native-cat on house_sales)
- `leaf_model="embedded_linear"` default re-confirmed for regression with
  real-data evidence; binary remains the weak quadrant (adult: constant
  still ahead by ~0.5%) — follow-up open
- Next priorities unchanged: native categorical splits (measured 0.3-2.5%
  headroom), then capacity knobs

## Phase 8 — native categorical subset splits ✅ (2026-06-11)

- One bin per category for declared categoricals; gradient-sorted prefix
  scan (LightGBM trick, cat_smooth=10) in the backend; per-node
  `Tree.left_categories`; serialization format v3 (v1/v2 readable);
  ordered-threshold fallback above `max_bins` categories
- Real data: diamonds gap to LightGBM-native-cat **closed and reversed**
  (embedded 0.0928, constant 0.0944 vs lgb native-cat 0.0948);
  house_sales neutral; adult −0.5-0.7% from high-cardinality overfit —
  `min_data_per_group` / `max_cat_threshold` guards are the named follow-up
- router_extraction still rejects external `==` splits (mapping them onto
  `left_categories` is now possible — open item)

## Phase 8b — categorical guards + `==` route extraction ✅ (2026-06-11)

- `min_data_per_group=100` and `max_cat_threshold=32` (bidirectional prefix
  scan) added to the subset kernel; together with `cat_smooth=10` all three
  are public estimator parameters with LightGBM defaults
- Real data: high-cardinality regressions resolved (adult constant back to
  ordinal parity; house_sales now *better* than ordinal) while the diamonds
  win over LightGBM-native-cat survives (0.0940 vs 0.0948); remaining
  native-cat gap ≤0.7% (adult only)
- `extract_routes` maps LightGBM `==` splits onto `Tree.left_categories`
  with exact prediction reproduction (NaN routing included) —
  router_extraction now accepts categorical-native bases
- Headroom recorded: per-dataset tuning of `min_data_per_group` (defaults
  favor robustness over the last diamonds decimals)

## Phase 12 — binary embedded-leaf gain study ✅ (2026-06-12, null result)

- Hypothesis "logistic h = p(1-p) starves leaf-linear fits" tested with
  paired-target diagnostics + a remedy grid (l2 sweep, Hessian floor,
  damped h^alpha leaf weighting): ESS decay is real (~36%) but no remedy
  beats constant leaves; defaults unchanged
  (experiments/results/binary_leaf_gain.md)
- Key diagnostic: binary leaves keep fitting sizable ||w|| late in boosting
  while regression ||w|| collapses with the residuals — within-leaf logit
  structure is mostly exhausted by routing; remaining fits are noise that
  the guard/ridge merely contain
- Documented guidance: leaf_model="constant" is an equal-accuracy, cheaper
  choice for binary tasks; future binary gains route through learned
  encoders / calibrated leaf outputs, not reweighting

## Phase 13 — learned encoders (PyTorch optional) ✅ (2026-06-12)

- `torch_periodic` / `torch_plr`: frequencies/projections trained by
  supervised pretraining on the initial Newton residual, then frozen —
  torch needed only at fit time (transform/serialization stay NumPy; saved
  models predict without torch); native path never imports torch
- Result (experiments/results/encoder_variants.md): learning beats the
  frozen counterpart in 9/9 cells; torch_periodic is best-overall on 2/3
  datasets incl. periodic_mix (0.3405 vs identity 0.3933) — the Phase 1b
  "frozen frequencies don't work" finding is resolved as predicted
- Defaults unchanged (torch optional). Phase 13's "prefer torch_periodic
  when installed" guidance was **withdrawn in Phase 14**

## Phase 14 — learned encoders on real data ✅ (2026-06-12, negative result)

- 4/4 real datasets: learned encoders never beat identity; torch_periodic
  is the worst embedded variant everywhere with a uniform overfit signature
  (lowest train, highest test) — the synthetic oscillatory structure that
  learned frequencies exploit is absent in these targets
  (real_data_validation.md Phase 14)
- Binary route from Phase 12 also closed: pretrained representations find
  nothing on adult beyond routing + constant leaves
- Final guidance: identity first on real data; torch encoders are
  specialist tools for known smooth/oscillatory structure
- ~~Open: pretraining regularization (validation early stopping, weight
  decay) as the one plausible fix before blaming the architecture~~ tested
  and closed in Phase 14b; interaction-aware features remain open

## Phase 14b — pretraining regularization ✅ (2026-06-12, follow-up closed)

- `weight_decay` (AdamW), `val_fraction` + `patience` (validation early
  stopping with best-epoch restore) added to `torch_periodic` / `torch_plr`
  pretraining; conservative defaults on (1e-3 / 0.15 / 5);
  `pretrain_epochs_used_` diagnostic; knobs serialized in encoder config
- Result (experiments/results/torch_pretrain_regularization.md): early
  stopping engages (14-21 of 30 epochs) but real-data accuracy is unchanged
  4/4 — identity stays best by the Phase 14 margins; the periodic_mix
  synthetic win survives regularization intact (0.3430 vs identity 0.3933)
- Verdict: the Phase 14 overfit is architectural, not a missing
  regularizer — per-feature pretrained representations find nothing the
  router doesn't on these targets. Defaults keep regularization on (equal
  accuracy, 30-50% fewer pretraining epochs, principled guard)
- The remaining encoder direction on real data is interaction-aware
  features (cross-feature blocks)

## Phase 16 — interaction-aware encoders ✅ (2026-06-12, negative on real data)

- Two new encoders close the interaction hypothesis Phase 14 left open:
  `cross` (deterministic: standardized features + 16 residual-correlated
  pairwise products, NumPy-only) and `torch_mlp` (learned: 64-hidden /
  16-output MLP + linear passthrough, Phase 14b-regularized pretraining,
  frozen to NumPy)
- Result (experiments/results/encoder_interactions.md): on the
  `interaction_mix` home turf both decisively beat `identity` (cross
  0.4598, torch_mlp 0.5367 vs 0.6147) — leaves *can* carry cross-feature
  structure. On real data identity stays best 4/4; torch_mlp shows the
  Phase 14 overfit signature even with early stopping engaging, and
  cross's full-train pair selection is unstable (diamonds outlier)
- Verdict: per-feature (13/14/14b) and cross-feature (16) learned
  representations are both already served by the router on typical real
  tabular targets. `identity` remains the default; learned encoders are
  opt-in specialists for known structure (oscillations → torch_periodic,
  dominant products → cross/torch_mlp)
- Possible refinement if ever needed: holdout-based pair selection for
  `cross`; not pursued absent a motivating dataset

## Phase 15 — v0.1 robustness ✅ (2026-06-12)

- Categorical preprocessing (docs/categorical_features.md):
  `pandas.Categorical` declared category order respected (unobserved
  declared categories keep stable codes); opt-in **frequency encoding**
  (`frequency_encoded_features` — column becomes numerical: threshold
  splits + encoder visibility, unseen → 0.0); clear cast errors for
  mistyped numerical columns; UserWarning above 256 categories
- Serialization (docs/serialization.md): schema validation on load
  (required files/keys, leaf params cross-checked against tree leaf
  counts, paired extrapolation bounds); `model.summary()` +
  `summary.txt` written on save; format_version 4 for frequency maps
  (ordinal-only models keep writing v3); v2 migration test added
- User-supplied eval metrics: `eval_metric` accepts a name, a
  `BaseMetric` instance, or a plain callable; `repleafgbm.make_metric`
  for named/greater-is-better wrapping; custom metrics serialize by name
- Target encoding deliberately deferred (needs OOF leakage protection,
  ties to `oof_predictions`)

## v0.1 — robustness ✅ (closed by Phase 15)

- ~~Early stopping on eval sets~~ done in Phase 1a
- ~~More metrics (MAE, AUC, accuracy)~~ done in Phase 1a;
  ~~user-supplied callable metrics~~ done in Phase 15 (`make_metric`)
- ~~Improved categorical preprocessing~~ done in Phase 15 (frequency
  encoding, `pandas.Categorical` dtype support, clearer dtype errors);
  target encoding deferred to the OOF utilities line
- ~~Better serialization (schema validation, format migration tests,
  human-readable model summary)~~ done in Phase 15
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

## v0.3 — more external backends ✅ (closed by Phases 19 + 21, 2026-06-12)

- ~~XGBoost external_model backend~~ done in Phase 19:
  `XGBoostExternalModel` with the same duck-typed contract as the LightGBM
  one (fit / predict_score / predict_leaf_indices, native early stopping
  with predictions pinned to the best iteration) — works unchanged with
  `oof_predictions` / `augment_features`; custom XGBoost objectives pass
  through `xgb_params`. Route extraction stays LightGBM-only
- ~~CatBoost external_model backend~~ done in Phase 21 (deliberately
  shallow, as planned): `CatBoostExternalModel` with the same contract
  (`use_best_model=False` + explicit `ntree_end` pinning under early
  stopping, `calc_leaf_indexes` for leaf features); CatBoost-native
  categorical handling reachable via `cb_params(cat_features=...)`, but
  the recommended categorical path remains RepLeafDataset

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

## Phase 17 — multiclass classification ✅ (2026-06-12)

- `MulticlassSoftmax` objective (diagonal-Hessian softmax, log-prior init)
  + `MulticlassBooster` (core/multiclass.py): one tree per class per round,
  mirroring the scalar boosting loop lifted to (n_rows, K) score matrices;
  every class reuses the same frozen embedding matrix and the unchanged
  Newton-target leaf machinery (constant and embedded_linear both work)
- `RepLeafClassifier` switches automatically at 3+ classes (binary path
  unchanged, including the p >= 0.5 tie rule); `n_estimators` counts rounds;
  `multi_logloss` metric (default monitor), `accuracy` extended to
  probability matrices; early stopping counts rounds
- Serialization format v5 (`n_classes` + vector `init_score`,
  round-major trees) — written only by multiclass models, so
  binary/regression models keep v3/v4 readable by older builds
- Learned-encoder supervised pretraining was scalar-target here (multiclass
  encoders fit unsupervised); since generalized to an `(n, K)` matrix target
  so multiclass encoders pretrain supervised (see the trainable-embeddings
  open-items note and docs/math.md)
- Tests (tests/test_multiclass.py), example
  (examples/multiclass_classification_basic.py), math.md softmax section

## Phase 18 — regression objectives ✅ (2026-06-12)

- `objective` parameter on the regressor: "huber" (clipped-residual
  gradients, h=1 LightGBM convention, median init), "quantile" (pinball,
  alpha-quantile init), "poisson" (log-mean raw score, exp transform,
  non-negative target validation) — names or parameterized instances
  (`Huber(delta=...)`, `Quantile(alpha=...)`), exported from `repleafgbm`
- All three reuse the scalar Newton-target leaf machinery unchanged;
  `RepLeafRegressor.predict` now applies the objective's output transform
  (identity except poisson's exp)
- Objective instances serialize by registry name (metric precedent):
  predictions reload exactly; refitting a reloaded model needs the instance
  again for non-default delta/alpha
- The classifier rejects the parameter (its objective follows from the
  target); router_extraction subclasses keep their reduced __init__ (and
  RouterExtractionClassifier now rejects 3+ classes explicitly, since its
  replay loop is scalar)
- Tests: gradient/Hessian unit checks, huber-beats-L2-under-outliers,
  quantile ordering/coverage, poisson positivity + baseline,
  save/load round-trips (tests/test_regression_objectives.py)

## Phase 22 — vector leaves (multi-output regression) ✅ (2026-06-15)

- `MultiOutputBooster` (core/multioutput.py): **one shared routing tree per
  round** whose leaves emit an `(n_outputs,)` vector — distinct from
  multiclass (one tree *per class* per round). Routing splits use the raw
  features shared across outputs; the split gain is the per-output Newton gain
  summed over outputs (`backends/numpy_backend.find_best_split_multioutput`,
  dispatched from `Splitter` when grad/hess are 2-D). The encoder stays frozen
  and all outputs reuse the same embedding matrix Z.
- Vector leaves reuse the leaf machinery lifted to a trailing output axis:
  `LeafValues` carries (n_leaves, K) bias and (n_leaves, emb, K) weights;
  constant and embedded_linear both work. Because multi-output is
  squared-error (Hessian = 1), the embedded-linear leaf's centered Gram is
  shared across outputs (one factorization, K right-hand sides). Per-leaf
  extrapolation guards (z_min/z_max) are shared across outputs.
- API: `RepLeafRegressor` auto-detects a 2-D `y` and returns
  (n_rows, n_outputs); `predict`/`score`/eval_set/early stopping all work.
  Serialization format v6 (`n_outputs` + vector init_score, 2-D bias / 3-D
  weights). Tests (tests/test_multioutput.py), example
  (examples/multioutput_regression_basic.py), math.md vector-leaf section
- Scope (documented limitations): multi-output was **squared-error only** at
  Phase 22 (Huber/quantile multi-output now shipped, see Phase 31 below),
  categorical features route via **ordered thresholds** on the ordinal code
  rather than gradient-sorted subset splits (single-output keeps subset
  splits), and the Rust backend builds the per-output histograms but the
  multi-output split scan stays NumPy (Rust kernel is a possible v2 follow-up)

## Phase 31 — multi-output robust losses + opt-in GPU encoder pretraining ✅ (2026-06-17)

- **Multi-output Huber/quantile** (`core/objectives.py`:
  `MultiOutputHuber`, `MultiOutputQuantile`): the constant-Hessian (`h = 1`)
  family extends cleanly to vector leaves — only the gradient (clipped /
  pinball residual) and the init score (per-output median / alpha-quantile)
  differ from squared error, so `fit_vector_leaves`' shared-Gram solve and the
  multi-output split scan are reused **unchanged** (docs/math.md).
  `RepLeafRegressor` now maps `objective="huber"`/`"quantile"` (or instances
  like `Quantile(alpha=0.9)`) onto these for 2-D `y`; poisson stays rejected
  (its Hessian is not constant across outputs). Serialization (format v6,
  unchanged) reconstructs the loss by name on load; every multi-output loss has
  an identity transform, so a fitted model predicts identically regardless.
  Tests: fit/predict, determinism, save/load, NumPy↔Rust parity, and an
  outlier-robustness check (tests/test_multioutput.py, test_rust_backend.py).
  Validated (5 seeds) in
  experiments/results/2026-06-17-multioutput-real-and-robust.md: under 8%
  heavy-tailed contamination of the training targets, huber (real RMSE 2.22,
  r² 0.94) and quantile (3.87) decisively beat squared error (12.49, r² −0.77),
  with the same picture on a synthetic clean-signal target. The study also
  closes the v1.4.0 loose end that the `(n, K)` vector target was synthetic-only:
  on real multi-output (energy-efficiency) RepLeaf beats a per-output LightGBM
  reference (RMSE 1.32 vs 1.71) and the before/after vector-pretraining gap is
  seed noise — identity stays the best default, as on the single-output real data.
- **Opt-in GPU pretraining for learned encoders** (`encoders/torch_encoders.py`):
  a `device` knob (`"cpu"` default, `"cuda"`, `"auto"`) selects where the torch
  pretraining matmuls run; all random draws stay on a CPU generator so the
  stream is device-independent and `device="cpu"` reproduces prior pretraining
  byte-for-byte. `transform`/serialization remain NumPy-frozen (no torch at
  predict). GPU is **allclose, not bitwise** (GPU reductions reorder), so CPU
  stays the deterministic default and GPU correctness is validated only on the
  Colab T4 loop (docs/cuda.md). Closes the "only `split_backend="cuda"` uses the
  GPU" gap; payoff is scale-dependent (large n / wide periodic embeddings).

## Phase 23 — label smoothing ✅ (2026-06-15)

- `label_smoothing` estimator parameter (classification only): binary targets
  soften to `y*(1-eps) + eps/2`, multiclass one-hot to `(1-eps)*onehot + eps/K`
  in both the init score and the gradients of `BinaryLogistic` /
  `MulticlassSoftmax`. `eps = 0` reproduces the unsmoothed objective exactly
- The objective serializes by registry name only, so eps is restored from the
  estimator config on reload (same convention as Huber's delta); predictions
  reload exactly. Tests (tests/test_label_smoothing.py)

## Phase 28 — sample weights + class weights + balanced accuracy ✅ (2026-06-16)

- `fit(..., sample_weight=)` (all estimators) and `class_weight` (classifier
  only: `{label: weight}` dict or `"balanced"`). Class weights are folded
  multiplicatively into the per-row sample weight before boosting.
- Implemented by scaling each row's `g, h` (and the optimal init score) via
  `core.booster.weight_grad_hess` — the split backends, leaf kernels, and
  NumPy/Rust parity are untouched (weighting happens upstream of the
  histogram). `min_samples_leaf` keeps counting raw rows; uniform weights
  cancel exactly at `l2_leaf=0` (the principled invariant, not row
  duplication — see docs/math.md "Sample weights").
- New `balanced_accuracy` eval metric (mean per-class recall, greater-better)
  for monitoring/early stopping on imbalanced targets; matches
  `sklearn.metrics.balanced_accuracy_score`.
- `class_weight` serializes with the estimator config (`"balanced"`/None
  round-trip exactly). Tests (tests/test_sample_weight.py); experiment
  (experiments/imbalanced_multiclass_class_weight.py).
- Follow-up: capability layer (`_supports_sample_weight`) — estimators that
  cannot reweight rows (`RouterExtraction*`, frozen-route replay) drop weights
  with a `UserWarning` instead of raising; documented fallback is plain loss +
  built-in early-stopping metric + external balanced accuracy. Usage guide
  (docs/weighting_and_metrics.md), ADR 0004, `get_metric` export. Tests
  (tests/test_weight_capability.py).

## Phase 25 — OpenML benchmark suite ✅ (2026-06-15)

- `benchmarks/openml_suite.py`: a reproducible breadth-first leaderboard over 9
  curated OpenML datasets (4 regression, 3 binary, 2 multiclass) comparing
  RepLeafGBM (constant, embedded_linear, the adaptive LOO-gated leaf, a
  fixed-PLR arm, and the learned `torch_*` encoders) against LightGBM, XGBoost,
  CatBoost, and sklearn HistGradientBoosting. Every model trains on the **same**
  ordinal-encoded matrix, fixed seed, 60/20/20 split, early stopping. Report:
  experiments/results/openml_benchmark.md (regenerate with one command);
  cached via `~/scikit_learn_data` for offline reruns. Part of the v1.0 OSS
  quality track.
- Findings (mean rank over **11 models**, 3 seeds, lower better; refreshed
  2026-06-25 with the expanded arm set): the tuned external GBMs lead —
  regression LightGBM 2.50 < CatBoost 3.00 < XGBoost 4.25; classification
  CatBoost 3.40 first. Among the RepLeaf arms the **adaptive** leaf (per-leaf
  weighted-LOO gate) is strongest — 2nd overall on classification (4.20, behind
  only CatBoost) and tied-best with constant on regression (4.50); plain
  `embedded_linear` is competitive on classification (4.80). The
  higher-dimensional encoders (`plr`, learned `torch_*`) rank low on real data.
- Confirms at breadth what Phases 14/16 found in depth: **higher-capacity
  representation leaves add no real-data accuracy over a constant or
  adaptively-gated leaf** here — their advantage is specific to smooth/periodic
  synthetic structure (there `embedded_linear`+`identity` edges constant and
  LightGBM), not typical tabular targets. The **adaptive** gate is the most
  robust RepLeaf configuration on unknown real data; a plain constant leaf
  remains a safe, honest baseline.

## Phase 26 — documentation completion (v1.0) ✅ (2026-06-15)

- API reference generation: `scripts/build_docs.sh` renders the public API
  from docstrings with pdoc (`[docs]` extra); a dedicated CI job builds it on
  every push so the reference can never silently break.
- `CHANGELOG.md` (Keep a Changelog) summarizing v0→1.0.
- README: the "APIs will change without notice" warning is replaced by the
  SemVer stability policy (ADR 0003); PyPI install instructions and a version
  badge added; honest real-data benchmark highlight (Phase 25).
- Roadmap intro corrected (v0→v1.5 + v2 core shipped) and the
  capability-tier-vs-package-version distinction stated explicitly.

## v1.5 — outputs and objectives ✅ (closed by Phases 22 + 23)

- ~~Multiclass classification (softmax)~~ done in Phase 17 (one tree per
  class per round)
- ~~Vector leaves (multi-output regression)~~ done in Phase 22 (shared-routing
  vector leaves; squared error), extended to Huber/quantile in Phase 31
- ~~Improved objectives (Huber, quantile, Poisson)~~ done in Phase 18;
  ~~label smoothing~~ done in Phase 23

## Phase 33 — tree growth policies (`grow_policy`) ✅ (2026-06-23)

- New `grow_policy ∈ {"leafwise" (default), "depthwise", "symmetric"}`
  (ADR 0006; docs/research/2026-06-22-tree-growth-policies.md):
  - **leafwise** — unchanged best-gain-first growth (byte-identical default).
  - **depthwise** — XGBoost-style level-order growth to `max_depth`; reuses the
    existing split scan, so it covers regression / binary / multiclass /
    multi-output.
  - **symmetric** — CatBoost-style oblivious trees: one shared `(feature,
    threshold)` per level chosen by **summed per-node gain** (host-side scan,
    automatic NumPy⇄Rust parity), complete `2**depth` tree, strong implicit
    regularization.
- Expanded into the existing flat `Tree` (no `format_version` bump); `depthwise`
  and `symmetric` require `max_depth >= 1`; thesis preserved (raw-feature routing
  only, representation-conditioned leaves untouched).
- **Implemented limitations (honest):** `symmetric` is numeric/ordered + scalar
  only in v0 — categorical features route as ordered thresholds (no subset
  splits) and multi-output `symmetric` raises `NotImplementedError`. Symmetric
  inference uses the general `Tree.apply` (no compact oblivious fast path yet).
  CUDA + symmetric is plumbed but unvalidated on GPU.
- **Usefulness study (synthetic, 5 seeds):** leafwise vs depthwise vs symmetric
  across 6 synthetic datasets × {constant, embedded_linear} × {reg, binary, mc}
  (`experiments/grow_policy_comparison.py` →
  `experiments/results/grow_policy_comparison.md`). Verdict
  (`experiments/results/2026-06-23-grow-policy-verdict.md`): **keep `leafwise`
  default**. Symmetric's broad synthetic wins are an oblivious-friendly-design
  artifact (planted low-order axis-aligned structure, no real data, leaf-wise
  capacity plausibly handicapped); it wins decisively only on clean/large
  piecewise + multiclass and loses *harder* than it wins on smooth/interaction
  and noisy-embedded targets. Provisional guidance: symmetric for strong
  low-order shared structure / multiclass / oblivious regularization; depthwise
  as a balanced, never-worst depth-bounded middle; leafwise otherwise.
- **Open follow-ups:** a **real-data** policy comparison (the gate before any
  default change — `benchmarks/openml_suite.py` + `benchmark_real_data.py`, a
  capacity-match sensitivity sweep, ≥5 seeds, a ≥1σ-separation decision rule);
  categorical-subset and multi-output symmetric; a compact oblivious storage +
  bitwise-indexed predictor.

## v2 — native high-performance backend (Phase 10: core shipped ✅)

- ✅ Rust kernels for the `BaseSplitBackend` contract (histogram building +
  split scan incl. categorical subset logic) as an optional pyo3/maturin
  extension under `native/`; `split_backend="auto"|"numpy"|"rust"` estimator
  parameter; parity tested (bitwise histograms, allclose predictions);
  dedicated CI job. Measured: constant 5.8x (LightGBM-parity), embedded ~2x,
  constant @100k rows 4.7x
- ✅ Phase 11: batched normal equations (one `np.linalg.solve` per tree)
  + fused Rust `leaf_linear_stats` pass (embeddings ≤32 dims) + clip-free
  training updates — embedded_linear ~2.6x over NumPy (2.9x vs the
  pre-batching baseline), wide-PLR ~1.5x; parity vs the centered reference
  implementation tested at rtol 1e-9
- ✅ Subsequent perf passes: rayon feature-major histograms, feature-parallel
  binning, multiclass leaf pooling + fused scalar linear prediction, and native
  `partition_rows` (native 0.2.0 / PR #30). The row partition kernel is
  index-identical to NumPy and reduced medium/large multiclass fit by 9-12% in
  the PR #30 benchmark.
- Open: native Gram/vector-leaf fast paths, compiled `Tree.apply` / forest
  predictor, and broader multi-output backend scans.

## v3 — GPU and scale

- CUDA histogram + adaptive numeric split scan **(Phase A + B1 + B2 shipped)**:
  experimental `split_backend="cuda"` builds per-node histograms on the GPU via
  CuPy, caches the binned matrix on-device (B1), and keeps the histogram resident
  while running the numeric gain sweep + argmax on the GPU for large histograms
  (B2, adaptive) — only the winning split's scalars return to the host; small
  histograms scan on the host (no narrow regression). Categorical subset splits
  and multi-output scans stay on the host (allclose, not bitwise; ADR 0005,
  docs/cuda.md). Measured on a Tesla T4: ~52x histogram micro-benchmark; **~2.1x
  end-to-end on a wide fit (50k×200)**, ~1.5x on narrow (100k×30, host path).
  Validated via the Colab dev loop. GPU leaf fitting (C1) was evaluated and **deferred**
  (leaf stats are already Rust-accelerated; low marginal value).
- GPU training (`device="cuda"`), multi-GPU (`multi_gpu=True`,
  `distributed_strategy="data_parallel"`)
- Distributed histogram building and leaf assignment
- Out-of-core training; large-scale tabular dataset support
- Distributed / data-parallel encoder computation and pretraining

## Encoder evolution track (cross-cutting, post-v0)

Listed here because v0 deliberately freezes the encoder:

- ~~PyTorch encoders (learned periodic frequencies, PLR projection)~~
  shipped in Phase 13; ~~interaction-aware features~~ shipped in Phase 16
  (`cross`, `torch_mlp` — negative on real data, see
  encoder_interactions.md); ~~full rtdl `PeriodicEmbeddings`
  (`torch_periodic_plr`: periodic basis + per-feature Linear+ReLU)~~ shipped in
  the trainable-embeddings track; still open: RealMLP-style blocks, category
  embeddings
- Encoder pretraining before boosting — supervised version shipped in
  Phase 13, regularized in Phase 14b, extended to cross-feature targets in
  Phase 16, made `sample_weight`/`class_weight`-aware in the
  trainable-embeddings track, and ~~generalized to a **`(n, K)`
  multiclass/multi-output pretraining target**~~ (those encoders no longer fit
  unsupervised — the throwaway head emits K outputs, docs/math.md "Supervised
  encoder pretraining target"); self-supervised variants are still open, and
  the accumulated evidence suggests the router already covers typical real
  tabular structure on reg/binary
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
- ~~Issue templates, SECURITY.md~~ done (Phase 20, 2026-06-12); ~~PyPI
  release~~ shipped from Phase 27 (v1.0.0; v1.0.1 followed) via OIDC trusted
  publishing on `v*` tags
- ~~Benchmark suite (OpenML/tabular benchmarks) under `benchmarks/`~~ done in
  Phase 25 (`benchmarks/openml_suite.py`, 9-dataset reproducible leaderboard);
  Phase 28 added a `--strict` release mode (fail on missing GBM) and a
  reproducibility manifest
- ~~API reference generation~~ done in Phase 26 (pdoc, `scripts/build_docs.sh`,
  CI docs job); ~~versioned/hosted docs~~ published via GitHub Pages
- ~~PEP 561 typing marker (`py.typed`)~~, ~~cross-platform CI
  (ubuntu/macos/windows)~~, ~~coverage gate (`pytest-cov` `fail_under`)~~, and
  ~~prebuilt native wheels for `repleafgbm-native` (Linux/macOS/Windows ×
  CPython 3.10-3.12 via maturin + OIDC trusted publishing)~~ done in Phase 28
  (v1.0.2). Optional: a Codecov badge (needs an external service token) and a
  Windows/macOS source-build smoke for the docs/torch lanes remain nice-to-haves.
