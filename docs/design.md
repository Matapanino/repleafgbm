# RepLeafGBM Design

Status: v0. Everything described as "implemented" exists and is tested;
future plans are explicitly marked.

## Core architecture

RepLeafGBM is a stage-wise additive ensemble:

```text
F_T(x) = F_0 + eta * sum_t f_t(x)
f_t(x) = b_{t, l_t(x_raw)} + w_{t, l_t(x_raw)}^T z_theta(x)
```

where `l_t(x_raw)` is the leaf index produced by routing on **raw features**
and `z_theta(x)` is a representation produced by a **frozen encoder**.

The design is deliberately asymmetric:

| Concern | Component | Why |
|---|---|---|
| Discontinuities, interactions, local regions | Tree (raw splits) | Axis-aligned splits on raw features are the proven strength of GBDTs |
| Smooth interpolation / extrapolation inside a region | Leaf model over `z_theta(x)` | Numerical embeddings capture smooth nonlinear structure cheaply |

## Component responsibilities

```text
RepLeafDataset  -> owns data + metadata, lazy/cached embeddings
Encoder         -> owns z_theta(x); fitted once, then frozen
Booster         -> owns gradients, tree growth, leaf fitting, prediction
sklearn wrapper -> glues the three together; public API
```

The booster consumes a dataset and an encoder (`booster.fit(dataset,
encoder, leaf_model, ...)`) and internally calls
`dataset.get_raw_features()` and `dataset.get_embeddings(encoder)`. It never
constructs encoders or parses pandas objects. This keeps the path open for
batch / GPU / out-of-core data access without touching boosting logic.

Internal training state is kept in separate, plainly-typed buffers —
raw feature matrix, binned feature matrix (uint16), embedding matrix,
gradient/Hessian buffers, per-leaf row indices, flat tree arrays, leaf
parameter arrays, prediction cache — exactly the layout a future native
backend would want.

## Raw-feature routing

- Histogram-based: features are quantized once per `fit` into ≤ `max_bins`
  bins (`core/histogram.py`); split search scans per-bin gradient/Hessian
  sums (`backends/numpy_backend.py`).
- Leaf-wise (best-gain-first) growth like LightGBM, controlled by
  `num_leaves`, with optional `max_depth` and `min_samples_leaf`.
- Split gain is the standard Newton gain `G_L^2/(H_L+λ) + G_R^2/(H_R+λ) -
  G^2/(H+λ)`.
- Declared categorical features are routed with gradient-sorted **subset
  splits** (one bin per category, categories sorted by `G/(H+λ)` and scanned
  as prefixes — the LightGBM trick); ordered-threshold fallback above
  `max_bins` categories. See docs/categorical_features.md.
- Missing values (NaN) always route left in *native training*; learned
  default directions are future work. The `Tree` structure itself carries a
  per-node `missing_left` flag, used by extracted LightGBM routes
  (router_extraction) to represent learned directions exactly.

## Why not split on embeddings

1. GBDT strength lies in axis-aligned splits on raw features.
2. Splitting on embedding dimensions multiplies histogram-building cost by
   the embedding width.
3. Interpretability of routing is lost.
4. High-dimensional split search invites the curse of dimensionality.
5. Missing-value and categorical handling become entangled with the encoder.

Embeddings therefore appear only in leaf outputs.

## Why the encoder is frozen in v0

Every fitted leaf weight `w_{t,l}` is tied to the encoder parameters theta at
the time tree `t` was fitted. If theta changes later, the outputs of **all
previous trees** change, the cached predictions used to compute gradients
become stale, and the stage-wise additive assumption of boosting silently
breaks. v0 therefore fits the encoder once before boosting and enforces
`freeze_encoder=True` (passing False raises `NotImplementedError`).
Alternating optimization, stage-wise snapshot encoders, per-stage low-rank
adapters, and prediction-cache invalidation schemes are roadmap items.

## Leaf model variants

| `leaf_model` | Leaf output | Z source |
|---|---|---|
| `constant` | `b` (Newton step) | — |
| `embedded_linear` | `b + w^T z` | encoder (`identity`, `plr`, `periodic`) |
| `raw_linear` | `b + w^T x_num` | standardized raw numericals (LightGBM `linear_tree` analogue) |
| `adaptive` | `embedded_linear`, gated per leaf | encoder, with a weighted-LOO fallback to `constant` where the linear fit does not generalize |

Encoders (all NumPy, frozen; see `encoders/`):

- `identity` — standardized raw numericals; the most robust default.
- `plr` — simplified quantile piecewise-linear basis (Gorishniy et al. 2022),
  `n_bins` components per feature plus, by default, an appended standardized
  linear term so leaves can extrapolate beyond the training range.
- `periodic` — PBLD-style (RealMLP) sinusoidal features with random frozen
  frequencies/phases plus a linear term. With frozen random frequencies it
  never beat `identity`; kept as the initialization/baseline for the learned
  version.
- `torch_periodic` / `torch_plr` / `torch_periodic_plr` (optional `[torch]`
  extra) — the learned versions, mirroring rtdl-num-embeddings (Gorishniy
  et al. 2022): `torch_plr` is the full *PiecewiseLinearEmbeddings* (a learned
  per-feature Linear+ReLU over the PLE basis), `torch_periodic` is the *lite*
  *PeriodicEmbeddings* (learned frequencies/phases only), and
  `torch_periodic_plr` is the full *PeriodicEmbeddings* (periodic basis + a
  learned per-feature Linear+ReLU). Parameters are pretrained on the initial
  Newton residual and then frozen, so the v0 frozen-encoder rule holds — they
  are pretrained-then-frozen, **not** joint-trainable. torch is required only
  at fit time; transform and serialization are NumPy. **Specialist tools**:
  decisively best on targets with genuine per-feature smooth/oscillatory
  structure (encoder_variants.md), but they overfit and lose to `identity`
  on all real datasets tested (real_data_validation.md Phase 14) — keep
  `identity` as the first choice on real tabular data. Pretraining is
  regularized by default (AdamW weight decay + validation early stopping
  with best-epoch restore, Phase 14b); this preserves the synthetic win and
  cuts pretraining cost but does not change the real-data verdict
  (torch_pretrain_regularization.md).
- `cross` / `torch_mlp` (Phase 16) — the *interaction-aware* line: `cross`
  appends the pairwise products most correlated with the initial residual
  (NumPy, deterministic); `torch_mlp` learns a small feature-mixing MLP
  (frozen after pretraining like the other torch encoders). Both decisively
  beat `identity` when the target is dominated by feature products
  (encoder_interactions.md home turf) and both lose to `identity` on every
  real dataset tested — the router already captures the interactions that
  matter there. Opt-in specialists, like the per-feature learned encoders.

Learned-encoder pretraining is **supervised on a scalar Newton residual** and
honors `sample_weight`/`class_weight`: the residual is taken at the *weighted*
initial score and the pretraining loss is per-row weighted. It is therefore
active for regression and binary classification only — for **multiclass and
multi-output** the residual is a per-row matrix, so those encoders fit
**unsupervised** (a scalar multiclass/multi-output pretraining target is a
roadmap item, docs/roadmap.md). Fixed encoders (`identity`/`plr`/`periodic`/
`cross`) ignore weights.

All linear leaves are fitted by Hessian-weighted ridge regression on Newton
targets (docs/math.md). Overfitting guards, all implemented:

- `l2_leaf` ridge penalty (intercept unpenalized),
- constant fallback when the leaf has fewer than
  `max(2*min_samples_leaf, emb_dim + 2)` rows or the normal equations are
  singular,
- **extrapolation guard**: each linear leaf stores the per-dimension min/max
  of the embeddings it was fitted on; at prediction time Z is clipped to
  that range, so beyond its training support a leaf extrapolates as a
  constant. Added in Phase 7 after real-data validation showed unguarded
  leaf-linear extrapolation blowing up on feature outliers
  (experiments/results/real_data_validation.md),
- `max_leaf_emb_dim`: encoders wider than this are reduced by a fixed,
  seeded Gaussian random projection. Note: experiments showed the projection
  consistently *hurts* accuracy (experiments/results/plr_projection_gap.md),
  so defaults are chosen to avoid it (PLR n_bins=4, max_leaf_emb_dim=64) and
  a UserWarning fires when it engages; it remains as an OOM/cost guard, not
  an accuracy feature.

## Multiclass training

Targets with three or more classes use softmax boosting with one tree per
class per round (`core/multiclass.py`): the per-round softmax
gradients/Hessians are computed once on the (n_rows, n_classes) score
matrix, then class k's tree is grown on column k with the same splitter,
grower, and Newton-target leaf models as scalar training. The loop is a
deliberate mirror of `Booster._run_boosting` rather than a generalization of
it, keeping the scalar boosting loop readable. Trees are stored round-major
in a flat list so serialization and feature importance reuse the per-tree
code paths; `best_iteration_` counts rounds.

## Dataset abstraction

`RepLeafDataset` (docs/dataset_and_memory.md) holds the encoded raw matrix,
target, and `FeatureMetadata` (names, types, category maps). Embeddings are
computed lazily per encoder and cached. Prediction-time inputs are re-encoded
with the training metadata so train/predict preprocessing always matches.

## Backend abstraction

`backends/BaseSplitBackend` isolates the numeric split-search kernel: it sees
only integer bin matrices and float buffers. v0 ships `NumPySplitBackend`;
the contract is designed to be implementable in Rust/C++/CUDA without
changing `TreeGrower`. Multi-GBM backends (LightGBM/XGBoost/CatBoost) are a
separate, higher-level axis described in docs/backend_strategy.md.

## Known risks

- **Leaf linear overfitting** on small leaves — mitigated as above, but the
  right defaults need empirical study.
- **Embedding dimension blow-up** — random projection is a blunt instrument;
  learned projections / per-leaf feature selection are open questions.
- **Python tree construction speed** — acceptable for research-scale data
  only; the native backend is the long-term answer.
- **Frozen encoder quality** — supervised pretraining on the initial Newton
  residual wins only when the target has structure trees route poorly:
  per-feature oscillations (Phases 13/14b) or dominant feature products
  (Phase 16). On real tabular targets both encoder families found nothing
  the router doesn't, even regularized — `identity` is the evidence-backed
  default, and the learned encoders are opt-in specialists.
- **eval on embedded leaves costs one transform per eval set** — fine in
  memory, needs batching out-of-core.

## Future extensions

See docs/roadmap.md for the full shipped-vs-planned breakdown. Early stopping,
multiclass/vector leaves, external GBM backends (`external_model`,
`router_extraction` modes), PyTorch encoders with optional pretraining, and the
native high-performance backend have all **shipped** (Phases 1–22); the
remaining genuinely future work is GPU / multi-GPU / distributed training.
