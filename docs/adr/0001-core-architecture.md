# ADR 0001: Core Architecture

- Status: accepted
- Date: 2026-06-11

## Context

We are building RepLeafGBM, a tabular ML library whose core idea is a boosted
ensemble of raw-feature routers with representation-conditioned local
predictors. v0 must validate the idea as both a software design and an
experimental prototype, while not blocking later goals: external GBM
backends, PyTorch encoders, native (Rust/C++/CUDA) kernels, and
GPU/distributed training.

## Decision

1. **Asymmetric model structure.** Splits use raw features only; leaf outputs
   may depend on encoder representations. Embeddings never enter split
   search.
2. **Frozen encoder in v0.** The encoder is fitted once before boosting;
   `freeze_encoder=False` raises `NotImplementedError`. Rationale: encoder
   updates retroactively change earlier trees' outputs and invalidate
   gradient computation (docs/math.md).
3. **Three-way responsibility split.** `RepLeafDataset` (data + metadata +
   lazy embeddings) / `Encoder` (representation) / `Booster` (gradients,
   tree growth, leaf fitting). The sklearn wrapper composes them and owns the
   public API.
4. **NumPy core, PyTorch optional.** Tree/booster/leaf fitting in
   NumPy/SciPy for readability and portability to native code. v0 encoders
   (identity, simplified PLR) are NumPy too — they are unlearned and frozen,
   so requiring PyTorch would add an install burden with no benefit. PyTorch
   becomes an optional extra when learned encoders arrive. (This is a
   deliberate narrowing of the original "encoders in PyTorch" plan, recorded
   here; the `BaseEncoder` interface is framework-agnostic.)
5. **Histogram-based training from day one.** Features are pre-binned
   (uint16) and split search scans per-bin g/h sums behind a narrow
   `BaseSplitBackend` contract. This keeps v0 fast enough to experiment with
   and is the exact seam for native/GPU kernels later.
6. **Newton-target leaf fitting.** All leaf models fit `t = -g/h` with
   weights `h` and ridge penalty `l2_leaf`; constant fallback for small or
   degenerate leaves. One code path serves squared error exactly and other
   losses to second order.
7. **Leaf-wise growth** controlled by `num_leaves` (LightGBM-style), with
   `max_depth` and `min_samples_leaf` as additional caps.
8. **Directory-based serialization** (JSON structure + npz arrays) with an
   explicit `format_version`, anticipating binary encoder weights.
9. **Missing values route left** at every split in v0; learned default
   directions are future work. *(Amended in Phase 3 / ADR 0002: the tree
   structure now stores a per-node `missing_left` flag so extracted external
   routes can carry learned directions; native training still always routes
   left.)*

## Consequences

- The core idea is testable end-to-end with ~2.5k lines of readable Python;
  the test suite runs in seconds.
- Python-level tree growth limits v0 to research-scale data; this is
  accepted and bounded by the backend seam (decision 5).
- Freezing the encoder caps achievable accuracy; the encoder-evolution track
  in docs/roadmap.md lists the designs that may lift this, all of which must
  address the stage-wise consistency problem explicitly.
- Ordinal categorical encoding is weaker than native subset splits; accepted
  for v0 and documented in docs/categorical_features.md.
- Adding an external GBM backend later requires mapping external trees into
  the unified ensemble representation (routing trees + leaf params), which
  the serialization format already assumes.
