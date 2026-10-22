# ADR 0011: Leaf-only features and per-leaf expert selection

**Status:** Accepted (2026-10-22)

## Context

Teacher logits, cross-fitted target encodings, and pretrained row embeddings
can be useful leaf inputs without being legitimate tree split candidates. The
old dataset exposed one numerical block both to routing and to one frozen
encoder, so it could neither enforce that boundary nor compare several
representation spaces within the same routed leaf.

The existing `adaptive` leaf already compares one ridge-linear candidate with
an intercept-only candidate using Hessian-weighted leave-one-out (LOO) error.
Extending that gate to named blocks preserves the raw-routing thesis and the
existing `LeafValues` prediction format: selected coefficients occupy their
block and every other coefficient is zero.

## Decision

- `leaf_only_features` removes named/positional numerical columns from the raw
  routing matrix while appending them, in declared order, to the encoder's
  ordered numerical block. Leaf-only values must be finite floats and cannot be
  categorical. Metadata persists their names.
- `leaf_experts` registers named `(columns, encoder)` pairs. An
  `ExpertBundleEncoder` fits each child once, applies dimensionality reduction
  per child, concatenates their outputs, and publishes stable block slices.
  The public `encoder="identity"` default remains unchanged for API stability.
  With `leaf_experts`, that exact string default and no top-level
  `encoder_params` is allowed and the bundle takes precedence without a
  warning. A non-default name, encoder instance, or top-level `encoder_params`
  remains mutually exclusive with `leaf_experts`.
- `BaseEncoder.fit_transform_train` defaults to the historical
  `fit`-then-`transform` behavior. A supervised encoder may override it with
  cross-fitted/ordered training rows while retaining full-data state for eval
  and prediction. Only the bundle path seeds this returned matrix into the
  embedding cache of a private training view of the dataset, so the caller's
  dataset object still yields ``transform`` output when reused as an eval set
  or as predict input; the single-encoder lifecycle is unchanged, and an
  encoder nested inside a wrapper (e.g. a projection around a bundle) also
  keeps ``fit`` → ``transform``.
- `leaf_model="select"` treats constant plus every expert block as candidates.
  A directly supplied `encoder=ExpertBundleEncoder(...)` also exposes every
  bundle block as a separate candidate. A single non-bundle encoder supplies
  one block, so selection is equivalent to the adaptive constant-vs-linear
  choice.
  Each eligible block gets its own Hessian-weighted ridge fit and LOO score.
  The lowest-error block is used only if it beats constant by the configured
  `leaf_gate_margin`; ties remain constant. Blocks and backend handles are
  fit-local and are not serialized.
- The existing full-width `LeafValues` and prediction path remain unchanged.
  Models using leaf-only metadata or an expert bundle write format v9; older
  configurations keep their earlier format.

## Consequences

Routing, feature sampling, and feature importance operate only on the routing
block. Eval and prediction always call the fitted experts' ordinary
`transform`, preventing OOF training values from leaking across the lifecycle.
The concatenated bundle can also be used unchanged by `embedded_linear` and
`adaptive`, providing the full-space ridge comparison arm.

Selecting the minimum among K noisy LOO estimates has a stronger winner's
curse than the two-arm adaptive gate. Version 1 deliberately adds no
complexity/multiple-comparison penalty: users must control conservatism with
`leaf_gate_margin`, and tests include a row-permuted false block. This is a
known generality debt, not evidence that LOO is unbiased after selection.

Version 1 is scalar-only: regression and binary classification are supported;
multiclass pooled leaves and multi-output vector leaves raise
`NotImplementedError` and direct users to `adaptive` or `embedded_linear`.
No OOF target encoder, ICL leaf, automatic expert generation, selection-bias
correction, or native selector kernel is implemented. The selector's leaf loop
is NumPy; NumPy/Rust split backends remain untouched. At least one input
column must stay routable: a root-only router whose columns are all leaf-only
would need zero-feature support in the splitter and sampler, which this change
leaves untouched.
