# ADR 0009: Explicit leaf-embedding reduction and numerical-column routing

**Status:** Accepted (2026-09-12)

## Context

The compatibility path silently selected a seeded Gaussian projection whenever
an encoder exceeded `max_leaf_emb_dim`. It warned, but callers could not choose
an uncapped or learned path and fitted models did not expose the resolved
dimensions. PLR also expanded every numerical column, which is wasteful for
wide banks dominated by target-encoded/noise columns.

## Decision

- Keep `max_leaf_emb_dim=64` and make the existing behavior explicit through
  `leaf_reduction="random_projection"` as the default.
- Add `"none"` and deterministic supervised `"target_correlation"` policies.
  Reduction is Python-side and leaves raw-feature tree routing unchanged.
- Expose resolved dimensions/policy and the retained linear-leaf fraction as
  fitted diagnostics.
- Register a composable `column_subset` encoder. Its indices always address
  the ordered numerical block returned by `get_numerical_features()`.
- Write model format v8 only when a routed or learned-reduction encoder occurs
  in the recursive encoder config. Existing encoder/model combinations keep
  their prior written format; versions 1–7 remain readable.

## Consequences

Default predictions remain unchanged. Users can remove the projection, but
must accept the quadratic/cubic leaf-fit cost and the `emb_dim + 2` sample gate.
The target-correlation option uses the frozen initial Newton residual and is
therefore supervised, deterministic, and compatible with scalar/vector targets.
Older releases reject v8 models rather than guessing how to reconstruct them.
