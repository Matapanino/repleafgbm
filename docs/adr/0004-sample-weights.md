# ADR 0004 — Sample weights via gradient/Hessian scaling

Status: accepted (2026-06-16, Phase 28)

## Context

Users training on imbalanced targets (especially multiclass tasks evaluated by
balanced accuracy) need a way to up-weight rare rows/classes. The standard
sklearn levers are `fit(sample_weight=)` and a classifier `class_weight`
parameter, plus a `balanced_accuracy` eval metric for early stopping.

The question is *where* to apply the weight in a histogram gradient-boosting
pipeline that has two compute backends (NumPy reference + optional Rust) which
must stay parity-tested, and a fused `leaf_linear_stats` native kernel.

## Decision

Apply the weight by scaling each row's gradient and Hessian — `g_i -> w_i g_i`,
`h_i -> w_i h_i` — at the top of the boosting loop, *before* the histogram is
built (`core.booster.weight_grad_hess`, called identically in the scalar,
multiclass, and multi-output boosters). The optimal init score `F_0` is
computed as the weighted optimum (`objective.init_score(y, weight=)`).

Consequences:

- **No backend changes.** `NumPySplitBackend`, `RustSplitBackend`, and the
  fused `leaf_linear_stats` kernel consume the already-weighted `g, h`, so the
  weighted Newton step `-Σw g / (Σw h + λ)` and weighted split gain fall out
  for free. NumPy/Rust parity is preserved trivially (both paths unchanged) and
  the native path still imports neither torch nor the external backends.
- **`min_samples_leaf` counts raw rows.** The histogram's third (count) channel
  stays unweighted, so the leaf-size guard reflects actual sample support
  (LightGBM's `min_child_samples` semantics), independent of weight mass.
- **Integer weights are not row duplication.** Duplicating rows would also move
  the per-feature quantile bin edges and the raw-count guard, so the two are
  not bitwise-equal. The principled exact invariant we test instead is *uniform
  scale invariance* at `l2_leaf = 0`: a constant positive weight cancels in
  numerator and denominator.

`class_weight` (classifier only, `{label: weight}` or `"balanced"`) is resolved
to per-row weights via `sklearn.utils.class_weight.compute_sample_weight` and
multiplied into any explicit `sample_weight`. It lives on the base estimator's
`__init__` and is documented as classifier-only, following the existing
`label_smoothing` precedent; it serializes with the estimator config.

## Alternatives considered

- **Threading `sample_weight` into each backend kernel** (NumPy + Rust +
  `leaf_linear_stats`). Rejected: more code in the parity-critical path for no
  behavioral gain, since pre-scaling `g, h` is mathematically identical.
- **Weighting the eval metric.** Not done: `balanced_accuracy` already corrects
  for imbalance, and unweighted validation metrics are the least-surprising
  default. Could be revisited if users ask for `eval_sample_weight`.
