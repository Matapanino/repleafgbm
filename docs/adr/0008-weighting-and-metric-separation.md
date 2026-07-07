# ADR 0008 — Sample/class weighting and the loss/metric separation

Status: accepted (2026-06-16, Phase 28 + capability follow-up). Renumbered
from a duplicate "ADR 0004" on 2026-07-07 — it was filed the same day as
ADR 0004 (sample weights via gradient/Hessian scaling) under the same number;
0004 keeps the *mechanism* decision, this ADR keeps the *policy* decision
(loss/metric separation + the `_supports_sample_weight` capability layer).

## Context

Users running imbalanced multiclass problems (e.g. Kaggle) often optimize a
**balanced-accuracy** target. The idiomatic recipe is:

- reweight the *training loss* by class (so the minority class is not ignored),
- early-stop on a smooth *built-in* metric (multi_logloss), and
- compute *balanced accuracy* externally for CV / Optuna / final reporting.

Before Phase 28 RepLeafGBM had none of the weighting controls, conflated the
monitoring/early-stopping metric with the (absent) report metric, and risked
users treating `label_smoothing` as a class rebalancer (it is not). We needed
weighting that is correct across every native path (regression, binary,
multiclass, multi-output) without disturbing the NumPy/Rust split-backend
parity that ADR 0001 protects.

## Decision

### 1. Separate four roles explicitly

Training loss (reweighted objective) · early-stopping metric (`eval_metric`) ·
report metric (external `balanced_accuracy_score`) · regularization
(`label_smoothing`) are independent knobs and documented as such
(docs/weighting_and_metrics.md). We deliberately did **not** add an internal
`primary_metric` plumbing path: the report metric is computed externally on
`predict(...)`, keeping the estimator surface small. `balanced_accuracy` is
registered (so it *can* be an `eval_metric`), but logloss remains the
recommended early-stopping signal because balanced accuracy is discrete/noisy.

### 2. Weighting via gradient/Hessian scaling

`sample_weight` scales per-row `g`, `h` (`core.booster.weight_grad_hess`) and
the init score. Since the Newton leaf target `-g/h` is invariant to the common
factor, weighting changes aggregation (split gains, leaf magnitudes) but never
the per-row target — the standard GBDT semantics (docs/math.md). The scaling
happens **upstream of the histogram**, so the split backends receive pre-scaled
statistics and stay parity-identical; `native/` is untouched.

Weights are **not renormalized** — used as given. The principled exact
invariant is *uniform scale invariance* at `l2_leaf=0` (a constant positive
weight cancels in every leaf). This is intentionally **not** equivalent to row
duplication: `min_samples_leaf` counts raw rows and duplication would shift
histogram bin edges.

### 3. `class_weight` is classifier-only sugar

`None` / `"balanced"` / `{label: weight}` expands to per-row weights via sklearn
`compute_sample_weight` and composes multiplicatively with `sample_weight`. The
regressor rejects it. It reweights the loss only — not the eval metric.

### 4. Capability flag, warn-don't-raise fallback

`_supports_sample_weight` (class attribute, default True). The native path is
always True. `RouterExtraction*` (frozen-route replay) is False: it cannot
reweight rows, so passing weights emits a `UserWarning` and drops them rather
than raising (`_enforce_weight_capability`). This keeps pipelines that pass a
uniform `sample_weight` working, and the documented fallback is exactly the
metric separation above.

## Consequences

- New optional parameters (`fit(sample_weight=...)`, `class_weight`,
  `RepLeafDataset(sample_weight=...)`) are additive — a MINOR change under
  ADR 0003. Defaults reproduce prior behavior (uniform weight cancels exactly
  at `l2_leaf=0`).
- `balanced_accuracy` joins the metric registry (greater-is-better);
  `get_metric` is exported.
- `class_weight` is serialized (`"balanced"`/`None` round-trip); it affects
  retraining only, so reloaded models predict identically.
- Backend parity is unaffected by construction (no native changes).

## Alternatives considered

- **Thread `sample_weight` into the splitter/native kernels.** Rejected: wider
  signature churn and a parity risk, for no benefit over scaling `g`/`h`
  upstream.
- **Normalize weights to mean 1.** Rejected: hides the user's chosen scale and
  breaks the clean `l2_leaf=0` cancellation invariant; users who want a stable
  effective learning rate can normalize themselves.
- **Internal `primary_metric` + per-round balanced-accuracy tracking.**
  Rejected for now: balanced accuracy is a poor early-stopping signal and
  external computation keeps the API smaller. Left open as a monitor-only
  `eval_metric` list.
- **Raise on unsupported models.** Rejected: breaks pipelines passing a
  uniform/neutral weight; a warning + fallback is friendlier and still honest.
