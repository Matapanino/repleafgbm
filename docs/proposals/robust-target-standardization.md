# Proposal: robust-objective target standardization (huber / quantile)

Status: proposed (spec only — implementation handed to orchestrator/implementer)
Author: research-proposer
Date: 2026-06-29
Validated by: `experiments/robust_delta_diagnosis.py` →
`experiments/results/2026-06-29-robust-delta-diagnosis.md` (commit 70b909d)

## 1. Problem

The robust regression objectives have **saturated, scale-relative gradients**:

- `Huber.grad_hess` clips the residual to `±delta` with `delta=1.0`
  (`core/objectives.py:112-113`); `MultiOutputHuber` does the same
  (`core/objectives.py:327-329`).
- `Quantile.grad_hess` emits a gradient of exactly `±alpha` regardless of
  residual magnitude (`core/objectives.py:137-139`); `MultiOutputQuantile`
  the same (`core/objectives.py:366-369`).

Because the per-tree Newton step is bounded by these saturated gradients, the
effective `delta/σ` ratio varies ~25x across datasets. On large-scale targets
the model cannot traverse the target range in `n_estimators` trees and
**underfits even on clean data**: the diagnosis measured a clean-fit RMSE
penalty of 15.0x (rf1, per-output σ~60) and 2.4x (scm20d, σ~250) for huber vs
squared, with quantile worse still.

The validated fix is **per-output target standardization** applied *only* to
the saturated-gradient robust family: fit in `(y - median) / (1.4826·MAD)`
space, un-standardize predictions. The diagnosis showed this eliminates the
clean penalty (rf1 15.0x→0.9x, scm20d 2.4x→1.0x) while keeping robustness
(std-huber/quantile beat squared under 16% contamination on rf1, scm20d, and
energy). `squared_error` is the scale-equivariant control and must be left
exactly as-is (standardizing it would perturb the `l2_leaf` regularization
interaction and break bitwise reproducibility/parity for the default path).

## 2. Hypothesis

A robust per-output affine target transform `y' = (y - loc) / scale`, with
`loc = median(y)` and `scale = 1.4826·MAD(y)` (zero-floored), makes the
residuals O(1) so that `delta=1` (huber) and the fixed `±alpha` step (quantile)
operate in **robust-σ units** for every dataset. This exploits a structure the
current path ignores: the saturated robust gradients are *not* scale-equivariant
(unlike squared error, whose `g = F - y` scales linearly), so the only loss
family that needs the transform is exactly the saturated one. Trees still route
on raw features and leaves still fit Newton targets — the transform changes only
the *units* the robust objective sees.

## 3. Recommended implementation layer

**Compute the transform at the estimator layer (`RepLeafRegressor`), apply it to
the training target before the encoder is fit, and let the booster *carry*
`(loc, scale)` as the single source of truth for prediction, eval, and
serialization.** This is the cleanest split for three reasons:

1. The estimator is the only layer that knows both the resolved objective family
   *and* the raw training `y` before any downstream component consumes it
   (`BaseRepLeafModel.fit`, `sklearn.py:261-350`).
2. Standardizing `dataset.y` *before* `_build_and_fit_encoder`
   (`sklearn.py:308-313`) means the encoder's supervised pretraining target
   (`_pretrain_target`, `sklearn.py:563-579`; regressor override
   `regressor.py:93-110`) is computed in std space too. This is important:
   `_pretrain_target` rebuilds the robust objective and clips at `delta=1`, so
   on a σ~250 target the *raw-space* pretrain residual is ~all `±1` (useless
   signal). Standardizing first makes the learned-encoder path coherent for
   free, and avoids introducing a new inconsistency.
3. The booster already owns `init_score_` and the prediction/eval loops, so
   storing `(loc, scale)` there keeps un-standardization in one place and lets
   serialization extend `tree_ensemble.json` alongside `init_score`
   (`serialization.py:100-108`).

Detection of the robust family is by **isinstance against the known objective
classes**, not by name string, so a parameterized instance such as
`Huber(delta=2.0)` or `Quantile(alpha=0.9)` is caught:

```
robust = (Huber, Quantile, MultiOutputHuber, MultiOutputQuantile)
```

`SquaredError`, `PoissonRegression`, any custom `BaseObjective` subclass not in
that tuple, and the entire classification path return **no transform** (identity
`loc=0, scale=1`). See §8 for the `Huber(delta=2.0)` semantic note.

### 3.1 Where exactly

New regressor hook (base returns `None`, regressor overrides):

```python
# sklearn.py — BaseRepLeafModel
def _resolve_target_transform(self, dataset) -> tuple[np.ndarray, np.ndarray] | None:
    return None   # non-robust default; classifier inherits this (always None)
```

```python
# regressor.py — RepLeafRegressor
def _resolve_target_transform(self, dataset):
    obj = (self._build_multioutput_objective()
           if getattr(self, "n_outputs_", 1) > 1 else self._build_objective())
    if not isinstance(obj, (Huber, Quantile, MultiOutputHuber, MultiOutputQuantile)):
        return None
    y = dataset.y
    loc = np.median(y, axis=0)                       # float (1-D y) or (n_outputs,)
    mad = np.median(np.abs(y - loc), axis=0) * 1.4826
    scale = np.where(mad < 1e-9, 1.0, mad)           # zero-floor
    return loc, scale
```

(The `robust_scale` helper at `experiments/robust_delta_diagnosis.py:44-48` is
the validated reference; the only addition is the multi-output `axis=0` path,
which already matches that file's array form.)

## 4. Fit path

In `BaseRepLeafModel.fit` (`sklearn.py:291-344`), immediately after
`_prepare_target` (`sklearn.py:293`) and the `n_outputs_` resolution it triggers
(`regressor.py:53-57`), **before** the sample-weight resolution and the encoder
fit:

```python
self._target_transform_ = self._resolve_target_transform(dataset)
if self._target_transform_ is not None:
    loc, scale = self._target_transform_
    dataset.y = (dataset.y - loc) / scale          # std-space training target
```

- **Do not** standardize `sample_weight` (it scales gradients/Hessians
  multiplicatively and is scale-free; confirmed out of scope).
- `loc`/`scale` are computed from the (possibly contaminated) training `y` using
  median/MAD, which is exactly what makes them robust.
- Pass the transform to the booster before `fit` (`sklearn.py:336-344`):

```python
self.booster_ = self._make_booster(params)
if self._target_transform_ is not None:
    self.booster_.target_loc_, self.booster_.target_scale_ = self._target_transform_
self.booster_.fit(...)
```

**No objective change is needed.** Once `dataset.y` is standardized:
- `objective.init_score(y)` (`booster.py:225-226`, `multioutput.py:229-230`)
  becomes the std-space median (~0) — correct.
- `objective.grad_hess(y, F)` (`booster.py:244`, `multioutput.py:247`) now clips
  at `delta=1` in σ-units, which is precisely the fix. `delta` is no longer
  raw-target units; it is robust-σ units (document this).
- Leaf Newton targets `t = -g/h` and the per-leaf `l2_leaf` ridge act in std
  space (intended — residuals are O(1), so `l2_leaf=1` is meaningful again).
- The extrapolation guard (per-leaf z-clip bounds) is on the *embedding* z, not
  on y, so it is unaffected.

Both `Booster` (`booster.py:86-103`) and `MultiOutputBooster`
(`multioutput.py:148-166`) get two new attributes defaulting to identity:

```python
self.target_loc_: float | np.ndarray = 0.0
self.target_scale_: float | np.ndarray = 1.0
```

Scalar defaults broadcast correctly over both 1-D (`booster.py`) and `(n, K)`
(`multioutput.py`) score arrays, so squared/poisson/loaded-old models need no
special-casing.

## 5. Predict path

Un-standardize inside the booster's `predict_raw` method — the single place all
prediction flows through:

```python
# booster.py:289-303  and  multioutput.py:290-304
raw = predict_raw(...free function unchanged...)   # prediction.py:16-38 / 65-85
return self.target_loc_ + self.target_scale_ * raw
```

Notes:
- Keep the free functions in `core/prediction.py` **pure** (they back the
  NumPy↔Rust parity tests); apply the affine only in the booster *method*.
- The robust family has an **identity output transform**
  (`Huber.transform`/`Quantile.transform` return `raw`,
  `objectives.py:115-116, 141-142`; multi-output likewise). So
  `regressor.predict` (`regressor.py:112-120`) doing
  `objective.transform(predict_raw(...))` reduces to
  `loc + scale·raw` — order-independent, no double application. Poisson keeps
  `scale=1, loc=0`, so its `exp` transform still applies to the un-touched raw
  log-mean. This is why un-standardizing in `predict_raw` is safe: it only ever
  activates for the identity-transform robust family.
- Staged prediction (`n_trees`) is unaffected — the affine is applied to the
  final accumulated score regardless of how many trees contributed.

## 6. eval_set / early-stopping / eval_metric — report in RAW scale

**Decision: standardize only the *training* target; keep eval-set `y` raw; the
booster un-standardizes its eval predictions before calling the metric, so
`evals_result_`, `best_score_`, and any custom `eval_metric` see raw target
scale.**

Justification:
- A user reading `model.evals_result_["valid_0"]["rmse"]` or passing a custom
  `eval_metric` expects the metric on *their* target scale, not σ-units.
- For multi-output, std-space pooled RMSE weights every output equally while
  raw-space pooled RMSE weights by `scale²`; their argmin can differ, so a
  std-space metric would silently change early-stopping decisions. Computing in
  raw space makes early stopping identical to the documented behavior.
- The affine is already on the booster, so this is one extra op per eval round.

Concretely, the eval datasets must **not** be standardized.
`_prepare_eval_sets` (`sklearn.py:488-499`) routes eval `y` only through
`_prepare_target(is_train=False)`, which for the regressor merely validates
numeric (`sklearn.py:509-517`, `regressor.py:53-57`) — so eval `y` already stays
raw; **do not** add standardization there. Inside the eval loops
(`booster.py:261-268`, `multioutput.py:263-270`) change:

```python
pred = self.objective.transform(Fe)                       # std space (identity)
pred = self.target_loc_ + self.target_scale_ * pred       # -> raw scale
self.evals_result_[name][eval_metric.name].append(eval_metric(ye, pred))  # ye RAW
```

The eval score-cache `Fe` stays in std space (it is accumulated from std-space
leaf outputs); only the value handed to the metric is un-standardized. The
training score-cache `F` (`booster.py:259`, `multioutput.py:261`) stays in std
space untouched — it only feeds `grad_hess` on std-space `y`. Early stopping
(`booster.py:269-283`, `multioutput.py:271-284`) then compares raw-scale metric
values exactly as before; `eval_metric.minimize` direction is unchanged.

This introduces a deliberate, documented asymmetry: **train `y` is std-space,
eval `y` is raw-space.** Eval `y` is used *only* by the metric (never by
`grad_hess`), so this is safe.

## 7. Serialization + SemVer

Persist `(loc, scale)` in `tree_ensemble.json` next to `init_score`
(`serialization.py:100-108`), written **only** when the booster carries a
non-identity transform:

```python
if not _is_identity_transform(booster):     # i.e. robust family was used
    ensemble["target_loc"]   = np.asarray(booster.target_loc_).tolist()
    ensemble["target_scale"] = np.asarray(booster.target_scale_).tolist()
```

**Format version (`serialization.py:41-53`):**
- `FORMAT_VERSION = 7`; `READABLE_VERSIONS = (1, 2, 3, 4, 5, 6, 7)`.
- Bump-on-use, matching the existing ladder (`serialization.py:84-91`): write
  version **7 iff `target_loc`/`target_scale` are present** (scalar *or*
  multi-output robust model); otherwise keep the current selection
  (multioutput→6, multiclass→5, freq→4, else→3). A multi-output *huber* model
  thus writes 7 (superset of 6: it still emits `n_outputs` + vector
  `init_score`); a multi-output *squared* model still writes 6; a scalar squared
  model still writes 3.

**Load (`serialization.py:183-205`):** after constructing the booster, read the
transform with an identity default (back-compat):

```python
booster.target_loc_   = np.asarray(ensemble["target_loc"])   if "target_loc"   in ensemble else 0.0
booster.target_scale_ = np.asarray(ensemble["target_scale"]) if "target_scale" in ensemble else 1.0
```

- **Old models (v3–v6) load with identity transform**, so they predict exactly
  as they did when saved — the broken-but-deterministic legacy behavior is
  preserved bit-for-bit. This satisfies the read-ladder invariant (new format ⇒
  new `format_version`, still reads old).
- An old build reading a v7 directory correctly errors with the existing
  "Unsupported model format version" message (`serialization.py:159-163`) —
  forward-incompat as intended.
- Loading branches on **key presence** (`n_outputs`, `n_classes`, now
  `target_loc`), consistent with the current loader; `version` remains the
  compat gate.
- The estimator does not need `_extra_config`/`_restore_extra_config`
  (`sklearn.py:754-756, 778-779`) for this — the booster is the source of truth
  and prediction flows through `booster.predict_raw`. (Optionally restore
  `self._target_transform_` for introspection, but it is not load-bearing.)

**SemVer: MINOR bump.** No public API signature changes; serialization is
backward-compatible (reads all old versions). Training *outputs* for
huber/quantile change (they were underfit/broken before) and the *units of
`delta`/`alpha`-driven robustness change to robust-σ — a documented behavior
change shipped as a minor with a CHANGELOG note, not a breaking API change.

## 8. Guardrail check

| Invariant | Status |
|---|---|
| Tree routing uses raw features only | Untouched — the transform applies to `y` only; splits and histograms are unchanged. |
| Leaf prediction may use `z_theta(x)` | Unchanged. |
| Encoder frozen during boosting | Unchanged — encoder is still fit once (`sklearn.py:308-313`); standardization happens *before* that fit and changes only the pretraining-target *scale*, not the freeze contract. |
| No split on embedding dims | N/A — no split logic touched. |
| Newton targets `t=-g/h` weighted by `h`; per-leaf z-clip guard | Preserved; targets are now std-space (the point). z-clip is on z, unaffected. |
| No wrapper-only product; native path imports no torch/external | Unchanged — pure NumPy host-side affine. |
| Determinism / `check_random_state` | Unchanged — median/MAD are deterministic; no RNG introduced. |
| SemVer + serialization read-ladder | Honored — new `format_version=7`, reads 1–7; old models → identity. |

**Backend parity (host-side standardization):** `dataset.y` is standardized in
NumPy at the estimator boundary *before* the splitter/backends ever see it. Both
`NumPySplitBackend` and `RustSplitBackend` receive identical std-space
gradients/Hessians, so **bitwise NumPy↔Rust parity holds in std space**. No
backend code changes. CUDA stays allclose-not-bitwise as today. Existing parity
tests (`tests/test_rust_backend.py`) use squared error and are entirely
unaffected (`scale=1`).

**Classification excluded:** the classifier's `_resolve_target_transform`
inherits the base `None` (`classifier.py`), and its objectives
(`BinaryLogistic`, `MulticlassSoftmax`) are not in the robust tuple. No
interaction with `class_weight` or `label_smoothing` — those operate on
gradients/priors, never on a continuous target, and the classifier path never
calls the regressor hook.

**Router extraction excluded:** `_RouterExtractionMixin.fit`
(`router_extraction.py:192-257`) bypasses `BaseRepLeafModel.fit` entirely and
builds a squared-error booster (`router_extraction.py:247`); the booster keeps
the identity transform default and is unaffected. (It is squared-only by design,
so the fix is correctly inert there.)

**Custom `BaseObjective` instance:** detection is by `isinstance` of the four
known robust classes, so a user-supplied `Huber(delta=2.0)`/`Quantile(alpha=0.9)`
*is* standardized. This means **their explicit `delta`/`alpha` now act in
robust-σ units, not raw-target units** — a semantic change. This is the
recommended, more-portable contract (delta in σ-units transfers across
datasets), but it must be documented prominently in the `objective`/`Huber`
docstrings (`sklearn.py:136-146`, `objectives.py:93-107`) and the CHANGELOG. An
arbitrary custom `BaseObjective` subclass *not* in the tuple is **not**
standardized (safe default), which is correct — we only standardize objectives
we have proven saturate.

## 9. Required tests

Add to `tests/test_regression_objectives.py` and `tests/test_serialization.py`
(and a multi-output case in `tests/test_multioutput.py`):

1. **Large-scale clean-fit fix (the core regression guard).** Build a clean
   linear target scaled to σ~100 (e.g. `make_linear_data(...) * 100`). Assert
   `RepLeafRegressor(objective="huber")` clean-test RMSE is within ~1.5x of
   `objective=None` (squared) — and add a contrasting assertion that *without*
   the fix the ratio would be »10x (document the expected pre-fix number in a
   comment; the test asserts the post-fix bound). Repeat for
   `objective="quantile"`. Multi-output variant with a `(n, 2)` target whose
   columns have σ~1 and σ~200.
2. **`squared_error` unchanged (scale-equivariance guard).** Fit squared on a
   large-scale target with a fixed seed; assert predictions are bitwise-identical
   to a pre-fix reference (or simpler: assert `booster_.target_scale_ == 1.0`
   and `target_loc_ == 0.0`, and that saved `tree_ensemble.json` has **no**
   `target_loc` key and `format_version == 3`). Also assert poisson is unchanged.
3. **fit/predict shapes** for scalar and multi-output robust models (existing
   `test_regressor_basic`/`test_multioutput` patterns) — predictions on the
   target scale, correct shape `(n,)` / `(n, n_outputs)`.
4. **save/load round-trip carrying (loc, scale).** Extend the existing
   `test_objective_instance_save_load_roundtrip`
   (`test_regression_objectives.py:116-126`): assert
   `loaded.predict(X) == model.predict(X)` (`atol=1e-12`), assert
   `model_config.json` `format_version == 7`, and assert the loaded booster's
   `target_loc_`/`target_scale_` equal the originals. Multi-output huber
   round-trip (v7) too.
5. **Backward-compat load.** Load a checked-in v6 multi-output-huber fixture (or
   construct one by writing without the transform keys) and assert it predicts
   with identity transform (`target_scale_ == 1.0`) — i.e. legacy behavior
   preserved. Mirror the existing
   `test_poisson_save_load_keeps_transform` style.
6. **eval/early-stopping reports raw scale.** Fit a robust model with
   `eval_set` + `early_stopping_rounds` on a large-scale target; assert the
   recorded `evals_result_` RMSE magnitude matches an externally computed
   raw-scale RMSE (not σ-units, i.e. it is ~`scale`x larger than the std-space
   value), and that a custom callable `eval_metric` receives raw `y_true`.
7. **Encoder transform shapes / constant-fallback for small leaves** unchanged —
   reuse the existing leaf-model coverage; add one robust+`embedded_linear`
   small-data fit to confirm the constant fallback still triggers in std space.
8. **NumPy↔Rust parity still green.** No new test needed beyond confirming
   `tests/test_rust_backend.py` passes (squared, unaffected); optionally add a
   huber parity case asserting both backends produce allclose predictions in
   std space (both see identical std-space inputs).

## 10. Validation experiment

The direction is already validated by
`experiments/results/2026-06-29-robust-delta-diagnosis.md` (out-of-model
standardization). The in-`src/` confirmation experiment (for
`experiment-runner` → `results-analyst`):

- **Datasets:** energy, rf1, scm20d (the diagnosis set) + 2–3 additional
  small-σ real regression targets to quantify the energy-style robustness
  trade-off (§11).
- **Baselines:** (a) pre-fix `main` huber/quantile, (b) squared error
  (scale-equivariant control), (c) the out-of-model standardization from the
  diagnosis (sanity: the in-`src/` path should match it to numerical noise).
- **Seeds:** ≥3 (the diagnosis used `[0,1,2]`); `n_estimators=200`.
- **Conditions:** contamination ∈ {0%, 16%} at 8x σ (match the diagnosis).
- **Metric:** mean per-output RMSE on the clean test set.
- **Expected effect:** clean-fit penalty collapses (rf1 ~15x→~1x, scm20d
  ~2.4x→~1x); std-huber/quantile beat squared under 16% contamination on
  rf1/scm20d/energy; squared and poisson numbers unchanged within seed noise;
  in-`src/` results match the diagnosis's out-of-model std column.

## 11. Risks

1. **Small-σ robustness trade-off (accepted, monitor).** On energy (σ~10) the
   diagnosis showed standardized huber loses *peak* robustness at 16%
   contamination (5.68 vs baseline-huber 2.51) while still beating squared
   (18.76) and improving the clean fit (1.59 vs 2.23). Unconditional
   standardization is recommended for simplicity and portability, but the
   validation experiment must quantify this across more small-σ datasets; if the
   loss is material, a follow-up could expose an opt-out — *not* in this change
   (avoid premature API surface).
2. **`Huber(delta=2.0)` semantic change.** Explicit `delta`/`alpha` become
   robust-σ units. Mitigation: prominent docstring + CHANGELOG note (§8). The
   escape hatch for a user wanting raw-target-unit delta is to pre-scale `y`
   themselves.
3. **Degenerate scale.** Constant or near-constant targets give MAD≈0; the
   zero-floor (`scale=1.0` where `mad<1e-9`) prevents divide-by-zero and falls
   back to a pure shift. Covered by a unit test (constant target).
4. **Multi-output version churn.** Multi-output huber/quantile bumps from v6→v7;
   ensure the v6→identity load path is fixture-tested so older saved
   multi-output-huber models keep predicting (broken-but-deterministic) exactly.

## 12. Recommendation

Implement at the estimator layer (compute + apply to training `y` before encoder
fit), with the booster carrying `(loc, scale)` as the single source of truth for
prediction, raw-scale eval, and serialization. Bump `format_version` to 7
(bump-on-use; old models load as identity). Ship as a **minor** release with a
documented behavior change for huber/quantile (delta/alpha now in robust-σ
units). Gate the in-`src/` confirmation through `experiment-runner` →
`results-analyst` before merge, then `qa-verifier` + `core-reviewer`.
