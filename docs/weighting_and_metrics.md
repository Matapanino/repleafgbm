# Sample weighting, class weighting, and the metric split

This page covers RepLeafGBM's loss-reweighting controls (`sample_weight`,
`class_weight`) and how they relate to evaluation metrics — in particular the
"optimize a balanced-accuracy target with imbalanced classes" workflow common
on Kaggle. The mechanics are in docs/math.md ("Sample weights"); this page is
the usage and decision guide.

## Four distinct knobs — keep them separate

A balanced-accuracy workflow only works if you stop conflating four things:

| Role | What controls it | Recommended for balanced accuracy |
|---|---|---|
| **Training loss** | objective (`logloss`/softmax) reweighted by `sample_weight` / `class_weight` | `class_weight="balanced"` (or explicit `sample_weight`) |
| **Early-stopping metric** | `eval_metric` + `early_stopping_rounds` (needs `eval_set`) | `"multi_logloss"` / `"logloss"` — smooth, stable |
| **Objective / CV / report metric** | computed externally on `predict(...)` | `sklearn.metrics.balanced_accuracy_score` |
| **Regularization** | `label_smoothing` | optional; *not* a rebalancer |

The trap is "I want balanced accuracy, so I'll early-stop on balanced
accuracy." Balanced accuracy is discrete and noisy round-to-round, which makes
early stopping erratic. Instead: **train a reweighted loss, early-stop on a
smooth built-in metric, and score balanced accuracy outside the model** for
cross-validation, Optuna objectives, and final reporting.

```python
from sklearn.metrics import balanced_accuracy_score
from repleafgbm import RepLeafClassifier

model = RepLeafClassifier(
    class_weight="balanced",      # reweights the TRAINING loss
    eval_metric="multi_logloss",  # early-stopping metric (smooth)
    early_stopping_rounds=50,
    n_estimators=2000,
)
model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])

# Optuna objective / CV score / final report: external balanced accuracy.
score = balanced_accuracy_score(y_va, model.predict(X_va))
```

## `sample_weight`

`fit(X, y, sample_weight=w)` takes one non-negative weight per row. Each row's
gradient and Hessian are scaled by `w` (`core.booster.weight_grad_hess`) before
tree growth and leaf fitting. Because the Newton leaf target is `t = -g/h`,
scaling `g` and `h` by the same `w` **leaves the per-row target unchanged**
while reweighting split gains and leaf magnitudes — the standard GBDT
sample-weight semantics. It works for the regressor, the binary and multiclass
classifier, and multi-output regression. The init score `F_0` becomes the
weighted optimum (weighted mean / log-odds / class prior / quantile).

Weights are **not** renormalized — they are used as given (you control the
effective scale). The principled exact invariant is *uniform scale
invariance*: with `l2_leaf=0` a constant positive weight cancels in every leaf's
numerator and denominator and leaves predictions unchanged. Note this is **not**
the same as row duplication: `min_samples_leaf` counts raw rows (not weight
mass), and duplication would also shift the per-feature histogram bin edges. See
docs/math.md.

A zero weight removes a row's contribution to the loss but the row still counts
toward `min_samples_leaf`.

## `class_weight` (classifier only)

`class_weight` is a convenience that expands to per-row `sample_weight`:

- `None` — no reweighting (default).
- `"balanced"` — weight each class inversely to its frequency
  (`n_samples / (n_classes * count_c)`), via
  `sklearn.utils.class_weight.compute_sample_weight`.
- `{label: weight}` — explicit per-class weights, keyed by the **original**
  class labels (strings, ints, …); remapped internally to the encoded classes.

`class_weight` and `fit(sample_weight=...)` **compose multiplicatively**. It
reweights the *training loss only* — it does not change the `eval_metric` used
for early stopping or any reported metric. The regressor has no classes; it
rejects `class_weight` (use `sample_weight`).

## `label_smoothing` is not class weighting

`label_smoothing` softens the hard targets (`y*(1-eps)+eps/2` for binary,
`(1-eps)*onehot+eps/K` for multiclass) to temper over-confident
probabilities. It is a **regularizer**, applied symmetrically across classes —
it does **not** rebalance a skewed class distribution. Use
`class_weight`/`sample_weight` for imbalance. The two are independent and
compose freely.

## `balanced_accuracy` as a registered metric

`balanced_accuracy` is available via `get_metric("balanced_accuracy")` and as
`eval_metric="balanced_accuracy"` (mean per-class recall, matching
`sklearn.balanced_accuracy_score`; greater-is-better). You *can* monitor or
early-stop on it, but prefer logloss for early stopping and reserve balanced
accuracy for the externally computed report metric.

## Capability and the unsupported-model fallback

The native boosting path always supports weights (it owns the gradients).
Estimators that cannot reweight rows — frozen-route replay
(`RouterExtractionRegressor` / `RouterExtractionClassifier`, which replay
LightGBM routes) — advertise `_supports_sample_weight = False`. Passing
`sample_weight`/`class_weight` to such a model **emits a `UserWarning` and drops
the weights** rather than raising, so pipelines keep running. The fallback there
is exactly the metric split above: train the (unweighted) loss, early-stop on a
built-in metric, and compute balanced accuracy externally.

```python
# RouterExtractionClassifier cannot reweight rows:
model.fit(X, y, sample_weight=w)   # -> UserWarning, weights ignored, fit proceeds
```

## Serialization

`class_weight` is stored in the model config (`"balanced"` / `None` round-trip
exactly). It only affects (re)training, not prediction, so a reloaded model
predicts identically. `sample_weight` is fit-time data and is not serialized.
