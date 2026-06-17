# RepLeafGBM Mathematical Notes

## Additive boosting formulation

Given a loss `L(y, F)`, RepLeafGBM builds

```text
F_0(x) = argmin_c sum_i L(y_i, c)
F_t(x) = F_{t-1}(x) + eta * f_t(x)
```

At round `t`, with per-row gradients and Hessians of the loss at the current
raw score,

```text
g_i = dL(y_i, F)/dF |_{F=F_{t-1}(x_i)}
h_i = d2L(y_i, F)/dF2 |_{F=F_{t-1}(x_i)}
```

the second-order (Newton) approximation of the round-`t` objective is

```text
sum_i [ g_i f(x_i) + (1/2) h_i f(x_i)^2 ] + regularization
  = (1/2) sum_i h_i ( f(x_i) - t_i )^2 + const,    t_i = -g_i / h_i .
```

So every leaf model — constant or linear — is fitted by **h-weighted least
squares on the Newton targets** `t_i`. For squared error (`h_i = 1`,
`t_i = residual_i`) this is exact, not an approximation.

### Sample weights

Per-row sample weights `w_i >= 0` scale each row's loss term, so the round-`t`
objective becomes `sum_i w_i [ g_i f + (1/2) h_i f^2 ]`. Equivalently, every
appearance of `g_i, h_i` is replaced by `w_i g_i, w_i h_i`: the Newton target
`t_i = -g_i/h_i` is unchanged but its fitting weight becomes `w_i h_i`, the
constant leaf is `b_l = -sum w_i g_i / (sum w_i h_i + λ)`, and the split gain
uses the weighted sums `G = sum w_i g_i`, `H = sum w_i h_i`. The optimal init
score `F_0` is likewise the weighted optimum (weighted mean / log-odds /
class-prior / quantile). Because the weighting is folded into `g, h` *before*
the histogram is built, the split backends and leaf-fitting kernels are
untouched and NumPy/Rust parity is preserved. `min_samples_leaf` continues to
count **raw rows** (it guards leaf sample size, not weight mass), so integer
weights are *not* identical to row duplication in a histogram GBM (duplication
would also shift the per-feature quantile bin edges). Uniform weights, however,
cancel exactly when `λ = 0` (`core.booster.weight_grad_hess`, `docs` tests).

## Tree growth (routing)

Splits are scored with the standard Newton gain

```text
gain = G_L^2/(H_L + λ) + G_R^2/(H_R + λ) - G^2/(H + λ)
```

over histogram bins of raw features, with `min_samples_leaf` enforced on both
children and missing values fixed to the left child. Trees grow leaf-wise
(best gain first) up to `num_leaves`.

## Constant leaf model

For leaf `l` with row set `I_l`:

```text
b_l = - sum_{i in I_l} g_i / ( sum_{i in I_l} h_i + λ )
```

the classic XGBoost-style Newton step with L2 damping `λ = l2_leaf`.

## Embedded linear leaf model

For leaf `l`, an affine model over the representation `z_i = z_theta(x_i)`:

```text
min_{b, w}  sum_{i in I_l} h_i ( b + w^T z_i - t_i )^2  +  λ ||w||^2
```

The intercept `b` is not penalized. Implementation: center `z` and `t` by
their h-weighted means, then solve the ridge normal equations

```text
( Z_c^T H Z_c + λ I ) w = Z_c^T H t_c ,
b = t_mean - w^T z_mean .
```

### Extrapolation guard (prediction time)

Each linear leaf stores ``z_min, z_max`` — the per-dimension range of the
embeddings it was fitted on. Prediction uses ``clip(z, z_min_l, z_max_l)``,
i.e. the leaf's response is the fitted affine function inside its training
support and constant outside it. Training rows are inside by construction,
so the training trajectory is unchanged; only out-of-support queries differ.

### Ridge-regularized fitting and fallback

The system can be ill-conditioned when the leaf is small or `Z` is locally
degenerate (e.g. PLR components constant within the leaf). Guards:

- if `|I_l| < max(2 * min_samples_leaf, d_z + 2)`, fit a constant leaf;
- if the solve fails or returns non-finite weights, fit a constant leaf;
- `λ > 0` is required in practice (default `l2_leaf = 1.0`).

`raw_linear` is the same fit with `z_i` replaced by standardized raw
numerical features.

## Objectives

**Regression (squared error).** `L = (1/2)(y - F)^2`, `g = F - y`, `h = 1`,
`F_0 = mean(y)`. Output transform: identity.

**Binary classification (logistic).** Labels mapped to {0, 1},
`p = sigmoid(F)`:

```text
L = -[ y log p + (1-y) log(1-p) ],   g = p - y,   h = p (1 - p)
F_0 = log( p_bar / (1 - p_bar) )
```

Output transform: sigmoid. Hessians are floored at 1e-12 for stability.

**Huber (robust regression).** `g = clip(F - y, -delta, delta)`, `h = 1`,
`F_0 = median(y)`. The true Hessian vanishes beyond `delta`; using `h = 1`
(the LightGBM convention) keeps outlier-only leaves bounded and makes the
Newton targets *clipped residuals*, so linear leaf fits see outliers with
bounded influence. Output transform: identity.

**Quantile (pinball).** The model estimates the alpha-quantile:
`g = (1 - alpha)` where `F >= y`, `-alpha` where `F < y`, `h = 1`,
`F_0 = quantile(y, alpha)`. The loss is piecewise linear (no curvature);
unit Hessians give fixed-size steps whose sign balance converges to the
alpha-quantile within each leaf. Output transform: identity.

**Poisson (counts).** `F` is the log-mean, `mu = exp(F)`:

```text
L = exp(F) - y F,   g = mu - y,   h = mu,   F_0 = log(mean(y))
```

Requires `y >= 0` with positive mean; raw scores are clipped to [-30, 30]
inside exp for overflow safety, and `h` is floored at 1e-12. Output
transform: exp (predictions are positive means).

**Multiclass classification (softmax).** Labels mapped to {0, ..., K-1},
`p_k = softmax(F)_k` over per-class raw scores `F in R^K`:

```text
L = -log p_y,   g_k = p_k - 1{y=k},   h_k = p_k (1 - p_k)
F_0,k = log(prior_k)
```

`h_k` is the diagonal of the true softmax Hessian (`diag(p) - p p^T`) — the
standard GBDT approximation (LightGBM/XGBoost). Each round grows one tree
per class on column `k` of (g, h); each class's tree is fitted with exactly
the scalar Newton-target machinery above, so leaves (constant or linear over
`z_theta(x)`) carry over unchanged. Output transform: row-wise softmax.
Hessians are floored at 1e-12.

**Multi-output regression (vector leaves).** Targets and raw scores are
matrices `Y, F in R^{n x K}`, `g = F - Y`, `h = 1`, `F_0,k = mean(Y[:,k])`.
Unlike multiclass, **one shared tree per round** routes all outputs: splits
use the raw features (shared routing), and a leaf emits a vector. The split
gain sums the per-output Newton gains over a shared partition,

```text
gain = sum_k [ G_{k,L}^2/(H_{k,L}+lambda) + G_{k,R}^2/(H_{k,R}+lambda) ]
       - sum_k G_k^2/(H_k+lambda) ,
```

so a split is chosen to reduce all outputs' residuals jointly. A constant
vector leaf is the per-output Newton step `b_k = -G_k/(H_k+lambda)`. A linear
vector leaf solves, per leaf, `K` ridge problems over the shared embedding
`Z`; since `h = 1` the centered Gram `Z_c^T Z_c + lambda I` is identical
across outputs, so it is one factorization with `K` right-hand sides
`Z_c^T t_c` (Newton targets `t = -g/h` centered per output). The
extrapolation guard's `z_min/z_max` are the leaf's `Z` range, shared across
outputs. Output transform: identity.

**Label smoothing (classification).** Hard targets are softened before the
gradients: binary `y -> y(1-eps) + eps/2`, multiclass one-hot
`-> (1-eps) onehot + eps/K`, applied to both `F_0` and `g` (so
`g_k = p_k - target_k`). With `eps = 0` the objective is exactly the
unsmoothed one; `eps > 0` caps how confident the fitted probabilities can
become, a regularizer against over-confidence.

## Why encoder updates break stage-wise assumptions

The ensemble after `T` rounds is

```text
F_T(x) = F_0 + eta * sum_{t<=T} [ b_{t,l_t(x)} + w_{t,l_t(x)}^T z_theta(x) ] .
```

Every `w_{t,l}` was fitted against the value of `z_theta` **at round t**. If
theta is updated at round `T' > t`:

1. the outputs of all trees `t <= T'` change retroactively;
2. the cached raw scores `F` used to compute `g, h` no longer equal the model
   output, so subsequent gradients are computed at the wrong point;
3. the interpretation of boosting as coordinate descent in function space —
   each stage greedily fitting the current residual — no longer holds.

Handling this correctly requires re-evaluating (or re-fitting) earlier leaf
weights after each encoder update, i.e. alternating optimization with
prediction-cache invalidation. That is a roadmap item, not a v0 feature.

## Supervised encoder pretraining target

A *learned* encoder (the optional torch extras) is fit once, before boosting,
on a supervised target and then frozen — the fit-then-freeze rule above. The
target is the **negative gradient of the loss at the constant initial score
`F_0`**, i.e. the residual the first tree would chase:

```text
regression:   t = -g = y - mean(y)            (h = 1)
binary:       t = -g = y - sigmoid(F_0)
multiclass:   T = -g = onehot - softmax(F_0)  in R^{n x K},  F_0,k = log(prior_k)
multi-output: T = -g = Y - mean(Y)            in R^{n x K},  F_0,k = mean(Y[:,k])
```

A scalar target trains a `Linear(d, 1)` head on the embedding `z_theta(x)`; the
`(n, K)` matrix trains a `Linear(d, K)` head and the MSE is averaged over rows
and outputs, so the representation is pushed to be linearly predictive of every
class/output residual at once (the head is thrown away after fit). Each column
is standardized to unit variance so no class/output dominates the loss.

Two design choices worth recording:

- **Why `-g`, not the Newton target `-g/h`.** Pretraining learns a
  representation aligned with the loss's steepest-descent direction; the
  Hessian reweighting belongs to the leaf solve, not to feature learning. Using
  `-g` keeps scalar and vector targets consistent. At `F_0` the multiclass
  Hessian `h_k = p_k(1 - p_k)` is *column-constant* (because `softmax(F_0)` is
  the row-independent prior), so `-g/h` equals `-g` up to a per-column scale and
  **coincides with it after per-output standardization** — the two formulations
  are identical here.
- **At `F_0` the multiclass target is a centered one-hot.** Since
  `softmax(F_0)` does not depend on the row, `T = onehot - prior`; standardized,
  this is a scaled, centered class-membership indicator. Pretraining therefore
  teaches the encoder a representation from which class membership is linearly
  recoverable — exactly the signal the per-class leaves consume.

## Limitations

- The Newton-target leaf fit is a second-order approximation for non-squared
  losses; no line search or leaf-wise damping beyond `λ` is performed.
- Histogram quantiles are computed once per `fit` on the full data
  (no per-node re-quantization).
- Native training routes missing values left at every split (no learned
  default direction); extracted external routes may carry per-node
  directions.
- The fixed `plr` encoder here is a simplified, unlearned piecewise-linear
  basis — not the full embedding of Gorishniy et al. (2022). The *learned*
  `torch_plr` is that full embedding (a per-feature Linear+ReLU over the basis,
  pretrained then frozen).
- Random projection to `max_leaf_emb_dim` preserves structure only in
  expectation; informative dimensions may be diluted.
