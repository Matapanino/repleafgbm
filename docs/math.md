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

## Limitations

- The Newton-target leaf fit is a second-order approximation for non-squared
  losses; no line search or leaf-wise damping beyond `λ` is performed.
- Histogram quantiles are computed once per `fit` on the full data
  (no per-node re-quantization).
- Native training routes missing values left at every split (no learned
  default direction); extracted external routes may carry per-node
  directions.
- PLR here is a simplified, unlearned piecewise-linear basis — not the full
  embedding of Gorishniy et al. (2022).
- Random projection to `max_leaf_emb_dim` preserves structure only in
  expectation; informative dimensions may be diluted.
