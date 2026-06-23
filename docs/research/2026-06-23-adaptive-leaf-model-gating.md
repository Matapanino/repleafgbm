# Literature Note: Adaptive Leaf Model Gating (Constant vs Embedded-Linear)

- **Date:** 2026-06-23
- **Author:** literature-scout
- **Question:** What does the external literature say about (1) piecewise-linear / model
  trees as GBDT base learners and when linear leaves fail, (2) closed-form
  weighted-ridge LOO/GCV generalization estimates and the role of heterogeneous leverage,
  and (3) adaptive per-leaf or per-region model-complexity selection in boosting/trees?
  How do these map onto RepLeafGBM's approved `leaf_model="adaptive"` feature?
- **Decision it feeds:** `docs/proposals/adaptive-leaf-model.md` (approved design, 2026-06-23).
  This note supplies the external-knowledge backing; it does not redesign the feature.

---

## 1. Rationale

RepLeafGBM currently selects `leaf_model` globally: either every leaf is a Newton-step
constant, or every leaf is a ridge-regularized linear model over the frozen representation
`z_theta(x)`. The roadmap evidence is clear that the right choice is *leaf-dependent*:
regression leaves benefit from the linear term; binary classification leaves — particularly
in late rounds with confident predictions — absorb noise via a linear fit that carries no
generalizable signal (`experiments/results/binary_leaf_gain.md`).

The approved feature (`leaf_model="adaptive"`) selects constant vs embedded-linear
*per leaf* using a closed-form weighted-ridge leave-one-out (LOO) gate. This note grounds
that gate in external literature.

---

## 2. Sources

| Title | URL | Date |
|---|---|---|
| Gradient Boosting With Piece-Wise Linear Regression Trees (GBDT-PL) — Shi, Li et al. | https://arxiv.org/abs/1802.05640 | arXiv 2018-02-15; revised 2019-06-25; AAAI 2019 |
| LightGBM Parameters documentation (linear_tree, linear_lambda) | https://lightgbm.readthedocs.io/en/latest/Parameters.html | accessed 2026-06-23 |
| LightGBM Issue #5131: LGBMRegressor with linear_tree has constant leafs | https://github.com/lightgbm-org/LightGBM/issues/5131 | 2022 |
| M5' Model Trees — Quinlan 1992 / Wang & Witten 1997 (Machine Learning 32:63-76) | https://link.springer.com/content/pdf/10.1023/A:1007421302149.pdf | 1998 |
| Logistic Model Trees (LMT) — Landwehr, Hall, Frank | https://link.springer.com/chapter/10.1007/s10994-005-0466-3 | Machine Learning 59, 2005 |
| Generalized Cross-Validation as a Method for Choosing a Good Ridge Parameter — Golub, Heath, Wahba | http://www.stat.yale.edu/~jtc5/312_612/readings/generalized-cross-validation-and-ridge_Golub-Heath-Wahba_79.pdf | Technometrics 21(2):215-223, 1979 |
| Ridge Regularization: an Essential Concept in Data Science (LOO formula) — McDonald et al. | https://arxiv.org/html/2006.00371v2 | arXiv 2020 |
| A New Formula for Faster Computation of the K-Fold Cross-Validation and Good Regularisation Parameter Values in Ridge Regression | https://arxiv.org/pdf/2211.15128 | arXiv 2022 |
| Leverage (statistics) — Wikipedia | https://en.wikipedia.org/wiki/Leverage_(statistics) | accessed 2026-06-23 |
| Kish (1965) effective sample size — explained with formula | https://aakinshin.net/posts/kish-ess-weighted-quantiles/ | accessed 2026-06-23 |
| An information criterion for automatic gradient tree boosting — Lunde, Kleppe, Skaug | https://arxiv.org/abs/2008.05926 | arXiv 2020 |
| Model Selection in Omnivariate Decision Trees — Gama, Brazdil | https://link.springer.com/chapter/10.1007/11564096_45 | ECML 2005 |
| GCV theory documentation (Golub-Heath-Wahba 1979 derivation) | https://cherab-inversion.readthedocs.io/en/v0.3.0/user/theory/gcv.html | accessed 2026-06-23 |

---

## 3. Key Findings

### 3.1 Piecewise-linear leaves in GBDT: when they help and when they fail

**GBDT-PL (Shi, Li et al., arXiv 1802.05640 / AAAI 2019).**
The central claim: replacing constant leaves with per-leaf ridge-regularized linear
regression models in gradient boosting accelerates convergence and improves accuracy
on numerical dense data, without proportionally increasing training cost. The GBDT-PL
trees split on raw features (splitting uses the same Newton gain as standard GBDT —
splits are scored on raw-feature histograms), and the linear model is fitted *after*
the tree structure is determined. On 10 public datasets GBDT-PL is competitive with
XGBoost, LightGBM, and CatBoost; Microsoft Research implemented an approximation
(`linear_tree=True`) in LightGBM.

Evidence strength: peer-reviewed (AAAI 2019); a production-quality approximation shipped
in LightGBM; affirmed by the LightGBM parameter docs as the reference for `linear_lambda`.

**Relevance to RepLeafGBM:** GBDT-PL's architecture is the closest published analog to
RepLeafGBM's embedded-linear leaf. The key structural difference is that GBDT-PL's linear
model uses the raw input features within the routing branch, while RepLeafGBM's linear
model uses the frozen representation `z_theta(x)` — a higher-dimensional or more
structured feature space. The split criterion is identical in both: raw-feature Newton
gain. This confirms the thesis is consistent with the literature's cleanest model-tree
success story. The GBDT-PL finding that linear leaves help on "numerical dense data" is
the positive evidence for the `embedded_linear` default on regression; the unaddressed
question GBDT-PL does not tackle is *per-leaf* selection, which is what the LOO gate adds.

**LightGBM `linear_tree` parameter (official docs, accessed 2026-06-23).**
LightGBM's approximation: "tree splits are chosen in the usual way, but the model at each
leaf is linear instead of constant." Key constraints surfaced in the docs:
- "The first tree has constant leaf values" (the cold-start case, when no warm residuals
  are available, defaults to constant).
- `linear_lambda` is the ridge penalty on the leaf linear model, defaulting to `0.0`
  (unregularized; this is aggressive and different from RepLeafGBM's `l2_leaf=1.0`
  default).
- The docs do not describe a per-leaf fallback to constant when a leaf is ill-conditioned;
  LightGBM issue #5131 (2022) confirms that LightGBM's `linear_tree` can produce leaves
  that are effectively constant when the linear fit has no support, but this is an observed
  behavior, not a guaranteed fallback mechanism with a documented condition.

Evidence strength: official docs + confirmed community issue; no formal per-leaf selection
criterion in the LightGBM linear-tree implementation.

**Relevance to RepLeafGBM:** LightGBM's design gap — a global `linear_tree` flag with no
per-leaf generalization check — is exactly the gap the LOO gate fills. The `min_data_in_leaf
>= 2` constraint when `path_smooth > 0` is a minimal size guard, not a generalization
signal. RepLeafGBM's existing `min_samples_linear = max(min_samples_linear, emb_dim + 2)`
pre-filter is already stricter than LightGBM's; the LOO gate adds a generalization
quality check *after* the pre-filter.

**Classic model trees: M5 / M5' (Quinlan 1992; Wang & Witten 1997/1998).**
M5 was the first algorithm to grow a regression tree and fit multivariate linear
regression at each leaf. M5' (Wang & Witten, Machine Learning 32:63-76, 1998) added
pruning: each internal node is evaluated as a candidate leaf — if the estimated error
of a linear model at that node is lower than the error of the subtree, the subtree is
pruned and replaced by the leaf's linear model. The smoothing procedure blends the leaf
model's prediction with the parent's model to avoid discontinuities at split boundaries.

The critical model-selection criterion in M5 is the *estimated error* at each node vs its
subtree — an information-criterion-like comparison that selects between more-complex and
less-complex models at each local node. M5 uses the standard deviation of the training
cases at a node as a proxy for the error, which is a rough but fast in-sample signal. M5'
does not use cross-validation.

Evidence strength: foundational algorithm, widely cited; no per-leaf LOO gate (only
post-hoc pruning by estimated error, which is in-sample). The minimum instances per leaf
in M5/M5' practice is documented as four (`mno=4`), motivated by the linear model
needing at least `d + 2` samples to be well-posed.

**Relevance to RepLeafGBM:** M5/M5' is a precedent for *per-node model selection* (linear
vs simpler model), but the selection signal is in-sample, not LOO. This supports the
"insample" naive baseline arm in the proposal (`leaf_gate="insample"`) as a historical
reference point, and affirms that per-leaf model selection is a principled design — but
also motivates why a cross-validated signal (the LOO gate) is the stronger choice.

**Logistic Model Trees — LMT (Landwehr, Hall, Frank; Machine Learning 59, 2005).**
LMT extends the model-tree concept to classification by replacing linear regression at
leaves with logistic regression, fitted via LogitBoost (additive logistic regression).
The key practical finding: LMTs use a minimum-instances-per-leaf guard (default `mno=4`,
range 1–64) and grow the tree with a constraint that leaf logistic models must have
sufficient support to avoid overfitting. LMT uses a cross-validation-based pruning
strategy to select the number of LogitBoost iterations per leaf, which is computationally
expensive but adapts leaf model complexity to the local data density.

For the classification failure mode of interest: LMT's logistic leaf models face the same
h = p(1-p) collapse for confident leaves. The LMT cross-validation pruning implicitly
guards against this by selecting fewer iterations (shallower logistic models) for leaves
with low effective curvature — which is conceptually equivalent to the LOO gate's
demotion of noisy logistic leaves in RepLeafGBM.

Evidence strength: peer-reviewed (Machine Learning journal); describes a cross-validation
selection signal but the published method applies it at the iteration level (how many
LogitBoost steps), not as a closed-form gate per leaf.

**Relevance to RepLeafGBM:** LMT is the classification analog of M5. The LOO gate
generalizes and closes the gap: instead of a full cross-validation scan over iteration
count, a single closed-form PRESS computation decides keep-linear vs constant per leaf.

---

### 3.2 The generalization signal: closed-form ridge LOOCV and heterogeneous leverage

**The standard LOO formula for ridge regression (McDonald et al., arXiv 2020; Golub, Heath, Wahba 1979).**
For an ordinary ridge regression `min_w ½ ||y - Xw||² + ½λ||w||²`, the LOO prediction
error for row `i` can be computed from the full-data fit without refitting:

```
LOO_i  = ( y_i - x_i^T w_hat ) / ( 1 - H_ii )
H_ii   = x_i^T (X^T X + λI)^{-1} x_i      [ridge leverage]
LOO-CV = (1/n) Σ_i LOO_i²
```

The key identity: the ridge hat matrix is `R^λ = X(X^T X + λI)^{-1} X^T`, with
diagonal entries `H_ii = R^λ_{ii}`. Using the SVD `X = UDV^T`, the leverages factor as
`H_ii = u_i^T S(λ) u_i` where `S(λ)_{jj} = d_j² / (d_j² + λ)` — so `λ > 0` shrinks
every leverage away from 1, preventing the denominator `(1 - H_ii)` from collapsing.

The **weighted-ridge** extension (Newton leaf context): the leaf objective is
`½ Σ_i h_i (b + w·z_i - t_i)² + ½λ||w||²`. This is equivalent to OLS on a
re-weighted problem, so the effective design matrix is `Z_tilde = diag(sqrt(h)) Z`
and the weighted hat matrix diagonal is
`H_ii = h_i * z̃_i^T (A + λI)^{-1} z̃_i + h_i / h_sum`
where `A = Σ_i h_i (z_i - z̄)(z_i - z̄)^T` is the centered Gram and the `h_i/h_sum`
term is the intercept contribution from centering. This is the exact form used in the
approved proposal (`docs/proposals/adaptive-leaf-model.md §3`).

Evidence strength: the unweighted LOO formula is textbook (Golub-Heath-Wahba 1979,
Technometrics 21(2):215-223, is the canonical citation); the weighted extension follows
directly from the substitution `X → diag(sqrt(h)) Z`, `y → diag(sqrt(h)) t`. Both are
algebraically exact (no approximation), well-supported, and uncontested.

**GCV as a LOO approximation and when they diverge (Golub, Heath, Wahba 1979).**
GCV replaces every per-row leverage `H_ii` with the mean leverage `tr(H)/n` in the
denominator:

```
GCV(λ) = ||( I - H )y||² / ( n - tr(H) )²     [aggregate form]
```

This makes GCV rotationally invariant and computationally cheaper (one scalar: `tr(H)`
vs `n` diagonal entries). GCV is equivalent to LOO when the leverages are homogeneous
(all `H_ii ≈ tr(H)/n`).

The failure mode: when leverage is **heterogeneous** — a few rows have `H_ii >> tr(H)/n`
while most are small — GCV averages the high-leverage signal into the bulk. It then
underestimates the influence of the high-leverage rows and underestimates the LOO error
on those rows. The approved proposal documents this failure mode directly: the binary
classification case where `h_i = p_i(1-p_i)` is near zero for confident rows has
high leverage `H_ii ≈ h_i z̃_i^T (A + λI)^{-1} z̃_i` precisely for those rows (because
the small `h_i` shrinks the Gram accumulation but *does not shrink* the test-point
leverage term `z̃_i^T (A + λI)^{-1} z̃_i` proportionally). GCV replaces those
high-leverage denominators with the mean and misses the signal.

Evidence strength: the GCV-vs-LOO divergence under heterogeneous leverage is noted in the
Golub-Heath-Wahba paper and is the standard theoretical reason to prefer LOO over GCV
in the "badly designed experiment" case (high leverage concentration). The application
to the logistic-Hessian collapse in GBDT is RepLeafGBM-specific reasoning that follows
from the established theory.

**Kish effective sample size (Kish 1965, Survey Sampling).**
For a weighted sample with weights `w_i`, the Kish ESS is:

```
n_eff = ( Σ w_i )² / Σ w_i²
```

Under uniform weights `n_eff = n`; under unequal weights `n_eff < n`. In the Newton
leaf context with `w_i = h_i`, this becomes `(Σ h_i)² / Σ h_i²`, which is exactly the
diagnostic used in `experiments/results/binary_leaf_gain.md` (the "ESS" column). The
ESS provides a scalar intuition for "how many effective samples does this leaf's fit see":
a leaf with `n = 500` rows but `n_eff = 320` is fitting a regression on about 320
independent pseudo-samples. The Phase-12 experiment measured `n_eff` declining from ~490
(round 1) to ~320 (round 76-100) for binary classification, while staying at 500 for
regression (squared-error `h = 1` keeps all weights equal).

The connection to LOO: Kish ESS is the inverse of the *average squared leverage* (up to
normalization). High per-row `H_ii` correspond to low contribution to `n_eff`; when a
few rows dominate the leverage, `n_eff` drops well below `n`. The LOO gate makes this
diagnosis per-row, not just in the aggregate.

Evidence strength: Kish (1965) is a foundational reference in survey statistics; the
formula is standard and uncontested. Its application as a GBDT leaf diagnostic is a
domain-specific extension that follows directly from the formula.

---

### 3.3 Adaptive model complexity / per-leaf model selection in boosting

**An information criterion for automatic gradient tree boosting — Lunde, Kleppe, Skaug (arXiv 2008.05926, 2020).**
This paper derives an information-theoretic criterion for automatically deciding (a)
whether to split a tree node and (b) how many trees to build in gradient boosting. The
criterion is derived from the optimism of the greedy splitting procedure, shown to
follow a Cox-Ingersoll-Ross process; the resulting GEC (generalization error criterion)
penalizes each split by an analytic complexity cost without requiring cross-validation.
The agtboost R package implements this, achieving 10-1400x speedups over XGBoost at
comparable accuracy.

The key distinction from the LOO gate: agtboost selects *node structure* (to split or
not) using a criterion on the gain; the LOO gate selects *leaf model form* (constant vs
linear) using a criterion on the fitted parameters. The two are complementary; the
information-criterion approach does not address per-leaf model type selection.

Evidence strength: arXiv preprint (not yet published in a top venue as of the search
date); the speedup claims are relative to untuned XGBoost and are implementation-specific.
The theoretical basis is sound (optimism analysis is standard); the applicability to
leaf-model selection (as opposed to node splitting) is indirect.

**Relevance to RepLeafGBM:** The agtboost direction is orthogonal to the LOO gate: it
automates tree *depth/leaf count* selection, not leaf *model form* selection. It is not
a competitor or a direct input to the adaptive-leaf feature; it is cited here to establish
that information-criterion-based complexity selection in boosting trees is an active
research direction, strengthening the motivation for data-driven per-leaf choices.

**Model Selection in Omnivariate Decision Trees — Gama, Brazdil (ECML 2005).**
Omnivariate decision trees place different model types at different nodes: univariate
splits, linear multivariate splits, and nonlinear (MLP) nodes, each selected per node
using AIC, BIC, cross-validation, or Structural Risk Minimization. The finding: CV
produces simpler trees than AIC/BIC without sacrificing expected error; quadratic
(MLP) nodes are selected rarely; univariate nodes dominate. The per-node model selection
is explicitly described as AIC/BIC/CV comparisons of in-sample fit quality.

Evidence strength: peer-reviewed (ECML 2005); applies to node *split* type selection,
not leaf *predictor* type. The methodology is analogous: compare two candidate model
forms at each node and select by a generalization criterion.

**Relevance to RepLeafGBM:** Omnivariate trees establish that per-node model selection
using a generalization criterion (not in-sample) is known to outperform global model
type choices. The LOO gate is RepLeafGBM's leaf-level analog: per-leaf rather than
per-node, and using closed-form LOO rather than full CV (cheaper and exact for linear
models). The finding that simpler models dominate in most nodes corroborates the Phase-12
result that binary leaves predominantly benefit from the constant fallback.

---

## 4. Relevance to RepLeafGBM

### 4.1 Thesis mapping

The adaptive-leaf feature does not violate any thesis invariant. Specifically:

| Invariant | Status |
|---|---|
| Splits use raw features only | PASS — the LOO gate operates on already-fitted leaf statistics; it does not affect the split criterion or add embedding-dim split candidates |
| Leaf prediction may use `z_theta(x)` | PASS — a kept-linear leaf uses the same `b + w·z_theta(x)` form; a demoted leaf uses the constant form (both already exist) |
| Encoder frozen during boosting | PASS — the LOO gate is a post-solve verdict; `z_theta(x)` is computed once and cached before boosting begins |
| Newton targets `t = -g/h`, weights `h` | PASS — the LOO gate computes `E_lin` and `E_const` from the same `h`-weighted PRESS formula; it does not change how `t` or `h` are computed |
| No wrapper around LightGBM/XGBoost | PASS — the gate is implemented in `core/leaf_models.py` using only NumPy; it reuses statistics the batched Gram solve already materializes |

### 4.2 Code-map touch points

The proposal identifies all touch points precisely; this note adds only the literature
context that each touch point is grounded in:

- `core/leaf_models.py::EmbeddedLinearLeafModel._solve_and_assemble` — the existing
  per-leaf constant fallback (lines 357-359) is the demotion path. Literature grounding:
  M5/M5' and LMT both use a per-leaf fallback to a simpler model when the linear fit
  is under-supported; this is the standard design.

- The LOO gate formula (`H_ii = h_i z̃_i^T (A + λI)^{-1} z̃_i + h_i/h_sum`,
  `E_lin = Σ h_i (r_i / (1 - H_ii))²`) — this is the direct application of Golub-Heath-
  Wahba (1979) to the h-weighted ridge problem. The formula is algebraically exact (no
  approximation); only the weighted substitution is novel (mapping `sqrt(h) z̃` → standard
  ridge). The `h_i/h_sum` intercept term is the contribution of the unpenalized intercept
  under the centering reparameterization.

- The comparison signal `E_lin < (1 - mu) * E_const` where `E_const = Σ h_i (t_bar - t_i
  )² / (1 - h_i/h_sum)²` — `E_const` is the LOO error of the intercept-only fit (constant
  leaf), derived from the same hat-matrix formula with the constant-leaf leverage
  `H_ii^const = h_i / h_sum`. This two-sided PRESS comparison is the natural extension of
  M5's estimated-error comparison to a LOO-corrected form.

- The use of LOO rather than GCV — directly motivated by the heterogeneous-leverage failure
  mode (Golub-Heath-Wahba 1979, §3.2 above): binary leaves with confident rows (`h_i ~ 0`)
  have concentrated leverage that GCV would average away. The per-row LOO formula keeps
  this signal intact.

### 4.3 Guardrail check

No thesis violation found. The gate selects *between two existing leaf forms* (constant
and embedded-linear) per leaf; it does not introduce a new leaf type, split on embedding
dimensions, or unfreeze the encoder.

The one implementation correctness point the literature highlights: when `H_ii` approaches
1 (high-leverage row), the denominator `(1 - H_ii)` approaches zero and `LOO_i` blows up.
The proposal addresses this by clamping `H_ii <= 1 - 1e-6`. The `lambda > 0` requirement
(already enforced; `l2_leaf > 0` is a pre-existing guard in `docs/math.md`) bounds
ridge leverage away from 1 by construction (`H_ii = h_i z̃_i^T (A + λI)^{-1} z̃_i <= h_i
||z̃_i||² / lambda <= h_sum ||z̃||_max² / lambda`), but the clamp is a cheap defensive
guard for numerics.

---

## 5. Implications for the adaptive-leaf feature

**Does the literature support LOO as the gate signal?**
Yes, strongly. The closed-form LOO formula for ridge regression is textbook
(Golub-Heath-Wahba 1979); its application to the weighted-ridge leaf problem is a direct
extension. The literature provides three independent lines of support:

1. *Algebraic exactness:* LOO via the hat-matrix diagonal is not an approximation — it is
   the exact leave-one-out error for any linear predictor. No approximation error is
   introduced.
2. *Computational efficiency:* the formula reuses statistics (`A`, `g_sum`, `h_sum`,
   `z_mean`, `t_mean`, the solved `w`) that the batched-solve path (`_solve_and_assemble`)
   already materializes. The marginal cost is a per-row pass over each leaf to accumulate
   `Σ h_i r_i² / (1 - H_ii)²` — dominated by the existing Gram solve for typical
   embedding dimensions.
3. *Superiority over the alternatives in this specific failure mode:* GCV is cheaper but
   averages away the heterogeneous-leverage signal that distinguishes binary's confident
   leaves from regression leaves (§3.2). An eval-set holdout would be exact in principle
   but requires threading a validation split through every fit path and introduces a
   new RNG surface; the proposal's code-map analysis correctly identifies this as a design
   cost that closed-form LOO avoids.

**Known failure modes to watch.**

- **High-leverage degeneracy.** A single row with `h_i << h_sum` (near-zero Hessian,
  i.e. a very confident logistic prediction) that is also an embedding outlier (large
  `||z̃_i||`) can produce `H_ii` near 1 even with `lambda > 0`. The clamp `H_ii <= 1 -
  1e-6` is the guard. Empirically, the Phase-12 experiment showed that binary leaves in
  late rounds have `ESS ~ 320` (out of 500), meaning some rows have leverage ~ 5x the
  average — high but not extreme. The clamp should be benign in practice; a test that
  verifies the gate does not produce `NaN`/`Inf` LOO errors on a planted high-leverage
  leaf is advisable.

- **Regression false-negatives (demoting good linear leaves).** On regression tasks with
  `h = 1`, the Kish ESS is always `n` (no Hessian collapse). LOO leverages are uniformly
  bounded by `||z̃_i||² / lambda` for centered embeddings. The main risk is that on small
  leaves (close to `min_samples_linear`) a high-variance linear fit is correctly identified
  as worse than a constant, demoting a leaf that *could* improve in later rounds if more
  data arrived. This is the right behavior (the gate is a generalization estimate, not a
  capacity estimate), but it implies that on small datasets `leaf_gate_margin` may need
  tuning — a larger `mu` makes the gate more conservative (prefers constant), a smaller
  `mu` more permissive. The default `mu = 0.01` requires a 1% LOO improvement for the
  linear fit to be kept, which is deliberately conservative.

- **Binary classification false-positives (keeping noisy linear leaves).** This is the
  primary failure mode the gate is designed to catch. The Phase-12 evidence shows binary
  leaves in late rounds keep `||w|| ~ 0.45` with no generalizable signal. The LOO gate
  should detect this: a noisy `w` fit will have large in-sample residuals on the
  high-leverage rows (exactly the rows the linear model over-fits), producing a large
  `E_lin`. The experiment in §6 of the proposal (the `leaf_gate_margin` sensitivity sweep)
  is the validation path.

- **Multiclass / multi-output heterogeneous verdict.** For multiclass, each `(class, leaf)`
  gets its own LOO verdict — the per-class Hessians `h_k = p_k(1-p_k)` differ across
  classes, so one class's leaf may demote while another keeps the linear fit. This is
  the correct behavior per the proposal. For multi-output (squared-error Hessian `h = 1`),
  the shared Gram means the leverage is identical across outputs and the gate sums `E_lin`
  over outputs — a single verdict for the whole vector leaf. No literature precedent exists
  for this exact multi-output gate form, but it follows algebraically from the shared-Gram
  structure already in `multioutput.py`.

- **`lambda = 0` edge case.** The LOO formula requires `(1 - H_ii) > 0`, which is
  guaranteed only when `lambda > 0` (with `lambda = 0`, a single-row interpolating fit
  can achieve `H_ii = 1`). The existing `l2_leaf > 0` default enforces this; the gate
  should validate and raise if `l2_leaf = 0` with `leaf_model="adaptive"`.

---

## 6. Concrete Next Steps for research-proposer

1. **Implement and test the LOO gate with the Phase-12 diagnostic.** The canonical
   correctness check is to verify that the adaptive gate agrees with the Phase-12 outcome:
   on the binary-classification late-round leaves where `||w||` stays `~0.45`, the gate
   should demote to constant; on regression leaves where `||w||` collapses to `~0.09`,
   the gate should keep the linear fit. This maps exactly onto the "recovers constant on
   noise / recovers linear on signal" required tests in `docs/proposals/adaptive-leaf-
   model.md §6`.

2. **Validate the LOO-vs-insample gap on the binary task.** The proposal includes a
   `leaf_gate="insample"` naive baseline (in-sample PRESS without leverage correction).
   The literature predicts the LOO gate will outperform on binary but not necessarily on
   regression (where `h = 1` makes leverage less heterogeneous). An experiment that
   compares `insample` vs `loo` gate across regression/binary/multiclass (at least 5
   seeds, on the OpenML suite) directly validates the core theoretical prediction.

3. **Guard `l2_leaf = 0` with `leaf_model="adaptive"`.** The LOO formula is only well-
   posed when `lambda > 0`. Add a `ValueError` in `BaseRepLeafModel.fit` when
   `leaf_model="adaptive"` and `l2_leaf = 0`. Document this in `docs/math.md` as a
   required constraint for the LOO gate (it is already stated for general stability but
   not yet for the gate specifically).

---

Sources:
- [Gradient Boosting With Piece-Wise Linear Regression Trees (arXiv 1802.05640)](https://arxiv.org/abs/1802.05640)
- [LightGBM Parameters (linear_tree)](https://lightgbm.readthedocs.io/en/latest/Parameters.html)
- [LightGBM Issue #5131: linear_tree constant leaves](https://github.com/lightgbm-org/LightGBM/issues/5131)
- [M5' Model Trees, Wang & Witten 1998 (Machine Learning 32:63-76)](https://link.springer.com/content/pdf/10.1023/A:1007421302149.pdf)
- [Logistic Model Trees, Landwehr, Hall, Frank (Machine Learning 59, 2005)](https://link.springer.com/chapter/10.1007/s10994-005-0466-3)
- [Generalized Cross-Validation as a Method for Choosing a Good Ridge Parameter, Golub, Heath, Wahba (Technometrics 1979)](http://www.stat.yale.edu/~jtc5/312_612/readings/generalized-cross-validation-and-ridge_Golub-Heath-Wahba_79.pdf)
- [Ridge Regularization: an Essential Concept in Data Science (arXiv 2006.00371)](https://arxiv.org/html/2006.00371v2)
- [A New Formula for Faster Computation of the K-Fold CV in Ridge Regression (arXiv 2211.15128)](https://arxiv.org/pdf/2211.15128)
- [Leverage (statistics), Wikipedia](https://en.wikipedia.org/wiki/Leverage_(statistics))
- [Kish effective sample size formula](https://aakinshin.net/posts/kish-ess-weighted-quantiles/)
- [An information criterion for automatic gradient tree boosting (arXiv 2008.05926)](https://arxiv.org/abs/2008.05926)
- [Model Selection in Omnivariate Decision Trees, Gama & Brazdil (ECML 2005)](https://link.springer.com/chapter/10.1007/11564096_45)
- [GCV theory (Cherab-Inversion docs, Golub-Heath-Wahba 1979 derivation)](https://cherab-inversion.readthedocs.io/en/v0.3.0/user/theory/gcv.html)
