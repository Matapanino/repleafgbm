# Literature Note: Ensemble Value of an Architecturally-Diverse Mid-Pack Member

- **Date:** 2026-06-29
- **Author:** literature-scout
- **Question:** Does a mid-pack but architecturally distinct model (RepLeafGBM) add
  ensemble value when stacked with tuned GBDTs? What does the literature say about
  diversity metrics, weaker-but-diverse members, and how to design a rigorous
  experiment from the leaderboard ledgers?
- **Feeds:** `research-proposer` to write a formal experiment spec in `docs/proposals/`.

---

## 1. Rationale

Under fair same-budget HPO on the Grinsztajn 2022 numeric suites, RepLeafGBM ranks
~3.2/5 on average — never first, consistently competitive. The question is whether its
_architectural_ distinctiveness (raw-feature tree routing with representation-conditioned
leaf models — not splits on embeddings, not a NN over features) produces prediction
errors that are sufficiently decorrelated from GBDT errors to improve a hetero ensemble
even though RepLeafGBM is not the best single model.

The hypothesis has two parts:
(a) architecturally-diverse models produce lower cross-family prediction correlations
    than within-family (inter-GBDT) correlations;
(b) lower pairwise correlation with GBDT members translates into measurable ensemble
    accuracy gain when RepLeaf is added to a GBDT-only blend.

---

## 2. Sources

| Title | URL | Date |
|---|---|---|
| Wolpert, D.H. — Stacked Generalization | https://www.semanticscholar.org/paper/Original-Contribution:-Stacked-generalization-Wolpert/bbc25a700e51984e560eae27df1587baa92e3afe | 1992 |
| Krogh & Vedelsby — Neural Network Ensembles, Cross Validation, and Active Learning | https://www.semanticscholar.org/paper/Neural-Network-Ensembles,-Cross-Validation,-and-Krogh-Vedelsby/910688d01c01856dd20715907af44157de8d3d1d | NIPS 1995 |
| Dietterich — Ensemble Methods in Machine Learning | https://dl.acm.org/doi/10.5555/648054.743935 | 2000 |
| Caruana et al. — Ensemble Selection from Libraries of Models (ICML 2004) | https://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf | 2004 |
| Kuncheva & Whitaker — Measures of Diversity in Classifier Ensembles | https://link.springer.com/article/10.1023/A:1022859003006 | Machine Learning 2003 |
| Grinsztajn et al. — Why tree-based models still outperform deep learning on tabular data | https://www.researchgate.net/publication/362123616_Why_do_tree-based_models_still_outperform_deep_learning_on_tabular_data | NeurIPS 2022 |
| Erickson et al. — AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data | https://arxiv.org/abs/2003.06505 | 2020 |
| Purucker et al. — QDO-ES: Population-based Quality-Diversity Optimisation for Post-Hoc Ensemble Selection | https://arxiv.org/abs/2307.08364 | AutoML-Conf 2023 |
| Agtabular-Tabular arXiv live HTML version | https://ar5iv.labs.arxiv.org/html/2003.06505 | accessed 2026-06-29 |
| TabArena — A Living Benchmark for Machine Learning on Tabular Data | https://arxiv.org/abs/2506.16791 | 2025 |
| TabArena HTML full paper | https://arxiv.org/html/2506.16791v4 | 2025 |
| Brown — Diversity in Neural Network Ensembles (PhD/survey) | https://www.semanticscholar.org/paper/Diversity-in-neural-network-ensembles-Brown/b2329bfeaff2c9edbe4891ad56e4a4e03ad4fa59 | 2004 |
| JMLR 2023 — A Unified Theory of Diversity in Ensemble Learning | https://arxiv.org/abs/2301.03962 | 2023 |
| Kaggle Grandmasters Playbook (NVIDIA blog) | https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/ | accessed 2026-06-29 |
| Kuncheva — Combining Pattern Classifiers (2nd ed, Wiley 2014) | https://onlinelibrary.wiley.com/doi/book/10.1002/9781118914564 | 2014 |

---

## 3. Key Findings

### 3.1 Foundational theory: why diverse members help even if individually weaker

**Krogh & Vedelsby (1995) — the ambiguity decomposition.**
For a uniformly-weighted ensemble of M regressors, the ensemble MSE decomposes exactly:

```
E_ens = E_avg - A_avg
```

where `E_avg` is the average individual MSE and `A_avg = (1/M) sum_i E[(f_i - F_ens)^2]`
is the "ambiguity" — variance of members around the ensemble mean, independent of the
target. This is a hard mathematical identity (not a bound) under squared loss. It
implies:

- `E_ens < E_avg` always: the ensemble is strictly better than the average member.
- A weaker individual model (higher `E_i`) can still reduce `E_ens` if it contributes
  positive ambiguity `A_i` — as long as it does not make the same errors as the existing
  members.
- Adding a member with high individual error but low correlation to existing members
  can still lower `E_ens` if `Delta_A > Delta_E_avg`.

This is the theoretical justification for the hypothesis: RepLeafGBM does not need to be
the best single model to reduce ensemble MSE — it needs to be sufficiently decorrelated
from the GBDT members that its ambiguity contribution outweighs its per-sample error.
(Source: Krogh & Vedelsby 1995; reviewed in JMLR 2023 unified theory paper.)

**Dietterich (2000)** gives three complementary reasons for ensemble effectiveness:
statistical (averaging reduces variance), computational (multiple optima explored), and
representational (functions outside any single H_i can be expressed). The representational
argument directly applies here: RepLeafGBM's representation-conditioned leaf predictions
explore a function class not representable by any pure GBDT (axis-aligned split +
constant leaf), so it could correct errors that all GBDTs share on certain inputs.

**Caruana et al. (2004) — ensemble selection evidence.**
The ICML 2004 ensemble selection algorithm (greedy forward selection on a held-out
validation set from a library of thousands of models) routinely picks weaker individual
models because they provide complementary error coverage. The key insight is that the
marginal value of adding a model to an ensemble is NOT its standalone accuracy but its
correlation with the current ensemble's errors. A model ranking 10th individually may
be the most valuable addition to an ensemble of models ranked 1-3 if it fails on
different examples. This is directly the RepLeafGBM scenario.
(Source: Caruana et al. 2004; confirmed in "Getting the Most Out of Ensemble Selection"
follow-up paper: http://www.niculescu-mizil.org/papers/enssel_most_long.pdf)

### 3.2 Diversity metrics — what to measure and its limits

**Kuncheva & Whitaker (2003)** identified 10 pairwise and non-pairwise diversity
statistics for classifier ensembles: Q-statistic, correlation coefficient rho,
disagreement measure, double fault, entropy, difficulty index, Kohavi-Wolpert variance,
inter-rater agreement, generalized diversity, and coincident failure diversity. Their
experiments found no _consistent_ monotone relationship between any single metric and
ensemble accuracy improvement: diversity is necessary but not sufficient. The paper
cautions against treating a high diversity score as a proxy for ensemble gain.

**Practical implication for the experiment:** Do not report a single diversity number
and claim it predicts gain. Instead, (a) compute pairwise prediction correlations across
all model pairs, (b) compare the inter-GBDT correlation block with the RepLeaf-vs-GBDT
block, and (c) report actual accuracy differences and test their significance. If
RepLeaf-GBDT correlations are lower than inter-GBDT, that is the mechanistic story; the
causal test is whether adding RepLeaf improves the ensemble metric on held-out test data.

**For regression (the primary RepLeafGBM target):**
The ambiguity decomposition makes diversity measurement concrete:
- Compute per-dataset: `A = mean_individual_MSE - ensemble_MSE` (the diversity harvest).
- Compute the pairwise Pearson r of test-set predictions for all pairs.
- A lower RepLeaf-vs-GBDT r relative to inter-GBDT r is the diversity signal; positive A
  when RepLeaf is included (vs. GBDT-only ensemble) is the outcome.

**For classification (binary/multiclass):**
Pairwise correlation of probability predictions (or logit scores) is the regression
analogue. Disagreement measure (fraction of test examples on which models disagree) is
cleanest for hard-label diversity. Note that averaging probabilities requires calibration;
if individual models are not calibrated, simple average of probabilities can be
miscalibrated. The benchmarks use log-loss (cross-entropy) as primary metric for
classification, which is sensitive to calibration; for pure accuracy evaluation,
calibration matters less.

### 3.3 Prior art: weaker-but-diverse member adds value

**AutoGluon-Tabular ablation (Erickson et al. 2020).**
The paper's ablation shows that removing neural networks from the heterogeneous ensemble
causes the _largest_ performance drop of any single component removal:
`NoNetwork` raises average rescaled loss from 0.1660 to 0.8171 — far worse than
removing bagging or stacking. Neural networks in AutoGluon are not the individually
strongest model (GBDTs typically outperform standalone NNs on tabular data under the
Grinsztajn 2022 evaluation) but they contribute the most to ensemble performance because
"decision boundaries learned by neural networks differ from the axis-aligned geometry of
tree-based models." This is direct prior art for the hypothesis: mid-pack individual
performance + architectural diversity + different error regions = high ensemble marginal
value.

**TabArena (2025, NeurIPS poster).**
Confirmed at benchmark scale on 16 models: a simulated ensembling pipeline covering all
TabArena models "outperforms all individual models and AutoGluon." The paper explicitly
rejects the GBDT-vs-DL framing: "both model families contribute to ensembles that
strongly outperform individual model families." Critically, models with the highest
individual leaderboard rank are not necessarily those with the highest ensemble weights
because the ensemble construction favors models that complement each other's errors,
specifically noting that ModernNCA and RealMLP show high ensemble weights despite
moderate individual rankings.

RepLeafGBM's situation maps onto this pattern: mid-pack rank individually, but a
genuinely distinct inductive bias (raw-feature routing + representation-conditioned leaf)
that could serve the same complementarity role that NNs serve in AutoGluon.

**QDO-ES (Purucker et al. 2023).**
Diversity-aware post-hoc ensemble selection outperforms greedy ensemble selection (GES)
on 71 AutoML benchmark datasets, though only statistically significantly on validation
data. The important nuance is that "diversity can be beneficial for post-hoc ensembling
but also increases the risk of overfitting" to the validation set used for selection
weights. This is a warning about the meta-learner design choice below.

**Kaggle competition practice.**
Multi-family ensembles (GBDT + NN + linear) are standard in winning Kaggle solutions.
Grandmaster advice: "combining different model families is essential" because they "often
push performance beyond what any one model can achieve." Hill climbing (weighted blend
search on a validation set) and stacking are both used, with the common observation
that diverse baselines are more important than deep tuning of a single family.

### 3.4 Practical experiment design for RepLeafGBM

**What we have.** The leaderboard JSONL files
(`benchmarks/results/leaderboard_grinsztajn_num_reg.jsonl`,
`leaderboard_grinsztajn_num_cls.jsonl`) contain, per (dataset, model, seed):
- Best hyperparameters from 50-trial Optuna HPO.
- Test-set primary metric value and secondary metric value.
- Validation value used for HPO.
- Fit seconds, n_used.

**What we do NOT have.** Actual test-set prediction arrays. The JSONL records metric
scalars only. This means ensemble computation cannot be done purely from the JSONL: the
experiment needs to refit models with stored hyperparameters and record raw predictions.

**Recommended experiment design.**

Step 1 — Refit with stored hyperparameters.
For each (dataset, model, seed) in the JSONL, refit the model with the recorded `params`
on the same train/val/test split (use the same `seed` as the random_state and the same
n_used cap). Save the test-set predictions as arrays. This is deterministic from the
logged params and seeds, so no new HPO is needed.

Step 2 — Build ensembles from test predictions.
For each (dataset, seed), compute:
- `best_single`: best individual model by val_value among the 5 models.
- `gbdt_avg`: simple average of {LightGBM, XGBoost, CatBoost, HistGBM} predictions.
- `gbdt_repleaf_avg`: simple average of all 5 model predictions.
- `gbdt_repleaf_weighted`: greedy Caruana-style forward selection on the validation
  predictions (requires saving val predictions too) — optional higher-effort variant.

For regression: convert predictions to RMSE. For binary classification: average
probabilities (check per-model calibration first; consider using logit average if
uncalibrated). For multiclass: average probability vectors.

Step 3 — Diversity measurement.
For each (dataset, seed), compute the 5x5 pairwise Pearson correlation matrix of
test predictions (regression) or log-odds (classification). Report the inter-GBDT block
mean r and the RepLeaf-vs-GBDT block mean r. Report the ambiguity harvest:
`A = mean_individual_RMSE^2 - ensemble_RMSE^2` for the two ensembles.

Step 4 — Statistical comparison.
Compare metric values across datasets using Wilcoxon signed-rank test (consistent with
the Friedman + Wilcoxon protocol already in the harness). Compare:
- `gbdt_repleaf_avg` vs `gbdt_avg` (primary test: does RepLeaf add value?)
- `gbdt_repleaf_avg` vs `best_single` (is the ensemble also best overall?)
Report effect size (mean rank difference) and p-value. With 5 seeds x N datasets, power
depends on N; the Grinsztajn numeric suites give ~7 reg + ~7 cls datasets.

**A defensible "RepLeaf adds ensemble value" claim requires:**
1. `gbdt_repleaf_avg` significantly beats `gbdt_avg` on Wilcoxon across datasets (p <
   0.05, corrected if multiple suites).
2. RepLeaf-vs-GBDT pairwise r is systematically below inter-GBDT r (diversity story).
3. The ambiguity harvest increases when RepLeaf is added (diversity is being exploited).
4. Results hold across both regression and classification suites (not a single domain).

**Anti-patterns to avoid.**

*Leakage via stacking on test:* Never fit a meta-learner's weights using test-set
predictions. If testing OOF stacking, the meta-learner must be fitted on OOF train
predictions; the test is evaluated once only with those meta weights.

*Calibration for probability averaging (classification):* The Grinsztajn suite primary
metric is log-loss. If any model's predicted probabilities are systematically too
confident or too uncertain, simple averaging can hurt. Consider: (a) use logit-averaging
instead of probability-averaging, or (b) apply isotonic regression or temperature scaling
per model on the validation set before averaging.

*Validation-set overfitting for weighted blends:* If fitting blend weights on validation,
report the test metric (not val) as the primary claim. The QDO-ES paper warns explicitly
that diversity-aware ensemble selection overfits validation in a way that does not
generalize fully to test.

*Small-N significance:* With 7 datasets the Wilcoxon has limited power. Supplement with
per-dataset rank tables and bootstrap confidence intervals on the mean rank difference.

---

## 4. Relevance to RepLeafGBM

### Code map

| Concern | Module |
|---|---|
| Refit models with stored params | extend `benchmarks/benchmark_grinsztajn.py` or new script |
| Save test/val predictions | new `--save-predictions` flag on the harness |
| OOF stacking | `src/repleafgbm/external/oof.py` (already implemented) |
| Augment features (stacking) | `src/repleafgbm/external/features.py` (already implemented) |
| External base model interface | `src/repleafgbm/external/lightgbm_model.py`, `xgboost_model.py`, `catboost_model.py` |
| OOF stacking recipe example | `examples/stacking_lightgbm.py` |
| Leaderboard ledger schema | `benchmarks/results/leaderboard_grinsztajn_num_reg.jsonl` |

All necessary external model utilities (`oof_predictions`, `augment_features`,
`LightGBMExternalModel`, `XGBoostExternalModel`, `CatBoostExternalModel`) are
already implemented. The main harness gap is that predictions are not saved — only
metric scalars are logged to the JSONL. The experiment requires either: (a) a new script
that refits with stored hyperparameters and records predictions, or (b) extending the
harness to optionally dump prediction arrays.

The RepLeafGBM `external/` module's `oof_predictions` function already handles K-fold
OOF correctly (no leakage) and works for any estimator, including the native
`RepLeafRegressor`/`RepLeafClassifier`. The `examples/stacking_lightgbm.py` is a
working end-to-end recipe that the experiment can adapt.

### Roadmap

- `docs/backend_strategy.md` explicitly names "hybrid ensemble" as a planned mode:
  "same encoder family, different tree-building algorithms, combined to maximize
  diversity" — the ensemble-diversity experiment directly validates whether this matters
  empirically.
- The `v0.2` stacking utilities were built for exactly this use case.
- Phase 25 (OpenML benchmark) already showed that the `adaptive` leaf RepLeafGBM arm is
  competitive with GBDTs; the ensemble-diversity experiment tests whether that competitiveness
  plus architectural diversity is enough for ensemble value.

---

## 5. Guardrail check

All findings below are COMPATIBLE with the thesis. No violated invariants.

- The proposed experiment uses RepLeafGBM in its standard mode: raw-feature routing,
  frozen encoder (identity by default), representation-conditioned leaf. No change to
  the core architecture.
- Simple prediction averaging and OOF stacking do not require unfreezing the encoder or
  splitting on embeddings.
- The external model utilities are already behind the optional `[external]` dependency
  boundary; the native path is not touched.
- The `oof_predictions` utility fits each fold's model from scratch — no leakage,
  encoder stays frozen, stage-wise assumptions hold per fold.
- Stacking with a meta-learner over OOF predictions is philosophically distinct from
  updating the encoder during boosting; the stage-wise assumption within each base model
  is untouched.

One design choice to flag to research-proposer: if the meta-learner is another
RepLeafGBM model fitted on OOF predictions as input features, the encoder pretraining
target for that second-stage model is the stacking residual, not the original target.
The docs/math.md "supervised encoder pretraining target" section applies in that case —
fine, but worth noting.

---

## 6. Concrete next steps for research-proposer

**Step A (required): design the prediction-saving extension to the benchmark harness.**
The harness already has the HPO ledger. Add a `--refit-and-save-predictions` mode to
`benchmarks/benchmark_grinsztajn.py` (or a companion script) that loads best params from
the JSONL and refits each (dataset, model, seed) triple, saving `{key}_test_preds.npy`
and `{key}_val_preds.npy` alongside the JSONL. This is purely additive to the harness
and involves no changes to `src/`.

**Step B (primary experiment): ensemble comparison study.**
Script: `experiments/ensemble_diversity_study.py`.
Inputs: prediction arrays from Step A.
Outputs: (i) metric table — best_single, gbdt_avg, gbdt+repleaf_avg per dataset x seed;
(ii) diversity matrix — 5x5 pairwise prediction correlations per dataset x seed;
(iii) ambiguity harvest — `A` per dataset x seed for both ensemble variants;
(iv) Wilcoxon p-value for `gbdt+repleaf_avg` vs `gbdt_avg` across datasets.
The report lands in `experiments/results/`.

**Step C (optional, higher-effort): OOF stacking with meta-learner.**
If Step B shows positive simple-average gain, extend to OOF stacking with a ridge
meta-learner (or another RepLeafRegressor with `leaf_model="constant"`) fitted on
concatenated OOF probability/prediction vectors. Use `oof_predictions` and
`augment_features` from the existing `external/` utilities. Compare OOF-stacked vs
simple-average to quantify whether a learnable combination adds beyond uniform weighting.
This answers whether RepLeafGBM can also _discover_ a better combination rule.

---

## 7. Summary judgment

The literature provides strong theoretical and empirical support for the hypothesis.
Theoretically, the Krogh-Vedelsby ambiguity decomposition guarantees that a weaker but
decorrelated member reduces ensemble MSE if its ambiguity contribution exceeds its error
penalty. Empirically, AutoGluon's ablation shows that architecturally distinct NNs
provide the most irreplaceable ensemble value even when they are not individually the
best model on tabular data. TabArena (2025) confirms this at benchmark scale. The
Caruana ensemble selection algorithm routinely picks non-SOTA members for diversity.

The open question is whether RepLeafGBM's architectural diversity from GBDTs is large
enough to manifest as measurable prediction decorrelation. That question is empirical and
not answerable from the literature alone — it requires the experiment in Step B.

If the prediction correlation of RepLeafGBM with any given GBDT is similar to the
inter-GBDT correlation (a plausible null, since RepLeaf still builds trees on raw
features), the diversity case collapses. If instead RepLeaf's representation-conditioned
leaf model systematically changes the residual structure relative to GBDT constant
leaves, the correlation will be lower and the diversity gain will be real.

The experiment is low-cost (no new HPO), builds directly on existing infrastructure, and
is a defensible scientific claim if Step B shows a Wilcoxon-significant improvement
across the Grinsztajn datasets. Hand off to research-proposer for the formal spec.
