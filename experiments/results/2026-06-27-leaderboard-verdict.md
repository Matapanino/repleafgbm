# Verdict — Fair same-budget leaderboard (Grinsztajn numerical reg + binary)

**Analyst:** results-analyst · **Date:** 2026-06-27
**Inputs:** `experiments/results/leaderboard-grinsztajn_num_reg.md` (run_id 20260626T024858Z, git f6dc44a-dirty),
`experiments/results/leaderboard-grinsztajn_num_cls.md` (run_id 20260626T160115Z, git 5514a3a-dirty).
Harness: `benchmarks/leaderboard.py` + `benchmarks/hpo.py`. 5 seeds, 70/15/15 split,
identical 50-trial Optuna budget per family, score-once-on-test. RepLeaf arm tunes
`leaf_model ∈ {constant, embedded_linear, adaptive}` × `encoder ∈ {identity, plr}`
(learned `torch_*` encoders deliberately excluded from the budgeted search).

## Question

Under fair, identical-budget HPO against tuned LightGBM / XGBoost / CatBoost /
HistGradientBoosting, where does RepLeafGBM actually land on the Grinsztajn numerical
suites — statistically and practically — and does the result warrant any default change?

## Evidence

### Regression (19 datasets)
- Friedman χ² = 22.53, p = 1.6e-4 (models differ). Nemenyi CD = 1.399.
- Mean ranks: catboost **2.11** < lightgbm 2.32 < xgboost 3.00 < **repleaf 3.32** < histgb 4.26.
- RepLeaf sits in the **top non-significant clique {catboost, lightgbm, xgboost, repleaf}**
  (rank gap to best = 1.21 < CD 1.40). HistGB is the only model separably worse.
- Wilcoxon repleaf-vs-catboost (best baseline): **p = 0.0546 — not significant**.
- Practical (1% MRD band) vs catboost: win/tie/loss **2 / 6 / 11**. So on 11/19 datasets
  catboost beats repleaf by >1% relative RMSE, but the signed magnitudes are small/mixed
  enough that the paired test does not reach significance. For comparison xgboost is 1/9/9
  and histgb 0/7/12 vs the same baseline — repleaf's practical profile sits between them,
  matching its 4th rank.
- RepLeaf is listed **1st** on diamonds (0.2334) and medical_charges (0.0783); effectively
  tied-1st (within rounding) on elevators; 2nd on abalone. Clear bottom only on sulfur and
  yprop_4_1.

### Binary (16 datasets)
- Friedman χ² = 15.80, p = 3.3e-3 (models differ). Nemenyi CD = 1.525.
- Mean ranks: lightgbm **1.75** < xgboost 3.00 = catboost 3.00 < **repleaf 3.38** < histgb 3.88.
- Cliques: {lightgbm, xgboost, catboost} and **{xgboost, catboost, repleaf, histgb}**.
  RepLeaf is in the chasing clique and is **Nemenyi-separated from lightgbm** (rank gap
  1.625 > CD 1.525), but statistically tied with xgboost / catboost / histgb.
- Wilcoxon repleaf-vs-lightgbm: **p = 0.0092 — significant in sign**, but win/tie/loss
  **0 / 13 / 3**: repleaf never beats lightgbm by >1%, ties within 1% MRD on 13/16, loses
  >1% on only 3. This is the textbook *statistically-significant-but-practically-negligible*
  pattern — lightgbm is consistently a hair better in sign, almost never materially better.
- RepLeaf is listed **1st** on bank-marketing (0.4207) and Bioresponse (0.4788); 2nd on
  electricity, covertype, Higgs. Bottom only on credit, MiniBooNE, Diabetes130US.

## Verdict

1. **Competitive, not SOTA-on-average — and now quantified.** RepLeafGBM is inside the
   leading non-significant clique on numerical regression and the chasing clique on binary,
   never the top model on average (catboost leads regression, lightgbm leads binary), and
   only *separably* behind one model in either suite (lightgbm on binary, by a within-MRD
   margin). It is never the worst — it beats HistGB on regression and is practically tied
   with the xgboost/catboost pack on binary. **Statistically:** indistinguishable from the
   best on regression (clique + Wilcoxon p = 0.055), separated from but practically tied
   with the binary leader. **Practically (MRD):** a genuine but small step behind
   catboost/lightgbm on regression (>1% on ~11/19), a near-wash vs lightgbm on binary
   (within 1% on 13/16).

2. **Consistent with — and a stronger version of — the standing "competitive but not SOTA"
   claim.** This corroborates Phase 25 (OpenML: CatBoost leads overall; RepLeaf competitive
   on classification) under a more rigorous protocol: 35 datasets vs 9, 5 seeds, *identical*
   HPO budget per family, CD diagrams + Wilcoxon + MRD gating instead of tuned-vs-default
   ranks. The paper/roadmap framing ("competitive but not state-of-the-art on average; niche
   support in robust multi-output and router-extraction regimes") is upheld and now better
   evidenced. No standing verdict is overturned.

3. **No model default change is warranted.** This is a *positioning* result, not a
   default-behavior experiment, and structurally it **cannot justify a default change**: the
   repleaf arm re-tunes `leaf_model` and `encoder` per Optuna trial, so the outcome is not
   attributable to any single default (encoder=identity, leaf_model=embedded_linear remain
   untouched and unchallenged). The report logs only the winning composite, not which arm
   won — so it does not even reveal the within-RepLeaf best configuration. The learned
   `torch_*` encoders were excluded from the search, so this run says **nothing new** about
   the identity-vs-learned-leaf verdicts (Phases 14/16/25) either. **Keep all current
   defaults.**

4. **Confidence: medium-high for the positioning, high for "no default change."** The
   design is sound (no test leakage, omnibus + post-hoc + practical band). The exact ordering
   is *provisional*: both decisive p-values are borderline (regression 0.0546, binary 0.0092)
   and rest on 5 seeds × 16–19 datasets, so a 10-seed rerun could move repleaf across either
   threshold without changing the qualitative story.

## Caveats (honest)

- **5 seeds, not 10** — limits CD/Wilcoxon power; the two load-bearing p-values are
  borderline and could flip with more seeds.
- **Numerical suites only.** The Grinsztajn categorical suites (334/335, reg + cls) are
  **not yet run**. RepLeaf's native gradient-sorted categorical subset splits are untested
  here, and categorical-heavy data is exactly where CatBoost's edge — or RepLeaf's
  raw-feature routing — could shift the ranking. The positioning is incomplete until those run.
- **50-trial budget is a CPU substitute** for Grinsztajn's compute-hour random search
  (~hundreds of iters). Equal trial count ≠ equal wall-clock, and RepLeaf's search space has
  two extra categorical branches (`leaf_model`, `encoder`) splitting the 50 trials across
  sub-spaces the pure-continuous GBDT spaces don't have — this can mildly *under*-tune RepLeaf
  relative to the GBDTs, i.e. its true ceiling may be slightly higher than shown.
- **Small-data regime:** train capped at 10k rows, max_rows 20k. Large-n scaling is not probed.
- **Manifests are git-dirty** (f6dc44a / 5514a3a, dirty=True) — uncommitted changes at run
  time; reproduce against a clean tree before quoting in the paper.

## Next action + owner

- **experiment-runner** — run the Grinsztajn **categorical** suites (334/335, cls + reg)
  through the same `benchmarks/leaderboard.py` harness, ideally at **10 seeds** for the
  borderline cells, to complete the positioning and firm up the two marginal p-values. This
  is the gating run before any paper-table claim of "competitive across Grinsztajn."
- **harness-optimizer (optional, default-relevant follow-up):** have the leaderboard log the
  per-trial winning `leaf_model`/`encoder` distribution across datasets. That — not this run —
  is the artifact that could eventually inform whether `embedded_linear` should remain the
  default leaf; today's reports cannot speak to it.
- **No core change.** Do not touch `src/` defaults on the basis of this run.
