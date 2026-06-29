# Consolidated verdict — Fair same-budget leaderboard across ALL FOUR Grinsztajn 2022 suites

**Analyst:** results-analyst · **Date:** 2026-06-29
**Supersedes/extends:** `experiments/results/2026-06-27-leaderboard-verdict.md` (numerical-only).

**Inputs (all four suites complete):**
- `experiments/results/leaderboard-grinsztajn_num_reg.md` — numerical regression, 19 datasets (run 20260626T024858Z, git f6dc44a-dirty)
- `experiments/results/leaderboard-grinsztajn_num_cls.md` — numerical binary, 16 datasets (run 20260626T160115Z, git 5514a3a-dirty)
- `experiments/results/leaderboard-grinsztajn_cat_reg.md` — categorical regression, 17 datasets (run 20260627T111344Z, git 607d8d9-dirty)
- `experiments/results/leaderboard-grinsztajn_cat_cls.md` — categorical binary, 7 datasets (run 20260627T111344Z, git 607d8d9-dirty)

**Common protocol:** `benchmarks/leaderboard.py` + `benchmarks/hpo.py`. Identical 50-trial
Optuna budget per family (repleaf / lightgbm / xgboost / catboost / hist_gradient_boosting),
70/15/15 split (train capped 10k, max_rows 20k), 5 seeds, score-once-on-test. Friedman omnibus
+ Nemenyi CD + Wilcoxon-vs-best-baseline + per-dataset win/tie/loss at 1% MRD. Single local
environment (macOS, Rust backend, `OMP_NUM_THREADS=1`). RepLeaf arm tunes
`leaf_model ∈ {constant, embedded_linear, adaptive}` × `encoder ∈ {identity, plr}`;
learned `torch_*` encoders deliberately excluded from the budgeted search.

Total: **59 dataset-slots across 4 suites** (= 19+16+17+7; the "~58" headline is off-by-one,
and unique datasets are fewer still — diamonds / abalone / medical_charges / nyc-taxi /
delays_zurich / Brazilian_houses / Bike_Sharing / house_sales / covertype / electricity /
eye_movements / default-of-credit-card-clients recur across suites).

---

## Question

With all four Grinsztajn suites now run under one fair, identical-budget protocol, where does
RepLeafGBM land overall — statistically and practically — and does anything here warrant a
model-behavior default change?

---

## Evidence — the four suites side by side

| suite | n | RepLeaf rank | RepLeaf avg rank | leader (avg rank) | Friedman p | RepLeaf clique status | Wilcoxon p vs best baseline | win/tie/loss @1% MRD |
|---|---|---|---|---|---|---|---|---|
| num_reg | 19 | 4th | **3.32** | catboost (2.11) | **1.6e-4 (sig)** | in top non-sig clique {cat, lgb, xgb, repleaf} | 0.0546 (NS) vs catboost | 2 / 6 / 11 |
| num_cls | 16 | 4th | **3.38** | lightgbm (1.75) | **3.3e-3 (sig)** | chasing clique {xgb, cat, repleaf, hgb}; Nemenyi-separated from lgb only | 0.0092 (sig sign) vs lightgbm | 0 / 13 / 3 |
| cat_reg | 17 | 3rd | **3.18** | lightgbm = catboost (2.59) | 0.375 (NS) | all-five-in-one clique | 0.517 (NS) vs lightgbm | 4 / 10 / 3 |
| cat_cls | 7 | 5th | **3.43** | lightgbm (2.43) | 0.788 (NS) | all-five-in-one clique (CD = 2.305) | 0.688 (NS) vs lightgbm | 2 / 4 / 1 |

**Reading the consolidated picture:**

1. **Tight mid-pack band.** RepLeaf's average rank across all four suites sits in a narrow
   **3.18–3.43** window. For five models the pack midpoint is 3.0, so RepLeaf is consistently
   *just below the midpoint* — squarely mid-pack, never the average leader anywhere.

2. **Never the top model on average; never significantly-and-practically worse than the best.**
   In no suite is RepLeaf rank 1. In no suite does it fail the combined Wilcoxon-AND-MRD gate
   against the best baseline. The single "significant in sign" cell (num_cls vs lightgbm,
   p = 0.0092) is the textbook *significant-but-negligible* pattern: win/tie/loss 0/13/3 means
   lightgbm is a consistent hair better in sign but materially better (>1%) on only 3 of 16.

3. **Significance lives only in the numerical suites.** Friedman detects *any* model difference
   only on num_reg (p = 1.6e-4) and num_cls (p = 3.3e-3). **Both categorical suites show NO
   significant difference among any of the five models** (p = 0.375, p = 0.788). On categorical
   data this protocol simply cannot separate RepLeaf from CatBoost / LightGBM / XGBoost / HGB —
   "competitive" there means "statistically indistinguishable," for everyone.

4. **Relatively stronger on categorical than numerical (in raw win-count), though never
   significant.** vs each suite's best baseline the practical win/tie/loss is net-negative on
   numerical (2/6/11; 0/13/3) but net-positive in raw wins on categorical (4/10/3; 2/4/1).
   The cat_reg 4/10/3 vs lightgbm (p = 0.517) is RepLeaf's most "competitive-looking" showing —
   but it sits inside a no-difference omnibus (Friedman p = 0.375), so it is competitiveness by
   indistinguishability, not by a demonstrated edge.

5. **Where RepLeaf wins outright is consistent with the standing thesis.** Its rank-1 datasets
   skew to smooth / high-structure targets: diamonds (R²≈.99), medical_charges (R²≈.98) in
   num_reg; visualizing_soil (R²≈1.0), diamonds, SGEMM_GPU (R²≈.9998), medical_charges in
   cat_reg; covertype in cat_cls. This matches the established verdict that
   representation-conditioned leaves help on smooth/structured signal.

---

## (1) Where RepLeafGBM lands across all of Grinsztajn

RepLeafGBM is a **consistent, robust mid-pack tabular learner**: average rank 3.18–3.43 out of
five across 4 suites / 59 dataset-slots, inside the top non-significant clique on numerical
regression and both categorical suites, and inside the chasing clique on numerical binary
(separated only from lightgbm, by a within-MRD margin). It is **never the average leader and
never significantly-plus-practically worse than the best baseline.** Practically it is a small
genuine step behind catboost on numerical regression (>1% on 11/19), a near-wash with lightgbm
on numerical binary (within 1% on 13/16), and statistically indistinguishable from everyone on
both categorical suites.

## (2) "Competitive-but-not-SOTA-on-average" — confirmed, with a much stronger base

**Confirmed, high confidence.** The numerical-only verdict (2026-06-27) is upheld and broadened
to all four suites: 59 dataset-slots vs 35, four suites vs two, the same rigorous protocol
(identical HPO budget, CD + Wilcoxon + MRD). The standing roadmap/paper framing — *"competitive
but not state-of-the-art on average; defensible support in niche regimes (robust multi-output,
router extraction)"* — is exactly what the consolidated evidence shows. This also corroborates
Phase 25 (OpenML: CatBoost leads overall; RepLeaf competitive on classification) under a far
more rigorous design. **No standing verdict is overturned.** RepLeaf is competitive; it is not
SOTA on average; it is significantly *better* than the best baseline **nowhere**.

## (3) The categorical-fairness caveat — CORRECTED by reading the harness

The task framed a caveat that "the harness gives RepLeaf native categorical subset splits while
the GBDT baselines get the ordinal-encoded matrix (an asymmetry favoring RepLeaf)." **Reading
the code, that asymmetry does NOT exist in this leaderboard harness.** The harness:

- builds `RepLeafDataset(Xtr, ytr, categorical_features=cats)` only to produce the ordinal codes,
  then extracts `Xtr_e = get_raw_features()` — a bare numpy float matrix
  (`benchmarks/leaderboard.py:112–115`, `src/repleafgbm/data/dataset.py:104–106`);
- feeds that **same ordinal-encoded array to every family**, RepLeaf included, both during
  tuning and final refit (`benchmarks/hpo.py:213–215`, `benchmarks/leaderboard.py:127`);
- builds the RepLeaf estimator with **no `categorical_features`** argument
  (`benchmarks/hpo.py:148–152`), so `fit` on a numpy array routes through
  `RepLeafDataset(X, y)` with no categorical metadata (`src/repleafgbm/sklearn.py:479–486`) →
  **RepLeaf treats the ordinal codes as ordered numerics and never engages native subset
  splits.** This is the harness's deliberate, documented design ("every model still sees the
  same ordinal-encoded matrix", `benchmarks/suites.py:12`, `benchmarks/hpo.py:13`).

So the categorical suites are a **symmetric ordinal-code fight** — clean and fair, with *no*
asymmetry favoring RepLeaf. Two consequences, both honest:

- **What it lets us claim:** RepLeaf's categorical standing (3rd cat_reg, 5th cat_cls, both in
  no-difference cliques) is *not* inflated by any native-categorical advantage. It is a fair
  result on equal footing. The user's worry — "it gets its special weapon and is still only
  mid-pack" — is *moot*: the special weapon was never drawn.
- **What it does NOT let us claim:** these suites neither validate nor refute RepLeaf's native
  gradient-sorted categorical subset splits (Phase 8) — that feature is simply untested here.
  Symmetrically, **CatBoost's and LightGBM's native categorical machinery is also OFF** (they
  too get ordinal codes as numerics), so the categorical suites *under-use the GBDTs'*
  categorical strength — if anything they understate CatBoost's true categorical ceiling, not
  RepLeaf's. The asymmetry the task describes IS real in the library's *recommended* categorical
  path (the `RepLeafDataset` + `categorical_features` route used in the older
  real-data/OpenML harnesses) — but it was deliberately neutralized in this fair leaderboard.

**Action item flowing from this:** the standing belief / any report text asserting "the
leaderboard gives RepLeaf a native-categorical edge" should be corrected to "the leaderboard
runs all models on the identical ordinal-encoded matrix; no model's native-categorical path is
engaged."

## (4) Statistical-power caveats

- **5 seeds, not 10.** Limits CD/Wilcoxon power; the two load-bearing numerical p-values are
  borderline (num_reg 0.0546, num_cls 0.0092) and could cross threshold with more seeds —
  without changing the qualitative mid-pack story.
- **cat_cls is effectively powerless: 7 datasets → Nemenyi CD = 2.305** (nearly the full rank
  range) and **Friedman p = 0.788.** RepLeaf's "5th" there spans ranks 2.43–3.43 inside a
  single no-difference clique; that rank ordering is **noise, not signal** and must not be
  quoted as a standalone result.
- **cat_reg also non-significant** (Friedman p = 0.375): every model is in one clique; "3rd" is
  positional, not a demonstrated difference.
- **50-trial budget is a CPU substitute** for Grinsztajn's compute-hour random search. Equal
  trial count ≠ equal wall-clock; RepLeaf's space carries two extra categorical branches
  (`leaf_model`, `encoder`) splitting the 50 trials across sub-spaces the pure-continuous GBDT
  spaces don't have → may mildly *under*-tune RepLeaf (true ceiling possibly slightly higher).
- **Small-data regime** (train ≤ 10k, max_rows 20k); large-n scaling unprobed.
- **Single environment**, and **manifests are git-dirty** (f6dc44a / 5514a3a / 607d8d9, all
  dirty) — reproduce on a clean tree with pinned SHAs before any paper table.

## (5) Default change

**No model-behavior default change is warranted — high confidence.** Same structural argument as
the numerical-only verdict, now reinforced: this is a *positioning* study, not a default
experiment. The RepLeaf arm re-tunes `leaf_model` × `encoder` every Optuna trial, so no outcome
is attributable to any single default (`encoder="identity"`, `leaf_model="embedded_linear"`,
`grow_policy="leafwise"` — confirmed current in `src/repleafgbm/sklearn.py:208,210,214` — remain
untouched and unchallenged). The reports log only the winning composite, not which arm won, so
they cannot even reveal the within-RepLeaf best config. Learned `torch_*` encoders were excluded,
so nothing here speaks to the identity-vs-learned-leaf verdicts (Phases 14/16/25). **Keep all
current defaults.**

## (6) Recommended next steps for a paper-grade claim

1. **experiment-runner — rerun all four suites at 10 seeds.** Top priority: firms the two
   borderline numerical p-values and adds power to the categorical omnibus. Gating run before any
   paper table.
2. **harness-optimizer — decide and document the categorical protocol.** Either (a) keep the
   symmetric ordinal-code protocol and state plainly in the paper that *native-categorical
   handling is OFF for all models* (fair, simple); or (b) add a *second* categorical comparison
   where each model uses its native categorical path (RepLeaf via `RepLeafDataset` +
   `categorical_features`; CatBoost via `cat_features`; LightGBM via `categorical_feature`) — the
   more interesting science, and the only way to actually test the native-categorical thesis.
   Either way, correct the "asymmetry favoring RepLeaf" framing (see §3).
3. **cat_cls — expand or demote.** Either run the fuller Grinsztajn categorical-classification
   set or drop cat_cls from any headline claim; 7 datasets / p = 0.788 carries no weight.
4. **harness-optimizer — log per-trial winning `leaf_model`/`encoder` distribution.** This is the
   artifact that could *eventually* inform whether `embedded_linear` should stay the default
   leaf; today's reports cannot speak to it.
5. **Reproduce on a clean (non-dirty) tree, pin SHAs**, before quoting in the paper.
6. **Run the niche studies at scale** (robust multi-output huber/quantile under contamination;
   router extraction) at the same fair-budget rigor — that is where RepLeaf's *defensible* edge
   lives, and a paper that pairs "competitive mid-pack on Grinsztajn" with "decisive in these
   niches" is far stronger than the leaderboard alone.

---

### One-line verdict

**Competitive, robustly mid-pack (avg rank 3.18–3.43 of 5 across 59 Grinsztajn dataset-slots),
SOTA nowhere, significantly-and-practically worse than the best baseline nowhere; categorical
results are a fair symmetric ordinal-code fight with no RepLeaf advantage engaged (correcting the
stated caveat). No default change. Next: 10-seed rerun + an explicit categorical-protocol
decision.** Confidence: high for the positioning and for "no default change"; the exact rank
ordering is provisional (5 seeds; cat suites non-significant; cat_cls powerless).
