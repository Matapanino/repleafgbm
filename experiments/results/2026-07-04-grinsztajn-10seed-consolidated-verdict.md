# Consolidated verdict — 10-seed fair leaderboard across ALL FOUR Grinsztajn 2022 suites

**Analyst:** results-analyst · **Date:** 2026-07-04
**Supersedes:** `experiments/results/2026-06-29-grinsztajn-consolidated-verdict.md` (5-seed).
Its step-1 demand was *exactly* this 10-seed rerun; this verdict discharges it.

**Inputs (all four suites complete, fresh run):**
- `benchmarks/results/colab10seed/leaderboard-grinsztajn_num_reg-10seed.md` — numerical regression, 19 datasets
- `benchmarks/results/colab10seed/leaderboard-grinsztajn_num_cls-10seed.md` — numerical binary, 16 datasets
- `benchmarks/results/colab10seed/leaderboard-grinsztajn_cat_reg-10seed.md` — categorical regression, 17 datasets
- `benchmarks/results/colab10seed/leaderboard-grinsztajn_cat_cls-10seed.md` — categorical binary, 7 datasets
- CD PNGs sit next to each report: `leaderboard-cd-grinsztajn_{suite}-{regression|binary}.png`

---

## Provenance (recorded verbatim)

- **2,950 / 2,950 cells complete** = (19 + 16 + 17 + 7) datasets × 5 model families × 10 seeds
  = 59 dataset-slots × 50 cells.
- **50 Optuna TPE trials per model per dataset** — identical budget for every family
  (repleaf / lightgbm / xgboost / catboost / hist_gradient_boosting).
- **70/15/15 split** (train stratified for classification, train capped at 10k).
- **`max_rows = 20000`** (dataset cap before split). **NOTE — the paper appendix currently says
  10,000; this is a required fix** (see paper edit #6; the 10k that is correct is the *train* cap,
  not the dataset cap).
- **Single local environment**: macOS arm64, Python 3.11.1, numpy 1.26.4, scikit-learn 1.9.0,
  lightgbm 4.6.0, xgboost 3.2.0, catboost 1.2.10, optuna 4.6.0, `OMP_NUM_THREADS=1`,
  pandas 1.5.2, scipy 1.10.0, matplotlib 3.6.2.
- **Git worktree pinned at `c7caa46`**, `dirty=False` in the manifest (only dirtiness = untracked
  run logs). **80 early cells were computed on a Colab VM with different numpy/sklearn and were
  DISCARDED and recomputed locally** for single-environment purity.
- **Resumable sharded ledgers merged and verified complete by key**:
  `benchmarks/results/colab10seed/merged_grinsztajn_{num_reg,num_cls,cat_reg,cat_cls}.jsonl`.
  Honesty note: raw merged line counts are 950 / 801 / 851 / 351 = 2,953; the 3 surplus rows are
  duplicate keys from the shard-resume merge and collapse to **2,950 unique keys** — the report
  tables are built from the deduped grid, so "2,950/2,950" stands.

---

## (1) Four-suite summary — 5-seed vs 10-seed

Lower average rank is better (5 model families → pack midpoint = 3.0). "vs best" = Wilcoxon
signed-rank of RepLeaf against each suite's best-avg-rank baseline; RepLeaf is the *bold*
(significant-and-practical) winner in **no** suite, so every cell is "not sig." for a RepLeaf win.

| suite | n | rank 5→10 | place 5→10 | Friedman p 5→10 | Wilcoxon vs best 5→10 | 10-seed w/t/l vs best @1% MRD |
|---|---|---|---|---|---|---|
| num_reg | 19 | 3.32 → **3.000** | 4th → **3rd** | 1.6e-4 → **2.19e-3** (sig) | 0.055 vs cat → **0.374 vs cat** (NS) | 1 / 10 / 8 |
| num_cls | 16 | 3.38 → **3.812** | 4th → 4th | 3.3e-3 → **4.26e-4** (sig) | 0.0092 vs lgb → **3.36e-3 vs lgb** (sig-in-sign, RepLeaf *worse*) | 1 / 13 / 2 |
| cat_reg | 17 | 3.18 → **3.000** | 3rd → 3rd | 0.375 FLAT → **0.0249** (now SIG) | 0.517 vs lgb → **0.207 vs cat** (NS) | 3 / 9 / 5 |
| cat_cls | 7 | 3.43 → **3.714** | 5th → 5th | 0.788 FLAT → **0.438** (still FLAT) | 0.688 vs lgb → **0.219 vs lgb** (NS) | 1 / 5 / 1 |

10-seed leaders / CD / cliques (RepLeaf clique membership in brackets):
- **num_reg**: catboost 2.211 · lightgbm 2.368 · **repleaf 3.000** · xgboost 3.421 · hgb 4.000; CD 1.399;
  RepLeaf in top clique {cat, lgb, repleaf, xgb} — **not separated from the leader (catboost)**.
- **num_cls**: lightgbm 1.875 · catboost 2.562 · xgboost 2.750 · **repleaf 3.812** · hgb 4.000; CD 1.525;
  top clique {lgb, cat, xgb}, RepLeaf clique {cat, xgb, repleaf, hgb} — **RepLeaf IS Nemenyi-separated
  from the single leader lightgbm** (as it was at 5 seeds).
- **cat_reg**: catboost 2.412 · lightgbm 2.471 · **repleaf 3.000** · xgboost 3.118 · hgb 4.000; CD 1.479;
  RepLeaf in top clique {cat, lgb, repleaf, xgb} — **not separated from the leader**.
- **cat_cls**: lightgbm 2.429 = catboost 2.429 · xgboost 3.000 · hgb 3.429 · **repleaf 3.714**; CD 2.305
  (nearly the full rank range) — **all five in one clique**; the "5th" is noise, not signal.

Outright rank-1 datasets for RepLeaf (thesis check): num_reg → cpu_act, diamonds, medical_charges,
abalone (4); num_cls → covertype (1); cat_reg → visualizing_soil, diamonds, SGEMM_GPU_kernel_performance,
abalone, medical_charges (5); cat_cls → none. The wins skew smooth / very-high-R² targets
(diamonds ≈.99, medical_charges ≈.98, SGEMM ≈.9998, visualizing_soil ≈1.0), exactly consistent with
the standing verdict that representation-conditioned leaves help on smooth/structured signal.

---

## (2) Evidence-backed reading — does "consistent mid-pack" survive?

**Partly. The qualitative claim survives; the tight-band claim does not, and a clean
regression-vs-classification split emerges.**

- **Survives:** RepLeaf is the average leader in **no** suite, is **never** both significantly and
  practically worse than the best baseline, and sits inside (or one Nemenyi step out of) the top
  non-significant clique in every suite. Real-data, 10 seeds, single clean env — this carries weight.
- **Does NOT survive:** the paper's "mean rank **3.18–3.43** in *every* suite" is now false. The
  10-seed band is **3.00–3.81**, and it is not a uniform mid-pack blur — it splits by task type.

**The regression-vs-classification split is the headline finding of the rerun.**
Averaging the two regression suites gives **3.00** (clear 3rd, ahead of xgboost and hgb in *both*);
averaging the two classification suites gives **3.76** (4th and 5th). At 5 seeds this asymmetry was
buried in noise (3.32/3.18 vs 3.38/3.43 ≈ flat); 10 seeds resolves it into a ~0.76-rank gap.

**This is a direct, predicted consequence of the documented binary-Hessian limitation** (Phase 12;
paper §"binary classification"). The logistic Hessian `h = p(1-p) ≤ 0.25` starves the `h`-weighted
ridge fit of the embedded-linear leaf, so RepLeaf's one differentiator — leaf expressivity — is muted
exactly on binary targets. On regression the Hessian is well-conditioned (`h = 1` for squared error),
the leaf fit is healthy, and RepLeaf climbs to a clean, ahead-of-half-the-field 3rd. The 10-seed data
is the clearest empirical confirmation to date that the leaf-expressivity edge is a
**regression-side** phenomenon.

**Two honest downgrades at 10 seeds (RepLeaf got worse):**
- **num_cls 3.38 → 3.812** (still 4th). The Wilcoxon deficit vs lightgbm *sharpened*
  (p 0.0092 → 3.36e-3): RepLeaf is now more firmly a small, consistent, sign-significant step behind
  the top-3 GBDTs on numerical binary (w/t/l 1/13/2 — the gap clears 1% MRD on only 2 of 16, so it is
  the textbook *significant-but-negligible* pattern, but the direction is now unambiguous).
- **cat_cls 3.43 → 3.714** (still 5th). This suite remains **statistically powerless**
  (7 datasets, Friedman p=0.438, CD=2.305): the 5th place spans a single no-difference clique and
  **must not be quoted as a standalone result.**

**One honest upgrade:** **cat_reg flipped from flat to significant** (Friedman 0.375 → 0.0249) and
RepLeaf holds a clean 3rd (3.000) inside the top clique — the omnibus now *detects* a difference and
RepLeaf is on the right side of xgboost/hgb. num_reg also improved (3.32 → 3.00, 4th → 3rd), and its
gap to the best baseline went from borderline (p=0.055 vs catboost) to comfortably non-significant
(p=0.374).

**Verdict on the claim:** replace "consistent mid-pack, 3.18–3.43 in every suite" with the more
accurate and better-supported "**mid-pack overall (rank 3.00–3.81), a clear 3rd on regression and a
4th–5th on classification — the split the binary-Hessian limitation predicts.**" Confidence: **high**
for the regression/classification split and the positioning; **provisional** only for the exact
cat_cls ordering (7 datasets, flat).

---

## (3) Exact paper edits required (`docs/paper/repleafgbm-algorithm.tex`)

1. **Table 3 (`tab:openml`, lines 662–665)** — replace the mean-rank and Friedman columns:
   - Numerical regression (19): `3.32 → 3.00`; `1.6×10⁻⁴ → 2.2×10⁻³`
   - Numerical classification (16): `3.38 → 3.81`; `3.3×10⁻³ → 4.3×10⁻⁴`
   - Categorical regression (17): `3.18 → 3.00`; `0.375 (flat) → 0.025` **(drop "(flat)" — now significant)**
   - Categorical classification (7): `3.43 → 3.71`; `0.788 (flat) → 0.438 (flat)`
2. **Table 3 caption (line 654)** — `$5$ seeds` → `$10$ seeds`. Consider adding one clause noting the
   regression (3.00/3.00) vs classification (3.81/3.71) split.
3. **The "3.18–3.43" sentence (line 626)** — `Its mean rank is $3.18$--$3.43$ out of five in
   \emph{every} suite` → `$3.00$--$3.81$`, and reframe from "consistent mid-pack" to "mid-pack
   overall — a clear 3rd on regression, 4th–5th on classification."
4. **Wilcoxon-and-flat prose (lines 628–631)**:
   - `the Wilcoxon test against the best baseline is $p=0.055$` → `$p=0.37$` (num_reg vs catboost).
   - `The categorical suites are statistically flat (Friedman $p=0.375$ and $0.788$...)` →
     categorical **regression is now significant** (`$p=0.025$`); **only categorical classification is
     flat** (`$p=0.438$`, seven datasets, low power). Rewrite accordingly.
5. **Figure 2 (`fig:cd`, lines 638–649)** — **regenerate all four CD diagrams from the new PNGs.**
   Source → paper-figure name mapping (note the suffix change):
   - `benchmarks/results/colab10seed/leaderboard-cd-grinsztajn_num_reg-regression.png` →
     `docs/paper/figures/leaderboard-cd-grinsztajn_num_reg.png`
   - `...num_cls-binary.png` → `.../leaderboard-cd-grinsztajn_num_cls.png`
   - `...cat_reg-regression.png` → `.../leaderboard-cd-grinsztajn_cat_reg.png`
   - `...cat_cls-binary.png` → `.../leaderboard-cd-grinsztajn_cat_cls.png`
   - **Caption (lines 646–647)**: soften `never separated from the leading GBDTs` — on numerical
     classification RepLeaf **is** Nemenyi-separated from the single leader (lightgbm); state
     "separated only from the single best model, on numerical classification." Also change
     "categorical suites are statistically flat" → "the categorical-classification suite is flat"
     (cat_reg is no longer flat).
6. **Protocol line (line 615)** — `$50$ trials $\times$ $5$ seeds` → `$\times$ $10$ seeds`.
   Leave the "10k-row training cap" wording as-is (the train cap really is 10k).
7. **Appendix manifest (lines 977–980)** — `over five seeds` → `over ten seeds`;
   `datasets capped at $10{,}000$ rows` → `$20{,}000$ rows` (this is the `max_rows` cap; do not
   confuse with the 10k train cap).
8. **Appendix package versions (line 988)** — update to the actual 10-seed leaderboard env:
   `numpy 1.23.5 → 1.26.4`, `scikit-learn 1.2.0 → 1.9.0` (lightgbm/xgboost/catboost already match;
   torch 2.9.1 stays — it belongs to the niche studies, not the leaderboard). Secondary but required
   for single-env honesty.
9. **Limitations item (vi) (lines 891–894)** — currently "The fair leaderboard uses five seeds ...
   (a ten-seed replication is the obvious firming-up step)." **The ten-seed replication is now DONE.**
   Rewrite to: the leaderboard uses **ten** seeds and fifty HPO trials; note that the ten-seed run
   *sharpened* two facts — RepLeaf's clean 3rd on both regression suites and its small,
   sign-significant deficit to the top GBDTs on binary — and that cat_cls (7 datasets) remains
   low-power. Drop the "obvious firming-up step" clause.
10. **Abstract / intro "mid-pack" framing (lines 51–52)** — the phrase "competitive *mid-pack* learner
    — not state-of-the-art on average" **survives unchanged**; optionally add that the mid-pack sits at
    a clear 3rd on regression. No numeric edit needed there.

---

## (4) README `## Benchmarks` guidance

The README `## Benchmarks` section (lines 194–238) is the **legacy OpenML 3-seed / 60-20-20 breadth
study**, not the Grinsztajn fair-budget leaderboard — the paper already marks that study *superseded*.
It does **not** quote the "3.18–3.43" band anywhere, so **no numeric edit is forced.** Guidance:

- **Keep** the intro framing (line 24) "RepLeafGBM is competitive mid-pack" — it survives at 10 seeds.
- **Do not** import the old 3.18–3.43 band into the README.
- **Optional (recommended):** add a small four-row Grinsztajn 10-seed table (mean rank 3.00 / 3.81 /
  3.00 / 3.71) under a clear "fair same-budget leaderboard, 10 seeds" label, and state the
  regression-vs-classification split. If added, hand-sync it to the four `*-10seed.md` reports (the
  README table is hand-maintained and drifts — re-transcribe, don't assume).
- Leave the legacy OpenML/synthetic snapshots as-is (labeled "development progress, not performance
  claims").

---

## (5) Default change?

**No model-behavior default change is warranted — high confidence.** Same structural argument as the
5-seed verdict, now on 2× the seeds and a clean-tree pin:

- This is a **positioning** study, not a default experiment. The RepLeaf arm re-tunes
  `leaf_model ∈ {constant, embedded_linear, adaptive}` × `encoder ∈ {identity, plr}` **every Optuna
  trial**, so no aggregate rank is attributable to any single default.
- Confirmed-current defaults are **unchanged and unchallenged**:
  `grow_policy="leafwise"` (`src/repleafgbm/sklearn.py:214`),
  `leaf_model="embedded_linear"` (`:216`), `encoder="identity"` (`:220`).
- Learned `torch_*` encoders were deliberately excluded from the budgeted search, so nothing here
  speaks to the identity-vs-learned verdicts (Phases 14/16/25).
- The binary result **reinforces, not overturns,** the standing binary-Hessian verdict (Phase 12):
  `constant` remains the cheap, correct default for binary; the leaderboard gives no reason to change it.

**Keep all current defaults.**

---

## Next actions (who takes them)

1. **paper author / core-reviewer** — apply paper edits #1–#10 above; regenerate the 4 CD figures
   from the new PNGs (edit #5). This is the load-bearing next step.
2. **paper author** — optionally add the 10-seed README table (§4).
3. **experiment-runner (optional, low priority)** — the *only* remaining power gap is **cat_cls**
   (7 datasets, p=0.438, CD=2.305). Either run a fuller Grinsztajn categorical-classification set or
   keep demoting cat_cls from any headline; 10 seeds did not rescue its power. Not gating for the paper.
4. **harness-optimizer (standing, unchanged from 5-seed verdict)** — if the native-categorical thesis
   is ever to be tested, add a *second* categorical comparison where each model uses its native
   categorical path; the current suites keep all native-categorical handling OFF for all models
   (fair, but leaves that thesis untested — already stated correctly in paper §exp-openml lines 620–623).

---

### One-line verdict

**At 10 seeds RepLeafGBM is mid-pack overall (mean rank 3.00–3.81 of 5 across 59 Grinsztajn
dataset-slots) but no longer uniformly so: a clean 3rd on both regression suites (3.00/3.00, ahead of
xgboost and hgb) and a 4th/5th on classification (3.81/3.71) — the split the logistic-Hessian
limitation predicts. It is the leader nowhere and significantly-and-practically worse than the best
baseline nowhere. The paper's "3.18–3.43 consistent mid-pack" must become "3.00–3.81 with a
regression/classification split"; Table 3 numbers, the CD figures, and the appendix manifest
(seeds 5→10, cap 10k→20k) all need updating. No default change.** Confidence: **high** for positioning,
the reg/cls split, and no-default-change; **provisional** only for the exact cat_cls ordering
(7 datasets, flat).
