# Consolidated Niche + Ensemble Verdict — RepLeafGBM's defensible edge

**results-analyst · 2026-06-29.** Companion to
`2026-06-29-grinsztajn-consolidated-verdict.md` (RepLeaf = robust mid-pack
standalone, SOTA nowhere). Tests whether the promised "defensible edge in niche
regimes" holds. Inputs: `router_extraction.md`,
`2026-06-29-robust-multioutput-suite.md`,
`2026-06-29-ensemble-diversity-grinsztajn_{num_reg,num_cls,cat_reg,cat_cls}.md`,
survey `docs/research/2026-06-29-diverse-ensemble-stacking.md`.

## One-line verdict
**Robust multi-output objectives are RepLeafGBM's one decisive, paper-grade edge**
(huge consistent wins under contamination on 4/6 suites, with an honest
conditional exception). **Router leaf-channel isolation is a solid secondary
edge** (identity-embedded leaf beats constant on the same routes, 3/3 real
datasets, never a loss, ~1–2%). **The ensemble-diversity hypothesis is the weakest
leg** (never hurts; sign-consistent but practically negligible; marginal
decorrelation). **No default changes** — all three are opt-in usages.

## (A) Robust multi-output — DECISIVE, with an honest exception
- energy / jura / wq / synthetic: huber/quantile beat squared **decisively** at
  ~every contamination level (mostly 5/0/0; effect sizes multiples-of-std, e.g.
  energy@8% 12.49→2.22). This is the strongest niche in the program.
- **Exception (rf1, scm20d):** robust losses are **worse than squared at 4–8%
  contamination (0/0/5)**, winning only at 16%. Root cause is a large **clean-fit
  penalty** — at 0% contamination rf1 huber=21.26 vs squared=1.48 (14×),
  scm20d 250 vs 105 — i.e. the robust loss over-clips real signal on these target
  scales. Almost certainly a **huber-delta / quantile mis-scaling**, not an
  intrinsic defeat. So robustness is **conditional** (helps once contamination
  exceeds the clean-fit cost), and a blanket "use huber under contamination" would
  hurt rf1/scm20d at moderate contamination.

## (B) Router / leaf-channel isolation — SOLID secondary edge
- identity-embedded linear leaf vs constant on the *same* extracted LightGBM
  routes: california −0.0085 (5/0/0), diamonds −0.0021 (4/1/0), wine_quality
  −0.0060 (2/3/0) — **3/3, never a loss**, median ~1–2%.
- PLR variant adds **nothing** (≈0, p 0.125/0.812/0.812) → **reinforces the
  identity-encoder default**; the gain is the linear leaf, not a fancier encoder.
- Real-data effect (~1–2%) is the low end of the prior 2–12% (synthetic) range →
  state as **~1–2%, provisional**.

## (C) Ensemble diversity — WEAKEST leg
| suite | median Δ | p | w/t/l | r(GBDT) → r(repleaf-GBDT) | mean-gain 95% CI |
|---|---|---|---|---|---|
| num_reg | -0.0003 | **0.0446** | 1/18/0 | 0.980 → 0.978 | [-0.0063, +0.0045] |
| num_cls | -0.0002 | 0.083 | 0/16/0 | 0.977 → 0.977 | [+0.0000, +0.0007] |
| cat_reg | -0.0002 | 0.098 | 2/15/0 | 0.984 → 0.982 | [-0.0030, +0.0024] |
| cat_cls | +0.0001 | 0.938 | 0/7/0 | 0.965 → 0.967 | [-0.0004, +0.0010] |

- Adding RepLeaf to a GBDT average **never hurts** anywhere; sign-consistent
  improvement on num_reg (p=0.0446) — but **practically negligible**: only 1/19
  clears the 1% MRD (18 ties), and the bootstrap mean-gain CI **straddles zero**.
- "Adds value *because diverse*" is **not claimable**: decorrelation is marginal
  (Δr≈0.002 on num_reg, none on classification) — RepLeaf still routes on raw
  features, so its errors are ~0.98-correlated with GBDTs. Positive ambiguity
  harvest confirms diversity is *technically* exploited but too small to surface.
- The robustly-supported claim is generic: **ensembling beats the val-selected
  best single on 15/19, 15/16, 13/17, 5/7** — "ensembling helps", not "RepLeaf
  specifically helps".

## 5-seed power ceiling
Every niche cell sits at **p=0.0625** = the hard floor of a two-sided Wilcoxon at
n=5 (all-same-sign). The **ensemble** p-values (across 16–19 datasets) are real,
limited by effect size not seeds. A **7–10-seed rerun** would convert robust-MO
(energy/jura/wq/synthetic — very high likelihood) and router (california/diamonds
— high) to genuine significance, and **sharpen** the rf1/scm20d exception. It will
**not** rescue the ensemble magnitude (effect-size-limited, not power-limited).

## Default change: NONE (high confidence)
All opt-in: robust `objective=` (default stays `squared_error`; rf1/scm20d penalty
is itself a reason not to default to huber); router extraction is an opt-in
external_model workflow that *validates the existing* `embedded_linear`+`identity`
default; ensembling is user-side. Defaults untouched.

## Prioritized next steps
1. **robust-MO at 7–10 seeds** (highest, surest) — converts the 4 decisive suites
   to genuine significance + sharpens the rf1/scm20d exception.
2. **Diagnose the rf1/scm20d clean-fit penalty** (huber-delta / quantile auto-
   scaling sweep) — could make robustness *general*; may touch a default → gate
   any change back through results-analyst.
3. **Router at 7–10 seeds + more real datasets** — significance + honest effect
   bound.
4. **Strengthen-or-downscope the ensemble claim** — Caruana greedy / OOF stacking;
   report whether RepLeaf's *selected* ensemble weight is systematically > 0
   (TabArena test). If not, frame as generic "ensembling beats best-single" and
   the RepLeaf contribution as "never hurts, marginal on num_reg, Δr≈0.002".
