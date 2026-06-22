# Verdict: tree growth policies (`grow_policy`) — keep `leafwise` default

- Date: 2026-06-23
- Analyst: results-analyst
- Evidence read: `experiments/results/grow_policy_comparison.md` (auto-report),
  `experiments/grow_policy_comparison.py` (data construction + capacity matching),
  ADR `docs/adr/0006-tree-growth-policies.md`, `docs/roadmap.md` Phase 33.
- Default confirmed in code: `grow_policy="leafwise"` in
  `src/repleafgbm/core/booster.py:63` (`BoosterParams`) and
  `src/repleafgbm/sklearn.py:183`. This verdict does **not** change it.

## Question

Does the multi-seed `grow_policy` comparison justify changing the default away
from `leafwise` to `symmetric`, which won 10/12 (dataset, leaf_model) cells?

## Verdict (one line)

**Keep `leafwise` as the default. Do NOT change it on this evidence.** The
symmetric sweep is 100% synthetic and its design structurally favors oblivious
trees; document per-policy use-cases now, and run a real-data follow-up before
any default decision. **Confidence: high** that the default should stay;
**provisional** on the per-policy guidance (synthetic-only).

## Evidence and effect sizes (honest reading of the numbers)

The auto-summary ("symmetric 10, leafwise 1, depthwise 1") is a raw win-count and
must not be read as "symmetric is better." Looking at mean separation vs pooled
std across 5 seeds:

| cell | best | gap over leafwise | ~std | separation | read |
|---|---|---|---|---|---|
| reg_piecewise_clean_n3000 / constant | symmetric 0.380 | 0.060 | 0.026–0.028 | ~2σ | real on this cell |
| reg_piecewise_clean_n3000 / embedded | symmetric 0.380 | 0.063 | 0.027–0.042 | ~1.5–2σ | real on this cell |
| mc3_n2000 / constant | symmetric 0.433 | 0.037 | 0.026 | ~1.5σ | real on this cell |
| mc3_n2000 / embedded | symmetric 0.423 | 0.048 | 0.029 | ~1.6σ | real on this cell |
| bin_piecewise_n2000 / constant | symmetric 0.358 | 0.022 | 0.031–0.041 | ~0.6σ | within noise |
| bin_noisy_n600 / constant | symmetric 0.434 | 0.041 | 0.057–0.062 | ~0.7σ | within noise |
| bin_noisy_n600 / embedded | symmetric 0.450 | 0.017 | 0.056–0.058 | ~0.3σ | noise |
| reg_piecewise_noisy_n600 / constant | symmetric 2.562 | 0.033 | 0.042–0.076 | ~0.5σ | tie w/ leafwise |
| **reg_piecewise_noisy_n600 / embedded** | **leafwise 2.593** | symmetric **+0.143 worse** | 0.067–0.103 | ~1.4–2σ | **symmetric clearly worse** |
| **reg_smooth_n2000 / constant** | **depthwise 0.815** | symmetric **+0.127 worse** (0.942, σ 0.17) | high-variance | **symmetric worst + unstable** |

Where symmetric "wins" decisively (≥~1.5σ) it is exactly the **clean / large
piecewise** regression cells and the **multiclass** cell. Several of its other
wins (both binary cells, the noisy-constant reg cell) are **within one std** —
ties, not wins. And in the two cells where it loses, it loses **harder** than it
wins elsewhere (+0.14 RMSE on noisy-embedded; worst + 3–6× the variance of the
other policies on smooth-constant). So even taken at synthetic face value this is
**"symmetric is strong on low-order piecewise/multiclass structure," not
"symmetric is the better general default."**

## Why this almost certainly over-favors symmetric (the disqualifying caveats)

1. **The signal is built to suit oblivious trees.** `_signal(..., "piecewise")`
   in `grow_policy_comparison.py` is
   `np.where(x0>0, 3, -2) + 2*x1 - x2 + 1.5*np.where(x3>0.5, ±1)` — a low-order,
   axis-aligned, *globally shared* threshold structure on x0 and x3. A symmetric
   (oblivious) tree applies one shared `(feature, threshold)` per level, which is
   the *exact* hypothesis class for "the same split on x0 and x3 partitions
   everyone." The multiclass scores are likewise shared linear contrasts on
   x0–x3. This is close to a best-case planted model for symmetric trees; it tells
   us little about messy real targets.

2. **100% synthetic, 0 real datasets.** The script's docstring promises "a couple
   of real datasets when their loaders are available," but `datasets()` only ever
   returns the 6 synthetic specs — no real loader is wired in, and the report
   shows only synthetic cells. The project rule is explicit: real data carries the
   weight for defaults (Phases 14/16/25 precedent). There is **zero** real-data
   evidence here.

3. **Only 8 features, all i.i.d. Gaussian, only x0–x3 active.** Real tabular data
   has many correlated/irrelevant/categorical features and higher-order
   interactions, where leaf-wise's adaptive, gain-greedy deepening typically
   pulls ahead — the standard LightGBM-style finding.

4. **Capacity matching plausibly handicaps leaf-wise on the small/noisy sets.**
   leaf-wise runs `num_leaves=31` with *no depth cap*; depth policies run
   `max_depth=5` (≤32 leaves) which for symmetric is a *complete, balanced* tree
   with heavy implicit regularization. On n=600 noisy data this lets leaf-wise
   grow lopsided high-variance branches while symmetric's structure regularizes —
   so part of symmetric's edge is a regularization-tuning artifact, not a growth
   advantage. (Tellingly, leaf-wise's *only* outright win is the noisy n=600
   embedded cell, and the noisy-constant cell is a tie — consistent with "leaf-wise
   was slightly over-capacity, not beaten.") A fair comparison must vary the
   capacity/`min_samples_leaf`/`max_depth` match.

5. **Standing literature + project priors.** Leaf-wise (LightGBM-style) is the
   usual winner on larger/real tabular data; CatBoost's oblivious trees win on
   speed/regularization, not raw accuracy on arbitrary structure. Nothing here
   overturns that on real data because no real data was tested.

Net: symmetric's headline is consistent with **an artifact of the synthetic
design** (planted low-order shared thresholds + a regularization-favoring capacity
match), not a demonstrated general improvement. That is not enough to move a
default under the project's evidence rule.

## Per-policy guidance (provisional — synthetic-only, document as such)

- **`leafwise` (keep default).** Best/again-tied where high-capacity adaptive
  fitting matters: the noisy small-n regression with `embedded_linear` leaves
  (its one clear win, 2.593 vs symmetric 2.736) and ties on the other noisy cell.
  Recommend as the general default, especially with `embedded_linear` leaves on
  noisy data and (by literature) on wide/real data.
- **`symmetric`.** Worth trying when you have **strong low-order, globally-shared
  axis-aligned structure and/or clean, larger data**, and on **multiclass** here
  (best in both mc cells, ~1.5σ). Also the natural pick when you later want
  oblivious-tree **inference speed / strong implicit regularization** (the
  CatBoost rationale; the compact fast predictor is still a follow-up per ADR
  0006). **Caution:** it was the *worst and most unstable* policy on smooth
  structure (reg_smooth constant 0.942 ± 0.17) and degraded most with
  `embedded_linear` on noisy data — do not recommend it for smooth/interaction
  targets or noisy embedded-leaf setups.
- **`depthwise`.** The balanced middle: best on `reg_smooth_n2000` (0.815, lowest
  there) and never the worst in any cell — a reasonable conservative alternative
  to leaf-wise when you want a depth-bounded tree without symmetric's all-or-none
  rigidity.
- **Leaf-model interaction (note for users).** Policy interacts with leaf model on
  noisy data: on `reg_piecewise_noisy_n600`, `embedded_linear` *helps* leaf-wise
  hold the win but *hurts* symmetric (2.736 vs its own 2.562 constant). On
  `reg_smooth_n2000`, `embedded_linear` is where symmetric finally wins (0.707).
  So leaf-model choice should be co-tuned with `grow_policy`, not fixed first.

## Recommendation

**(i) Keep `leafwise` default + document the per-policy use-cases above (mark
provisional/synthetic).** Then **(ii) run a real-data follow-up before any default
reconsideration.** Do **not** change the default now (option iii is unjustified:
no real data, and the synthetic suite is structurally biased toward symmetric).

### Follow-up experiment to design (owner: experiment-runner; design below)

Goal: test whether symmetric's edge survives on real data and under a fair
capacity match. Reuse the existing real loaders — they already exist:
`benchmarks/openml_suite.py` (`DATASETS`: california, house_sales, diamonds,
wine_quality, credit_g, phoneme, adult, wine, vehicle — regression/binary/
multiclass, several with categorical features) and
`benchmarks/benchmark_real_data.py` (california, house_sales, diamonds, adult).

1. **Datasets:** all 9 OpenML-suite datasets (or the 5-dataset `--quick` set for a
   first pass), covering reg/binary/multiclass *with categorical + many features* —
   the dimension entirely missing here. Keep the 6 synthetic as a reference column
   so the contrast (synthetic-favors-symmetric vs real) is visible in one report.
2. **Policies × leaves:** same {leafwise, depthwise, symmetric} × {constant,
   embedded_linear}; identity/plr encoder as appropriate (identity is the
   real-data default per standing verdicts — do not let encoder choice confound).
   Note ADR 0006: symmetric is numeric/ordered + scalar only, so on categorical-
   heavy sets it routes categoricals as ordered thresholds (a real, documented
   disadvantage vs leaf-wise subset splits — that asymmetry is part of the test).
3. **Capacity-match sensitivity (the key fix):** sweep at least two regimes, e.g.
   `(num_leaves=31 vs max_depth=5)` *and* a tighter leaf-wise cap
   (`num_leaves` ≈ 2**max_depth with `max_depth` also set) plus a
   `min_samples_leaf ∈ {20, 50}` sweep, so we can tell a genuine growth effect
   from a regularization-tuning artifact. Report whether symmetric's wins persist
   when leaf-wise is given a matching depth cap.
4. **Protocol:** ≥5 seeds, early stopping (as now), means ± std, bold-best per
   cell, and an explicit ≥1σ-separation flag so ties aren't miscounted as wins.
5. **Decision rule (state up front):** change the default only if symmetric (or
   depthwise) beats leaf-wise on a **majority of real datasets by ≥1σ across the
   capacity regimes**; otherwise keep `leafwise` and ship the use-case guidance.

Until that report exists, the default stays `leafwise` per ADR 0006 and the
project rule.
