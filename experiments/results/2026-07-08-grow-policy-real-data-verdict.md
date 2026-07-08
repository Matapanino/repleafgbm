# Verdict: grow_policy real-data gate (Phase 33) — keep `leafwise` default

- Date: 2026-07-08
- Analyst: results-analyst
- Evidence read:
  - `experiments/results/2026-07-08-grow-policy-real-data.md` (new real-data
    report: 9 legacy OpenML datasets x 5 seeds x 24 arms, capacity-match sweep)
  - `experiments/results/2026-06-23-grow-policy-verdict.md` (prior synthetic-only
    verdict; source of the pre-registered decision rule + follow-up design)
  - `docs/adr/0006-tree-growth-policies.md` (policy invariants, symmetric v0 scope)
  - `docs/roadmap.md` Phase 33 (the open follow-up = "the gate before any default
    change")
  - Raw per-seed appendix in the report (cross-checked wins/ties against it)
- Default confirmed in code (this verdict does **not** change it):
  `grow_policy: str = "leafwise"` at `src/repleafgbm/core/booster.py:64`
  (`BoosterParams`) and `grow_policy: str = "leafwise"` at
  `src/repleafgbm/sklearn.py:228`.

## Verdict (one line)

**KEEP `grow_policy="leafwise"` as the default.** The pre-registered decision rule
was not met on any regime; `depthwise` is a null (it collapses onto capped
leaf-wise); `symmetric` has two genuine but narrow real-data niches and loses just
as hard elsewhere. **Confidence: high** on keeping the default; the per-policy
niches below are now real-data-backed (no longer provisional).

## Was the pre-registered decision rule met? No.

Rule (fixed before the run): change the default only if `symmetric` or `depthwise`
beats **both** capacity-matched leaf-wise shapes (free + capped) on a **majority
of datasets (>=5/9) by >=1σ** paired, across the capacity regimes. Actual counts
from the report's decision-rule summary:

| leaf_model | challenger | wins | losses | datasets |
|---|---|---|---|---|
| constant | depthwise (all 3 regimes) | **0** | 0 | 9 |
| constant | symmetric d5 msl20 | 1 (credit_g) | 4 | 9 |
| constant | symmetric d5 msl50 | 1 (credit_g) | 4 | 9 |
| constant | symmetric d6 msl20 | 1 (credit_g) | 2 | 9 |
| adaptive | depthwise (all 3 regimes) | **0** | 0 | 9 |
| adaptive | symmetric d5 msl20 | 2 (credit_g, vehicle) | 2 | 9 |
| adaptive | symmetric d5 msl50 | 3 (credit_g, house_sales, vehicle) | 2 | 9 |
| adaptive | symmetric d6 msl20 | 2 (credit_g, vehicle) | 2 | 9 |

The best any challenger reaches is **3/9** (symmetric, adaptive, d5 msl50). The
majority threshold is 5/9. No challenger clears it in any leaf-model x capacity
regime. `depthwise` is 0/9 everywhere. **Rule decisively not met → keep leafwise.**

Aggregates agree: the only regime with a significant Friedman test is constant
d5 msl20 (p=0.0161), where the top ranks are depthwise (1.778) and capped leaf-wise
(2.000) with symmetric **worst** (3.444) — and the Nemenyi CD groups still overlap
(no arm separates from leaf-wise). Every adaptive regime is non-significant
(p >= 0.37). No Wilcoxon shows symmetric significantly better than leaf-wise in any
regime (best p = 0.25).

## Honest effect-size reading

**Where symmetric genuinely wins (real, not noise):**
- **credit_g (binary, n=1000, 13 categoricals):** symmetric wins in *every* regime
  and *both* leaf models — the only dataset it wins on the constant leaf. Deltas
  are large and consistent: constant d5 msl20 -0.0338 (3.0σ) vs free / -0.0258
  (1.6σ) vs capped; adaptive d5 msl20 -0.0328 (3.0σ) / -0.0319 (3.9σ). This is a
  small, noisy, categorical-heavy set where the complete-balanced-tree implicit
  regularization pays off — and it wins *despite* symmetric's documented
  ordered-threshold categorical handicap (13 categoricals). Its strongest,
  most robust real-data niche.
- **vehicle (multiclass, n=846, 0 categoricals) under the adaptive leaf:** decisive
  — adaptive d5 msl50 -0.1433 logloss (6.3σ) vs free / -0.1174 (5.5σ) vs capped
  (+5-7pp accuracy). This is where the synthetic "symmetric wins on multiclass"
  claim survives real data.
- **house_sales (regression) under adaptive d5 msl50:** a marginal win (-0.0069,
  1.2σ vs both) — only in that one leaf-model x regime cell.

**Where symmetric loses (as hard as it wins):**
- **phoneme (binary, 0 cat):** loses badly and consistently — constant d5 msl20
  +0.0239 (5.0σ) vs free / +0.0140 (3.3σ) vs capped.
- **california, wine_quality (regression, 0 cat):** consistent losses, 1.5-6.8σ
  (wine_quality constant d6 msl20 +0.0279, 6.8σ vs free).
- **diamonds (regression):** loses under the constant leaf (up to +0.0212, 1.7σ);
  a tie under adaptive.
- **adult (binary):** tie/mixed — symmetric never beats *both* leaf-wise shapes
  (the depthwise/capped shape is the best arm on adult, and free leaf-wise is
  weak there).

Net: symmetric helps on **small / noisy / categorical / multiclass-adaptive** sets
(credit_g always; vehicle-adaptive) and hurts on the **larger, cleaner numeric
regression/binary** sets (phoneme, california, wine_quality, diamonds-constant),
exactly where leaf-wise's adaptive gain-greedy deepening is supposed to win. Under
the constant leaf it is net-negative (1 win vs 2-4 losses per regime); under
adaptive it is roughly break-even on its niches but still short of a majority.

**Did the synthetic "symmetric wins on multiclass" claim survive real data?**
Partially and conditionally. It survives on **vehicle only, and only under the
adaptive leaf** (5-6σ). On **wine** (multiclass, n=178) symmetric-adaptive is
numerically best (0.0612 vs 0.0872) but only 0.7σ — a tie at that tiny n / huge
variance, and on the constant leaf wine is a flat tie for every policy. And on
vehicle under the *constant* leaf symmetric is not a clean win (beats free by 1.6σ
but not capped). So "symmetric is better on multiclass" is **not** a general rule;
it is "symmetric can win on a specific multiclass set when paired with the adaptive
(embedded_linear) leaf." The design note confirms symmetric *does* cover multiclass
(one scalar routing tree per class per round) — the old "n/a on multiclass"
assumption is now disproven — but coverage is not a general advantage.

**Did the capacity sweep expose the synthetic edge as an artifact? Yes, largely.**
The 2026-06-23 verdict hypothesized that symmetric's synthetic edge was partly a
regularization-tuning artifact: leaf-wise ran uncapped (`num_leaves=31`, no depth
cap) against symmetric's complete balanced depth-5 tree. This run gives leaf-wise a
**capped** shape (matching `max_depth`) as well as free. Result: the **capped
leaf-wise shape is at or near the top of every table**, and symmetric's wins require
beating *both* free and capped. In the many cells where symmetric beats free but not
capped (e.g. adult, vehicle-constant, several d6 rows), the "advantage" is just that
free leaf-wise was over-capacity — matching the depth cap neutralizes it. So a
substantial part of the synthetic symmetric edge was indeed a capacity/regularization
artifact, now corrected. The `min_samples_leaf` sweep is consistent: heavier
regularization (msl50) *helps* symmetric on its small/noisy niches (vehicle-adaptive
d5 msl50 is its biggest win) but *hurts* it on the large clean sets (california
symmetric d5 msl50 loses by 3.7-4.3σ) — a regularization knob, not a growth
advantage.

## depthwise: 0 wins / 0 losses everywhere — expected null

`depthwise` returns **0 wins and 0 losses against both matched leaf-wise shapes in
all six challenger rows**, and its per-row Δ vs capped leaf-wise is `+0.0000 (0.0σ)`
almost universally (confirmed against the raw per-seed appendix — e.g. adult,
credit_g, diamonds depthwise seeds are identical or off-by-noise to capped
leaf-wise). This is the correct read given the design: depthwise ran at
`num_leaves = 2^d - 1` (stock budget); at `num_leaves = 2^d` depthwise is
*deterministically identical* to the capped leaf-wise shape (both grow every valid
split to depth `d`), and at `2^d - 1` it is that shape minus one leaf. On real data,
at a budget where the depth cap governs, **depthwise is a re-parameterization of
capped leaf-wise, not a distinct competitor.** The both-shapes rule correctly nulls
it: it can never beat capped leaf-wise because it *is* (essentially) capped
leaf-wise. Its occasional "-1σ vs LW-free" rows are just the capped shape beating
the free shape, a leaf-wise tuning effect, not a depthwise effect. This supersedes
the prior synthetic framing of depthwise as a "balanced, never-worst middle" — on
real data it carries no independent signal.

## Updated per-policy guidance (real-data-backed; guidance, NOT defaults)

This replaces the provisional synthetic-only guidance in the 2026-06-23 verdict and
ADR 0006.

- **`leafwise` (keep as default).** General-purpose winner; no policy beats it on a
  majority of real datasets under a fair capacity match. Practical tuning note from
  this run: on real data a **depth-capped** leaf-wise shape (set `max_depth`, e.g.
  d5, with `num_leaves ≈ 2^max_depth`) is consistently at/near the top and often
  beats the *uncapped* `num_leaves=31/63` shape (uncapped free was the weakest
  leaf-wise arm on adult, credit_g, wine_quality). This is a leaf-wise
  hyperparameter insight, not a default change.
- **`symmetric` (opt-in, narrow niches — now real-data-confirmed).** Worth trying on
  (1) **multiclass with the adaptive (`embedded_linear`) leaf** (vehicle: -0.14
  logloss, 5-6σ), and (2) **small, noisy, categorical-heavy** sets where its
  complete-tree implicit regularization helps (credit_g: 1.5-4σ across all regimes
  and both leaf models). Do **not** use it on larger, cleaner numeric
  regression/binary (phoneme, california, wine_quality, diamonds) — it loses there
  by 1.5-6.8σ. Two standing caveats: it routes categoricals as ordered thresholds
  (a documented handicap that confounds the categorical-set comparison, yet credit_g
  still wins), and its real long-term draw — compact oblivious *inference speed* —
  remains unbuilt/untested here.
- **`depthwise` (no evidence-backed reason to prefer).** On real data, at a budget
  where the depth cap governs, it is indistinguishable from capped leaf-wise
  (0 wins / 0 losses, ~0σ everywhere). If you want a depth-bounded tree, set
  `max_depth` on the default `leafwise` policy and get the same behavior under the
  primary knob. The prior "balanced never-worst middle" recommendation does not
  survive the real-data test.

## Phase 33 gate: can it be closed? Yes.

The roadmap Phase 33 open follow-up — "a **real-data** policy comparison (the gate
before any default change)" — has now run to its pre-registered specification
(9 OpenML datasets covering reg/binary/multiclass with categoricals, capacity-match
sensitivity sweep across d5/d6 x free/capped x msl {20,50}, 5 seeds, >=1σ paired
decision rule). The rule was evaluated and **not met**; the default stays
`leafwise`. **The gate can be closed** with the outcome "keep default; per-policy
niches documented."

### Follow-ups still justified by evidence (none block the gate)

1. **Categorical-subset symmetric** (already a listed v0-scope follow-up in ADR
   0006). credit_g wins *despite* symmetric's ordered-threshold categorical
   handicap; removing that confounder would clarify whether symmetric's categorical
   niche is larger than it looks (and whether it closes the gap on adult/diamonds).
   This is a *feature* follow-up, not a default gate. Owner: research-proposer →
   experiment-runner.
2. **Compact oblivious storage + bitwise-indexed predictor** (ADR 0006 follow-up):
   symmetric's genuine rationale is inference speed / model size, which this
   accuracy-only study did not measure. Owner: native-optimizer / research-proposer.
3. **No accuracy re-run is needed to move the default** — the evidence to keep
   `leafwise` is sufficient and the rule was decisively not met (max 3/9 vs 5/9
   required). Do not reopen the default decision without materially new evidence
   (e.g. a categorical-subset symmetric that flips the majority — unlikely given the
   losses on numeric sets are unrelated to categorical handling).

## Bottom line

Keep `grow_policy="leafwise"`. The decision rule was not met (best 3/9, need 5/9);
`depthwise` collapses onto capped leaf-wise (null); `symmetric` earns two real but
narrow niches (multiclass-adaptive on vehicle; small/noisy/categorical on credit_g)
while losing 1.5-6.8σ on the larger clean numeric sets, and the capacity sweep shows
much of its synthetic edge was a regularization artifact that a depth-capped
leaf-wise shape neutralizes. Document the niches as opt-in guidance and close the
Phase 33 gate.
