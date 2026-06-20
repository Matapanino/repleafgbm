# Leaf fit: native rayon leaf-parallel + dispatch gate 32 → 64

**Date:** 2026-06-19 (measurements taken 2026-06-20)
**Type:** parity-preserving performance optimization (CPU / native path)
**Scope:** `native/src/lib.rs::leaf_linear_stats`,
`src/repleafgbm/core/leaf_models.py` (`_NATIVE_STATS_MAX_DIM`)
**Verdict:** **Ship (keep the gate at 64).** Large, output-preserving fit
speedup at `--size medium` (default emb=64): **regression −29.3%, binary −26.6%,
multiclass −24.8%, pooled −25.8%.** Every task clears the ≥15% ship bar by a wide
margin, the win localizes cleanly to `leaf_fit`, and quality is identical at
seed 0. Not a null result.

## Question

Perf session 2. The default `emb=64` leaf fit was running on the
single-threaded per-leaf BLAS loop because the native fused `leaf_linear_stats`
helper was gated to `emb_dim ≤ 32` (the prior histogram study, 2026-06-19,
explicitly noted leaf_fit-on-BLAS was *not* a rayon target at the old gate). The
change under test makes that helper **rayon leaf-parallel** (each leaf writes a
disjoint output chunk; per-leaf row order preserved) and raises the dispatch gate
`_NATIVE_STATS_MAX_DIM` from **32 → 64**, so the default emb=64 leaf fit now runs
on the native-rayon path. Does this clear the ≥15% fit bar, and does the win
actually come from `leaf_fit`? Keep or revert the gate default?

## Change under test (perf-only, output-preserving)

1. Native `leaf_linear_stats` parallelized **across leaves** with rayon — the
   right axis for the small per-leaf Gram matrices, which thread poorly inside
   BLAS. Each leaf accumulates into a disjoint output chunk in its own fixed row
   order, so the per-leaf summation order is identical to the serial native scan
   → **bitwise-identical to serial native**, and allclose vs the BLAS path.
2. Dispatch gate `_NATIVE_STATS_MAX_DIM` raised **32 → 64**, routing the default
   emb=64 leaf fit onto the native-rayon path instead of the per-leaf BLAS loop.

Predictions and quality are unchanged by construction; the verification below
confirms it empirically.

## Method

`benchmarks/gpu_profile.py`, `--size medium`, **seed 0**, **5 repetitions** per
task (each a fresh process, same seed). `OMP_NUM_THREADS=1`,
`RAYON_NUM_THREADS=8`. **Predict phase excluded** from the bar (fit wall-clock
only). Hardware: 8-core Apple Silicon (arm64), macOS. `repleafgbm-native` 0.1.0,
NumPy 1.23.5. `git_dirty=true` (uncommitted perf branch — measurement caveat).

- **medium** config: 100k train × 100 features, 50k test; `num_leaves=31`,
  `n_estimators=100`, `max_bins=256`; `leaf_model="embedded_linear"`,
  `encoder="identity"` → deterministic random projection to **emb=64** (same
  seed both arms); multiclass K=5.
- baseline (gate=32 → emb=64 on BLAS):
  `artifacts/gpu_bench/leaf/baseline.jsonl` (15 rows = 3 tasks × 5 reps).
- treatment (gate=64 → native rayon):
  `artifacts/gpu_bench/leaf/treatment.jsonl` (15 rows = 3 tasks × 5 reps).

Reported as mean ± population std over the 5 reps. Significance is judged by
effect size vs run-to-run noise: Welch t on fit means and the absolute
fit-saving expressed in units of baseline σ.

## Results (mean of 5; ± population std)

| task | fit base [s] | fit treat [s] | **fit Δ** | leaf_fit base [s] | leaf_fit treat [s] | leaf_fit Δ | leaf_fit speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| regression | 12.613 ± 0.586 | 8.913 ± 0.128 | **−29.3%** | 5.201 ± 0.241 | 1.659 ± 0.037 | −68.1% | 3.13× |
| binary | 12.612 ± 0.166 | 9.262 ± 0.076 | **−26.6%** | 5.084 ± 0.071 | 1.738 ± 0.020 | −65.8% | 2.92× |
| multiclass (K=5) | 55.274 ± 1.174 | 41.579 ± 0.519 | **−24.8%** | 25.420 ± 0.436 | 12.394 ± 0.536 | −51.2% | 2.05× |
| **pooled (sum of 3)** | 80.499 ± 1.040 | 59.754 ± 0.649 | **−25.8%** | 35.705 | 15.791 | −55.8% | 2.26× |

Pooled fit Δ is identical whether computed as a per-rep sum-of-tasks (−25.8%) or
as total-saved / total-baseline (20.75s / 80.50s = −25.8%); the mean of the three
per-task deltas is −26.9%.

### Effect size vs noise (significance)

| task | fit saving [s] | baseline σ [s] | **saving / σ_base** | Welch t |
|---|---:|---:|---:|---:|
| regression | 3.699 | 0.586 | 6.3 | −13.8 |
| binary | 3.351 | 0.166 | 20.1 | −40.9 |
| multiclass | 13.695 | 1.174 | 11.7 | −23.9 |

The fit saving is 6–20× the baseline run-to-run noise on every task; the
treatment is significant well beyond any reasonable threshold (n=5/arm, same
seed, fresh processes). Treatment variance is also **lower** than baseline
(regression fit σ 0.13 vs 0.59; multiclass 0.52 vs 1.17), i.e. the native-rayon
path is both faster and steadier than the BLAS loop.

## Win localizes to leaf_fit

The full fit improvement is attributable to `leaf_fit`: the per-task fit-saving
equals the leaf_fit-saving to within noise, and every non-leaf phase is flat.

| task | fit saving [s] | leaf_fit saving [s] | leaf_fit / fit | residual non-leaf Δ [s] |
|---|---:|---:|---:|---:|
| regression | 3.699 | 3.542 | 0.96 | +0.16 |
| binary | 3.351 | 3.346 | 1.00 | +0.01 |
| multiclass | 13.695 | 13.025 | 0.95 | +0.67 |

Non-leaf phases (regression, baseline → treatment): histogram 1.997 → 1.886,
binning 1.982 → 1.984, split_scan 0.380 → 0.368, partition 1.032 → 1.008, eval
0.905 → 0.896, encoder 0.313 → 0.310. All within ±6% (run noise); none is a
systematic regression. The small positive residual non-leaf Δ (≤0.67s, well
inside per-phase noise) means the leaf_fit saving slightly *under*-counts the fit
saving rather than borrowing from elsewhere. **The change does exactly one
thing: it accelerates leaf fitting.**

## Output invariance (quality identical at seed 0)

Across all 5 reps per task, quality is identical between arms:

- regression: rmse 0.2531719558201733, r2 0.9924121709427628 (both arms);
  mae 0.1651212129298631.
- binary: logloss 0.06875763565306939, auc 0.9977921063811496, accuracy 0.97526.
- multiclass: accuracy 0.89232; multi_logloss 0.27483026454449994.

The only differences are trailing-digit float-repr artifacts (regression mae
…298631 vs …2986306, ≈1e-16; multiclass multi_logloss …449994 vs …45, ≈1e-13).
These are the expected consequence of a different reduction order
(native-rayon leaf accumulation vs BLAS), which is **allclose, not bitwise**, vs
the BLAS path — consistent with the change's design contract (bitwise-identical
to *serial native*; allclose vs *BLAS*). No metric moves at reported precision.

## Recommendation

**Keep the gate default at 64** (the working tree already has
`_NATIVE_STATS_MAX_DIM = 64`; this report blesses it). The change is a large,
significant, output-preserving fit speedup on the default emb=64 configuration
(−25.8% pooled, every task ≥24%, every task clears the ≥15% bar by ~10+ points),
the entire win localizes to `leaf_fit`, and quality is identical. Reverting to 32
would re-strand the default leaf fit on the slower single-threaded BLAS loop for
no benefit.

This is confirmation-strength enough to move the default; it should not, however,
be read as a general "native always beats BLAS" claim — the gate is a measured
crossover (see Limitations) and emb ≫ 64 was not tested here.

## Limitations

- **Multiclass gets the smallest %** (−24.8% fit / −51.2% leaf_fit / 2.05×, vs
  ~3× for single-output). Single-output `leaf_fit` parallelizes ~3×; multiclass
  only ~2× because the per-class path retains un-parallelized Python/NumPy glue:
  the per-class loop, the serial `g_sum`/`h_sum` reductions, and NumPy
  post-processing of the K solves. The observed ~2–3× leaf_fit speedup (not the
  full 8× of the core count) is bounded by **memory bandwidth** (the per-leaf Z
  gather is bandwidth-bound on 8 shared-bandwidth cores), **serial glue**, and
  **per-leaf load imbalance** (leaves hold unequal row counts, so leaf-parallel
  work is uneven). Squeezing multiclass further would mean moving the per-class
  glue and the g_sum/h_sum reductions into the native pass — a larger,
  separate change, deferred.
- **`git_dirty=true`** on both arms (uncommitted perf branch); same code state
  per arm, so the A/B is valid, but these exact numbers are not pinned to a
  committed SHA.
- **Single hardware / single seed / n=5 per arm.** 8-core Apple Silicon only;
  one seed (0) with fresh-process reps. The effect is enormous relative to noise
  (6–20σ), so the *direction and order of magnitude* are robust, but the precise
  percentage is hardware- and thread-count-specific (`RAYON_NUM_THREADS=8`).
- **Gate value is a measured crossover, not proven optimal.** 64 was tuned at
  `OMP_NUM_THREADS=1`; this study validates 64 beats the BLAS path *at emb=64*.
  It does not establish where native stops winning above 64, nor the behavior
  when BLAS is given threads. A future sweep over emb ∈ {64, 96, 128} would pin
  the true upper edge of the gate.

## Parity & invariants

- Output unchanged at reported precision → determinism, sklearn API, and
  serialization back-compat intact. Differences vs BLAS are allclose-level
  float-reduction-order artifacts (≤1e-13), matching the change's contract
  (bitwise vs serial native, allclose vs BLAS). The native↔NumPy histogram
  bitwise parity from the prior study is untouched (this change is leaf-fit
  only).
- `rayon` is a pure-Rust dependency; no torch/lightgbm/cupy/external enters the
  native path. The encoder stays frozen and routing stays on raw features —
  this is a pure leaf-fit kernel/dispatch change, no thesis surface touched.

## Next action

- **Owner: experiment-runner / native-optimizer.** Land via the perf loop:
  `qa-verifier` (parity + suite, `OMP_NUM_THREADS=1`) then `core-reviewer`
  sign-off on the `native/src/lib.rs` + `leaf_models.py` diff, then commit (the
  branch is currently dirty). Confirm `tests/test_leaf_models.py` covers the
  native-rayon branch at emb=64 (gate boundary) with an allclose-vs-BLAS
  assertion.
- **Optional follow-up (defer): emb sweep** {64, 96, 128} at
  `OMP_NUM_THREADS=1` to pin the upper edge of `_NATIVE_STATS_MAX_DIM`; and a
  multiclass leaf-fit study moving per-class glue + g_sum/h_sum reductions into
  the native pass to lift the K=5 case above its current ~2× ceiling.
