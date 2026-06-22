# Binning: feature-parallel thread-pool (perf session 3)

**Date:** 2026-06-22
**Type:** parity-preserving performance optimization (CPU / shared core path)
**Scope:** `core/histogram.py` (`compute_bin_thresholds`, `bin_features`),
new `utils/parallel.py` (`map_features`). No native/Rust change.
**Verdict:** **Ship.** Output-invariant fit speedup on the default backend at
the chosen scale (`--size large`): **regression +24.7%, binary +24.3%**; at
medium **+16.5% / +17.9%**. Single-output clears the ≥15% bar at both scales;
the win is **fully localized to binning** (non-binning fit time flat within
±1.2%) and quality is **bitwise-identical**. Multiclass +3.9% (binning is a
small share of its fit — see Analysis); every task improves, parity is exact.

## Motivation

After perf sessions 1–2 (rayon histogram #23, rayon leaf_fit #24) the Rust fit
profile is balanced. A fresh `--size large` (500k×200) re-profile of the
optimized backend (Step 0, `OMP_NUM_THREADS=1`, `RAYON_NUM_THREADS=8`, embedded
linear, identity→proj 64) showed the new dominant phase is **`binning`** — for
single-output it is the #1 phase:

| phase (large, Step-0 probe) | regression | binary | multiclass |
|---|---:|---:|---:|
| **binning** | **34.5%** | **32.7%** | 7.9% |
| histogram | 22.0% | 26.2% | **38.2%** |
| leaf_fit | 9.6% | 11.5% | 24.7% |

`binning` is **once-per-fit, serial NumPy** (`compute_bin_thresholds` →
per-feature `np.unique`/`np.quantile`; `bin_features` → per-feature
`np.searchsorted`). Its dominant cost is the `np.unique` **sort (O(n·log n))**,
so it grows as a *share* with rows: ~22% at medium → ~30% at large (clean
baseline). It is embarrassingly parallel across features and — crucially —
parity-safe: parallelizing it does not reorder any float accumulation, so it
stays bitwise-identical (unlike the deferred histogram row-block rewrite).

## Change (perf-only, output-preserving)

`compute_bin_thresholds` and `bin_features` map their per-feature work across a
scoped `ThreadPoolExecutor` (`utils.parallel.map_features`). The per-feature
NumPy calls (`np.unique`/`np.quantile`/`np.searchsorted`) **release the GIL**,
so this gives real parallelism; and because each feature runs the *identical*
call and results are reassembled in feature order, the output is
**bitwise-identical to the serial loop regardless of thread count**. Pool size
is read once from `REPLEAFGBM_NUM_THREADS` (default `os.cpu_count()`), gated by
`PARALLEL_MIN_CELLS` (small inputs stay serial). Binning is upstream of the
split backend, so the NumPy⇄Rust histogram parity is untouched. No native build.

## Method

`benchmarks/gpu_profile.py --backend rust --max-leaf-emb-dim 64`, **seed 0**,
**5 reps** per cell (fresh process each), **predict excluded**.
`OMP_NUM_THREADS=1 RAYON_NUM_THREADS=8 REPLEAFGBM_NUM_THREADS=8`. 8-core Apple
Silicon, `repleafgbm-native` 0.1.0 (rebuilt from `main` for the baseline so
#23/#24 are present). Baseline = serial binning (current `main`); treatment =
this change. Artifacts: `artifacts/gpu_bench/s3/{baseline,treatment}_{large5,
medium5}.jsonl`. Reported mean ± population std; significance by Welch t (n=5).

## Results (mean of 5)

| scale / task | fit base [s] | fit treat [s] | **fit Δ** | Welch t (fit) | binning [s] | bin speedup | non-binning fit Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| **large** regression | 80.32 ± 10.8 | 60.44 ± 9.7 | **+24.7%** | 2.74 | 24.77 → 4.69 | **5.28×** | −0.4% |
| **large** binary | 81.29 ± 8.9 | 61.57 ± 12.8 | **+24.3%** | 2.54 | 23.61 → 4.55 | **5.19×** | +1.2% |
| medium regression | 9.21 ± 0.42 | 7.69 ± 0.90 | **+16.5%** | 3.08 | 1.95 → 0.37 | 5.30× | −0.9% |
| medium binary | 9.36 ± 0.27 | 7.69 ± 0.30 | **+17.9%** | 8.32 | 1.95 → 0.35 | 5.57× | +1.0% |
| medium multiclass | 44.25 ± 4.85 | 42.51 ± 4.21 | +3.9% | 0.54 | 1.95 → 0.36 | 5.40× | +0.4% |

Standalone binning probe (500k×200, isolated): serial 21.1s → 8-thread 4.4s =
**4.82×** (thresholds 5.46×, bin_features 4.01×), bitwise-identical.

## Analysis

- **Win localizes entirely to binning.** Across all five cells the
  **non-binning fit time is flat (±1.2%)** while binning drops 5.2–5.6×, and
  `binning_save ≈ fit_save` (ratio 0.91–1.04). The binning speedup itself is
  enormously significant (Welch t **7.6 / 16.6 / 46.7 / 108.0 / 76.1**) — a
  low-variance, deterministic 5.3×. The change does exactly one thing.
- **Single-output clears ≥15% at both scales.** The *saving* is clean (~20s
  large, ~1.6s medium); the fit *percentage* tracks binning's share (≈30% large,
  ≈21% medium) with 81% of it removed. Medium is the low-noise anchor
  (binary Welch t=8.3); large has the larger effect (+24%) but higher
  run-to-run variance on 60–80s fits (fit Welch t 2.5–2.7, still p<0.05). That
  σ is driven by a single high rep per large arm; the **median** fit Δ is
  **+24.9% (reg) / +31.0% (binary)**, so the mean +24% is conservative, not
  cherry-picked (independently reconfirmed from the raw JSONL).
- **Multiclass +3.9% (expected, not a miss).** Binning parallelizes the same
  5.4×, but it is only ~4–8% of the multiclass fit (the K=5 per-class path
  inflates histogram/leaf_fit), so the 1.6s saving is lost in the 42s±5s fit
  noise (fit t=0.54). Multiclass's lever is the **deferred histogram row-block
  rewrite** (Candidate B), not binning.

## Output invariance (quality bitwise-identical)

Per-rep quality matches to full float precision, e.g. large regression rmse
`0.22393674814877448`, mae `0.1415258548365576`, r2 `0.9940859397219488`
(base == treat); large binary logloss `0.06198151448676692`, auc
`0.9990377771811404`, accuracy `0.98407` (base == treat). Confirmed by the unit
test below (threaded == serial, bitwise).

## Parity & invariants

- `tests/test_binning_parallel.py`: thresholds and `binned` are
  `assert_array_equal` (bitwise) between `n_threads=1` and `8`, on data above
  the gate, covering every binning branch (quantile / midpoint / constant /
  fully-missing / scattered NaN). `tests/test_rust_backend.py` (numpy⇄rust
  bitwise histogram + end-to-end) stays green untouched — binning output is
  unchanged, so both split backends are unaffected.
- Determinism, sklearn API, serialization back-compat all intact (output
  identical). `ThreadPoolExecutor` is stdlib — no torch/lightgbm/cupy/external
  in any path; no rayon/native change; native crate stays 0.1.0. Thesis
  surfaces untouched (binning is feature quantization, not routing).

## Limitations

- **Large-fit variance** (σ ≈ 9–13% of mean on 60–80s fits) on this 8-core
  machine; the binning-phase measurement (t≥7.6) and the medium fits (t up to
  8.3) are the low-noise anchors. The *saving* is deterministic; only the total
  denominator is noisy.
- **Single hardware / one seed / n=5.** Effect direction and the binning 5.3×
  are robust; exact percentages are hardware/thread-count specific
  (`REPLEAFGBM_NUM_THREADS=8`).
- **Residual serial cost in the phase:** the `numeric_X.copy()` in
  `Splitter.__init__` is still serial (~the remaining ~4.5s of binning at
  large). Skipping it when there are no categoricals is a future micro-opt;
  not needed to clear the bar.
- `preprocessing` looked large (13%) only on the first process of a run
  (cold-start import inflation); its true share is ~2–3% (binary), so it is not
  a lever.

## Recommendation

**Ship** the feature-parallel binning. Real, significant, output-invariant fit
speedup on the default backend: single-output +24% (large) / +16–18% (medium),
fully localized to binning, quality bitwise-identical, parity exact. Land via
the perf loop: `qa-verifier` (full `scripts/check.sh`, `OMP_NUM_THREADS=1`) then
`core-reviewer` sign-off on `core/histogram.py` + `utils/parallel.py` +
`tests/test_binning_parallel.py`.

Follow-ups (defer): (1) multiclass via the histogram row-block rewrite
(Candidate B); (2) skip `numeric_X.copy()` when no categorical features.
