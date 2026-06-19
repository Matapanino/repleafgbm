# Rust histogram: rayon feature-parallel + feature-major layout

**Date:** 2026-06-19
**Type:** parity-preserving performance optimization (CPU / native path)
**Scope:** `native/src/lib.rs::build_histograms`, `backends/rust_backend.py`, `Cargo.toml`
**Verdict:** **Ship.** Real, bitwise-parity-safe fit speedup on the default
(`split_backend="auto"` → Rust) path: **regression 15.2%, binary 17.3%,
multiclass 13.8%** at `--size medium` (pooled 14.6%). Single-output clears the
≥15% ship bar; multiclass lands just under it (memory-bound histogram cap, see
Analysis). Not a null result — every task improves and parity is exact.

## Motivation

v1.7.0 profiling reported `histogram` at 68–70% of fit — but that was the
**NumPy** backend (`np.bincount` + `np.repeat`). The default path is
`auto`→**Rust**, and a fresh per-phase measurement on the Rust backend (medium,
`OMP_NUM_THREADS=1`, embedded_linear leaf, identity encoder) shows a very
different, more balanced profile:

| phase (rust, medium) | regression | binary | multiclass |
|---|---:|---:|---:|
| leaf_fit (BLAS) | 32.7% | 35.3% | 41.6% |
| **histogram (Rust)** | **25.6%** | **30.3%** | **36.7%** |
| binning / preprocessing / partition / eval / split_scan | rest | rest | rest |

`histogram` was the largest *single-threaded Rust* phase (`build_histograms` had
no rayon). `leaf_fit` is larger but is NumPy/BLAS (the native `leaf_linear_stats`
fast path is gated to `emb_dim ≤ 32`; this config projects to 64), and forcing
BLAS threads made fit **slower** (small-matrix thread overhead: 21.4s vs 15.6s),
so it is not a rayon target. Histogram is the clean lever.

## Change

1. `build_histograms` is parallelized **across features** with rayon
   (`par_chunks_mut` over the output, one disjoint `(n_bins_max, 3)` block per
   feature). Each feature accumulates its bins in row order, so every
   `(feature, bin)` cell sees the identical summation order as the serial scan →
   **bitwise-identical** to NumPy regardless of thread count. (Row-parallel with
   per-thread merge would reorder the float sums and break bitwise parity, so it
   is avoided.) A `PARALLEL_MIN_CELLS` threshold keeps tiny nodes serial.
2. The `binned` matrix is passed **feature-major** (`(n_features, n_rows)`,
   cached transpose in `RustSplitBackend`, keyed by object identity, never
   serialized). With sorted node rows this turns the per-feature gather from a
   strided row-major read (memory-bound) into a near-sequential one.

Step 1 alone gave only **1.2–1.4×** histogram speedup (strided reads, memory-
bound — exactly the contingency the plan called out). Step 2 (feature-major)
lifted it to **2.2×** for single-output.

## Method

`benchmarks/gpu_profile.py --backend rust --size medium --seed 0`, 5 repetitions
per task (each a fresh process), `OMP_NUM_THREADS=1`, `RAYON_NUM_THREADS=8`.
Hardware: 8-core Apple Silicon (arm64). Before = serial binary; after =
feature-parallel + feature-major binary. Artifacts:
`artifacts/gpu_bench/perf_hist/baseline_{before,after}.jsonl`.

## Results (mean of 5; ± population std)

| task | fit before [s] | fit after [s] | **fit improvement** | histogram before→after | hist speedup |
|---|---:|---:|---:|---:|---:|
| regression | 14.013 ± 0.32 | 11.878 ± 0.17 | **15.2%** | 3.933 → 1.797 | 2.19× |
| binary | 14.848 ± 1.14 | 12.274 ± 0.26 | **17.3%** | 4.254 → 1.892 | 2.25× |
| multiclass (K=5) | 60.145 ± 1.33 | 51.868 ± 0.29 | **13.8%** | 20.933 → 12.248 | 1.71× |
| **pooled** | 89.01 | 76.02 | **14.6%** | — | — |

Improvements are large relative to the 2–8% run-to-run noise → clearly
significant.

## Analysis

- **Single-output (reg/binary): clears the bar.** Histogram parallelizes ~2.2×
  (memory-bound scatter-add; ~2× is the practical ceiling for this kernel on
  8 shared-bandwidth cores), and at 26–30% of fit that yields 15–17%.
- **Multiclass: 13.8%, just under.** Its histogram phase speedup is only 1.71×.
  Two confounds: (a) the per-class path adds un-parallelized NumPy glue
  (`np.ascontiguousarray` of each grad/hess column + `np.stack`); (b) multiclass
  does K× the histogram work but the same memory-bound kernel.
- **Batched multi-output histogram was tried and rejected.** A fused native
  `build_histograms_multi` (one parallel pass over all K outputs, no column
  copies / no `np.stack`, bitwise-parity-tested) measured **no better** than the
  per-class path (≈12.2–12.8s histogram, fit ~12–14%) — the larger per-feature
  block and inner K-loop offset the glue savings. Reverted to keep the change
  minimal (no dead/unhelpful code).

## Parity & invariants

- `tests/test_rust_backend.py`: 10 passed, incl. `test_histogram_parity_exact`
  (serial branch) and the new `test_histogram_parity_parallel_branch` (large
  node → rayon branch, `RAYON_NUM_THREADS=8`) — both assert **bitwise**
  (`assert_array_equal`) equality to NumPy. End-to-end numpy⇄rust allclose
  preserved.
- Output unchanged → determinism, sklearn API, serialization back-compat all
  intact. `rayon` is a pure-Rust dependency (no torch/lightgbm/cupy/external in
  the native path). The transpose cache is a runtime-only handle (backend is
  dropped by `Booster.__getstate__`, never serialized).

## Recommendation / follow-on

Ship the feature-parallel + feature-major histogram (real win on the default
backend, exact parity). To push multiclass over 15%, the remaining lever is a
**row-block-parallel** histogram (read grad/hess once, scale past the per-feature
grad/hess re-read) — but that changes the float accumulation order and would
require a **coordinated NumPy + Rust** rewrite to keep bitwise parity, plus
re-validation. That is a larger, separate PR; deferred.
