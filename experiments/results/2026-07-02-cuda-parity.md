# CUDA backend parity report

- GPU: **Tesla T4**
- Parity tests (`tests/test_cuda_backend.py` + `tests/test_cuda_leaf_fit.py`): **FAIL** — `3 failed, 43 passed in 21.26s`
  - **Post-fix note (same session):** the 3 failures were the z_min/z_max guard
    bounds off by ~5e-8 — CuPy `cupyx.scatter_min/max` rounds float64 through
    float32. After replacing them with exact per-leaf slice reductions (commit
    d1ec97b) the full parity suite passes **53/53** (cuda_backend + cuda_leaf_fit
    + batched_scan) on the same VM; the benchmark numbers below were unaffected
    (all sum reductions were f64-exact throughout). See
    `2026-07-02-cuda-leaf-ridge-ab.md`.

## Histogram micro-benchmark

Single `build_histograms` over 200,000 rows x 50 features x 65 bins (mean of 5):

- NumPy: **145.55 ms**
- CUDA:  **2.82 ms**
- Speedup: **51.62x**

_Phase B1/B2: binned is uploaded once and cached on-device, and the histogram is returned resident (no per-build copy back)._

## End-to-end training (Phase B2: resident hist + GPU numeric scan)

`RepLeafRegressor.fit`, 50 trees, embedded_linear (GPU histogram + GPU numeric scan; host categorical scan + leaf fitting):

| config | rows x feat | numpy (s) | cuda (s) | speedup |
| --- | --- | --- | --- | --- |
| narrow | 100,000 x 30 | 12.97 | 5.62 | **2.31x** |
| wide | 50,000 x 200 | 31.64 | 8.21 | **3.85x** |

_B2's value grows with per-node histogram size: narrow d is its worst case (tiny scan, GPU launch/sync overhead), wide d its best (the big per-node histogram round-trip B1 paid is now avoided)._

## Per-fit transfer counters (`benchmarks.gpu_profile`)

End-to-end `gpu_profile` smoke; transfer columns are the CUDA backend's private H2D/D2H byte counters for one fit (numpy reports none). The grad/hess H2D column is the per-node host gather the next optimization targets — full rows saved to `gpu_bench/cases.jsonl`.

| case_id | backend | fit (s) | binned H2D | grad/hess H2D | winner D2H | hist D2H |
| --- | --- | --- | --- | --- | --- | --- |
| regression_30f_bins256_numpy | numpy | 2.53 | 0 | 0 | 0 | 0 |
| regression_30f_bins256_cuda | cuda | 2.25 | 3,000,000 | 61,288,048 | 0 | 337,883,040 |
| regression_200f_bins256_numpy | numpy | 12.16 | 0 | 0 | 0 | 0 |
| regression_200f_bins256_cuda | cuda | 5.44 | 12,000,000 | 37,099,200 | 58,496 | 0 |
| binary_30f_bins256_numpy | numpy | 2.49 | 0 | 0 | 0 | 0 |
| binary_30f_bins256_cuda | cuda | 2.18 | 3,000,000 | 58,521,232 | 0 | 338,438,160 |
| binary_200f_bins256_numpy | numpy | 12.48 | 0 | 0 | 0 | 0 |
| binary_200f_bins256_cuda | cuda | 5.62 | 12,000,000 | 35,368,272 | 58,432 | 0 |
| multiclass_c5_200f_bins256_numpy | numpy | 48.02 | 0 | 0 | 0 | 0 |
| multiclass_c5_200f_bins256_cuda | cuda | 15.21 | 12,000,000 | 155,974,288 | 285,280 | 0 |
| multioutput_k5_30f_bins256_numpy | numpy | 10.94 | 0 | 0 | 0 | 0 |
| multioutput_k5_30f_bins256_cuda | cuda | 5.28 | 3,000,000 | 324,267,120 | 0 | 1,693,116,000 |
| multioutput_k5_200f_bins256_numpy | numpy | 56.33 | 0 | 0 | 0 | 0 |
| multioutput_k5_200f_bins256_cuda | cuda | 10.35 | 12,000,000 | 195,370,640 | 58,560 | 0 |

_Expect `binned_uploads == 1` per fit (Phase B1 cache) and a non-zero grad/hess H2D total — that gather is what a device-resident grad/hess buffer would remove (docs/gpu_roadmap.md, Phase 1)._

## Multi-output device scan A/B (`REPLEAFGBM_CUDA_MO_DEVICE_SCAN`)

cuda multi-output fit with the on-device summed-gain scan **off** (host stack + host scan — the pre-device baseline) vs **on**. `hist+scan` sums the `histogram`+`split_scan` phase seconds (the two phases the device path keeps on the GPU); on-device should shrink them and replace the per-output histogram D2H with a 32-byte winner pack.

| case | features | fit off (s) | fit on (s) | speedup | hist+scan off (s) | hist+scan on (s) | hist D2H off | winner D2H on |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| multioutput_k5_30f_bins256 | 30 | 5.96 | 5.89 | **1.01x** | 4.85 | 4.89 | 0 | 0 |
| multioutput_k5_200f_bins256 | 200 | 30.35 | 10.13 | **3.00x** | 24.85 | 6.14 | 0 | 58,560 |

_Parity is covered by `tests/test_cuda_backend.py` (allclose); this table is the speed verdict — the device path must win (or at least not regress) on the wide shape, with the narrow shape protected by the adaptive small-scan crossover._
