# CUDA backend parity report

- GPU: **Tesla T4**
- Parity tests (`tests/test_cuda_backend.py`): **PASS** — `35 passed in 13.59s`

## Histogram micro-benchmark

Single `build_histograms` over 200,000 rows x 50 features x 65 bins (mean of 5):

- NumPy: **143.27 ms**
- CUDA:  **2.94 ms**
- Speedup: **48.76x**

_Phase B1/B2: binned is uploaded once and cached on-device, and the histogram is returned resident (no per-build copy back)._

## End-to-end training (Phase B2: resident hist + GPU numeric scan)

`RepLeafRegressor.fit`, 50 trees, embedded_linear (GPU histogram + GPU numeric scan; host categorical scan + leaf fitting):

| config | rows x feat | numpy (s) | cuda (s) | speedup |
| --- | --- | --- | --- | --- |
| narrow | 100,000 x 30 | 12.36 | 8.83 | **1.40x** |
| wide | 50,000 x 200 | 31.14 | 13.72 | **2.27x** |

_B2's value grows with per-node histogram size: narrow d is its worst case (tiny scan, GPU launch/sync overhead), wide d its best (the big per-node histogram round-trip B1 paid is now avoided)._

## Per-fit transfer counters (`benchmarks.gpu_profile`)

End-to-end `gpu_profile` smoke; transfer columns are the CUDA backend's private H2D/D2H byte counters for one fit (numpy reports none). The grad/hess H2D column is the per-node host gather the next optimization targets — full rows saved to `gpu_bench/cases.jsonl`.

| case_id | backend | fit (s) | binned H2D | grad/hess H2D | winner D2H | hist D2H |
| --- | --- | --- | --- | --- | --- | --- |
| regression_30f_bins256_numpy | numpy | 2.78 | 0 | 0 | 0 | 0 |
| regression_30f_bins256_cuda | cuda | 2.11 | 3,000,000 | 61,288,048 | 0 | 337,883,040 |
| regression_200f_bins256_numpy | numpy | 12.34 | 0 | 0 | 0 | 0 |
| regression_200f_bins256_cuda | cuda | 7.24 | 12,000,000 | 37,099,200 | 58,496 | 0 |
| binary_30f_bins256_numpy | numpy | 2.48 | 0 | 0 | 0 | 0 |
| binary_30f_bins256_cuda | cuda | 2.05 | 3,000,000 | 58,521,232 | 0 | 338,438,160 |
| binary_200f_bins256_numpy | numpy | 11.56 | 0 | 0 | 0 | 0 |
| binary_200f_bins256_cuda | cuda | 6.46 | 12,000,000 | 35,368,272 | 58,432 | 0 |
| multiclass_c5_200f_bins256_numpy | numpy | 45.07 | 0 | 0 | 0 | 0 |
| multiclass_c5_200f_bins256_cuda | cuda | 21.14 | 12,000,000 | 155,974,288 | 285,280 | 0 |
| multioutput_k5_30f_bins256_numpy | numpy | 10.86 | 0 | 0 | 0 | 0 |
| multioutput_k5_30f_bins256_cuda | cuda | 5.66 | 3,000,000 | 324,267,120 | 0 | 1,693,116,000 |
| multioutput_k5_200f_bins256_numpy | numpy | 54.48 | 0 | 0 | 0 | 0 |
| multioutput_k5_200f_bins256_cuda | cuda | 9.71 | 12,000,000 | 195,370,640 | 58,560 | 0 |

_Expect `binned_uploads == 1` per fit (Phase B1 cache) and a non-zero grad/hess H2D total — that gather is what a device-resident grad/hess buffer would remove (docs/gpu_roadmap.md, Phase 1)._

## Multi-output device scan A/B (`REPLEAFGBM_CUDA_MO_DEVICE_SCAN`)

cuda multi-output fit with the on-device summed-gain scan **off** (host stack + host scan — the pre-device baseline) vs **on**. `hist+scan` sums the `histogram`+`split_scan` phase seconds (the two phases the device path keeps on the GPU); on-device should shrink them and replace the per-output histogram D2H with a 32-byte winner pack.

| case | features | fit off (s) | fit on (s) | speedup | hist+scan off (s) | hist+scan on (s) | hist D2H off | winner D2H on |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| multioutput_k5_30f_bins256 | 30 | 5.67 | 5.52 | **1.03x** | 4.62 | 4.48 | 0 | 0 |
| multioutput_k5_200f_bins256 | 200 | 28.99 | 9.64 | **3.01x** | 23.71 | 5.91 | 0 | 58,560 |

_Parity is covered by `tests/test_cuda_backend.py` (allclose); this table is the speed verdict — the device path must win (or at least not regress) on the wide shape, with the narrow shape protected by the adaptive small-scan crossover._
