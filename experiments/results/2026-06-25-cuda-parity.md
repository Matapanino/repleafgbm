# CUDA backend parity report

- GPU: **Tesla T4**
- Parity tests (`tests/test_cuda_backend.py`): **PASS** — `36 passed in 13.99s`

## Histogram micro-benchmark

Single `build_histograms` over 200,000 rows x 50 features x 65 bins (mean of 5):

- NumPy: **174.21 ms**
- CUDA:  **3.47 ms**
- Speedup: **50.21x**

_Phase B1/B2: binned is uploaded once and cached on-device, and the histogram is returned resident (no per-build copy back)._

## End-to-end training (Phase B2: resident hist + GPU numeric scan)

`RepLeafRegressor.fit`, 50 trees, embedded_linear (GPU histogram + GPU numeric scan; host categorical scan + leaf fitting):

| config | rows x feat | numpy (s) | cuda (s) | speedup |
| --- | --- | --- | --- | --- |
| narrow | 100,000 x 30 | 12.33 | 8.86 | **1.39x** |
| wide | 50,000 x 200 | 29.93 | 14.82 | **2.02x** |

_B2's value grows with per-node histogram size: narrow d is its worst case (tiny scan, GPU launch/sync overhead), wide d its best (the big per-node histogram round-trip B1 paid is now avoided)._

## Per-fit transfer counters (`benchmarks.gpu_profile`)

End-to-end `gpu_profile` smoke; transfer columns are the CUDA backend's private H2D/D2H byte counters for one fit (numpy reports none). The grad/hess H2D column is the per-node host gather the next optimization targets — full rows saved to `gpu_bench/cases.jsonl`.

| case_id | backend | fit (s) | binned H2D | grad/hess H2D | winner D2H | hist D2H |
| --- | --- | --- | --- | --- | --- | --- |
| regression_30f_bins256_numpy | numpy | 2.36 | 0 | 0 | 0 | 0 |
| regression_30f_bins256_cuda | cuda | 1.90 | 3,000,000 | 61,288,048 | 0 | 337,883,040 |
| regression_200f_bins256_numpy | numpy | 11.51 | 0 | 0 | 0 | 0 |
| regression_200f_bins256_cuda | cuda | 7.03 | 12,000,000 | 37,099,200 | 58,496 | 0 |
| binary_30f_bins256_numpy | numpy | 2.34 | 0 | 0 | 0 | 0 |
| binary_30f_bins256_cuda | cuda | 2.44 | 3,000,000 | 58,521,232 | 0 | 338,438,160 |
| binary_200f_bins256_numpy | numpy | 11.18 | 0 | 0 | 0 | 0 |
| binary_200f_bins256_cuda | cuda | 6.36 | 12,000,000 | 35,368,272 | 58,432 | 0 |
| multiclass_c5_200f_bins256_numpy | numpy | 43.70 | 0 | 0 | 0 | 0 |
| multiclass_c5_200f_bins256_cuda | cuda | 21.67 | 12,000,000 | 155,974,288 | 285,280 | 0 |
| multioutput_k5_30f_bins256_numpy | numpy | 10.51 | 0 | 0 | 0 | 0 |
| multioutput_k5_30f_bins256_cuda | cuda | 5.58 | 3,000,000 | 324,267,120 | 0 | 1,693,116,000 |
| multioutput_k5_200f_bins256_numpy | numpy | 52.79 | 0 | 0 | 0 | 0 |
| multioutput_k5_200f_bins256_cuda | cuda | 9.59 | 12,000,000 | 195,370,640 | 58,560 | 0 |

_Expect `binned_uploads == 1` per fit (Phase B1 cache) and a non-zero grad/hess H2D total — that gather is what a device-resident grad/hess buffer would remove (docs/gpu_roadmap.md, Phase 1)._

## Multi-output device scan A/B (`REPLEAFGBM_CUDA_MO_DEVICE_SCAN`)

cuda multi-output fit with the on-device summed-gain scan **off** (host stack + host scan — the pre-device baseline) vs **on**. `hist+scan` sums the `histogram`+`split_scan` phase seconds (the two phases the device path keeps on the GPU); on-device should shrink them and replace the per-output histogram D2H with a 32-byte winner pack.

| case | features | fit off (s) | fit on (s) | speedup | hist+scan off (s) | hist+scan on (s) | hist D2H off | winner D2H on |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| multioutput_k5_30f_bins256 | 30 | 5.52 | 5.62 | **0.98x** | 4.48 | 4.71 | 0 | 0 |
| multioutput_k5_200f_bins256 | 200 | 28.35 | 10.21 | **2.78x** | 23.13 | 5.93 | 0 | 58,560 |

_Parity is covered by `tests/test_cuda_backend.py` (allclose); this table is the speed verdict — the device path must win (or at least not regress) on the wide shape, with the narrow shape protected by the adaptive small-scan crossover._
