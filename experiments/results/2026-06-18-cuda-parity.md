# CUDA backend parity report

- GPU: **Tesla T4**
- Parity tests (`tests/test_cuda_backend.py`): **PASS** — `15 passed in 7.38s`

## Histogram micro-benchmark

Single `build_histograms` over 200,000 rows x 50 features x 65 bins (mean of 5):

- NumPy: **157.22 ms**
- CUDA:  **3.38 ms**
- Speedup: **46.55x**

_Phase B1/B2: binned is uploaded once and cached on-device, and the histogram is returned resident (no per-build copy back)._

## End-to-end training (Phase B2: resident hist + GPU numeric scan)

`RepLeafRegressor.fit`, 50 trees, embedded_linear (GPU histogram + GPU numeric scan; host categorical scan + leaf fitting):

| config | rows x feat | numpy (s) | cuda (s) | speedup |
| --- | --- | --- | --- | --- |
| narrow | 100,000 x 30 | 12.60 | 9.01 | **1.40x** |
| wide | 50,000 x 200 | 30.55 | 15.10 | **2.02x** |

_B2's value grows with per-node histogram size: narrow d is its worst case (tiny scan, GPU launch/sync overhead), wide d its best (the big per-node histogram round-trip B1 paid is now avoided)._

## Per-fit transfer counters (`benchmarks.gpu_profile`)

End-to-end `gpu_profile` smoke; transfer columns are the CUDA backend's private H2D/D2H byte counters for one fit (numpy reports none). The grad/hess H2D column is the per-node host gather the next optimization targets — full rows saved to `gpu_bench/cases.jsonl`.

| case_id | backend | fit (s) | binned H2D | grad/hess H2D | winner D2H | hist D2H |
| --- | --- | --- | --- | --- | --- | --- |
| regression_30f_bins256_numpy | numpy | 2.85 | 0 | 0 | 0 | 0 |
| regression_30f_bins256_cuda | cuda | 1.93 | 3,000,000 | 61,288,048 | 0 | 337,883,040 |
| regression_200f_bins256_numpy | numpy | 11.86 | 0 | 0 | 0 | 0 |
| regression_200f_bins256_cuda | cuda | 7.38 | 12,000,000 | 37,099,200 | 58,496 | 0 |
| binary_30f_bins256_numpy | numpy | 2.42 | 0 | 0 | 0 | 0 |
| binary_30f_bins256_cuda | cuda | 1.97 | 3,000,000 | 58,521,232 | 0 | 338,438,160 |
| binary_200f_bins256_numpy | numpy | 11.56 | 0 | 0 | 0 | 0 |
| binary_200f_bins256_cuda | cuda | 6.77 | 12,000,000 | 35,368,272 | 58,432 | 0 |
| multiclass_c5_200f_bins256_numpy | numpy | 44.10 | 0 | 0 | 0 | 0 |
| multiclass_c5_200f_bins256_cuda | cuda | 21.55 | 12,000,000 | 155,974,288 | 285,280 | 0 |

_Expect `binned_uploads == 1` per fit (Phase B1 cache) and a non-zero grad/hess H2D total — that gather is what a device-resident grad/hess buffer would remove (docs/gpu_roadmap.md, Phase 1)._
