# CUDA backend parity report

> **Note (2026-06-19):** this captures the run of a **prototype** device-resident
> grad/hess cache that was subsequently **reverted/shelved as a null result** —
> the per-fit transfer table and its "Phase 1-1 now removes…" note describe the
> prototype, not the shipped backend. See
> [`2026-06-19-cuda-gradhess-cache-verdict.md`](2026-06-19-cuda-gradhess-cache-verdict.md)
> for why (the grad/hess H2D is <0.2% of fit wall-clock; `split_scan` is the real
> bottleneck). The 20/20 parity result itself stands.

- GPU: **Tesla T4**
- Parity tests (`tests/test_cuda_backend.py`): **PASS** — `20 passed in 8.67s`

## Histogram micro-benchmark

Single `build_histograms` over 200,000 rows x 50 features x 65 bins (mean of 5):

- NumPy: **138.05 ms**
- CUDA:  **1.34 ms**
- Speedup: **102.97x**

_Phase B1/B2: binned is uploaded once and cached on-device, and the histogram is returned resident (no per-build copy back)._

## End-to-end training (Phase B2: resident hist + GPU numeric scan)

`RepLeafRegressor.fit`, 50 trees, embedded_linear (GPU histogram + GPU numeric scan; host categorical scan + leaf fitting):

| config | rows x feat | numpy (s) | cuda (s) | speedup |
| --- | --- | --- | --- | --- |
| narrow | 100,000 x 30 | 12.36 | 8.62 | **1.43x** |
| wide | 50,000 x 200 | 30.61 | 14.16 | **2.16x** |

_B2's value grows with per-node histogram size: narrow d is its worst case (tiny scan, GPU launch/sync overhead), wide d its best (the big per-node histogram round-trip B1 paid is now avoided)._

## Per-fit transfer counters (`benchmarks.gpu_profile`)

End-to-end `gpu_profile` smoke; transfer columns are the CUDA backend's private H2D/D2H byte counters for one fit (numpy reports none). The grad/hess H2D column is the per-node host gather the next optimization targets — full rows saved to `gpu_bench/cases.jsonl`.

| case_id | backend | fit (s) | binned H2D | grad/hess H2D | g/h resident | winner D2H | hist D2H |
| --- | --- | --- | --- | --- | --- | --- | --- |
| regression_30f_bins256_numpy | numpy | 2.41 | 0 | 0 | 0 | 0 | 0 |
| regression_30f_bins256_cuda | cuda | 2.17 | 3,000,000 | 48,000,000 | 30 | 0 | 337,883,040 |
| regression_200f_bins256_numpy | numpy | 12.00 | 0 | 0 | 0 | 0 | 0 |
| regression_200f_bins256_cuda | cuda | 6.75 | 12,000,000 | 28,800,000 | 30 | 58,496 | 0 |
| binary_30f_bins256_numpy | numpy | 2.58 | 0 | 0 | 0 | 0 | 0 |
| binary_30f_bins256_cuda | cuda | 1.89 | 3,000,000 | 48,000,000 | 30 | 0 | 338,438,160 |
| binary_200f_bins256_numpy | numpy | 11.62 | 0 | 0 | 0 | 0 | 0 |
| binary_200f_bins256_cuda | cuda | 7.10 | 12,000,000 | 28,800,000 | 30 | 58,432 | 0 |
| multiclass_c5_200f_bins256_numpy | numpy | 43.53 | 0 | 0 | 0 | 0 | 0 |
| multiclass_c5_200f_bins256_cuda | cuda | 20.94 | 12,000,000 | 144,000,000 | 150 | 285,280 | 0 |

_Phase 1-1 (device-resident grad/hess) now removes the per-node host gather: after a (grad, hess) pair's first node the full buffers live on-device and only `rows` cross. Expect `binned_uploads == 1`, the grad/hess H2D column to collapse vs the v1.6.0 baseline (artifacts/gpu_bench/2026-06-18-T4), and `g/h resident` (per-tree promotions) non-zero on the scalar tasks (docs/gpu_roadmap.md, Phase 1)._
