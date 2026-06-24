# GPU backend suite — numpy vs cuda

- GPU: **Tesla T4**

`benchmarks.gpu_profile`, 30 trees, each task at a narrow (30f) and wide (200f) shape. Speedup = numpy / cuda (higher is better for cuda). The cuda histogram + on-device numeric scan (Phase B2) pays off on the wide shapes where the per-node histogram is large.

| case | task | features | numpy fit (s) | cuda fit (s) | fit speedup | numpy pred (s) | cuda pred (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| binary_30f_bins256 | binary | 30 | 3.10 | 2.03 | **1.53x** | 0.072 | 0.071 |
| binary_200f_bins256 | binary | 200 | 11.82 | 7.24 | **1.63x** | 0.212 | 0.235 |
| multiclass_c5_200f_bins256 | multiclass | 200 | 45.99 | 21.71 | **2.12x** | 0.365 | 0.350 |
| regression_30f_bins256 | regression | 30 | 2.51 | 1.97 | **1.27x** | 0.074 | 0.076 |
| regression_200f_bins256 | regression | 200 | 12.33 | 6.68 | **1.85x** | 0.391 | 0.233 |
