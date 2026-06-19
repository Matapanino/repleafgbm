# GPU backend suite — numpy vs cuda

- GPU: **Tesla T4**

`benchmarks.gpu_profile`, 30 trees, each task at a narrow (30f) and wide (200f) shape. Speedup = numpy / cuda (higher is better for cuda). The cuda histogram + on-device numeric scan (Phase B2) pays off on the wide shapes where the per-node histogram is large.

| case | task | features | numpy fit (s) | cuda fit (s) | fit speedup | numpy pred (s) | cuda pred (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| binary_30f_bins256 | binary | 30 | 2.58 | 1.89 | **1.37x** | 0.068 | 0.071 |
| binary_200f_bins256 | binary | 200 | 11.62 | 7.10 | **1.64x** | 0.221 | 0.215 |
| multiclass_c5_200f_bins256 | multiclass | 200 | 43.53 | 20.94 | **2.08x** | 0.348 | 0.368 |
| regression_30f_bins256 | regression | 30 | 2.41 | 2.17 | **1.11x** | 0.074 | 0.127 |
| regression_200f_bins256 | regression | 200 | 12.00 | 6.75 | **1.78x** | 0.234 | 0.229 |
