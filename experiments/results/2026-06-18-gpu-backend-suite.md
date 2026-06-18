# GPU backend suite — numpy vs cuda

- GPU: **Tesla T4**

`benchmarks.gpu_profile`, 30 trees, each task at a narrow (30f) and wide (200f) shape. Speedup = numpy / cuda (higher is better for cuda). The cuda histogram + on-device numeric scan (Phase B2) pays off on the wide shapes where the per-node histogram is large.

| case | task | features | numpy fit (s) | cuda fit (s) | fit speedup | numpy pred (s) | cuda pred (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| binary_30f_bins256 | binary | 30 | 2.42 | 1.97 | **1.23x** | 0.069 | 0.070 |
| binary_200f_bins256 | binary | 200 | 11.56 | 6.77 | **1.71x** | 0.361 | 0.221 |
| multiclass_c5_200f_bins256 | multiclass | 200 | 44.10 | 21.55 | **2.05x** | 0.345 | 0.357 |
| regression_30f_bins256 | regression | 30 | 2.85 | 1.93 | **1.48x** | 0.123 | 0.090 |
| regression_200f_bins256 | regression | 200 | 11.86 | 7.38 | **1.61x** | 0.223 | 0.248 |
