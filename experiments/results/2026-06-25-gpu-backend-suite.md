# GPU backend suite — numpy vs cuda

- GPU: **Tesla T4**

`benchmarks.gpu_profile`, 30 trees, each task at a narrow (30f) and wide (200f) shape. Speedup = numpy / cuda (higher is better for cuda). The cuda histogram + on-device numeric scan (Phase B2) pays off on the wide shapes where the per-node histogram is large.

| case | task | features | numpy fit (s) | cuda fit (s) | fit speedup | numpy pred (s) | cuda pred (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| binary_30f_bins256 | binary | 30 | 2.48 | 2.05 | **1.21x** | 0.084 | 0.069 |
| binary_200f_bins256 | binary | 200 | 11.56 | 6.46 | **1.79x** | 0.339 | 0.223 |
| multiclass_c5_200f_bins256 | multiclass | 200 | 45.07 | 21.14 | **2.13x** | 0.351 | 0.341 |
| multioutput_k5_30f_bins256 | multioutput | 30 | 10.86 | 5.66 | **1.92x** | 0.081 | 0.082 |
| multioutput_k5_200f_bins256 | multioutput | 200 | 54.48 | 9.71 | **5.61x** | 0.674 | 0.678 |
| regression_30f_bins256 | regression | 30 | 2.78 | 2.11 | **1.32x** | 0.123 | 0.091 |
| regression_200f_bins256 | regression | 200 | 12.34 | 7.24 | **1.70x** | 0.237 | 0.233 |
