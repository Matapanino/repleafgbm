# GPU backend suite — numpy vs cuda

- GPU: **Tesla T4**

`benchmarks.gpu_profile`, 30 trees, each task at a narrow (30f) and wide (200f) shape. Speedup = numpy / cuda (higher is better for cuda). The cuda histogram + on-device numeric scan (Phase B2) pays off on the wide shapes where the per-node histogram is large.

| case | task | features | numpy fit (s) | cuda fit (s) | fit speedup | numpy pred (s) | cuda pred (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| binary_30f_bins256 | binary | 30 | 2.38 | 1.94 | **1.23x** | 0.116 | 0.071 |
| binary_200f_bins256 | binary | 200 | 11.40 | 7.06 | **1.61x** | 0.235 | 0.213 |
| multiclass_c5_200f_bins256 | multiclass | 200 | 43.78 | 20.51 | **2.13x** | 0.389 | 0.352 |
| multioutput_k5_30f_bins256 | multioutput | 30 | 9.96 | 4.98 | **2.00x** | 0.137 | 0.084 |
| multioutput_k5_200f_bins256 | multioutput | 200 | 53.58 | 10.06 | **5.33x** | 0.451 | 0.441 |
| regression_30f_bins256 | regression | 30 | 2.63 | 1.94 | **1.36x** | 0.075 | 0.075 |
| regression_200f_bins256 | regression | 200 | 11.79 | 6.76 | **1.75x** | 0.214 | 0.220 |
