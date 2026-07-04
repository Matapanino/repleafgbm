# GPU backend suite — numpy vs cuda

- GPU: **Tesla T4**

`benchmarks.gpu_profile`, 30 trees, each task at a narrow (30f) and wide (200f) shape. Speedup = numpy / cuda (higher is better for cuda). The cuda histogram + on-device numeric scan (Phase B2) pays off on the wide shapes where the per-node histogram is large.

| case | task | features | numpy fit (s) | cuda fit (s) | fit speedup | numpy pred (s) | cuda pred (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| binary_30f_bins256 | binary | 30 | 2.47 | 2.03 | **1.22x** | 0.071 | 0.114 |
| binary_200f_bins256 | binary | 200 | 11.74 | 4.94 | **2.38x** | 0.361 | 0.232 |
| multiclass_c5_200f_bins256 | multiclass | 200 | 46.82 | 16.17 | **2.89x** | 0.372 | 0.461 |
| multioutput_k5_30f_bins256 | multioutput | 30 | 11.25 | 5.22 | **2.15x** | 0.082 | 0.087 |
| multioutput_k5_200f_bins256 | multioutput | 200 | 54.85 | 9.68 | **5.67x** | 0.442 | 0.472 |
| regression_30f_bins256 | regression | 30 | 3.03 | 1.99 | **1.53x** | 0.078 | 0.075 |
| regression_200f_bins256 | regression | 200 | 12.21 | 6.14 | **1.99x** | 0.273 | 0.221 |
