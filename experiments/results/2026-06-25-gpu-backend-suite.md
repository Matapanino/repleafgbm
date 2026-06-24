# GPU backend suite — numpy vs cuda

- GPU: **Tesla T4**

`benchmarks.gpu_profile`, 30 trees, each task at a narrow (30f) and wide (200f) shape. Speedup = numpy / cuda (higher is better for cuda). The cuda histogram + on-device numeric scan (Phase B2) pays off on the wide shapes where the per-node histogram is large.

| case | task | features | numpy fit (s) | cuda fit (s) | fit speedup | numpy pred (s) | cuda pred (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| binary_30f_bins256 | binary | 30 | 2.47 | 2.60 | **0.95x** | 0.069 | 0.072 |
| binary_200f_bins256 | binary | 200 | 11.90 | 7.22 | **1.65x** | 0.226 | 0.345 |
| multiclass_c5_200f_bins256 | multiclass | 200 | 44.83 | 21.72 | **2.06x** | 0.488 | 0.564 |
| multioutput_k5_30f_bins256 | multioutput | 30 | 10.87 | 5.28 | **2.06x** | 0.086 | 0.083 |
| multioutput_k5_200f_bins256 | multioutput | 200 | 55.44 | 10.27 | **5.40x** | 0.461 | 0.458 |
| regression_30f_bins256 | regression | 30 | 3.21 | 2.11 | **1.52x** | 0.083 | 0.097 |
| regression_200f_bins256 | regression | 200 | 12.20 | 7.37 | **1.66x** | 0.222 | 0.223 |
