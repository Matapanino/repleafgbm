# GPU backend suite — numpy vs cuda

- GPU: **Tesla T4**

`benchmarks.gpu_profile`, 30 trees, each task at a narrow (30f) and wide (200f) shape. Speedup = numpy / cuda (higher is better for cuda). The cuda histogram + on-device numeric scan (Phase B2) pays off on the wide shapes where the per-node histogram is large.

| case | task | features | numpy fit (s) | cuda fit (s) | fit speedup | numpy pred (s) | cuda pred (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| binary_30f_bins256 | binary | 30 | 2.34 | 2.44 | **0.96x** | 0.068 | 0.070 |
| binary_200f_bins256 | binary | 200 | 11.18 | 6.36 | **1.76x** | 0.203 | 0.216 |
| multiclass_c5_200f_bins256 | multiclass | 200 | 43.70 | 21.67 | **2.02x** | 0.339 | 0.323 |
| multioutput_k5_30f_bins256 | multioutput | 30 | 10.51 | 5.58 | **1.88x** | 0.081 | 0.079 |
| multioutput_k5_200f_bins256 | multioutput | 200 | 52.79 | 9.59 | **5.50x** | 0.443 | 0.446 |
| regression_30f_bins256 | regression | 30 | 2.36 | 1.90 | **1.24x** | 0.072 | 0.075 |
| regression_200f_bins256 | regression | 200 | 11.51 | 7.03 | **1.64x** | 0.254 | 0.238 |
