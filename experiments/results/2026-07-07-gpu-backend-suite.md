# GPU backend suite — numpy vs cuda

- GPU: **Tesla T4**

`benchmarks.gpu_profile`, 30 trees, each task at a narrow (30f) and wide (200f) shape. Speedup = numpy / cuda (higher is better for cuda). The cuda histogram + on-device numeric scan (Phase B2) pays off on the wide shapes where the per-node histogram is large.

| case | task | features | numpy fit (s) | cuda fit (s) | fit speedup | numpy pred (s) | cuda pred (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| binary_30f_bins256 | binary | 30 | 2.98 | 1.98 | **1.51x** | 0.072 | 0.074 |
| binary_200f_bins256 | binary | 200 | 11.67 | 5.60 | **2.08x** | 0.207 | 0.221 |
| multiclass_c5_200f_bins256 | multiclass | 200 | 45.15 | 14.70 | **3.07x** | 0.610 | 0.379 |
| multioutput_k5_30f_bins256 | multioutput | 30 | 10.81 | 5.93 | **1.82x** | 0.081 | 0.081 |
| multioutput_k5_200f_bins256 | multioutput | 200 | 54.15 | 10.58 | **5.12x** | 0.466 | 0.473 |
| regression_30f_bins256 | regression | 30 | 2.36 | 2.51 | **0.94x** | 0.073 | 0.138 |
| regression_200f_bins256 | regression | 200 | 12.09 | 5.38 | **2.25x** | 0.249 | 0.234 |
