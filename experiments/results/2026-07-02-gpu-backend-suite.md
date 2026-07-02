# GPU backend suite — numpy vs cuda

- GPU: **Tesla T4**

`benchmarks.gpu_profile`, 30 trees, each task at a narrow (30f) and wide (200f) shape. Speedup = numpy / cuda (higher is better for cuda). The cuda histogram + on-device numeric scan (Phase B2) pays off on the wide shapes where the per-node histogram is large.

| case | task | features | numpy fit (s) | cuda fit (s) | fit speedup | numpy pred (s) | cuda pred (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| binary_30f_bins256 | binary | 30 | 2.49 | 2.18 | **1.14x** | 0.068 | 0.079 |
| binary_200f_bins256 | binary | 200 | 12.48 | 5.62 | **2.22x** | 0.238 | 0.248 |
| multiclass_c5_200f_bins256 | multiclass | 200 | 48.02 | 15.21 | **3.16x** | 0.376 | 0.378 |
| multioutput_k5_30f_bins256 | multioutput | 30 | 10.94 | 5.28 | **2.07x** | 0.144 | 0.084 |
| multioutput_k5_200f_bins256 | multioutput | 200 | 56.33 | 10.35 | **5.44x** | 0.457 | 0.468 |
| regression_30f_bins256 | regression | 30 | 2.53 | 2.25 | **1.12x** | 0.076 | 0.115 |
| regression_200f_bins256 | regression | 200 | 12.16 | 5.44 | **2.24x** | 0.214 | 0.349 |
