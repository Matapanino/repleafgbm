# Colab T4 sizing — iter-010 (batched histogram) + Task-B (leafwise) 

embedded_linear, split_backend=cuda, batched depthwise scan default ON, median of 3, REPLEAFGBM_PROFILE.

| case | policy | fit(s) | histogram | split_scan | leaf_fit | hist% | scan% |
|---|---|---:|---:|---:|---:|---:|---:|
| depthwise-wide | depthwise | 14.10 | 0.452 | 1.171 | 9.186 | 3.2% | 8.3% |
| depthwise-mc5 | depthwise | 42.79 | 1.162 | 3.186 | 31.072 | 2.7% | 7.4% |
| leafwise-wide | leafwise | 19.66 | 0.481 | 6.327 | 9.551 | 2.4% | 32.2% |
