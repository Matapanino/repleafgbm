# Node-batched depthwise scan A/B (Colab T4)

REPLEAFGBM_CUDA_BATCHED_SCAN off (per-node device scan) vs on (level-batched). `scan` = split_scan phase seconds (median of 5, interleaved). Quality: r2 (reg) / accuracy (clf).

| case | shape | depth | fit off | fit on | fit× | scan off | scan on | scan× | |Δq| |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wide | 50000x200 | 8 | 14.59 | 3.73 | **3.91x** | 11.613 | 1.289 | **9.01x** | 0.0e+00 |
| narrow | 100000x30 | 8 | 4.96 | 2.60 | **1.91x** | 2.796 | 0.564 | **4.95x** | 0.0e+00 |
| multiclass | 50000x200 | 8 | 14.25 | 4.41 | **3.23x** | 11.164 | 1.625 | **6.87x** | 0.0e+00 |
