# Node-batched depthwise scan A/B (Colab T4)

REPLEAFGBM_CUDA_BATCHED_SCAN off (per-node device scan) vs on (level-batched). `scan` = split_scan phase seconds (median of 5, interleaved). Quality: r2 (reg) / accuracy (clf).

| case | shape | depth | fit off | fit on | fit× | scan off | scan on | scan× | |Δq| |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wide | 50000x200 | 8 | 14.61 | 3.79 | **3.86x** | 11.824 | 1.300 | **9.09x** | 0.0e+00 |
| narrow | 100000x30 | 8 | 4.99 | 2.50 | **1.99x** | 2.778 | 0.562 | **4.94x** | 0.0e+00 |
| multiclass | 50000x200 | 8 | 14.57 | 4.95 | **2.94x** | 11.484 | 1.774 | **6.47x** | 0.0e+00 |
