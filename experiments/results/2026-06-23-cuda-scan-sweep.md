# CUDA adaptive-scan threshold sweep

- GPU: **Tesla T4**
- Default threshold: **32768** (`_GPU_SCAN_MIN_CELLS`); override is the private `REPLEAFGBM_CUDA_SCAN_MIN_CELLS`.

Per workload, `benchmarks.gpu_profile` cuda fit (30 trees) across `REPLEAFGBM_CUDA_SCAN_MIN_CELLS` ∈ {0, 8192, 32768, 131072, very_large}. `0` forces every node onto the GPU scan; `very_large` forces the host scan. Lower fit is better; **bold** = fastest threshold for the workload. Measurement only — not a default change.

## Headline (fastest threshold vs default 32768)

- **binary, 30f**: best=very_large (1.57s) vs default 32768 (2.03s) → +22.7% headroom
- **binary, 200f**: best=131072 (5.97s) vs default 32768 (6.56s) → +9.0% headroom
- **multiclass K=5, 200f**: best=131072 (20.16s) vs default 32768 (21.27s) → +5.2% headroom
- **regression, 30f**: best=very_large (1.56s) vs default 32768 (1.59s) → +1.9% headroom
- **regression, 200f**: best=131072 (6.08s) vs default 32768 (6.82s) → +10.9% headroom

## binary, 30f

| threshold | fit (s) | split_scan (s) | n_small | n_gpu | quality |
| --- | --- | --- | --- | --- | --- |
| 0 | 4.72 | 3.55 | 0 | 1829 | logloss=0.1474, auc=0.9936, accuracy=0.9537 |
| 8192 | 1.57 | 0.81 | 1829 | 0 | logloss=0.1474, auc=0.9936, accuracy=0.9537 |
| 32768 | 2.03 | 1.07 | 1829 | 0 | logloss=0.1474, auc=0.9936, accuracy=0.9537 |
| 131072 | 1.64 | 0.85 | 1829 | 0 | logloss=0.1474, auc=0.9936, accuracy=0.9537 |
| very_large | **1.57** | 0.81 | 1829 | 0 | logloss=0.1474, auc=0.9936, accuracy=0.9537 |

## binary, 200f

| threshold | fit (s) | split_scan (s) | n_small | n_gpu | quality |
| --- | --- | --- | --- | --- | --- |
| 0 | 7.07 | 3.87 | 0 | 1826 | logloss=0.1652, auc=0.9869, accuracy=0.9407 |
| 8192 | 6.01 | 3.39 | 0 | 1826 | logloss=0.1652, auc=0.9869, accuracy=0.9407 |
| 32768 | 6.56 | 3.77 | 0 | 1826 | logloss=0.1652, auc=0.9869, accuracy=0.9407 |
| 131072 | **5.97** | 3.37 | 1826 | 0 | logloss=0.1652, auc=0.9869, accuracy=0.9407 |
| very_large | 6.78 | 3.52 | 1826 | 0 | logloss=0.1652, auc=0.9869, accuracy=0.9407 |

## multiclass K=5, 200f

| threshold | fit (s) | split_scan (s) | n_small | n_gpu | quality |
| --- | --- | --- | --- | --- | --- |
| 0 | 21.61 | 17.88 | 0 | 8915 | multi_logloss=0.4296, accuracy=0.858 |
| 8192 | 21.12 | 17.74 | 0 | 8915 | multi_logloss=0.4296, accuracy=0.858 |
| 32768 | 21.27 | 17.93 | 0 | 8915 | multi_logloss=0.4296, accuracy=0.858 |
| 131072 | **20.16** | 16.83 | 8915 | 0 | multi_logloss=0.4296, accuracy=0.858 |
| very_large | 20.38 | 16.73 | 8915 | 0 | multi_logloss=0.4296, accuracy=0.858 |

## regression, 30f

| threshold | fit (s) | split_scan (s) | n_small | n_gpu | quality |
| --- | --- | --- | --- | --- | --- |
| 0 | 8.56 | 7.33 | 0 | 1826 | rmse=0.5384, mae=0.3748, r2=0.966 |
| 8192 | 1.61 | 0.85 | 1826 | 0 | rmse=0.5384, mae=0.3748, r2=0.966 |
| 32768 | 1.59 | 0.83 | 1826 | 0 | rmse=0.5384, mae=0.3748, r2=0.966 |
| 131072 | 1.61 | 0.85 | 1826 | 0 | rmse=0.5384, mae=0.3748, r2=0.966 |
| very_large | **1.56** | 0.82 | 1826 | 0 | rmse=0.5384, mae=0.3748, r2=0.966 |

## regression, 200f

| threshold | fit (s) | split_scan (s) | n_small | n_gpu | quality |
| --- | --- | --- | --- | --- | --- |
| 0 | 6.76 | 3.62 | 0 | 1828 | rmse=0.6241, mae=0.4274, r2=0.9537 |
| 8192 | 7.44 | 4.38 | 0 | 1828 | rmse=0.6241, mae=0.4274, r2=0.9537 |
| 32768 | 6.82 | 3.98 | 0 | 1828 | rmse=0.6241, mae=0.4274, r2=0.9537 |
| 131072 | **6.08** | 3.46 | 1828 | 0 | rmse=0.6241, mae=0.4274, r2=0.9537 |
| very_large | 6.42 | 3.72 | 1828 | 0 | rmse=0.6241, mae=0.4274, r2=0.9537 |
