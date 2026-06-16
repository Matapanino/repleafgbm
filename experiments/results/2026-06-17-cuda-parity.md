# CUDA backend parity report

- GPU: **Tesla T4**
- Parity tests (`tests/test_cuda_backend.py`): **PASS** — `7 passed in 3.11s`

## Histogram micro-benchmark

Single `build_histograms` over 200,000 rows x 50 features x 65 bins (mean of 5):

- NumPy: **140.54 ms**
- CUDA:  **4.35 ms**
- Speedup: **32.31x**

_Phase B1: binned is uploaded once and cached on-device (keyed by identity); each call ships only its rows + gathered grad/hess, so repeated builds over the same matrix avoid re-transferring it._

## End-to-end training (the number that sizes Phase C)

`RepLeafRegressor.fit` over 100,000 rows x 30 features, 50 trees, embedded_linear (GPU histogram + host leaf fitting):

- numpy backend: **12.82 s**
- cuda backend:  **8.12 s**
- Speedup: **1.58x**

_A large speedup ⇒ histogram dominated, so Phase C1 (GPU leaf stats) adds little; a modest one ⇒ host leaf fitting is now the bottleneck and C1 is justified._
