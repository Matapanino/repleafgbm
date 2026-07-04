# Device leaf-fit extended to pooled multiclass + multi-output vector leaves — T4 A/B

Colab Tesla T4, 2026-07-05. Interleaved paired A/B, 5 reps per arm
(`REPLEAFGBM_CUDA_LEAF_FIT` off → on), `OMP_NUM_THREADS=1`, direct
`benchmarks.gpu_profile` invocations per arm (the `cuda_overnight_loop` A/B
CLI lacks `--n-classes/--n-outputs`; noted as harness follow-up), 20 trees,
`split_backend="cuda"`, `embedded_linear`.

## iter 014 — pooled multiclass (`leaf_fit_stats_mc`)

| case | A: host (native pooled Rust) | B: device | speedup | wins |
|---|---:|---:|---:|---|
| mc5, 30k×200f, emb 64 | 14.21 s | 12.46 s | **1.14×** (−12.3%) | 5/5 |

The baseline here is the *fast* pooled native Rust kernel, not the BLAS loop —
the device still wins on the Colab host's weak 2-vCPU CPU.

## iter 015 — multi-output vector leaves (`leaf_fit_stats_vector`)

| case | A: host (NumPy loop) | B: device | speedup | wins |
|---|---:|---:|---:|---|
| MO5, 30k×200f, emb 200 (wide) | 12.53 s | 9.95 s | **1.26×** (−20.6%) | 5/5 |
| MO5, 50k×30f, emb 30 (narrow), pre-fix | 5.27 s | 5.52 s | 0.95× (−4.7%) | 2/5 |

**Narrow regression → fixed with a vector-specific crossover.** The vector
path's per-leaf device work (one Gram + one (d, K) cross GEMM) is too small at
narrow embeddings: 50k×emb30 = 1.5M cells regressed under the shared 1e6-cell
threshold while 30k×emb200 = 6M cells wins. New
`_GPU_LEAF_FIT_MIN_CELLS_VECTOR = 4e6` splits the measured points; an explicit
`REPLEAFGBM_CUDA_LEAF_FIT_MIN_CELLS` override applies to both paths (tests and
forced-device profiling unchanged). Post-fix narrow recheck in both pair
orders: medians within ±6% with reps spanning ±13% and wins split 2–3/5 —
**parity within VM noise** (both arms now run the identical host path), the
consistent 5/5 loss is gone.

## Crossover sweep (iter 012 follow-up)

Scalar 20k×10 (200k cells/tree), `REPLEAFGBM_CUDA_LEAF_FIT_MIN_CELLS` ∈
{0, 1e5, 1e6, 4e6}: forced-device 0.93 s vs host 0.77–0.80 s median → the
**1e6 scalar default is validated** (tiny trees correctly stay on the host;
forcing the device there costs ~20%).

## Parity

50/50 GPU tests green (`test_cuda_backend.py` + `test_cuda_leaf_fit.py`,
including the new `leaf_fit_stats_mc` / `leaf_fit_stats_vector` parity at
rtol 1e-9 and the multiclass-accuracy / multi-output-R² forced-device
quality-equivalence e2e tests), re-run after the crossover hot-patch.

## Verdict

Ship all three device leaf-fit paths default-ON with the two-tier crossover
(scalar/mc 1e6, vector 4e6) and the existing kill switch. Follow-ups:
`cuda_overnight_loop --mode ab` should learn `--n-classes/--n-outputs`
(harness); a finer vector-crossover sweep between 1.5M and 6M cells is
optional polish.
