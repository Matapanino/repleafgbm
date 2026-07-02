# CUDA device leaf-fit (GPU leaf ridge) + leafwise children-pair batch — T4 A/B

Colab Tesla T4, 2026-07-02. `benchmarks/cuda_overnight_loop.py --mode ab`, 5 interleaved
reps per arm (paired; interleaving cancels thermal drift), `OMP_NUM_THREADS=1`,
`RepLeafRegressor` embedded_linear, `split_backend="cuda"`, default `grow_policy`
(leafwise), 30 trees. Raw JSONL: `artifacts/gpu_bench/2026-07-02-T4-ab/`.

## iter 012 — device leaf-fit statistics (`REPLEAFGBM_CUDA_LEAF_FIT` off → on)

| case | A: host leaf fit (p50) | B: device leaf fit (p50) | speedup | wins | signal |
|---|---:|---:|---:|---|---|
| wide 50k×200, emb 200 | 14.94 s | 8.70 s | **1.72×** | 5/5 | True |
| narrow 100k×30, emb 30 | 4.19 s | 3.41 s | **1.23×** | 5/5 | True |

Both arms otherwise default (batched scans ON). The narrow case clears the provisional
`REPLEAFGBM_CUDA_LEAF_FIT_MIN_CELLS = 1e6` crossover (100k×30 = 3M cells) and still wins,
so the provisional default is kept — no measured regression case yet; a finer crossover
sweep is future harness work.

## iter 013 — leafwise children-pair batched scan (`REPLEAFGBM_CUDA_LEAFWISE_BATCH` off → on)

| case | A: per-node scans (p50) | B: children-pair batch (p50) | speedup | wins | signal |
|---|---:|---:|---:|---|---|
| wide 50k×200, emb 200 | 10.24 s | 8.83 s | **1.16×** | 5/5 | True |

Both arms have the device leaf fit ON (B arm ≈ iter-012's B arm, 8.8 s — consistent).
−13.8% whole fit matches the ledger's projected ~14% ceiling for M=2 batching of the
32.2%-share leafwise scan.

## Parity / quality

- Full T4 parity: **53 passed / 0 failed** (`tests/test_cuda_backend.py` +
  `tests/test_cuda_leaf_fit.py` + `tests/test_batched_scan.py`) after fixing the
  z_min/z_max bug below.
- Forced-device e2e quality-equivalence: |Δr²| < 5e-3 for `embedded_linear` and
  `adaptive` (near-tied LOO-gate flips are the leaf-fit analog of near-tied split
  flips; ADR 0005).
- **Bug found by the first parity run:** CuPy's `cupyx.scatter_min`/`scatter_max` on
  float64 round through float32 (~5e-8 relative error), while `bincount`/
  `scatter_add`/cuBLAS GEMM are f64-exact (~1e-13, probed in isolation). The
  extrapolation-guard bounds now ride the per-leaf GEMM loop as exact slice
  reductions.

## Verdict

Ship both defaults ON (kill switches retained): the device leaf fit is the largest
single CUDA-fit win since the batched depthwise scan (leaf_fit was 49–73% of fit), and
the leafwise batch delivers its full measured ceiling. Combined, leafwise-wide CUDA fit
drops ~2.0× vs the pre-session build (host leaf fit + per-node leafwise scans). Same
session's backend suite vs NumPy @30 trees: regression-wide 1.99×, multiclass-200f
3.06×, multioutput-200f 5.34× (`experiments/results/2026-07-02-gpu-backend-suite.md`).
Follow-ups: multiclass-pooled + multi-output vector leaf-fit variants of
`leaf_fit_stats`, and a `REPLEAFGBM_CUDA_LEAF_FIT_MIN_CELLS` sweep.
