# Working assumptions — CUDA/GPU overnight loop

Decisions taken to avoid stalling. Each is reversible; if the user corrects one,
update here and re-evaluate any affected verdict.

| # | Assumption | Rationale | Reverse impact |
|---|---|---|---|
| A1 | Colab GPU = **T4** for validation passes. | Cheapest; the memory-validated baseline (docs/cuda.md numbers are T4). | L4/A100 needed only for a bigger-GPU threshold sweep. |
| A2 | **1** Colab T4 pass overnight (end/morning), batching accumulated CUDA candidates. | Local-first decision; minimise billable T4. | More passes need explicit pre-approval. |
| A3 | Base branch = `cuda-multioutput-device-scan` (work branch `perf/cuda-overnight-loop-20260624` forks it). | Latest CUDA code incl. MO device scan; `main` lacks it. | On `main`, the split_scan-batch experiment loses its MO base. |
| A4 | New perf paths (e.g. float32 cache) stay **default-off / internal**, default behaviour unchanged. | No public-API/default change without a results-analyst report. | A needed public param splits into a separate API proposal. |
| A5 | `benchmarks/results/latest.jsonl` reuses the `gpu_profile` schema (rolling scratch); `artifacts/gpu_bench/` stays the dated archive. | No schema fork; historical comparability. | — |
| A6 | Local runs use `OMP_NUM_THREADS=1`. | Avoids torch+lightgbm libomp deadlock; stable single-thread timing. | — |
| A7 | "Accept" requires +3% **median** over ≥5 reps + green + parity. | Noise-aware gate; matches schema.md. | — |

_(append new assumptions as the loop encounters unknowns)_
