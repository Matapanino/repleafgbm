# CUDA scan-threshold A/B confirmation

- GPU: **Tesla T4**
- A = **32768** (on-device scan at 200f) vs B = **131072** (host scan); 5 interleaved reps, identical data, paired diff `A - B` (positive ⇒ host faster). 200f, 30 trees, cuda backend.
- Measurement only — not a default change.

## Provenance

`artifacts/gpu_bench/2026-06-23-T4/scan_ab.jsonl` was **reconstructed from the
run's stdout log**, not emitted directly by `benchmarks.gpu_profile.run_case`: the
GPU run completed all 30 fits but crashed on the final JSONL write (a missing
parent dir, since fixed in `scripts/colab_scan_ab.py`). The `fit_seconds` /
`split_scan` figures below are the values the driver printed immediately after
each fit, so they are faithful, but the reconstructed rows lack `quality` /
`transfer_bytes`. Quality-invariance across thresholds was verified separately
from the sweep JSONL (`artifacts/gpu_bench/2026-06-23-T4/scan_sweep.jsonl`):
quality is identical across all thresholds — the threshold is a pure speed knob.
(Both JSONLs are gitignored under `artifacts/`; the tables here are the
version-controlled record.)

## Verdict (results-analyst)

- **Default threshold stays `_GPU_SCAN_MIN_CELLS = 32768`** — the change is
  **deferred**, not adopted.
- **131072 is a candidate / per-GPU tuning point, not the new default.**
- The T4 **multiclass evidence is strong** (host +4.9%, 5/5, p<0.01); regression
  and binary 200f are within fit-level noise. But the **global-default evidence
  is insufficient**: T4 only, one seed, one 200f boundary shape, no L4/A100, and
  no ≥131072-cell regime. A broader benchmark matrix is required before any
  default change.

| workload | A=32768 mean (s) | B=131072 mean (s) | paired Δ(A-B) mean | paired Δ % | host faster | signal |
| --- | --- | --- | --- | --- | --- | --- |
| binary, 200f | 8.02 | 8.09 | -0.08 ± 0.92 | -1.0% | 1/5 | within noise |
| multiclass K=5, 200f | 23.51 | 22.36 | +1.14 ± 0.24 | +4.9% | 5/5 | host edge confirmed |
| regression, 200f | 8.36 | 8.14 | +0.22 ± 0.98 | +2.6% | 3/5 | within noise |

## Per-rep fit times (s)

### binary, 200f

| rep | A=32768 (GPU) | B=131072 (host) | Δ(A-B) |
| --- | --- | --- | --- |
| 0 | 8.73 | 8.74 | -0.01 |
| 1 | 7.65 | 8.74 | -1.09 |
| 2 | 8.85 | 7.24 | +1.61 |
| 3 | 7.39 | 7.98 | -0.59 |
| 4 | 7.46 | 7.77 | -0.31 |

### multiclass K=5, 200f

| rep | A=32768 (GPU) | B=131072 (host) | Δ(A-B) |
| --- | --- | --- | --- |
| 0 | 23.30 | 21.98 | +1.32 |
| 1 | 23.58 | 22.26 | +1.32 |
| 2 | 23.79 | 22.43 | +1.36 |
| 3 | 23.40 | 22.63 | +0.77 |
| 4 | 23.46 | 22.52 | +0.94 |

### regression, 200f

| rep | A=32768 (GPU) | B=131072 (host) | Δ(A-B) |
| --- | --- | --- | --- |
| 0 | 7.63 | 7.65 | -0.02 |
| 1 | 9.01 | 7.24 | +1.77 |
| 2 | 7.43 | 8.70 | -1.27 |
| 3 | 8.87 | 8.33 | +0.54 |
| 4 | 8.84 | 8.77 | +0.07 |
