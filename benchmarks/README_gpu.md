# GPU Benchmarking Plan

This document defines the benchmark plan for RepLeafGBM GPU and native
acceleration work. It is a design/TODO document; the current runnable CUDA loop
is still `scripts/colab_gpu_test.sh`.

## Current Runnable CUDA Check

Run on a Colab GPU VM:

```bash
bash scripts/colab_gpu_test.sh --gpu T4
```

Optional larger GPUs:

```bash
bash scripts/colab_gpu_test.sh --gpu L4
bash scripts/colab_gpu_test.sh --gpu A100
```

The current script runs `tests/test_cuda_backend.py`, a histogram
micro-benchmark, two end-to-end fit cases, and the `benchmarks.gpu_profile`
matrix (numpy vs cuda across regression/binary/multiclass at narrow/wide
shapes). Two reports are downloaded:

```text
experiments/results/<date>-cuda-parity.md        # parity + micro-bench + transfer counters
experiments/results/<date>-gpu-backend-suite.md  # numpy vs cuda fit/predict speedups
```

## Benchmark Matrix To Implement

### Backends

- `split_backend="numpy"`
- `split_backend="rust"` when `repleafgbm_native` is installed
- `split_backend="cuda"` on GPU

### Tasks

- Regression: RMSE, MAE.
- Binary classification: logloss, AUC, balanced accuracy.
- Multiclass classification: 3, 5, 10 classes; multi-logloss, accuracy,
  balanced accuracy.
- Optional multi-output regression: 2, 8, 32 outputs; RMSE/MAE per output and
  averaged.

### Dataset Sizes

| size | train rows | test rows | feature counts | implemented in `gpu_profile._SIZES` |
|---|---:|---:|---|---|
| small | 20,000 | 10,000 | 30 | ✅ |
| medium | 100,000 | 50,000 | 100 | ✅ |
| large | 500,000 | 100,000 | 200 | ✅ |
| stress | 1,000,000 | 200,000 | 200 | ✅ |

Use T4 for correctness and medium benchmarks. Use L4/A100 for large/stress
cases when available.

### Model Settings

Sweep the following:

- `leaf_model`: `constant`, `embedded_linear`
- `encoder`: `identity`, `plr`, optionally `torch_mlp`
- `max_bins`: 32, 64, 256, 1024
- `num_leaves`: 8, 31, 127
- `max_leaf_emb_dim`: 16, 64, 256
- `n_estimators`: 30 for smoke, 100 for standard, 300 for prediction stress
- categorical fraction: 0%, 10%, 50%
- categorical cardinality: 8, 64, 256+

## Metrics To Log

Required:

- fit time
- predict time
- peak host RSS
- GPU memory allocated and peak GPU memory
- GPU utilization
- GPU memory utilization
- CPU utilization
- estimated transfer bytes
- quality metrics
- backend parity vs NumPy where applicable

Strongly recommended:

- phase timings: preprocessing, binning, histogram, split scan, partition, leaf
  fit, eval, predict
- kernel count and small/large scan path counts
- number of histogram builds
- number of categorical slices copied back
- number of multi-output histogram copies
- CuPy memory-pool stats before/after each fit

## Output Layout

Write all benchmark artifacts under:

```text
artifacts/gpu_bench/<date>/<gpu-name>/
```

Suggested files:

```text
cases.jsonl
nvidia_smi.csv
summary.md
environment.json
```

Each JSONL row should include:

```json
{
  "case_id": "regression_medium_100f_bins256_cuda",
  "task": "regression",
  "backend": "cuda",
  "n_train": 100000,
  "n_test": 50000,
  "n_features": 100,
  "max_bins": 256,
  "num_leaves": 31,
  "leaf_model": "embedded_linear",
  "encoder": "identity",
  "fit_seconds": 0.0,
  "predict_seconds": 0.0,
  "peak_rss_bytes": 0,
  "peak_gpu_bytes": 0,
  "quality": {},
  "phase_seconds": {},
  "transfer_bytes": {},
  "env": {}
}
```

## Commands (implemented)

`benchmarks/gpu_profile.py` exists and runs one case per invocation, appending a
JSONL row (schema above) and regenerating `summary.md` beside it:

```bash
python -m benchmarks.gpu_profile --task regression --size small --backend numpy --out artifacts/gpu_bench/dev/cases.jsonl
python -m benchmarks.gpu_profile --task regression --size small --backend cuda --out artifacts/gpu_bench/dev/cases.jsonl
python -m benchmarks.gpu_profile --task binary --size medium --backend cuda --max-bins 256 --out artifacts/gpu_bench/dev/cases.jsonl
python -m benchmarks.gpu_profile --task multiclass --n-classes 5 --size medium --backend cuda --out artifacts/gpu_bench/dev/cases.jsonl
```

`--size {small,medium,large,stress}` overrides `--n-train/--n-test/--n-features`;
other knobs: `--leaf-model`, `--encoder` (any encoder name, incl. learned
`torch_periodic_plr`), `--device {cpu,cuda,auto}` for learned-encoder pretraining
(v1.5.0; torch encoders only), `--num-leaves`, `--max-leaf-emb-dim`,
`--n-estimators`, `--quick`, and `--parity` (also fits a numpy twin and records
`parity_max_abs_diff`). `numpy`/`rust` backends run on CPU; `cuda` needs a GPU.

On the GPU, the Colab loop runs the `gpu_profile` matrix automatically, writes a
backend-comparison suite, and pulls everything back:

```bash
bash scripts/colab_gpu_test.sh --gpu T4
#  -> experiments/results/<date>-cuda-parity.md
#  -> experiments/results/<date>-gpu-backend-suite.md
#  -> artifacts/gpu_bench/<date>-T4/cases.jsonl
```

## Profiling TODO

- **Done:** `CudaSplitBackend` private transfer counters — binned / rows /
  grad-hess H2D bytes, full-histogram D2H (small scans), categorical-slice D2H,
  and winner-scalar D2H — surfaced via `get_transfer_stats()` and recorded in the
  JSONL `transfer_bytes` field.
- Add optional timers around `Splitter.__init__`, `TreeGrower.grow`,
  backend histogram build, backend split scan, `Splitter.partition`,
  `fit_leaves`, eval tree application, and prediction (populates the
  currently-empty `phase_seconds` field; deferred to keep the boosting loop
  untouched for now).
- Add `nvidia-smi` sampling during fit and predict:

```bash
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv -lms 200
```

- Record CuPy memory pool:

```python
pool = cupy.get_default_memory_pool()
used = pool.used_bytes()
total = pool.total_bytes()
```

- Add one Nsight Systems trace for a representative wide CUDA run after the
  benchmark runner is stable.

## Smoke Suite

Use this before and after every CUDA change:

- regression, small, 30 features, `max_bins=64`, `num_leaves=31`
- regression, small, 200 features, `max_bins=256`, `num_leaves=31`
- binary, small, 50 features, `max_bins=256`
- multiclass 5-class, small, 50 features, `max_bins=256`
- categorical regression, small, 10% categoricals, low cardinality

Pass criteria:

- CUDA predictions match NumPy within documented allclose tolerance.
- CUDA is not materially slower than NumPy on wide cases.
- Any narrow-case slowdown is explained by the adaptive host scan path.
- GPU memory returns to a stable baseline after repeated runs.

