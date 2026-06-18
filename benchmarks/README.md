# RepLeafGBM benchmarks

These benchmarks **track** RepLeafGBM across development and substantiate the
README's accuracy claims — they are not marketing. Synthetic datasets are small
and seeded; numbers are indicative. Every model trains on the same
ordinal-encoded feature matrix RepLeafGBM uses, so differences are in the model,
not the preprocessing.

External GBMs (LightGBM / XGBoost / CatBoost) are optional `[bench]` extras and
are skipped when not importable; learned encoders need the `[torch]` extra.
Always run with `OMP_NUM_THREADS=1` (avoids a torch+lightgbm libomp deadlock on
macOS):

```bash
pip install -e ".[bench,torch]"           # or: PYTHONPATH=src
OMP_NUM_THREADS=1 python3 benchmarks/<script>.py [--quick]
```

## Scripts

| script | task(s) | what it answers | output |
|---|---|---|---|
| `benchmark_synthetic_regression.py` | regression | leaf models, fixed+learned encoders, robust objectives, `--contamination` | stdout + `experiments/results/<date>-synthetic-regression.md` |
| `benchmark_synthetic_binary.py` | binary | leaf models, fixed+learned encoders | stdout + `experiments/results/<date>-synthetic-binary.md` |
| `benchmark_real_data.py` | reg + binary | where representation leaves help on real data; categorical handling; `--robust` | `experiments/results/real_data_validation.md` |
| `openml_suite.py` | reg + binary + multiclass | breadth-first leaderboard vs LightGBM/XGBoost/CatBoost/HistGB; `--learned-encoders`, `--strict` | `experiments/results/openml_benchmark.md` |
| `multioutput_suite.py` | multi-output regression | single-routing vector leaf vs per-output GBMs; robust multi-output (huber/quantile) under contamination | `experiments/results/multioutput_benchmark.md` |
| `trainable_embeddings.py` | reg + binary + multiclass | fixed vs learned encoder families, mean±std over seeds | `artifacts/trainable_embeddings/<date>/` + `experiments/results/<date>-trainable-embeddings.md` |
| `gpu_profile.py` | reg + binary + multiclass | one fit/predict case per invocation: timings, quality, peak memory, CUDA transfer counters | `artifacts/gpu_bench/cases.jsonl` + `summary.md` |

The GPU loop (`scripts/colab_gpu_test.sh --gpu T4`) drives `gpu_profile.py` on a
Colab GPU and writes `experiments/results/<date>-cuda-parity.md` and
`<date>-gpu-backend-suite.md` — see [`README_gpu.md`](README_gpu.md).

## Feature × benchmark coverage

Which suite exercises each capability added through v1.6.0:

| capability (since) | synthetic | real_data | openml_suite | multioutput_suite | trainable_embeddings | gpu_profile |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| constant / embedded_linear / raw_linear leaves | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| fixed encoders (identity/plr/periodic/cross) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| learned encoders `torch_periodic_plr` / `torch_mlp` (1.3.0) | ✅ | ✅ | `--learned-encoders` | ✅ | ✅ | `--encoder` |
| weighted / `(n,K)` vector pretraining (1.3–1.4) | — | — | — | ✅ (multi-output) | ✅ (multiclass) | — |
| robust objectives huber / quantile (scalar) | ✅ `--contamination` | `--robust` | — | ✅ | — | — |
| multi-output regression | — | — | — | ✅ | — | — |
| multi-output huber / quantile (1.5.0) | — | — | — | ✅ | — | — |
| GPU encoder pretraining `device="cuda"` (1.5.0) | — | — | — | — | — | `--device cuda` |
| CUDA split backend + transfer counters (1.6.0) | — | — | — | — | — | `--backend cuda` |
| categorical handling (native subset splits) | — | ✅ | ✅ | — | — | — |

## Reproducing the committed reports

```bash
# CPU suites (torch + external GBMs; a few minutes each)
OMP_NUM_THREADS=1 python3 benchmarks/openml_suite.py --learned-encoders --seeds 3 --strict
OMP_NUM_THREADS=1 python3 benchmarks/multioutput_suite.py --seeds 5
OMP_NUM_THREADS=1 python3 benchmarks/benchmark_real_data.py --seeds 3 --robust
OMP_NUM_THREADS=1 python3 benchmarks/benchmark_synthetic_regression.py
OMP_NUM_THREADS=1 python3 benchmarks/benchmark_synthetic_binary.py
OMP_NUM_THREADS=1 python3 benchmarks/trainable_embeddings.py --seeds 5

# GPU suite (Colab T4)
bash scripts/colab_gpu_test.sh --gpu T4
```

Add `--quick` to any script for a fast smoke run.
