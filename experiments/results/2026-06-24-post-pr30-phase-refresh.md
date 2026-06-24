# Post-PR #30 phase refresh

Date: 2026-06-24
Change context: PR #30 (`perf(native): add Rust fast path for row partitioning`)
is merged on `main` (`fcd7c9d`). PyPI publishing is deferred; this note is
measurement-only.

## Goal

Refresh the performance picture after the native `partition_rows` kernel, then
choose the next small performance direction from current phase evidence. This is
not a README headline refresh because it does not include paired NumPy/Rust
end-to-end rows.

## Method

CPU-safe short matrix, Rust backend only:

```bash
env OMP_NUM_THREADS=1 RAYON_NUM_THREADS=8 REPLEAFGBM_NUM_THREADS=8 \
  python3 -m benchmarks.gpu_profile --backend rust --task regression \
  --size medium --n-estimators 30 --seed 0 \
  --out artifacts/gpu_bench/post-pr30-short/cases.jsonl

env OMP_NUM_THREADS=1 RAYON_NUM_THREADS=8 REPLEAFGBM_NUM_THREADS=8 \
  python3 -m benchmarks.gpu_profile --backend rust --task binary \
  --size medium --n-estimators 30 --seed 0 \
  --out artifacts/gpu_bench/post-pr30-short/cases.jsonl

env OMP_NUM_THREADS=1 RAYON_NUM_THREADS=8 REPLEAFGBM_NUM_THREADS=8 \
  python3 -m benchmarks.gpu_profile --backend rust --task multiclass \
  --n-classes 5 --size medium --n-estimators 30 --seed 0 \
  --out artifacts/gpu_bench/post-pr30-short/cases.jsonl

env OMP_NUM_THREADS=1 RAYON_NUM_THREADS=8 \
  python3 -m benchmarks.partition_microbench
```

Environment from JSONL: Python 3.11.1, NumPy 1.23.5, scikit-learn 1.2.0,
`repleafgbm-native` 0.2.0, macOS arm64. `git_dirty=true` because pre-existing
untracked CUDA result notes were present locally.

## Rust phase results

Medium preset = 100k train rows, 50k test rows, 100 features, 30 estimators,
`leaf_model=embedded_linear`, `encoder=identity`, `max_leaf_emb_dim=64`.

| task | fit s | predict s | quality | top fit phases |
|---|---:|---:|---|---|
| regression | 2.686 | 0.301 | rmse=0.5520, r2=0.9639 | histogram 0.622, preprocessing 0.618, leaf_fit 0.458, binning 0.365 |
| binary | 3.509 | 0.323 | logloss=0.1427, auc=0.9933, acc=0.9543 | histogram 0.884, preprocessing 0.624, leaf_fit 0.609, binning 0.596 |
| multiclass K=5 | 8.860 | 1.222 | multi_logloss=0.4212, acc=0.8517 | histogram 4.005, leaf_fit 2.147, predict 1.153, split_scan 0.556 |

Partition is no longer a leading phase:

| task | partition s | partition / fit |
|---|---:|---:|
| regression | 0.064 | 2.4% |
| binary | 0.064 | 1.8% |
| multiclass K=5 | 0.257 | 2.9% |

## Partition microbench check

The isolated kernel still matches the PR #30 conclusion. Rust wins at every
tested node size from 16 to 262144 rows:

| split type | observed Rust speedup range |
|---|---:|
| numeric | 3.1x to 5.6x |
| categorical subset | 10.9x to 16.3x |

## Interpretation

- PR #30 moved row partitioning off the active bottleneck list for the Rust
  backend. The post-merge phase share is under 3% in the refreshed matrix.
- Current fit-time pressure is task-dependent: histogram and leaf fitting still
  dominate, while multiclass prediction traversal is now a visible separate
  cost even in a 30-tree short run.
- The next performance decision should not be another partition change.
- README native speed numbers should not be rewritten from this note alone; a
  paired NumPy/Rust matrix is still required for that claim.

## Next candidate signal

The best one-PR-sized next direction is a benchmark/docs PR for compiled
prediction planning: add a focused prediction benchmark that sweeps rows,
trees, classes, and categorical/missing routes, then use it to justify or reject
a Rust `Tree.apply` / `apply_forest` implementation. This is lower packaging
risk than GPU work, CPU CI-testable, and directly grounded in the post-PR #30
phase ranking.
