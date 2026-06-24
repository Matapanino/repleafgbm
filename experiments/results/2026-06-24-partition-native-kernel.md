# Rust native row-partition kernel (`partition_rows`)

Date: 2026-06-24
Change: fused single-pass Rust kernel for `Splitter.partition` behind
`split_backend="rust"` (native `0.1.0 -> 0.2.0`). The NumPy reference is
unchanged and now lives on `BaseSplitBackend`; parity is index-exact.

## Motivation

After the histogram, leaf-fit (rayon), multiclass leaf pooling, and fused linear
prediction kernels landed, row partitioning was the last pure-NumPy hot phase in
tree growth. `docs/gpu_audit.md` named a native partition the best first compiled
target ahead of a CUDA tree traversal.

## Method

- `benchmarks/gpu_profile.py --backend rust`, per-phase breakdown, multiclass
  K=5, 100 estimators, before vs after, medium and large presets (single seed;
  the partition saving is large relative to run-to-run noise).
- `benchmarks/partition_microbench.py`: isolated NumPy-vs-Rust `partition_rows`
  across node sizes, for a numeric and a categorical split.

## Pre-change partition share (rust backend)

| size | partition | fit | share |
|---|---|---|---|
| medium 100k×100 | 4.09s | 30.35s | 13.5% |
| large 500k×200 | 20.75s | 205.72s | 10.1% |

Higher than the NumPy-era 8.4% / 10.8% baseline because the other Rust kernels
already shrank the remaining phases — so partition's relative slice grew.

## Before -> after (rust backend, multiclass)

| size | partition | end-to-end fit | quality |
|---|---|---|---|
| medium 100k×100 | 4.09s -> 0.83s (4.9x) | 30.35s -> 26.68s (-12.1%) | bitwise-identical |
| large 500k×200 | 20.75s -> 4.57s (4.5x) | 205.72s -> 186.53s (-9.3%) | bitwise-identical |

`multi_logloss` and `accuracy` are unchanged to full precision, confirming the
kernel produces identical partitions in a real fit.

## Microbench (isolated kernel, 500k×200, 5% missing)

Rust wins at every node size from 16 to 262144 rows: ~3-5x numeric, ~10-15x
categorical (NumPy's `np.isin` is the slow path). The crossover is below 16
rows, so the planned min-rows FFI fallback gate was dropped as dead complexity.

## Parity

16 `tests/test_rust_backend.py` tests pass with strict `assert_array_equal` on
both children: numeric, categorical (incl. codes absent from the node), empty /
all-left / all-right / singleton edges, tiny-and-large node sizes, int32 /
non-contiguous rows, and the older-native `getattr` fallback. The three existing
end-to-end backend-agreement tests cover the integrated path.

## Decision

Keep. Partition is now off the CPU/native perf-candidate list; the remaining
native lever in this area is the `Tree.apply` / predictor kernel.
