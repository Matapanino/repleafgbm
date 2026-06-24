# `benchmarks/results/` schema

`benchmarks/results/latest.jsonl` is the **rolling** result file the overnight
perf loop appends to. It does **not** fork the measurement schema: every row is
produced by reusing `benchmarks/gpu_profile.run_case()`, so the descriptive +
measurement fields are exactly the `gpu_profile` schema documented in
`benchmarks/README_gpu.md`. The dated, per-GPU archive
(`artifacts/gpu_bench/<date>-<gpu>/cases.jsonl`) stays the source of truth for
committed GPU runs; `latest.jsonl` is the loop's working scratch + provenance.

## Base schema (from `gpu_profile`, per single sample)

`case_id, task, backend, n_classes, n_outputs, n_train, n_test, n_features,
max_bins, num_leaves, leaf_model, encoder, device, cuda_scan_min_cells,
n_estimators, fit_seconds, predict_seconds, quality{}, peak_rss_bytes,
peak_gpu_bytes, phase_seconds{}, transfer_bytes{}, env{}` (+ optional
`parity_max_abs_diff`).

- `phase_seconds`: preprocessing, encoder, binning, histogram, split_scan,
  partition, leaf_fit, eval, predict (from `REPLEAFGBM_PROFILE=1`).
- `transfer_bytes`: CUDA backend only (H2D/D2H counters); `{}` for numpy/rust.

## Orchestrator additions (`cuda_overnight_loop`, aggregated row)

Aggregated rows **replace** the single-sample `fit_seconds`/`predict_seconds`/
`phase_seconds` with medians + spread and add provenance:

| field | meaning |
|---|---|
| `harness_version` | e.g. `cuda_overnight_loop/0.1.0` — bumped only when aggregation/schema changes (never on product change). |
| `run_id` | UTC timestamp `YYYYMMDDThhmmssZ` grouping one orchestrator invocation. |
| `n_reps` | measured reps (warmup excluded). |
| `median_fit_seconds`, `median_predict_seconds` | p50 over reps. |
| `fit_spread`, `predict_spread` | `{p50, min, max, rel_spread}` where `rel_spread=(max-min)/p50`. |
| `median_phase_seconds{}` | per-phase p50 over reps. |

## A/B rows (`--mode ab`)

`{harness_version, mode:"ab", run_id, reps, task, size, variant_a, variant_b,
env_a{}, env_b{}, fit_a{agg}, fit_b{agg}, paired_delta_a_minus_b{median,stdev},
b_win_count, b_faster_pct, signal}`. `signal=true` iff B wins ≥ reps-1 AND
|median Δ| > 1σ (the noise-aware accept gate; interleaved order fights drift).

## Acceptance reminder

A local candidate is an **accept** only at **+3% median over ≥5 reps with green
tests** and parity held (NumPy↔Rust bitwise; CUDA allclose + quality-equivalent).
Always read `rel_spread`/`signal` before trusting a delta. Never compare rows
with different `harness_version` as a clean before/after.
