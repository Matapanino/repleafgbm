# Harness change log — measurement infrastructure only

`harness-optimizer` owns this file. The harness is improved to *find* real
speedups more reliably — never to manufacture them. Every harness change follows
the discipline below; product-code changes never appear here.

## Rules (non-negotiable)

1. State the **reason** the harness change improves measurement (noise ↓,
   before/after stability ↑, GPU/CPU-boundary visibility ↑, regression detection ↑,
   better next-experiment selection).
2. Record the **pre-change harness commit** (sha) so results stay attributable.
3. **Re-measure the baseline** after the change; attach the new `harness_version`.
4. Keep harness changes in a **separate commit** from product changes.
5. **Never** delete unfavorable benchmark cases or change benchmark conditions to
   flatter a product change.
6. **Never** accept a product win measured only on a freshly-changed suite —
   re-confirm against the prior harness where possible.

## Format

```
### <harness_version>  <date>
- Pre-change harness commit: <sha>
- Change: <what + which files in benchmarks/ scripts/ benchmarks/results/>
- Why (measurement objective): ...
- Baseline re-measured: <cmd> → <result file / numbers>
- Old vs new harness cross-check: <result or "n/a (additive only)">
- Commit: <sha>
```

## Entries

### cuda_overnight_loop/0.1.0  2026-06-24
- Pre-change harness commit: n/a (new file; orchestrator did not exist)
- Change: add `benchmarks/cuda_overnight_loop.py` (median+spread aggregation over
  ≥5 reps, interleaved A/B, rolling `benchmarks/results/latest.jsonl`),
  `benchmarks/results/schema.md`, `scripts/perf_loop.sh`. Reuses
  `gpu_profile.run_case`; no measurement of product code is altered.
- Why: single-sample `gpu_profile` rows are noisy; the loop needs median + spread
  + paired interleaved A/B to make +3% accept/reject decisions trustworthy.
- Baseline re-measured: `bash scripts/perf_loop.sh --quick` (numpy+rust) → see
  `benchmarks/results/latest.jsonl` (run_id of the dry-run).
- Old vs new harness cross-check: n/a (additive; gpu_profile schema reused verbatim).
- Commit: first harness commit on `perf/cuda-overnight-loop-20260624` (see `git log`).

## Future harness-iter candidates (not yet actioned)

- **Record BLAS thread config per row.** iter 002 found wide-emb leaf_fit is
  BLAS-bound; the share is thread-sensitive in principle. Empirically multi ≈
  single here (small per-leaf GEMMs), but the harness should log
  `OMP_NUM_THREADS` / BLAS vendor in `env` so BLAS-bound deltas stay attributable.
  Additive to `gpu_profile.collect_env`; bump `harness_version`; re-baseline.
