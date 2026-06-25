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

### cuda_overnight_loop/0.2.0  2026-06-25
- Pre-change harness commit: 5823446
- Change: (1) `gpu_profile.collect_env` now logs `env.threads` (OMP/OPENBLAS/MKL/
  VECLIB/NUMEXPR num-threads) + `env.blas` (numpy BLAS vendor); (2)
  `gpu_profile.build_estimator` + `--leaf-fit-precision` forward the opt-in
  precision when set; (3) `cuda_overnight_loop` forwards shared estimator knobs
  (`--n-features/--max-leaf-emb-dim/--leaf-model/--n-estimators/--n-train/--n-test`)
  in both modes + per-variant `--precision-a/--precision-b` for the float32 A/B.
- Why (measurement objective): the float32 wide-emb leaf-fit win lives in a
  BLAS-bound phase; logging thread config makes the A/B attributable, and the knob
  passthrough lets that A/B run through the orchestrator's median+spread+signal
  instead of a hand-rolled shell loop. Pure measurement enablement; no product code.
- Baseline re-measured: `bash scripts/perf_loop.sh --quick` → `latest.jsonl` rows
  now carry `env.threads`/`env.blas`, `harness_version=cuda_overnight_loop/0.2.0`.
- Old vs new harness cross-check: rust `regression_20f_bins256` fit_p50
  0.1.0=0.0436s vs 0.2.0=0.0439s (+0.7%, within noise) — additive change does not
  shift measurements.
- Commit: second harness commit on `perf/cuda-overnight-loop-20260624` (see `git log`).
