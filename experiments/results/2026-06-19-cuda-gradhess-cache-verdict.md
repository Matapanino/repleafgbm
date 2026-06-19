# Verdict: CUDA grad/hess device-resident cache (Phase 1-1 / PR #2)

- Date: 2026-06-19
- Branch: `perf/cuda-gradhess-device-cache`
- GPU: Tesla T4 (Colab dev loop)
- Artifacts: `artifacts/gpu_bench/2026-06-19-T4/cases.jsonl` (AFTER),
  `artifacts/gpu_bench/2026-06-18-T4/cases.jsonl` (BEFORE, v1.6.0, same shapes/same T4),
  `experiments/results/2026-06-19-cuda-parity.md` (parity),
  `artifacts/gpu_bench/dev-perf-cache/summary.md` (CPU/numpy size=medium baseline)
- Roadmap target: `docs/gpu_roadmap.md` Phase 1 §1 "Cache Full Grad/Hess Buffers On CUDA"

## Question

The GPU audit named the per-node host gather of `grad[rows]` / `hess[rows]` (the
`gradhess_h2d_bytes` H2D upload) as the "dominant remaining H2D upload" on the CUDA
path. Phase 1-1 caches the full contiguous `grad`/`hess` on device (promoted once per
`(grad,hess)` pair) so the kernel reads `grad[row]`/`hess[row]` and only `rows` cross
per node. Does removing this transfer move fit wall-clock, and is the change worth
shipping?

## Evidence

### Parity — perfect

`tests/test_cuda_backend.py`: **20 passed** on T4 (incl. 5 new cache tests; weighted
reg/clf, multiclass class-column views, repeated rounds for stale-cache reuse). Quality
dicts are **bit-identical** BEFORE vs AFTER across all 5 cases (rmse/logloss/accuracy
match to full precision; multiclass `multi_logloss` matches to its existing
GPU-atomic-add allclose tolerance, 0.429632430095377 both runs). The cache is correct.

### Transfer — engaged as designed, modest byte reduction

`gradhess_resident_uploads` = exactly one promotion per tree on scalar tasks (30 = 30
trees) and per tree×class on multiclass (150 = 30×5). `binned_uploads == 1`. The cache
engaged precisely as specified.

| case (cuda) | g/h H2D before | g/h H2D after | byte ratio | g/h resident uploads |
|---|---:|---:|---:|---:|
| regression_30f | 61.3 MB | 48.0 MB | 1.28x | 30 |
| regression_200f | 37.1 MB | 28.8 MB | 1.29x | 30 |
| binary_30f | 58.5 MB | 48.0 MB | 1.22x | 30 |
| binary_200f | 35.4 MB | 28.8 MB | 1.23x | 30 |
| multiclass_c5_200f | 156.0 MB | 144.0 MB | 1.08x | 150 |

The reduction is only 1.08–1.29x, not a collapse, because leaf-wise growth plus sibling
histogram subtraction already keeps the per-node grad/hess gather at roughly ~2.5×
`n_rows` per tree — the cache removes the gather but the residual `rows` traffic and the
remaining per-promotion full-buffer upload dominate the byte count.

### Fit wall-clock — unchanged

| case (cuda) | fit before (s) | fit after (s) | speedup (before/after) |
|---|---:|---:|---:|
| regression_30f | 1.93 | 2.17 | 0.89x |
| regression_200f | 7.38 | 6.75 | 1.09x |
| binary_30f | 1.97 | 1.89 | 1.05x |
| binary_200f | 6.77 | 7.10 | 0.95x |
| multiclass_c5_200f | 21.55 | 20.94 | 1.03x |

Speedups span 0.89–1.09x and straddle 1.0 with no directional signal — this is
single-run noise (these are 1-run cases, not multi-seed). The reason is direct: the
grad/hess H2D wall-time, estimated at ~12 GB/s PCIe, is **2.4–12.0 ms** against fits of
**1900–21000 ms**, i.e. **0.034–0.212% of fit time**. Removing all of it could not move
the needle. The audit's "dominant" upload was dominant by **bytes**, not by wall-clock.

### Real bottleneck — `split_scan`, not the histogram the cache touches

Per-phase timers from the AFTER cuda run (`phase_seconds`, share of summed phases):

| case (cuda) | split_scan | histogram |
|---|---:|---:|
| regression_30f | 48.8% | 20.4% |
| regression_200f | 53.5% | 5.7% |
| binary_30f | 48.0% | 22.8% |
| binary_200f | 54.4% | 6.1% |
| multiclass_c5_200f | 85.0% | 5.0% |

`split_scan` dominates the CUDA fit (48–85%, peaking on multiclass where per-class trees
multiply the per-node scans). `histogram` — the phase the cache feeds — is only 5–23%.
This inverts the CPU picture: on the numpy path the histogram dominates (53–64% in the
same 06-19 cases; 68–70% at size=medium in `dev-perf-cache/summary.md`), but on CUDA the
102x-faster histogram kernel (parity report) is cheap and `split_scan` — per-node
cumsum/gain/argmax, the per-node GPU→host winner sync, and the host scan on narrow fits —
is now the long pole.

## Verdict

**Null result for fit speedup; correct and parity-perfect.** The device-resident
grad/hess cache does exactly what it was specified to do — removes the per-node host
gather, cuts grad/hess H2D bytes 1.08–1.29x — but that transfer is <0.22% of fit
wall-clock, so fit time is statistically unchanged (0.89–1.09x, noise). The optimization
targeted a bytes bottleneck that is not a time bottleneck.

**Confidence: high** for the null fit-speedup finding and for the split_scan redirect.
The transfer-byte deltas, the resident-upload counts, and the parity bit-identity are
hard measurements; the fit-time non-effect is corroborated independently by the H2D
wall-time arithmetic (it is physically too small to matter), not by the noisy single-run
timings alone. The one provisional caveat: fit timings are single-run, so a small real
regression on `regression_30f` (0.89x) cannot be fully excluded — but the H2D arithmetic
makes a cache-caused slowdown implausible; it is far more likely run-to-run jitter.

## Next action — redirect the GPU roadmap to `split_scan`

This evidence does not justify any default change (CUDA stays explicit, `"auto"` never
picks it — unchanged). It redirects the *next* GPU optimization target. The lever is
`split_scan` (48–85% of fit), not further H2D-byte trimming. Candidates to **design**
next (do not design them here — owner: `research-proposer` for the spec, then
`experiment-runner` to validate on T4):

1. Cut the per-node GPU→host winner sync (one `winner_d2h` round-trip per node — keep the
   argmax/winner resident and batch the host pull).
2. Batch the multiclass per-class scans (roadmap Phase 3 §2 "Multiclass Batched
   Histogram"/scan) — multiclass is where `split_scan` peaks at 85% because per-class
   trees multiply the per-node scan count (`n_gpu_scans` 8915 vs ~1828 on scalar).
3. Revisit the narrow-fit host-scan path (`n_small_scans` 1826–1829 on the 30f cases run
   entirely on host) — wide cases already go GPU-resident (`n_gpu_scans`), narrow cases
   fall back to host.

## Ship vs shelve recommendation (final call is the maintainer's)

**Recommendation: ship the cache.** Reasoning: it is correct, parity-perfect (bit-identical
quality, 20/20 incl. stale-cache tests), and low-risk — it strictly removes a host gather
and reduces H2D bytes 1.08–1.29x with no measured fit-time cost. The byte reduction is
perf-neutral on an idle single-T4 today, but it is the right shape for the conditions the
roadmap is explicitly building toward (Phase 5: multi-GPU and large/out-of-core data),
where PCIe bandwidth becomes contended and fewer H2D bytes matter. It also keeps the
device-residency invariant consistent (grad/hess now live on-device like `binned`),
which de-risks the Phase 3 batched-scan work that this report recommends next. The
counter-argument — shelve it as perf-neutral complexity — is legitimate if the maintainer
wants to hold the line that only measured wall-clock wins land; in that case mark it
"correct, perf-neutral, revisit under multi-GPU" rather than discarding the code. Either
way, do **not** advertise it as a speedup in the changelog: it is a transfer-byte
reduction, not a fit-time win.
