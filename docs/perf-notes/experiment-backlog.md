# Experiment backlog — CUDA/perf overnight loop

~30 prioritized hypotheses. Discipline: **cheap evidence first**, implement only
on ≥3% median + green + parity. Status: `SHIPPED` / `REJECT` / `HOLD` / `TODO`.
`L`=locally verifiable (CPU), `G`=needs live GPU (Colab T4). Most will be quick
evidence verdicts; only bar-clearing items get implemented. Verdicts land in
`experiment-log.md` (iter NNN) + `reflections.md`.

Pre-investigated this session (cheap probes, before the backlog):
- **preprocessing phase is a COLD-START artifact**, not a lever: `_build_dataset`
  is ~8.7 ms steady-state (check_X_y 0.3 ms + RepLeafDataset 7.8 ms), but the
  single-shot `gpu_profile` attributes ~326 ms (first-fit lazy imports) to it. The
  orchestrator's warmup already absorbs this. → not a product lever (E29 harness).
- **eval phase is the mandatory boosting F-update** (`F += lr * predict`), not
  optional quality tracking (which only runs with an eval_set). Not skippable.
- **binned matrix is already `uint16`** (splitter.py:79); histogram is `float64×3`
  (count channel included).

## Tier 1 — local, leaf_fit (largest measured headroom; float64-safe or opt-in)

- **E01** `SHIPPED` (iter 004) — float32 wide-emb Gram (`leaf_fit_precision`). 1.18× wide fit.
- **E02 `L` TODO** — native-Rust wide-emb (emb>64) Gram, optionally float32. Extend
  `leaf_linear_stats` past the 64 gate; removes the per-leaf Python loop + could beat
  NumPy float32 BLAS. Cheap test: micro-bench native-vs-BLAS at emb 96/128/200. Risk:
  Rust+parity (allclose for f32); big-ish kernel. Top Tier-1 prize.
- **E03 `L` TODO** — re-measure the `_NATIVE_STATS_MAX_DIM=64` crossover on this
  machine/BLAS. Cheap: native-vs-BLAS leaf-fit sweep emb∈{48,64,96,128}. If native
  wins past 64, raise the gate (bitwise-safe — native path is allclose-vs-BLAS but
  bitwise serial=parallel). Risk: the gate was measured before; honest re-measure only.
- **E04 `L` TODO** — vectorize the per-leaf BLAS loop's *non-GEMM* work (`z_min/z_max`,
  centering `outer`, the Python `for j` overhead) across leaves. float64, bitwise-safe.
  Cheap: profile the loop body vs the GEMM share at emb=200.
- **E05 `L` REJECT-likely** — float32 the *solve* too. Cheap: measure deviation (cancellation
  in centered normal equations). Expect tolerance blow-up → reject; record the number.
- **E06 `L` TODO** — skip `z_min/z_max` extrapolation-bound compute when clip is a
  proven no-op (training rows are in-range). float64. Cheap: measure z_min/z_max share.

## Tier 2 — local, histogram / binning / split (memory- or sort-bound)

- **E07 `L` HOLD** — split the histogram count channel to `int32` (separate array) to cut
  `float64×3` memory bandwidth (histogram is memory-bound). Rust+NumPy+parity. Medium.
- **E08 `L` TODO** — confirm histogram is single-pass (grad+hess+count one sweep) on Rust;
  if not, fuse. Cheap: read the kernel. (Likely already done.)
- **E09 `L` REJECT-likely** — sibling subtraction (build smaller child, subtract). Memory
  says already shipped — verify, else implement. Cheap: read grower.
- **E10 `L` TODO** — binning quantile is super-linear (`np.unique` sort per feature). Try a
  partition/`np.partition`-based quantile or a fixed-stride sketch. Cheap: micro-bench
  unique-sort vs partition at 500k rows. Risk: bin-edge parity (bitwise) — gate carefully.
- **E11 `L` DONE** — int bin codes: binned matrix already `uint16`. No-op.
- **E12 `L` REJECT** — skip eval: it's the mandatory F-update (pre-investigated).

## Tier 3 — local, predict / eval / multi-output

- **E13 `L` HOLD** — forest-fused predictor (route+leaf-eval+accumulate one kernel). From
  iter 001: only >3% predict path, large layer-coupling change. Stays HELD.
- **E14 `L` TODO** — float32 leaf-eval in `predict_linear` (opt-in, like E01). Cheap:
  predict_profile float32 vs float64 leaf_eval at wide emb. Allclose-gated.
- **E15 `L` SHIPPED** (iter 009) — `float32_gram` extended to the shared-routing
  **vector** leaf fit (`multioutput.py::fit_vector_leaves`, the only multi-output path
  with NO float32 branch — multiclass-wide already reused the scalar branch via the
  per-class `fit_leaves` fallback; narrow multiclass is native float64 = E02). 1.055×
  wide-emb (emb=256) MO fit, 5/5 signal, quality-equivalent (|Δr2|=8e-9). Opt-in;
  default float64 bitwise.
- **E16 `L` TODO** — the training F-update (eval phase) `leaf_idx[rows]=i` Python loop over
  leaves → vectorize. float64, bitwise. Cheap: measure the loop vs predict share.

## Tier 4 — local, dataset / encoder / dtype / harness

- **E17 `L` TODO** — float32 embedding cache: store `Z` as float32 internally (default
  float64). Halves embedding memory + feeds E01/E14/CUDA. Opt-in. Cheap: A/B leaf_fit+mem.
- **E18 `L` REJECT** — preprocessing: cold-start artifact, not steady-state (pre-investigated).
- **E19 `L` TODO** — encoder phase (identity ~0.18s): is identity doing a copy? Cheap: profile
  identity transform; avoid copy if so. float64, bitwise.
- **E20 `L` HOLD** — `max_bins` sweep effect on histogram/scan vs quality (not a code change;
  a tuning-guidance experiment). Cheap: sweep 64/128/256/512.

## Tier 5 — CUDA (Colab T4-gated; design/queue locally, validate on GPU)

- **E21 `G` SHIPPED** (iter 007) — node-batched split scan. Colab T4: split_scan
  5–9×, whole depthwise fit 1.9–3.9×, quality identical, parity 35/35. Opt-in
  (`REPLEAFGBM_CUDA_BATCHED_SCAN`). The CUDA path was a plain CuPy M-axis lift (no
  RawKernel). narrow wins too (batching amortizes the launch).
- **E22 `G` TODO** — class-batched multiclass CUDA histogram (mc split_scan ~85%). Design
  locally; kernel on Colab.
- **E23 `G` TODO** — float32 device embeddings / binned upload (halve H2D). Design locally.
- **E24 `G` HOLD** — fuse categorical subset scan onto device (currently host). Parity-heavy.
- **E25 `G` TODO** — persistent/cached CuPy buffers across rounds to cut alloc churn. Design.

## Tier 6 — external-technique-seeded (merged from `research-transferable-perf-techniques.md`)

- **E26 `L` REJECT** (H7) — encode_features copy avoidance. The "preprocessing 27%"
  premise is the cold-start artifact; steady-state `_build_dataset` is ~8.7 ms
  (~0.7% of narrow fit) → below the bar.
- **E27 `L` TODO** (H8) — Cholesky solve `scipy.linalg.solve(assume_a="pos")` for the
  SPD ridge Gram (halves solve FLOPs). Now applies only to the emb>128 BLAS path
  (post-E03). scipy already transitively present (sklearn). Allclose vs LU. Cheap.
- **E28 `L` HOLD** (H2) — uint8 binned matrix (currently uint16). Halves DRAM
  bandwidth + CUDA H2D, but max_bins=256 default + missing bin > 255 → needs a
  cap/special-case; semantics risk. Medium.
- **E29 `L` REJECT-likely** (H1) — histogram allocation pool. ~10% of histogram phase
  ≈ 1-2% fit; GC-cycle risk. Below bar.
- **E30 `L` TODO** (H3) — quantized int8/int16 gradient histogram (NumPy feasibility
  first; allclose-gated). Higher effort; 2× histogram if it holds parity.
- (H5 constant-leaf vectorize → E-misc low; H4 k-means binning → quality not speed,
  out of scope; H6 forest-fused → same as E13 HOLD; H12 → E21; H13 → CUDA tuning.)

## Results so far (2026-06-25)

- **E01 SHIPPED** (iter 004): float32 wide-emb Gram, 1.18× wide fit (opt-in).
- **E03 SHIPPED** (iter 005): gate 64→128, **1.65× fit @emb=128** (default, float64,
  quality-identical) — the headline win; partly supersedes E01 (float32 now emb>128).
- **REJECT (pre-investigated):** E12/E18 (eval=mandatory F-update; preprocessing=cold-start),
  E09 (sibling subtraction already shipped), E11 (uint16 already), E26/H7.
- Next: E27 (Cholesky) → E04 (vectorize non-GEMM loop) → E16 → E14 → E15 → E02.

## Execution order (this session)

Cheap-evidence-first, local before GPU: **E03 → E09 → E08 → E10 → E04 → E16 → E19
→ E06 → E14 → E02 → E17 → E15**, recording each verdict; harness iter every 5
product iters; GPU items (E21–E25) stay queued for a Colab session. E05/E12/E18
already reasoned to reject (record + skip the build).

## Session 2 results (2026-06-25, iter 008–010)

- **iter 008 SHIPPED + T4-validated** — flipped `REPLEAFGBM_CUDA_BATCHED_SCAN` default
  ON (cuda+depthwise). Re-val: wide 3.86× / narrow 1.99× / mc 2.94× fit, quality
  identical; kill switch `=0`.
- **E15 SHIPPED** (iter 009) — float32_gram for multi-output **vector** leaves
  (`fit_vector_leaves`): 1.055× wide-emb MO fit, quality-equivalent. (Multiclass-wide
  was already float32 via the per-class fallback; narrow mc is native = E02.)
- **iter 010 HOLD/null** — batched `build_histograms`: histogram is only 2.7–3.2% of
  fit post-batched-scan (below the +3% gate). Sized out. (Also lowers E22's value.)
- **GPU bottleneck shifted to leaf_fit** (65–73% depthwise, 49% leafwise post-batch).
  The next CUDA lever is **leaf_fit** (E02 native-rust wide Gram, or GPU leaf-fit), not
  histogram/scan.
- **Task B (leafwise frontier-batch) → BUILD-NEXT** — leafwise split_scan is 32.2% of
  fit; M=2 batching at the measured ~89%-launch-bound ratio → ~1.8× scan → **~14%
  whole-fit ceiling**. Stage host-bitwise (reuse `_make_candidates_batched`) then the
  CuPy M-axis device lift.
- **Next-session order:** Task-B leafwise batch (G) → E02 native-rust wide Gram (L) →
  E14 float32 `predict_linear` (L) → E17 float32 embedding cache (L).
