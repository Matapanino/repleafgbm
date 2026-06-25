# GEPA-style reflections — one per iteration (≤15 lines each)

After every product iteration (and every harness iteration), append one
compressed reflection in the fixed format below. These feed
`experiment-strategist` when it picks the next 3 mutation candidates. Keep them
short — long prose defeats the purpose.

## Format

```
### NNN — <slug>   <date>
- What we tried:
- What happened:
- Why it likely happened:
- What rule we learned:
- Next mutation candidates:
- Should this affect the harness/prompt/code?:
```

## Reflections

### 001 — forest-batched predictor   2026-06-24
- What we tried: validate the memory's "forest-batched traversal + fused leaf-output" predict lever before implementing.
- What happened: profiling showed routing is already per-tree native and leaf-eval already fused native; `overhead_seconds` (the only batchable slack) is 4–16%, of which routing-call overhead is <3%.
- Why it likely happened: PR #32 (apply_tree) + Session 4 (predict_linear) already moved the heavy work into Rust; only Python loop/marshalling/accumulation remain.
- What rule we learned: profile the decomposition (routing/leaf/overhead) BEFORE building a batched kernel; "routing dominates" does not imply "routing is batchable" once it's already native.
- Next mutation candidates: forest-fused single-kernel predictor (HOLD, big); shift the night to FIT levers (split_scan, float32 cache) with more headroom.
- Should this affect the harness/prompt/code?: code unchanged; rejected-ideas updated so future sessions skip standalone forest-batched routing.

### 002 — float32 leaf-fit ceiling   2026-06-24
- What we tried: size the float32 wide-emb leaf-fit lever before any API/numeric commitment, with an isolated ceiling bench.
- What happened: leaf_fit is 69% of wide-emb fit; float32 Gram + float64 solve gives 1.6-1.9x leaf_fit (~30% of total fit) at rel weight deviation ~1e-6; robust to BLAS threading.
- Why it likely happened: emb>64 falls to a per-leaf NumPy BLAS Gram (irreducible float64 FLOPs); only lower precision cuts it, and float64 solve keeps the deviation tiny.
- What rule we learned: an isolated ceiling micro-bench is the right tool when the real win is blocked by an API/numeric decision — it sizes the prize without touching product code; and check thread-sensitivity before blaming OMP=1.
- Next mutation candidates: promote float32 to an API-gated proposal; consider a native-rust wide-emb float32 Gram (could beat NumPy float32 too).
- Should this affect the harness/prompt/code?: harness — log BLAS thread config per row for BLAS-bound phases (future harness iter). Code — gated by human API approval.

### 003 — node-batched split scan   2026-06-24
- What we tried: de-risk the headline CUDA split_scan lever without a local GPU.
- What happened: validated the batched-scan math bitwise locally; found the real cost is a grower frontier-batch interface, not the kernel; wrote a grounded design + Colab A/B plan instead of blind CUDA.
- Why it likely happened: the win is launch-count amortisation (per-node scan is launch-bound), so the math is unchanged (stacking on M) — the architecture (presenting M nodes) is where the work and review risk live.
- What rule we learned: when a lever is GPU-gated, validate the deterministic math locally and design the interface; never ship a CUDA kernel you cannot iterate on a GPU.
- Next mutation candidates: implement during a GPU-in-the-loop session (depthwise level batch first); core-reviewer signs off the frontier-batch interface before the kernel.
- Should this affect the harness/prompt/code?: code — queued, human/GPU gated. cuda_overnight_loop --mode ab already supports the device-on/off A/B it needs.

### 004 — float32 leaf-fit shipped (opt-in)   2026-06-25
- What we tried: implement the iter-002 float32 lever default-off behind a public param, measured via the C-extended A/B.
- What happened: 1.18× whole-fit (−15.2%) on wide 50k×200, 5/5 signal, narrow flat, quality-equivalent, default bitwise-green (414 passed). ACCEPT.
- Why it likely happened: float32 halves only the two GEMM reductions; the float64 solve/centering/gather (~40% of leaf_fit) are untouched, so 15% not the projected 30%.
- What rule we learned: project a precision win from the *reduction* fraction of a phase, not the whole phase; an isolated ceiling bench (equal-size synthetic leaves) over-states the real, imbalanced-leaf win.
- Next mutation candidates: a native-Rust wide-emb float32 Gram (could beat NumPy float32 + cut the per-leaf Python loop); revisit whether the solve/gather can shrink.
- Should this affect the harness/prompt/code?: harness C (precision A/B passthrough) proved its worth; keep. No default change.

### 005 — native gate was over-conservative (64 → 128)   2026-06-25
- What we tried: re-measure the _NATIVE_STATS_MAX_DIM=64 native-vs-BLAS crossover instead of trusting it.
- What happened: native wins to ~128 multi-threaded (crossover ~256) and ≥200 single-thread on the SAME hardware the gate was set on; e2e 1.65× fit at emb=128, quality identical. Bigger + cleaner than the float32 win (E01) it partly supersedes.
- Why it likely happened: the 2026-06-19 gate move (32→64) validated only the *default* emb=64 and never probed wider — a tuned-but-under-explored constant, not a measured crossover.
- What rule we learned: re-measure "tuned" gates/constants before building around them — an under-explored gate can hide a bigger, simpler win than the fancy optimization (float32) you were about to ship. Always test the cheap 1-line knob first.
- Next mutation candidates: H8 Cholesky solve (assume_a='pos') for emb>128 BLAS; native float32 at emb>128; revisit whether the gate should be adaptive to core count.
- Should this affect the harness/prompt/code?: code shipped (1-line + test fix). Memory note rust-leaf-fit-rayon (gate=64) now stale → update.

### 007 — node-batched CUDA scan shipped + validated   2026-06-25
- What we tried: implement the headline GPU lever (node-batched split scan) staged — host contract+grower (local, bitwise) then CUDA device scan (Colab).
- What happened: parity 35/35 on T4; A/B split_scan 5-9×, whole depthwise fit 1.9-3.9×, quality identical; narrow wins too (1.9×).
- Why it likely happened: the per-node device scan was launch-bound; batching M frontier nodes into one CuPy reduction amortizes the launch — and the device scan is plain CuPy (no RawKernel), so an M-axis lift was low-risk to write blind.
- What rule we learned: stage GPU work — prove the host contract + grower refactor BITWISE locally, then the device path is a thin, low-risk M-axis vectorization; "no blind CUDA" doesn't mean "no CUDA", it means "validate the deterministic structure first."
- Next mutation candidates: flip the gate default ON for cuda+depthwise (MO precedent); leafwise frontier-batching; batched build_histograms to feed it.
- Should this affect the harness/prompt/code?: scripts/colab_batched_ab.py is a reusable depthwise A/B harness; keep. Design note → mark implemented.

### 008 — flip batched-scan default ON   2026-06-25
- What we tried: ship the iter-007 node-batched CUDA scan as the cuda+depthwise default (1-line resolver flip + kill switch + tests + docs), mirroring the MO device-scan default-on precedent.
- What happened: local green (421 passed / 96 CUDA-skip / ruff clean); Colab re-val queued. Surgical surface — the grower/dispatch already read `supports_batched_scan`, so only the resolver default + test intent changed.
- Why it likely happened: iter 007 staged the work so the default is a pure policy flip on a validated, allclose-by-contract path; the adaptive `_scan_min_cells` crossover means the flip can't regress tiny frontiers (they still fall to the host loop).
- What rule we learned: when last session shipped a validated opt-in behind an env gate, flipping its default is a cheap, high-confidence iteration — but still a deliberate default change (core-reviewer + a re-validation), not a silent edit.
- Next mutation candidates: size Task B (leafwise split_scan share under the default grow_policy) on the same Colab run; iter-009 E15 float32 vector leaves (local).
- Should this affect the harness/prompt/code?: docs paid the iter-007 doc debt (cuda.md/ADR never documented the batched scan). No harness/prompt change.

### 009 — E15 float32 vector leaves shipped (opt-in)   2026-06-25
- What we tried: extend the approved float32_gram opt-in to multi-output, after a cheap-evidence pass to LOCATE the true surface first.
- What happened: the backlog's "vector branch is pure float64" held — but only for the shared-routing vector fit (`fit_vector_leaves`); the multiclass-wide case was ALREADY float32-covered via the per-class fallback. Shipped 1.055× MO fit (5/5, |Δr2|=8e-9).
- Why it likely happened: `fit_vector_leaves` has no native path (always NumPy BLAS), and its two centered GEMMs are 21–31% of MO fit; float32 halves only those, so 5.5% not 30% (reflection-004 rule held a third time).
- What rule we learned: LOCATE the exact surface before sizing — "multi-output leaf fit" was three different code paths (scalar-reuse / native-mc / vector); only one was the real lever. A 20-min trace stopped me writing float32 where it already existed.
- Next mutation candidates: E14 float32 `predict_linear`; E02 native-rust wide Gram (the only float64 wide leaf-fit left is the scalar BLAS solve + this vector path); could the vector Gram go native too?
- Should this affect the harness/prompt/code?: the orchestrator `--task multioutput --precision-a/-b` A/B worked cleanly; keep. No default change.

### 010 — batched histogram HOLD; bottleneck shifted to leaf_fit; Task-B re-prioritized   2026-06-25
- What we tried: SIZE the batched-histogram lever (iter 010) + Task-B (leafwise) on Colab T4 before building either, via embedded_linear phase shares.
- What happened: histogram is only 2.7–3.2% of fit post-batched-scan → iter-010 HOLD/null. Surprise: leaf_fit is now 65–73% of CUDA fit. Task-B's leafwise split_scan is 32.2% → M=2 batching projects ~14% fit (BUILD-next).
- Why it likely happened: the batched scan crushed split_scan (9×), so the remaining CUDA fit is host-leaf_fit-dominated; histogram was never large; leafwise still scans per-node (32.2%), a big launch-bound share to amortize even at M=2.
- What rule we learned: AFTER a big win, RE-SIZE the phase decomposition — the bottleneck moves. The "obvious next lever" (histogram, mirroring the scan win) was dead because the scan win itself reshaped the profile; a *deferred* lever (Task-B) became MORE attractive once measured.
- Next mutation candidates: BUILD Task-B leafwise frontier-batch (~14% ceiling); attack CUDA leaf_fit (E02 native-rust wide Gram / GPU leaf-fit) — now the dominant phase.
- Should this affect the harness/prompt/code?: `scripts/colab_sizing.py` is a reusable phase-share sizer; keep. No product code this iter.
