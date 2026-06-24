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
