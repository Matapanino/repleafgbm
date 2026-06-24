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
