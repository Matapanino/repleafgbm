# Experiment log — CUDA/GPU overnight optimization loop

Append-only ledger. One block per product-optimization iteration. Newest at the
bottom. Keep each block scannable; deep notes go in the dated
`overnight-loop-<date>.md`. Verdicts: **ACCEPT** / **REJECT** / **HOLD**.

Acceptance gate (see `benchmarks/results/schema.md`): local +3% median over ≥5
reps with green tests and parity held (NumPy↔Rust **bitwise**; CUDA **allclose +
quality-equivalent**); +1–3% → HOLD unless complexity is low; within noise /
accuracy regress / API change → REJECT; a scaffold for a future win may HOLD.

## Format

```
### NNN — <one-line hypothesis>   [ACCEPT|REJECT|HOLD]   <date>
- Surface: <files touched>           Backend: <numpy|rust|cuda|local>
- Hypothesis: ...
- Change: <1 hypothesis, 1–3 files>
- Measure: <bench cmd>; median Δ = <x%> over <N> reps (rel_spread <y%>); parity <status>
- Decision: <why accept/reject/hold> (Pareto: speed/parity/stability/simplicity/maint/context/future)
- Commit: <sha or "uncommitted/queued for Colab">
```

## Iterations

_(none yet — first iteration appended after the orchestrator baseline dry-run)_
