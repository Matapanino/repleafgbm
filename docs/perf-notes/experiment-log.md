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

### 001 — forest-batched Rust predictor (batch traversal + fused leaf-output)   [REJECT + HOLD]   2026-06-24
- Surface: `core/prediction.py`, `core/tree.py`, `native/` (proposed)   Backend: rust/local
- Hypothesis: routing is 60–100% of predict; batching all trees into one native
  call (+ fused leaf-output) cuts predict ≥3%.
- Evidence (`benchmarks/predict_profile.py`, rust, 200 trees, no-cat,
  `artifacts/predict_bench/exp1_baseline/`):
  - constant: 20k rows routing=53.3ms leaf=5.5ms predict=61.4ms (routing 86.8%, **overhead 4.2%**); 100k routing=232ms predict=280ms (**overhead 6.7%**).
  - embedded_linear: 20k routing=60.7ms **leaf=119ms** predict=213ms (routing 28.5%, overhead 15.6%); 100k routing=272ms **leaf=324ms** predict=660ms (overhead 9.8%).
  - Routing is already per-tree native (`apply_tree`); leaf-eval is already fused
    native (`leaf_models.LeafValues.predict` → `predict_linear`, Session 4).
- Decision:
  - **REJECT forest-batched routing alone** — addressable gain = per-call
    overhead only (pyo3 crossing + per-tree `ascontiguousarray`/`_cat_csr`), which
    is a fraction of the 4–7% `overhead_seconds`; routing's bulk is genuine
    descent work that batching does not reduce. ≤~3% ceiling vs a new forest API +
    concatenated-forest representation + parity surface → fails the complexity gate.
  - **HOLD forest-fused predictor** (route+leaf-eval+accumulate in ONE kernel) —
    this *would* capture the 4–16% overhead, but it couples the deliberately
    separate tree-router / leaf-model layers (`prediction.py` docstring) and must
    fuse constant/scalar-linear/vector/multiclass variants + clip + lr. Too large
    for an overnight accept; documented as a dedicated-PR scaffold.
- Commit: uncommitted (no product change). Evidence in `artifacts/predict_bench/exp1_baseline/`.
