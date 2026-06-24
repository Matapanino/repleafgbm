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

### 002 — float32 embedding/Gram leaf-fit option   [HOLD → human-gated proposal]   2026-06-24
- Surface: `core/leaf_models.py` wide-emb BLAS fallback (lines ~287-308)   Backend: rust/local
- Hypothesis: a float32 leaf-cache halves wide-embedding leaf_fit (BLAS Gram).
- Evidence:
  - Phase probe (`artifacts/gpu_bench/exp2_probe/`, rust, regression, 50k×200f,
    identity emb=200>64 → BLAS path, 50 trees): **leaf_fit = 69.2% of fit**
    (7.98s / 11.5s); histogram 9%, eval 7%, split_scan 3%.
  - Isolated ceiling micro-bench (scratchpad `f32_leaf_ceiling.py`, emb=200,
    float32 Gram accumulation + **float64 solve**): **1.94x** single-thread
    (−48.5%) / **1.64x** multi-thread (−39.1%) on leaf_fit; weight deviation vs
    float64 **rel 1.2e-6** (allclose, NOT bitwise).
  - Thread check: multi-threaded BLAS (11.67s) ≈ single-threaded (10.49s) for
    float64 — the per-leaf small-GEMM Python loop does not benefit from BLAS
    threads, so the OMP=1 leaf_fit share is real-world-representative (not an
    artifact). → ~30% of TOTAL wide-emb fit is addressable.
- Decision: **HOLD**. Strong, robust lever but blocked by: (1) opt-in needs a
  public constructor param → an API change, which is **human-approval-gated**
  (plan §8); (2) numerics are allclose ~1e-6 not bitwise → existing NumPy↔Rust
  parity tests must stay float64, the float32 path needs its own allclose test +
  a tolerance decision; (3) only helps emb>64 (narrow already uses native rayon
  `leaf_linear_stats`). Recommend promoting to a `docs/proposals/` spec
  (research-proposer) once the user approves the API direction.
- Commit: uncommitted (no product change). Top "promote to proposal" candidate.

### 003 — node-batched CUDA split scan   [DESIGNED + math-validated; kernel queued]   2026-06-24
- Surface: `backends/cuda_backend.py` + `BaseSplitBackend` + grower (proposed)   Backend: cuda (Colab-gated)
- Hypothesis: batching the per-node numeric scan across the frontier into ONE
  kernel launch amortises the launch-bound per-node cost (the measured GPU
  bottleneck) and finally beats the host scan on wide/multiclass.
- Evidence:
  - `split_scan` is 48–85% of CUDA fit (85% mc K=5); per-node on-device scan is
    launch-bound and loses to host (settled, `rejected-ideas.md`).
  - Target math validated locally: `scratchpad/batched_scan_parity.py` stacks M
    nodes on an M axis over the real `_numeric_split_table` →
    **BITWISE PARITY: PASS** (feature/bin/gain match the per-node loop exactly).
    The CUDA kernel changes only launch count, not the result.
- Decision: **DESIGNED + math-validated; CUDA kernel + grower wiring queued for a
  GPU-in-the-loop session.** No local GPU → blind CUDA is rejected by the loop
  rules. Full design + interface + Colab A/B plan in
  `docs/perf-notes/research-node-batched-split-scan.md`. Architectural (grower
  frontier-batch) → needs core-reviewer sign-off on the interface first.
- Commit: design note only (no product change). Queued for Colab T4 A/B.

### 004 — float32 wide-emb leaf-fit (`leaf_fit_precision="float32_gram"`)   [ACCEPT — opt-in, default-off]   2026-06-25
- Surface: `core/leaf_models.py` (BLAS Gram path), `sklearn.py` (param)   Backend: rust/local
- Change: public MINOR param `leaf_fit_precision="float64"`(default)|`"float32_gram"`;
  float32 confined to the two wide-emb (emb>64) reductions `Zl.T@hZ` and `g@Zl`;
  centering + solve + everything else stay float64; default path byte-identical.
- Measure (C-extended orchestrator `--mode ab`, rust, 50k×200f, emb=256, 50 trees, 5 reps interleaved):
  - **Wide: float64 10.821s → float32 9.160s = 1.18× (−15.2%), B wins 5/5,
    signal=True, rel_spread 2.5–3.4%.**
  - Narrow control (50k×30f, emb≤64): 0.3%, no signal → wide-only confirmed.
  - Quality-equivalent: tests assert preds rtol 1e-5 + |Δr2|<5e-3 (smoke Δr2=4.5e-10).
  - Default float64 stays bitwise: `test_rust_backend.py` 16 passed; full suite 414 passed.
- Decision: **ACCEPT**, shipped **default-off** (opt-in). Clears the gate (≥1.10×,
  signal, spread<10%, quality-equivalent, narrow flat). NOTE: real whole-fit win
  is **~15%**, below the proposal's optimistic ~25–30% — float32 only narrows the
  two GEMM reductions; the float64 solve/centering/gather inside leaf_fit are
  unaffected (they're ~40% of the phase). Still a clean, safe, opt-in win.
- Commit: impl (leaf_models.py + sklearn.py + tests/test_leaf_fit_precision.py).
  Proposal: dbe280f. Evidence: `artifacts/gpu_bench/exp2_float32_ab/`.

### 005 — raise `_NATIVE_STATS_MAX_DIM` 64 → 128 (native leaf-fit past emb=64)   [ACCEPT — default, float64]   2026-06-25
- Surface: `core/leaf_models.py` (1-line gate constant)   Backend: rust/local
- Hypothesis: the 64 gate is over-conservative — the rayon native `leaf_linear_stats`
  may still beat the per-leaf BLAS Gram well past emb=64.
- Evidence (`scratchpad/e03_gate_crossover.py`, 8-core arm64, 50k rows, imbalanced leaves):
  - **Crossover sweep, multi-threaded BLAS:** native/BLAS = 0.76× @64, 0.76× @128,
    **1.10× @256** (BLAS wins), 1.20× @384 → real crossover ~256, NOT 64.
  - Single-thread (OMP=1): native wins through ≥200 (0.58×). Δw ~1e-14 (allclose).
  - The 2026-06-19 report only validated the default emb=64 when it moved 32→64;
    it never probed 96–256. **Same hardware** — so 64 was simply under-explored.
  - **e2e A/B (orchestrator, rust, 50k×128f, emb=128, 50 trees, 5 reps, OMP=1):**
    gate64(BLAS) fit 5.939s / leaf_fit 3.960s → gate128(native) fit **3.595s**
    (**1.65×, −39.5%**) / leaf_fit 1.577s (**2.51×, −60.2%**); **quality r2 identical
    |Δ|=0.0**; spread 1.0%. (Multi-threaded est. ~1.19× fit.)
- Decision: **ACCEPT**. Raise gate to a **conservative 128** (native wins under
  BOTH threading regimes on this hw, well below the ~256 multi-threaded crossover);
  float64 precision (Δw~1e-14, not a numerical-caveat change like E01), deterministic
  (native is bitwise serial=parallel), quality-identical. Default-path improvement
  (no opt-in). Full suite 414 passed; ruff clean. **Complementary to E01**: float32
  (`float32_gram`) now applies only to the BLAS path at emb>128.
- Follow-up: test `WIDE` bumped 80→160 (must exceed the new gate to exercise the
  BLAS/float32 branch); the float32 allclose assertion made scale-relative (the
  deviation is ~1e-6 of the prediction range, a fixed atol failed near zero).
- Commit: impl (leaf_models.py gate + test update). Evidence: `artifacts/gpu_bench/e03_gate/`.
