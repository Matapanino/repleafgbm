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

### 006 — Cholesky solve for the SPD leaf Gram (E27/H8)   [REJECT]   2026-06-25
- Surface: `core/leaf_models.py::_solve_and_assemble` (emb>128 BLAS path)   Backend: local
- Hypothesis: the ridge Gram is SPD; `scipy.linalg.solve(assume_a="pos")` (Cholesky)
  halves the solve FLOPs vs `np.linalg.solve` (LU).
- Evidence (micro-bench, k=31 leaves, emb=160 SPD): batched LU `np.linalg.solve`
  **4.50 ms** vs per-leaf scipy assume_a=pos **5.64 ms** (allclose=True). scipy has
  no batched API → the per-leaf Python loop is *slower* than the existing batched
  LU; and the solve is a small part of leaf_fit (GEMM ~8× at emb=160).
- Decision: **REJECT** — slower in practice (batched LU beats a per-matrix Cholesky
  loop) and the solve isn't the bottleneck. Post-E03 it only affects emb>128 anyway.

### GPU validation pass (Colab T4, 2026-06-25)
Ran `scripts/colab_gpu_test.sh --gpu T4` on the current branch (validates the
shipped gate/float32 changes don't regress the CUDA path; reports under
`experiments/results/2026-06-25-{cuda-parity,gpu-backend-suite}.md`).
- **CUDA parity: PASS** — `tests/test_cuda_backend.py` 31 passed (gate 64→128 +
  float32 param are host-side leaf-fit → CUDA path unaffected, confirmed).
- Histogram micro: NumPy 139ms → CUDA 2.77ms = **50.2×**.
- e2e fit (50 trees): narrow 100k×30 **1.56×**, wide 50k×200 **2.08×**.
- Backend suite (30 trees): regression 200f 1.66×, multiclass-c5 200f 2.06×,
  **multioutput-k5 200f 5.40×**; narrow ~1× (host-scan crossover, by design).
- MO device-scan A/B: 200f **2.86×** on vs off (matches prior ~2.95×).
- NOTE: node-batched split scan (iter 003 / E21) is **design-only** — no kernel
  exists, so it was NOT in this run. It needs the grower frontier-batch refactor
  + kernel (core-reviewer-gated) as a focused GPU-in-the-loop session, not a batch run.

### 007 — node-batched CUDA split scan (E21 / iter 003 design)   [ACCEPT — opt-in, Colab T4-validated]   2026-06-25
- Surface: `backends/base.py` + `core/splitter.py` + `core/tree.py` (Stage 1, host,
  bitwise) + `backends/cuda_backend.py` (Stage 2, device kernel)   Backend: cuda
- Change: `find_best_split_batched` — the depthwise grower scans a whole level's M
  frontier histograms in ONE call. Host default loops per-node (bitwise-identical
  tree, proven local); CUDA vectorizes the per-node CuPy scan over a leading M axis
  (no RawKernel — lifts the proven `find_best_split` path). Gated
  `REPLEAFGBM_CUDA_BATCHED_SCAN` (default OFF, opt-in); host path untouched.
- Validation:
  - **Local (Stage 1):** batched == per-node loop bitwise (numpy+rust, numeric +
    categorical + tie-break); forced-batched depthwise tree byte-identical to FIFO
    (`tests/test_batched_scan.py`, 7 passed; full suite 421).
  - **Colab T4 (Stage 2):** parity **35 passed** (device batched == NumPy reference,
    numeric + categorical; gate-off loops per-node; e2e quality-equivalent).
  - **A/B (T4, depthwise, 5 reps interleaved, `experiments/results/2026-06-25-batched-scan-ab.md`):**

    | case | shape | fit off→on | fit× | scan off→on | scan× | \|Δq\| |
    |---|---|---|---|---|---|---|
    | wide | 50k×200 | 14.59→3.73s | **3.91×** | 11.61→1.29s | **9.01×** | 0.0 |
    | narrow | 100k×30 | 4.96→2.60s | **1.91×** | 2.80→0.56s | **4.95×** | 0.0 |
    | multiclass | 50k×200 | 14.25→4.41s | **3.23×** | 11.16→1.62s | **6.87×** | 0.0 |
- Decision: **ACCEPT** (opt-in). split_scan 5–9×, whole depthwise fit 1.9–3.9×,
  **quality identical** (Δ=0 — no near-tied flips occurred, though the allclose-not-
  bitwise caveat stands in general). NOTE: **narrow wins too** — batching amortizes
  the launch that made the per-node device scan a loser ([[gpu-cuda-bottleneck-split-scan]]
  predicted exactly this). Follow-up worth a beat: flip the gate default ON for
  cuda+depthwise (the MO device-scan precedent defaults on; CUDA is already allclose).
- Commit: Stage 1 aae7332, Stage 2 f73d6e9 + this A/B script/report.

### Campaign wrap (2026-06-25) — ~30-hypothesis backlog
Triaged ~30 hypotheses (E01–E30 + cuda-researcher H1–H13, `experiment-backlog.md`).
**Shipped 2:** E01 float32 (1.18× wide), **E03 gate 64→128 (1.65× @emb=128 — headline)**.
**Rejected w/ evidence:** iter001 forest-routing, E27 Cholesky, E12 eval(=F-update),
E18/E26 preprocessing(cold-start), E09 sibling-subtraction(shipped), E11 uint16(done).
**Held/queued:** node-batched CUDA scan (designed, Colab), forest-fused predictor,
uint8 bins, quantized-grad histogram, native-float32-wide. **Lower-ROI TODO** (post-E03,
emb>128 or small phases): E04/E14/E15/E16/E19. Meta-lesson: re-measure tuned gates.

## Session 2026-06-25 (iter 008+) — ship validated wins as defaults; broaden a local lever

### 008 — flip `REPLEAFGBM_CUDA_BATCHED_SCAN` default → ON (cuda+depthwise)   [ACCEPT — default change; Colab re-val queued]   2026-06-25
- Surface: `backends/cuda_backend.py` (`_resolve_batched_scan` default), `backends/base.py` (comment), `tests/test_cuda_backend.py` (kill-switch + default-on tests), docs (`cuda.md`, `CHANGELOG`, ADR 0005)   Backend: cuda
- Hypothesis: the node-batched depthwise scan (iter 007: T4 1.9–3.9× fit / 5–9× split_scan, quality-identical, parity 35/35) should be the cuda+depthwise default, mirroring the MO device-scan precedent (`REPLEAFGBM_CUDA_MO_DEVICE_SCAN`, default ON) — CUDA is already allclose + quality-equivalent by contract.
- Change: `_resolve_batched_scan` unset/empty → **True** (was False); a falsy value (`0/false/no/off`) is now the kill switch → per-node host loop. 1 logic line + comments/docstrings + tests (gate-off test → `test_batched_scan_on_by_default` + `test_batched_scan_kill_switch_loops_per_node`) + docs. Host NumPy/Rust path untouched; the dispatch guard (`grad.ndim==1 ∧ supports_batched_scan`, `tree.py:281`) and the adaptive `_scan_min_cells` crossover are unchanged, so only cuda+depthwise+scalar (non-tiny frontiers) changes behavior.
- Measure: evidence = iter 007 T4 A/B (wide 3.91× / narrow 1.91× / mc 3.23× fit; quality Δ=0). Local: ruff clean; `pytest tests/ -q` **421 passed / 96 skipped** (CUDA self-skip, no GPU) / 0 failed. Colab re-validation (default-on vs `=0` kill-switch, ≥5 reps interleaved) queued for the GPU session.
- Decision: **ACCEPT** (deliberate default change) — gated on core-reviewer sign-off + the Colab re-val above before it counts as shipped. Zero new risk: identical code path to iter 007, default-resolved on; kill switch preserved; host bitwise untouched. SemVer MINOR (new default on an optional GPU sub-feature; no public API / model-format change).
- Commit: 5695fbc (separate). Colab re-val (report → `experiments/results/2026-06-25-batched-scan-default-on.md`) queued for the GPU session.

### 009 — E15: `float32_gram` for multi-output (vector) leaves   [ACCEPT — opt-in, default-off]   2026-06-25
- Surface: `core/multioutput.py::fit_vector_leaves` (the two wide-emb GEMM reductions) + `tests/test_leaf_fit_precision.py` (3 MO tests)   Backend: numpy/local
- Hypothesis: extend the approved `leaf_fit_precision="float32_gram"` opt-in to the shared-routing vector-leaf fit, the one multi-output path with NO float32 branch.
- Investigation (re-scoped the backlog): the *multiclass* wide path already reuses the scalar float32 branch (per-class `fit_leaves` fallback, `leaf_models.py:457`); the *narrow* multiclass path is native Rust `leaf_linear_stats_mc` (float64 → E02 territory, not E15). The only genuine NumPy-BLAS target is `fit_vector_leaves` (`multioutput.py:38`) — pure NumPy, no native path, centered Gram `wZc.T@Zc` + projection `wZc.T@tc`.
- Change: float32 confined to those two reductions (gated on `leaf_model.leaf_fit_precision`); centering, the float64 solve, z_min/z_max, and the LOO-gate leverage stay float64. Default float64 byte-identical.
- Measure (orchestrator `--mode ab`, rust, multioutput, 30k×256f, emb=256, K=3, 25 trees, 5 reps interleaved, OMP=1):
  - **float64 17.89s → float32 16.83s = 1.055× (−5.5%), B wins 5/5, signal=True, spread 1.7–1.8%.**
  - Cheap evidence: isolated vector-reduction 1.28–1.38× @emb256 (weight dev ~1e-5); e2e leaf_fit share 21–31% of MO fit (rust, OMP=1) → projected +3.6–7.8%.
  - Quality-equivalent on the A/B config: per-output r2 identical (|Δr2|=7.8e-9), max |Δpred|/scale=1.8e-6; unit allclose + e2e |Δr2|<5e-3 tests pass.
- Decision: **ACCEPT**, opt-in default-off — broadens the already-approved `leaf_fit_precision` (no new public param → no new human gate). Clears the gate (≥+3%, 5/5, low spread, quality-equivalent). Default float64 stays bitwise (new test asserts default == explicit float64). Complements E01: scalar AND vector wide-emb leaves now honor the precision knob; the remaining float64 wide leaf-fit is the scalar BLAS solve + this vector path (→ E02 native).
- Commit: queued (separate). Evidence: `artifacts/gpu_bench/e15_mo_ab/`.
