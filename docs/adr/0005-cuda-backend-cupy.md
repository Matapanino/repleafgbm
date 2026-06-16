# ADR 0005: CUDA split backend via CuPy (Phase A + B1)

- Status: **accepted — Phase A + B1 implemented** (2026-06-17); GPU validation
  runs through the Colab dev loop (`scripts/colab_gpu_test.sh`), not CI. Phase C1
  (GPU leaf stats) was evaluated against an end-to-end benchmark and **deferred**
  (see Consequences).
- Date: 2026-06-17
- Depends on: ADR 0001, docs/backend_strategy.md (the narrow `BaseSplitBackend`
  contract + sibling-subtraction trick), the Rust backend (Phase 10) as the
  optional-backend template.

## Context

`docs/backend_strategy.md` reserves `backends/cuda_backend.py` as the GPU
compute backend ("planned (v3)"), and `docs/roadmap.md` lists "CUDA histogram
building" under v3. This ADR brings that item forward, **strictly behind the
existing `BaseSplitBackend` abstraction**. It does not touch the core thesis
(raw-feature routing, frozen encoder, no splitting on embedding dims) and does
not modify the boosting loop, splitter, or tree grower — it adds a third
compute backend selectable with `split_backend="cuda"`.

The constraint that shaped every decision: the dev box is macOS (no NVIDIA
GPU) and GitHub Actions has no GPU runner, so the CUDA path cannot be built or
tested locally or in CI.

## Decisions

1. **Tech: CuPy, not Rust-CUDA or Numba.** The histogram kernel is a
   `cupy.RawKernel` (CUDA C, float64 `atomicAdd`) JIT-compiled at runtime. No
   new compiled package, no maturin/nvcc/manylinux wheel pipeline — the whole
   backend is one Python file (`backends/cuda_backend.py`, the documented slot)
   plus an optional `[cuda]` extra (`cupy-cuda12x`). CuPy is torch-independent,
   so the "native compute path never imports torch/lightgbm/external"
   invariant holds.

2. **Scope (Phase A): histogram on GPU only; split scan on the host.**
   `build_histograms` runs on the GPU; `find_best_split` delegates verbatim to
   `NumPySplitBackend`. The `(n_features, n_bins_max, 3)` histogram is tiny, and
   the split scan carries all the branchy, parity-critical logic (categorical
   gradient-sorted subsets, stable sort, tie-break on lowest index,
   missing-routes-left). Delegating keeps that **byte-for-byte identical** to
   the reference and shrinks the parity surface to a single kernel.

3. **Parity: allclose, not bitwise.** GPU `atomicAdd` summation order is not
   fixed, so histogram sums differ from NumPy `bincount` in the low bits — and
   are not even reproducible run-to-run. We therefore:
   - assert histogram parity by `allclose` (rtol/atol ~1e-9), with the *count*
     channel still exact (integer sums < 2^53);
   - assert end-to-end agreement at `rtol=1e-6, atol=1e-8` (the same bar the
     Rust end-to-end tests use);
   - rely on subtractability holding to float noise (~1e-15 abs): the grower's
     `right = parent - left` stays within end-to-end tolerance over a tree.

   This is a deliberate, documented departure from the NumPy⇄Rust **bitwise**
   histogram rule. The Rust path keeps its bitwise tests unchanged; the rule in
   CLAUDE.md now reads "bitwise for numpy⇄rust; allclose for cuda."

4. **`"cuda"` is explicit-only.** `make_split_backend("auto")` is unchanged
   (Rust→NumPy) — it never probes for a GPU. Users opt in with
   `split_backend="cuda"`, which raises a clear `ImportError` when CuPy or a GPU
   is missing (never silently falls back, so a typo on a GPU box is visible).

## Consequences / non-goals

- **Phase A (shipped):** GPU histogram + host split scan. Correctness + parity
  milestone (allclose, 7/7 on a Tesla T4).
- **Phase B1 (shipped):** resident-data fast path. The binned matrix is the same
  object for every node of every tree, so `CudaSplitBackend` uploads it once and
  caches it on-device (keyed by `id` + shape); each node then ships only its
  `rows` + gathered grad/hess and the kernel gathers bins on-device. This needs
  **no interface or core change** (rejected the originally-planned
  `begin_tree/end_tree` seam as unnecessarily invasive). Measured on a T4:
  histogram micro-benchmark **32x** over NumPy (was ~8.5x before caching), and
  **1.58x end-to-end** `RepLeafRegressor.fit` (100k×30, 50 trees,
  embedded_linear). See `experiments/results/2026-06-17-cuda-parity.md`.
- **Phase C1 (evaluated, deferred):** GPU `leaf_linear_stats`. The 1.58x
  end-to-end implies histogram was only ~37% of the fit; the remaining host work
  (tree growth, split scan, leaf fitting) dominates. Crucially, leaf fitting is
  *already* accelerated by the Rust `leaf_linear_stats` kernel in real
  deployments (the Colab box lacked it, so the benchmark ran the slow NumPy
  fallback — inflating, not understating, the host share). So GPU leaf stats
  would target a small, already-fast slice while costing a wider core change
  (threading the split backend into the leaf model) plus a new Gram-matrix kernel
  with its own parity surface. Per the project's "change directions only with
  evidence" rule, **C1 is deferred**; revisit only if profiling shows leaf
  fitting is a bottleneck for some workload.
- **Phase B2 (future, if more speed is wanted):** keep histograms resident and
  port the *numeric* split scan to GPU to cut the per-node GPU→host round-trip;
  categoricals/tie-break stay on host for parity. Higher value than C1 per the
  benchmark; gated by a results-analyst-backed end-to-end measurement.

## Validation

CI and the macOS dev box skip the CUDA tests (`pytest.importorskip("cupy")` +
device check), so they stay green without a GPU. `tests/test_backends_registry.py`
covers the no-GPU dispatch (`"cuda"` → `ImportError`, unknown → `ValueError`,
`"auto"` never CUDA) on every lane. GPU parity + benchmark run via
`bash scripts/colab_gpu_test.sh --gpu T4`, which provisions a Colab VM, runs
`tests/test_cuda_backend.py` on the GPU, and downloads a dated report to
`experiments/results/`.
