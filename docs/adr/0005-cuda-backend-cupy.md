# ADR 0005: CUDA split backend via CuPy (Phase A + B1 + B2)

- Status: **accepted — Phase A + B1 + B2 implemented** (B2: 2026-06-17); GPU
  validation runs through the Colab dev loop (`scripts/colab_gpu_test.sh`), not
  CI. B2 (resident histograms + **adaptive** GPU numeric scan) is parity-verified
  on a T4 (9/9) and measured at **~2.1x end-to-end on a wide fit and ~1.5x on a
  narrow fit** (the adaptive threshold keeps narrow on the host path so it
  matches B1 instead of regressing — see Consequences). Phase C1 (GPU leaf stats)
  was evaluated against an end-to-end benchmark and **deferred**.
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
   *(Superseded for numeric features by Phase B2 below: the numeric
   cumulative-sum gain sweep + argmax now run on-device; categorical subsets,
   the branchiest part, stay on the host.)*

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
- **Phase C1 (evaluated, deferred — later SHIPPED 2026-07-02):** GPU
  `leaf_linear_stats`. The 1.58x end-to-end implies histogram was only ~37% of
  the fit; the remaining host work (tree growth, split scan, leaf fitting)
  dominates. Crucially, leaf fitting is *already* accelerated by the Rust
  `leaf_linear_stats` kernel in real deployments (the Colab box lacked it, so
  the benchmark ran the slow NumPy fallback — inflating, not understating, the
  host share). So GPU leaf stats would target a small, already-fast slice while
  costing a wider core change (threading the split backend into the leaf model)
  plus a new Gram-matrix kernel with its own parity surface. Per the project's
  "change directions only with evidence" rule, **C1 was deferred**; the evidence
  arrived once the batched scans shipped and leaf_fit became 49–73% of CUDA fit
  (2026-06-25 sizing) — see the device leaf-fit entry below.
- **Device leaf-fit statistics** (added + on by default 2026-07-02, iter 012).
  `leaf_fit_stats` computes the per-leaf weighted Gram stacks / projections /
  z-range guards on-device (Z identity-cached per fit); the centering, ridge
  solve, and adaptive LOO gate stay host float64 through the same assembly as
  the native path. This *extends the allclose-not-bitwise decision one layer
  down*: a **near-tied adaptive LOO-gate verdict can flip** under device
  reduction noise, exactly like near-tied splits, so its e2e tests assert
  quality-equivalence (|Δr²| bound). Two hard-won constraints: CuPy's float
  `scatter_min/max` rounds float64 through float32 (~5e-8) — guard bounds are
  exact per-leaf slice reductions — and device-sum parity is asserted at
  rtol/atol 1e-9 (atomic-order noise ~1e-12). T4: wide fit 1.72×, narrow 1.23×
  (`experiments/results/2026-07-02-cuda-leaf-ridge-ab.md`); kill switch
  `REPLEAFGBM_CUDA_LEAF_FIT=0`, crossover `REPLEAFGBM_CUDA_LEAF_FIT_MIN_CELLS`.
- **Phase B2 (shipped, adaptive):** resident histograms + an **adaptive**
  numeric split scan. `build_histograms` now *returns* the
  `(n_features, n_bins_max, 3)` histogram as a resident CuPy array instead of
  copying it to the host, so it never leaves the GPU during a tree's growth — the
  grower's sibling-subtraction `parent - child` is CuPy arithmetic. The numeric
  gain sweep + argmax run **on the GPU only when the per-node histogram is large**
  (`n_features * n_bins_max >= _GPU_SCAN_MIN_CELLS`, default 2^15); for small
  histograms the scan is copied back and delegated to the reference, which beats
  launching many tiny GPU kernels. When on-device, only the winning split's
  scalars cross back per node (cut to a single batched `asnumpy`).

  Why adaptive: the T4 benchmark showed the GPU scan is **~2.1x end-to-end on a
  wide fit (50k×200)** — the big per-node histogram round-trip B1 paid is
  avoided — but **~neutral / a regression on a narrow fit (100k×30)**, where the
  (30×257) scan is too small to amortize GPU launch/sync overhead and B1's bulk
  copy + vectorized host scan wins. With the threshold in place the CUDA backend
  measures **2.09x (wide) and 1.46x (narrow, host path)** — no regression, full
  wide win; tune the threshold with a per-GPU sweep. See
  `experiments/results/2026-06-17-cuda-parity.md`.

  This needs **no core change**: the grower already treats the histogram opaquely
  (only subtraction + indexing + pass to `find_best_split`), so CuPy duck-types
  through it. Parity stays **allclose, not bitwise**, but with a subtlety worth
  recording: when a node is scanned on the **host** (the adaptive default for
  narrow histograms) the chosen split — and thus the tree — is identical to the
  reference and only leaf values carry the histogram's low-bit float noise, so
  predictions agree to `rtol=1e-6`. When the numeric scan runs **on-device** its
  gains are reduced with CuPy whose low bits also differ, so on a *near-tied* node
  the lowest-index argmax can select a different — but equally good — split,
  leaving the trees structurally different and predictions allclose except on the
  few rows a flipped split reroutes. Those flips are quality-neutral (the gains
  were tied), so model quality matches even when the exact tree does not. (An
  earlier draft of this ADR claimed genuine ties are measure-zero so the selected
  split always agrees with the reference; that holds for the host scan but not the
  on-device scan, whose CuPy reductions perturb the gains.) Verified by
  `tests/test_cuda_backend.py`: `rtol=1e-6` on the host-scan path,
  quality-equivalence (RMSE + bulk-agreement) on the device path. Held to host for
  parity: only the categorical subset scan (stable sort / both-end prefix), which
  gets just its few feature slices copied back.

- **Multi-output (shared-routing) trees get the same B2 residency** (added
  2026-06-24). Previously the K per-output histograms were built on the GPU,
  copied to the host, `np.stack`ed, and scanned by the NumPy
  `find_best_split_multioutput` — losing residency. Now optional
  `BaseSplitBackend.build_histograms_multioutput` / `find_best_split_multioutput`
  carry the stack + scan, with **host defaults that reproduce that exact prior
  behavior** (so NumPy/Rust stay bitwise-unchanged), and `CudaSplitBackend`
  overrides them to keep the `(F, bins, 3, K)` stack resident and run the
  summed-gain scan on-device, returning only the winning split. It is node-local
  (multi-output is one shared tree, so the K histograms already co-locate at a
  node) — the grower is untouched. The `REPLEAFGBM_CUDA_MO_DEVICE_SCAN` env var
  (read once at construction, private) forces the host fallback as a kill switch,
  and the same `_scan_min_cells` crossover sends narrow nodes to the host. On a
  T4: ~2.95x wide-200f fit (device scan off→on) and ~5.3x vs NumPy; see
  `experiments/results/2026-06-24-cuda-parity.md`.

- **Node-batched depthwise scan** (added + on by default 2026-06-25). With
  `grow_policy="depthwise"` the grower hands a whole level's M frontier histograms
  to `BaseSplitBackend.find_best_split_batched` for one device scan instead of one
  call per node, amortizing the per-node kernel launch (the measured GPU scan
  bottleneck, 48–85% of CUDA fit). The host default loops `find_best_split`, so
  NumPy/Rust stay **bitwise** and the grower's batched path is bitwise-identical to
  the per-node FIFO; `CudaSplitBackend` overrides it with a CuPy M-axis scan (no
  RawKernel). Like the MO device scan it is **on by default** with a private
  `REPLEAFGBM_CUDA_BATCHED_SCAN=0` kill switch, and `_scan_min_cells` still routes
  tiny frontiers to the host. Scalar targets only (multi-output keeps the per-node
  path). T4: split_scan 5–9x, depthwise fit 1.9–3.9x, quality-equivalent; see
  `experiments/results/2026-06-25-batched-scan-ab.md`.

## Validation

CI and the macOS dev box skip the CUDA tests (`pytest.importorskip("cupy")` +
device check), so they stay green without a GPU. `tests/test_backends_registry.py`
covers the no-GPU dispatch (`"cuda"` → `ImportError`, unknown → `ValueError`,
`"auto"` never CUDA) on every lane. GPU parity + benchmark run via
`bash scripts/colab_gpu_test.sh --gpu T4`, which provisions a Colab VM, runs
`tests/test_cuda_backend.py` on the GPU, and downloads a dated report to
`experiments/results/`.
