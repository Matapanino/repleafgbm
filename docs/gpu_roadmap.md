# RepLeafGBM GPU and Native Acceleration Roadmap

This roadmap follows the audit in `docs/gpu_audit.md`. The guiding constraint is
unchanged: keep raw-feature routing plus representation-conditioned leaves, keep
the public sklearn-style API stable, and avoid a large rewrite until profiling
justifies it.

## Principles

- Optimize measured bottlenecks, not assumed bottlenecks.
- Preserve the NumPy reference path and existing parity tests.
- Prefer Rust/native for branchy CPU-bound orchestration before forcing it onto
  CUDA.
- Use CUDA where data is already resident or the kernel has enough work to
  amortize launch and synchronization.
- Keep `split_backend="cuda"` explicit. Do not make `"auto"` pick CUDA.
- Do not introduce public GPU tensor inputs until a device-aware dataset policy
  exists.

## Phase 0: Baseline And Measurement

Status: largely landed — the measurement harness and CUDA transfer counters are
in; per-phase internal timers remain deferred.

Tasks:

- [x] Add a GPU benchmark runner under `benchmarks/` that records JSONL results
  (`benchmarks/gpu_profile.py`).
- [x] Extend the Colab CUDA loop to run selected benchmark cases in addition to
  parity tests (`scripts/colab_remote_test.py` runs a `gpu_profile` smoke;
  `scripts/colab_gpu_test.sh` pulls back the JSONL).
- [ ] Add optional internal timers for preprocessing, binning, histogram build,
  split scan, partition, leaf fitting, eval, and predict. **Deferred** to keep
  the boosting loop untouched; the JSONL `phase_seconds` field is reserved but
  emitted empty for now.
- [x] Add CUDA transfer counters for binned upload, rows upload, grad/hess
  upload, histogram copy-back, categorical slice copy-back, and winner copy-back
  (`CudaSplitBackend.get_transfer_stats()`; surfaced as the JSONL
  `transfer_bytes` field, read off the fitted booster's `split_backend_`).
- [x] Record environment metadata: git SHA, dirty flag, GPU, CuPy, NumPy,
  scikit-learn, Python, and whether `repleafgbm_native` is installed.

Acceptance criteria:

- [x] `bash scripts/colab_gpu_test.sh --gpu T4` still runs the existing CUDA
  parity tests.
- [x] A benchmark summary is written beside the JSONL (`summary.md`); the Colab
  report (`experiments/results/<date>-cuda-parity.md`) gains a transfer-counter
  table.
- [x] Local CPU-only smoke execution does not require CuPy or a GPU
  (`tests/test_gpu_profile.py`; `--backend numpy`).

## Phase 1: Low-risk Performance Patches

### 1. Cache Full Grad/Hess Buffers On CUDA

Target:

- `src/repleafgbm/backends/cuda_backend.py`

Plan:

- Cache full contiguous `grad` and `hess` arrays on device, keyed by array
  identity plus shape/strides or by explicit cache version.
- Change the RawKernel to read `grad[row]` and `hess[row]` rather than
  pre-gathered `g_sel` and `h_sel`.
- Keep the current host path for arrays that cannot be cached safely.

Expected effect:

- Reduces per-node H2D transfer by about `16 * n_selected_rows` bytes.
- Removes CPU fancy-index gathers of `grad[rows]` and `hess[rows]`.

Tests:

- Existing CUDA parity tests.
- Weighted regression/classification.
- Multiclass class-column views.
- Repeated rounds to catch stale cache reuse.

### 2. Constant Leaf Vectorization

Target:

- `src/repleafgbm/core/leaf_models.py::ConstantLeafModel.fit_leaves`

Plan:

- Replace per-leaf list comprehension with concatenated row order plus
  `np.add.reduceat` or `np.bincount`.
- Keep exact output dtype and shape.

Expected effect:

- Small but broad CPU speedup for constant leaves and multiclass.

Tests:

- Existing leaf model tests.
- End-to-end regression/classification prediction parity.

### 3. Documentation Sync

Target:

- `docs/backend_strategy.md`
- `docs/cuda.md`
- `docs/adr/0005-cuda-backend-cupy.md`

Plan:

- Ensure every document says CUDA Phase B2 includes resident histograms and
  adaptive GPU numeric split scan.
- Keep explicit notes that categorical and multi-output scans remain host.

## Phase 2: Rust-native CPU Bottlenecks

### 1. Binning Kernel

Target:

- `src/repleafgbm/core/histogram.py`
- `native/src/lib.rs`

Plan:

- Add Rust-native bin assignment for dense `float64` features and `float64`
  thresholds.
- Preserve NumPy as the reference implementation.
- Consider threshold fitting later; start with bin assignment because it is
  simpler and used every fit.

Acceptance criteria:

- Exact bin parity with NumPy for NaN, constant columns, repeated values, and
  max-bin edge cases.

### 2. Row Partition Kernel

Target:

- `src/repleafgbm/core/splitter.py::partition`
- `native/src/lib.rs`

Plan:

- Add a native partition function for numeric and categorical subset splits.
- Return `rows_l` and `rows_r` as NumPy arrays with the same order as current
  NumPy masks.

Acceptance criteria:

- Tree structure and predictions remain identical for NumPy vs Rust-native
  partition under the same backend.

### 3. Batched Predictor

Target:

- `src/repleafgbm/core/tree.py::Tree.apply`
- `src/repleafgbm/core/prediction.py`
- `native/src/lib.rs`

Plan:

- First implement `apply_tree` / `apply_forest` in Rust returning leaf IDs.
- Then optionally compute leaf outputs in the same native pass.
- Keep Python fallback for portability.

Acceptance criteria:

- Leaf-id parity for numeric, categorical subset, missing, and externally
  extracted routes.
- Predict time improves materially for large ensembles.

## Phase 3: Backend Contract Extensions

### 1. Multi-output Histogram/Scan

Target:

- `src/repleafgbm/core/splitter.py`
- split backends

Plan:

- Add an optional backend method for multi-output histograms and split scans.
- CUDA path should keep `(features, bins, channels, outputs)` resident and scan
  summed gains on device.
- Rust path should avoid per-output Python loops and repeated host stacking.

Acceptance criteria:

- Multi-output squared, huber, and quantile models match the NumPy reference
  within existing tolerances.

### 2. Multiclass Batched Histogram

Target:

- `src/repleafgbm/core/multiclass.py`
- split backends

Plan:

- Add optional batched histogram construction for `grad/hess` matrices.
- For CUDA, use one pass over rows/features and accumulate per-class channels.
- For Rust, evaluate parallelization and memory layout first.

Acceptance criteria:

- Multiclass parity with and without sample/class weights.
- Scaling improves with `n_classes`.

## Phase 4: Representation-side Acceleration

### 1. float32 Embedding Cache

Target:

- `src/repleafgbm/data/dataset.py`
- encoders
- leaf models

Plan:

- Add an internal option for `Z` storage dtype, defaulting to current `float64`.
- Upcast where numerical solves need `float64`, or validate `float32` solves
  explicitly.

Acceptance criteria:

- Default behavior is unchanged.
- `float32` path passes tolerance-based accuracy tests and lowers peak memory.

### 2. Batched Encoder Transform

Target:

- `BaseEncoder`
- fixed and torch-frozen encoders
- `RepLeafDataset`

Plan:

- Add a private or experimental batch transform hook.
- Use it for prediction and future out-of-core training.
- Fuse random projection with base transform where practical.

Acceptance criteria:

- Batched and full transform outputs match within tolerance.
- Peak memory decreases on high-dimensional PLR/projection cases.

### 3. GPU Leaf Ridge

Target:

- `EmbeddedLinearLeafModel`
- `fit_vector_leaves`

Plan:

- Only start after profiling shows leaf fitting dominates.
- Implement batched statistics and solve on device for wide embeddings.
- Keep Rust and NumPy paths as reference/fallback.

Acceptance criteria:

- Parity and fallback behavior are documented.
- End-to-end fit improves for high `max_leaf_emb_dim` and many leaves.

## Phase 5: Device-aware Dataset And Large Data

Tasks:

- Design a dataset policy that can represent host NumPy, CuPy dense arrays,
  memory-mapped arrays, and future sparse matrices.
- Add chunked training/prediction primitives.
- Define a sparse-specific route rather than retrofitting sparse into the dense
  binned matrix.
- Evaluate multi-GPU only after single-GPU transfers and residency are solved.

## Near-term PR Sequence

1. Benchmark/profiler harness and docs sync.
2. CUDA grad/hess device cache.
3. Constant leaf vectorization.
4. Rust partition kernel.
5. Rust batched predictor.
6. Multi-output backend scan.
7. Multiclass batched histogram.
8. float32/batched embedding work.

## Explicit Non-goals For The Next Few PRs

- No public API change for GPU tensor inputs.
- No automatic CUDA selection from `split_backend="auto"`.
- No replacement of raw-feature routing with embedding-based splits.
- No joint encoder updates during boosting.
- No sparse GPU path until a separate sparse dataset design exists.

