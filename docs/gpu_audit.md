# GPU Utilization Audit for RepLeafGBM

## Executive Summary

- Current GPU utilization is partial and intentionally narrow. `split_backend="cuda"` accelerates scalar split search: the binned raw-feature matrix is cached on the GPU, per-node histograms are built by a CuPy RawKernel, large numeric split scans run on the GPU, and only the winning split scalars cross back. This preserves the core design: raw-feature routing plus representation-conditioned leaves.
- The largest remaining bottleneck is not a single kernel. End-to-end fit still spends substantial time in CPU orchestration: feature binning, tree growth, categorical split handling, leaf-model fitting, prediction traversal, objective/eval loops, and sklearn/pandas preprocessing. Row partitioning is now native on the Rust path, but the NumPy reference and host row-array orchestration still exist.
- The highest-priority performance issue in the existing CUDA path is avoidable per-node host work and transfer: `CudaSplitBackend.build_histograms` uploads `rows`, `grad[rows]`, and `hess[rows]` for every node. `binned` is resident, but gradients and Hessians are not.
- The most important CPU bottlenecks outside CUDA are prediction path traversal (`Tree.apply` per tree), multi-output split search falling back to host (`Splitter.build_histograms` calls `_as_host` per output), and per-leaf ridge/statistics for embedded leaves.
- Do not GPU-accelerate everything immediately. Sparse input is rejected by the sklearn layer, pandas/category preprocessing is not a GPU target, small numeric histograms already use the faster host scan path, and low-dimensional leaf statistics already have a Rust fast path.
- Historical first patch: add the GPU benchmark/profiling harness and measure the CUDA `grad`/`hess` cache idea. The harness shipped; the cache was measured and deferred because transfer bytes fell but wall-clock did not. Current CPU evidence points to compiled prediction as the next low-risk Rust target before another CUDA rewrite.

## Current Execution Map

| Area | Files / functions | Current execution | CPU/GPU/Rust boundary | Notes |
|---|---|---|---|---|
| Public sklearn API | `src/repleafgbm/sklearn.py`, `classifier.py`, `regressor.py` | CPU Python, NumPy, sklearn validation | `check_array` / `check_X_y` reject sparse; `RepLeafDataset` stores NumPy arrays | This layer is API glue, not a good GPU target. |
| Pandas/category preprocessing | `data/preprocessing.py::encode_features` | CPU Python loops over columns; list comprehensions for categories | DataFrame/object values become a dense `float64` NumPy matrix | Correctness and metadata stability matter more than GPU speed here. |
| Raw feature storage | `data/dataset.py::RepLeafDataset` | CPU NumPy `float64` | `get_raw_features()` returns host array | No GPU-resident dataset policy exists yet. |
| Encoder fit/transform, fixed encoders | `encoders/identity.py`, `plr.py`, `periodic.py`, `cross.py`, `projection.py` | CPU NumPy, dense `float64` | `get_embeddings()` caches a host NumPy `Z` | PLR and binning-like transforms loop over features; projection is CPU BLAS. |
| Torch encoder pretraining | `encoders/torch_encoders.py::_pretrain` | Optional torch CPU/CUDA during `fit` only | NumPy `X`/target copied to torch `float32`; learned params copied back to NumPy `float64` | `transform()` remains NumPy CPU after freezing. |
| Feature binning | `core/histogram.py`, `core/splitter.py::__init__` | CPU NumPy, feature loops | Produces host `uint16` binned matrix | CUDA backend uploads this matrix once, but binning itself is CPU. |
| Scalar histogram construction | `backends/numpy_backend.py`, `rust_backend.py`, `cuda_backend.py` | NumPy: CPU vectorized; Rust: native CPU; CUDA: GPU RawKernel | CUDA uploads `binned` once; per node uploads `rows`, `grad[rows]`, `hess[rows]` | CUDA histogram micro-benchmark is already strong (~52x on T4 for 200k x 50 x 65). |
| Scalar numeric split scan | `backends/numpy_backend.py::find_best_split`, `cuda_backend.py::find_best_split` | NumPy/Rust CPU; CUDA GPU only when `n_features * n_bins_max >= 32768` | Small CUDA histograms copy back to host; large scans return one packed winner | Adaptive path is correct: small scans do not amortize GPU launch/sync overhead. |
| Categorical split scan | `NumPySplitBackend._best_categorical_split`, CUDA delegates per feature slice | CPU host | CUDA copies categorical histogram slices with `cp.asnumpy(hist_d[f])` | Parity-critical branchy/stable-sort logic stays host. Good short-term choice. |
| Multi-output histogram/scan | `core/splitter.py::build_histograms`, `numpy_backend.py::find_best_split_multioutput` | Histograms can be built by backend per output; stacked and scanned on CPU | CUDA histograms are copied to host with `_as_host` before `np.stack` | Multi-output loses B2 residency and GPU scan benefits. |
| Tree growth / row partition | `core/tree.py::TreeGrower.grow`, `core/splitter.py::partition` | CPU Python heap; NumPy masks on the reference path; Rust `partition_rows` on the native path | CUDA histograms can stay device-resident, but rows remain host arrays | Partition is no longer the leading Rust bottleneck after PR #30; tree growth orchestration remains host-side. |
| Leaf assignment | `Tree.apply`, training `leaf_rows` lists | CPU vectorized loop over levels / trees | No GPU or Rust predictor path | Prediction repeats tree traversal for each tree. |
| Constant leaf fitting | `ConstantLeafModel.fit_leaves` | CPU Python list comprehension | None | Simple and low compute, but avoidable Python loop for many leaves/trees. |
| Embedded linear leaf fitting | `EmbeddedLinearLeafModel.fit_leaves`, `native::leaf_linear_stats` | CPU NumPy/BLAS; optional Rust stats for `emb_dim <= 32`; solve on CPU | `Z`, `grad`, `hess`, leaf rows are host arrays | Rust accelerates stats, but wide embeddings and solves remain CPU. |
| Vector leaves / multi-output | `core/multioutput.py::fit_vector_leaves` | CPU per-leaf Python loop plus NumPy solve | No Rust/GPU fast path | Scales with leaves x outputs x embedding dimension. |
| Objective / loss | `core/objectives.py` | CPU NumPy vectorized | No GPU state | Multiclass softmax materializes `(n_rows, n_classes)` grad/hess on host. |
| Early stopping / eval loop | `core/booster.py`, `multiclass.py`, `multioutput.py` | CPU; one new tree applied per eval round | Eval `Tree.apply` CPU and metrics CPU | Incremental raw-score cache is good; traversal remains costly. |
| Prediction | `core/prediction.py`, `Tree.apply`, `LeafValues.predict` | CPU loop over trees; NumPy `einsum` for leaf outputs | No compiled predictor | Likely dominant for large `n_trees` and multiclass. |
| Rust backend | `native/src/lib.rs`, `backends/rust_backend.py` | Native CPU kernels for histogram, split scan, leaf stats/predict helpers, and row partitioning | Python wrapper forces contiguous host NumPy arrays | Good target for compiled predictor and vector-leaf kernels before CUDA rewrites. |

## Bottleneck Analysis

### Feature Binning

- Complexity: threshold fitting is roughly `O(n_features * (n_rows log n_rows))` for quantile/unique work; bin assignment is `O(n_rows * n_features * log n_bins)` via per-feature `searchsorted`.
- Memory: host `X_raw` is `float64` (`8 * n_rows * n_features` bytes); `numeric_X = X_raw.copy()` duplicates the raw matrix in `Splitter.__init__`; binned output is `uint16` (`2 * n_rows * n_features` bytes).
- GPU potential: medium for large dense matrices, but GPU quantiles and category handling would need a device-resident dataset contract. Rust-native binning is lower risk first.
- Current issue: CPU feature loops in `compute_bin_thresholds` and `bin_features`; categorical features still require host metadata semantics.

### Split Candidate Evaluation

- Scalar numeric split scan is already vectorized in NumPy and adaptive CUDA for large histograms.
- Bottleneck remains for categorical and multi-output. Categorical split uses stable sorting and both-end prefix scanning per categorical feature; CUDA copies only categorical feature slices back, which is acceptable for few categoricals but costly for many/high-cardinality features.
- Multi-output stacks per-output histograms on host and scans with `find_best_split_multioutput`; this eliminates B2 GPU residency.

### Histogram Construction

- NumPy backend uses three `np.bincount` calls over a flattened `(row, feature)` index and repeats `grad`/`hess` per feature, causing large temporary memory.
- Rust backend avoids the NumPy temporaries and parallelizes the feature-major histogram path, but the kernel is still memory-bandwidth bound.
- CUDA backend is the best-developed GPU surface. It caches `binned` on device and gathers bins on device, but still gathers `grad[rows]` and `hess[rows]` on the CPU and uploads them for each node.
- Expected scaling: GPU helps most for large `n_rows`, wide `n_features`, high `max_bins`, and larger trees. It helps less for narrow datasets and small histograms.

### Leaf Assignment And Partitioning

- `TreeGrower.grow` stores row arrays for every active leaf. Row partitioning now has a native Rust kernel behind `split_backend="rust"` (`partition_rows`, native 0.2.0): a fused single pass replaces the NumPy multi-pass boolean masks (~4.5-5x per node, -9 to -12% end-to-end multiclass fit at medium/large; `benchmarks/partition_microbench.py`). The NumPy backend keeps the boolean-mask reference at exact (index-identical) parity.
- `Tree.apply` still routes all rows with a while loop over tree depth and a Python loop over categorical nodes at each level. These paths are memory-bandwidth and branch-heavy; a native predictor is the remaining compiled target here, ahead of a CUDA tree traversal.

### Representation / Embedding Construction

- All frozen encoders return dense host NumPy `float64`. `RepLeafDataset.get_embeddings()` materializes and caches the full `Z`.
- PLR/periodic/cross transforms can become memory-heavy for large `n_rows` and high output dimension. `RandomProjectionEncoder.transform()` materializes the full base embedding before multiplying by the projection.
- Torch encoders can pretrain on CUDA, but they freeze back into NumPy arrays. Inference-time transform and leaf fitting are CPU.

### Random Projection

- Complexity: `O(n_rows * base_dim * out_dim)`, CPU BLAS.
- Current design uses random projection only as a dimension cap when encoder output exceeds `max_leaf_emb_dim`.
- GPU benefit is workload-dependent and likely secondary because the projection already reduces downstream leaf-fit cost but may degrade accuracy. A fused/batched projection in an encoder transform pipeline is preferable to a standalone CUDA feature.

### Per-leaf Ridge / Linear Model Fitting

- Constant leaves use a Python list comprehension and per-leaf sums.
- Embedded leaves gather rows in leaf order, compute per-leaf stats, then solve batched normal equations. Rust `leaf_linear_stats` accelerates stats for `emb_dim <= 32`; wide embeddings use NumPy/BLAS per leaf; all solves are CPU.
- Complexity is roughly `O(total_leaf_rows * emb_dim^2 + n_linear_leaves * emb_dim^3)`. For high `max_leaf_emb_dim`, this can dominate once CUDA accelerates histograms.
- GPU ridge may help for large batches of leaves and medium/wide embeddings, but it requires careful batched Cholesky/solve and data-residency planning for `Z`, leaf indices, and grad/hess.

### Prediction Path Traversal

- Prediction loops over every tree, calls `Tree.apply`, then calls `LeafValues.predict`.
- Multiclass stores `n_rounds * n_classes` trees, so prediction traversal scales linearly in class count.
- This is a strong Rust-native target: a compiled batched predictor can traverse all trees with flat arrays and compute leaf outputs with less Python overhead.

### Multiclass Objective / Loss

- Softmax, gradients, and Hessians are CPU NumPy over `(n_rows, n_classes)`.
- Each boosting round grows one tree per class, reusing the same `Splitter` but building histograms per class. CUDA helps histograms but repeats row transfers and launches per class.
- Future improvement should batch class histograms as `(feature, bin, channel, class)` on GPU or Rust. This would reduce repeated memory reads and kernel launches.

### Early Stopping / Eval Loop

- The raw-score cache is incremental, which is good. But each round still routes eval rows through the new tree(s) on CPU and computes metrics on CPU.
- For small eval sets this is not worth GPU work. For large eval sets and multiclass, a compiled predictor path is likely valuable.

### sklearn API Wrapper

- Validation, metadata checks, and input conversion are CPU. Sparse inputs are explicitly rejected (`accept_sparse=False`), so there is no sparse GPU story today.
- This layer should remain stable and sklearn-compatible; do not introduce GPU tensors into public estimator inputs without a separate dataset/device policy.

### Python Loops Remaining In Heavy Paths

- `core/histogram.py`: per-feature bin threshold and bin assignment loops.
- `data/preprocessing.py`: per-column categorical/frequency list comprehensions.
- `core/tree.py`: heap-based growth, per-level categorical membership, final leaf row collection, and generic `Tree.apply` traversal.
- `core/booster.py` / `multiclass.py` / `multioutput.py`: per-round, per-leaf, per-class loops.
- `core/leaf_models.py` / `core/multioutput.py`: per-leaf stats/solve loops and fallback loops.
- `core/prediction.py`: per-tree prediction loops.
- `native/src/lib.rs`: native loops cover histogram, split scan, leaf stats/predict helpers, and row partitioning; remaining hot candidates include compiled tree traversal and vector-leaf paths.

### Rust Native Backend Boundary

- `RustSplitBackend` requires contiguous `uint16`, `int64`, and `float64` host arrays; wrappers use `np.ascontiguousarray`, which may copy.
- Rust kernels currently implement histogram, split scan, embedded-leaf stats, scalar linear prediction, and row partitioning. They do not implement native binning, `Tree.apply` / forest traversal, vector leaves, or multi-output backend scans.
- Rust is the best next step for CPU-bound branchy code because it preserves API compatibility and avoids CUDA-only behavior.

## Data Movement Analysis

### Host-side Baseline

- `RepLeafDataset` encodes inputs into a host `float64` raw matrix.
- `Splitter` creates a host `uint16` binned matrix.
- Encoders create a host dense `float64` embedding matrix `Z`.
- Gradients, Hessians, raw-score caches, leaf indices, trees, and leaf params are all host NumPy arrays.

### CUDA Split Backend Transfers

Current scalar CUDA path:

- One-time per backend/cache key: `binned` host `uint16` -> device `uint16`, about `2 * n_rows * n_features` bytes.
- Per histogram build: `rows` host -> device, about `8 * n_selected_rows` bytes.
- Per histogram build: `grad[rows]` and `hess[rows]` are gathered on CPU and uploaded, about `16 * n_selected_rows` bytes plus CPU gather cost.
- Per histogram build: histogram allocated on device, about `24 * n_features * n_bins_max` bytes (`float64` channels).
- Small scan path: full histogram copied device -> host, about `24 * n_features * n_bins_max` bytes.
- Large numeric scan path: one packed result copied device -> host, about four `float64` scalars. This is the desirable path.
- Categorical features: each categorical feature slice copied device -> host, about `24 * n_bins_max` bytes per categorical feature scanned.
- Multi-output: each output's histogram is copied device -> host before `np.stack`; total copied is about `24 * n_features * n_bins_max * n_outputs` per node.

Avoidable transfer: `grad[rows]` and `hess[rows]` should be gathered on device from cached full-round `grad`/`hess` buffers, just like `binned`.

### Torch Encoder Transfers

- `_pretrain` converts NumPy arrays to contiguous `float32` and moves them to the selected torch device.
- Learned parameters are copied back with `.detach().cpu().numpy().astype(np.float64)`.
- After fit, transform/predict use CPU NumPy. Therefore torch GPU pretraining does not create a GPU-resident boosting pipeline.

### dtype And Layout

- Core training buffers are mostly `float64`. This supports parity and numerical stability but doubles memory bandwidth compared with `float32`.
- CUDA histogram uses `float64 atomicAdd`. This is slower than `float32` on many GPUs and not bitwise deterministic, but it preserves current numerical behavior.
- `Z` is always `float64`. A `float32` embedding storage option is a strong memory lever but needs accuracy and parity tests.
- `binned` is `uint16`, which is good and matches `max_bins <= 65535` assumptions. There is no compressed bitset or sparse path.

## Improvement Roadmap

### Phase 1: Low-risk, Immediate Improvements

1. Add repeatable GPU benchmarking/profiling harness.
   - Target: `benchmarks/`, `scripts/colab_remote_test.py`.
   - Problem: current CUDA report measures only a few cases and does not log GPU utilization, transfer volume, peak memory, or CPU utilization.
   - Direction: add a JSONL benchmark runner with fixed dataset matrix, `nvidia-smi` sampling, CuPy memory-pool stats, and phase timers.
   - Effect: prevents optimizing the wrong path.
   - Difficulty: low.
   - API risk: none.
   - Tests: smoke run locally without CUDA; full run via Colab.
   - Benchmark: compare numpy/rust/cuda on small/medium/large grids.

2. Cache full `grad`/`hess` device buffers in `CudaSplitBackend` (measured and deferred).
   - Target: `src/repleafgbm/backends/cuda_backend.py::build_histograms`, RawKernel signature.
   - Problem: per-node CPU gather and H2D upload of `grad[rows]` and `hess[rows]`.
   - Why GPU is underused: the kernel gathers `binned` on device but not gradients/Hessians.
   - Direction: cache contiguous full `grad` and `hess` arrays on device keyed by `(id, shape, strides?)`; kernel reads `grad[row]` and `hess[row]`.
   - Effect: reduced transfer bytes in the follow-up experiment, but did not improve wall-clock enough to remain a near-term speed lever.
   - Difficulty: low to medium.
   - API risk: none if implemented inside backend.
   - Tests: existing CUDA parity tests plus weighted, multiclass, and class-view cases.
   - Benchmark: isolate histogram build, narrow/wide end-to-end, transfer counters.

3. Vectorize constant leaf fitting.
   - Target: `core/leaf_models.py::ConstantLeafModel.fit_leaves`.
   - Problem: Python list comprehension with per-leaf `grad[r]` / `hess[r]` indexing.
   - Why GPU is underused: not a GPU candidate; this is CPU overhead.
   - Direction: concatenate leaf rows once, use `np.add.reduceat` or `np.bincount`.
   - Effect: modest speedup for constant leaves and many leaves.
   - Difficulty: low.
   - API risk: none.
   - Tests: leaf model parity, end-to-end predictions.
   - Benchmark: constant-leaf fit with many trees/leaves.

4. Add phase timers and optional debug stats.
   - Target: `Booster._run_boosting`, `TreeGrower.grow`, `Splitter`, CUDA backend.
   - Problem: no built-in timing for binning, histogram, split scan, partition, leaf fit, eval, predict.
   - Direction: private opt-in profiler object or environment variable, not public API.
   - Effect: enables evidence-driven prioritization.
   - Difficulty: low.
   - API risk: low if private.
   - Tests: ensure disabled path has no behavior change.
   - Benchmark: verify phase sums match wall time.

### Phase 2: Medium-size Improvements

1. Rust-native binning and partition follow-up.
   - Target: `core/histogram.py`, `core/splitter.py::partition`, `native/src/lib.rs`.
   - Problem: feature-loop CPU binning remains; per-node partition masks remain only on the NumPy reference path.
   - Why GPU is underused: CUDA receives already-binned host data and host row arrays.
   - Direction: keep the landed `partition_rows` kernel; consider native bin assignment only with fresh evidence.
   - Effect: speeds CPU and CUDA paths without changing public API when the remaining host work is material.
   - Difficulty: medium.
   - API risk: none.
   - Tests: exact bin parity, categorical/missing parity, partition parity.
   - Benchmark: fit time with large `n_features`, high `max_bins`, many leaves.

2. Rust-native predictor.
   - Target: `Tree.apply`, `core/prediction.py`, `native/src/lib.rs`.
   - Problem: per-tree Python traversal dominates predict and eval on large ensembles.
   - Why GPU is underused: prediction never enters CUDA/Rust today.
   - Direction: compiled batched traversal over flat tree arrays; later include leaf output computation.
   - Effect: large predict speedup; also speeds early stopping eval.
   - Difficulty: medium.
   - API risk: low if internal fallback remains.
   - Tests: exact leaf-id parity for numeric/categorical/missing, prediction parity, serialization compatibility.
   - Benchmark: predict time vs `n_trees`, `n_rows`, `n_classes`. Shipped as `benchmarks/predict_profile.py`, which decomposes predict into routing (`Tree.apply`) vs leaf-eval (`LeafValues.predict`) so the routing share bounds this target's payoff; see `experiments/results/2026-06-24-prediction-traversal-bench.md`.

3. Multi-output CUDA/Rust split scan.
   - Target: `core/splitter.py::build_histograms`, `find_best_split_multioutput`.
   - Problem: CUDA histograms are copied to host and stacked per output.
   - Why GPU is underused: B2 device residency is lost for multi-output.
   - Direction: build/scan `(F, B, 3, K)` on device or Rust-native CPU; reduce gain across outputs on backend.
   - Effect: important for vector targets and high output counts.
   - Difficulty: medium to high.
   - API risk: low if backend contract is extended carefully or a new backend method is optional.
   - Tests: multi-output parity for squared, huber, quantile; CUDA allclose.
   - Benchmark: `n_outputs` sweep.

4. float32 embedding storage option.
   - Target: `RepLeafDataset.get_embeddings`, encoders, leaf models.
   - Problem: `Z` is dense `float64`, often a large memory-bandwidth bottleneck.
   - Why GPU is underused: even torch-pretrained encoders freeze to `float64` CPU.
   - Direction: optional internal `embedding_dtype`, default `float64`; solve can upcast if needed.
   - Effect: halves `Z` memory and bandwidth; may improve cache behavior.
   - Difficulty: medium.
   - API risk: low if default unchanged.
   - Tests: accuracy tolerance, serialization, leaf model parity within tolerance.
   - Benchmark: embedded leaves with high `max_leaf_emb_dim`.

### Phase 3: CUDA/Rust Native Acceleration

1. Batched class histogram and split scan for multiclass.
   - Target: `MulticlassBooster.fit`, `CudaSplitBackend`, Rust backend.
   - Problem: one tree per class repeats histogram work and per-node transfers.
   - Direction: backend builds histograms for multiple grad/hess columns in one pass where possible.
   - Effect: improves multiclass scaling, especially `n_classes >= 5`.
   - Difficulty: high.
   - API risk: medium if backend contract changes; can be optional method.
   - Tests: multiclass parity and quality metrics.
   - Benchmark: classes 3/5/10/20.

2. GPU-resident `Z`, leaf indices, and batched ridge.
   - Target: `RepLeafDataset`, `LeafValues`, `EmbeddedLinearLeafModel`.
   - Problem: GPU histograms speed up routing, but representation-conditioned leaves stay host.
   - Direction: only after profiling proves leaf fitting dominates; implement batched Cholesky/solve and stats with CuPy.
   - Effect: helps wide embeddings and many leaves.
   - Difficulty: high.
   - API risk: medium because dataset/device policy is needed.
   - Tests: leaf solve parity, finite fallback behavior, extrapolation guard.
   - Benchmark: `emb_dim`, `num_leaves`, `max_leaf_emb_dim`, projection on/off.

3. CUDA predictor for large batches.
   - Target: prediction path.
   - Problem: CPU traversal scales poorly for very large prediction batches.
   - Direction: consider only after Rust predictor and data-residency decisions; GPU tree traversal can be branch-divergent.
   - Effect: useful for large batch inference, less useful for small/online inference.
   - Difficulty: high.
   - API risk: low if internal.
   - Tests: prediction parity.
   - Benchmark: large batch predict throughput.

### Phase 4: Future Multi-GPU And Large-data Support

- Introduce a device-aware dataset policy: host NumPy, CuPy dense, torch tensor, memory-mapped/out-of-core.
- Add batch/chunked encoder transforms and chunked prediction to avoid materializing huge `Z`.
- Consider distributed or multi-GPU histogram partitioning only after single-GPU transfer boundaries are fixed.
- Add sparse/dense split at the dataset boundary. Do not retrofit sparse into the current dense assumptions.
- Add compatibility tests that preserve raw-feature routing and representation-conditioned leaf semantics under every backend.

## Proposed Benchmarks

### Dataset Grid

- Sizes:
  - small: 20k train / 10k test, 20 to 30 features.
  - medium: 100k train / 50k test, 50 to 100 features.
  - large: 500k train / 100k test, 200 features on L4/A100; scale down on T4 if memory-limited.
- Tasks:
  - regression: squared error plus MAE/RMSE.
  - binary classification: logloss, AUC, balanced accuracy.
  - multiclass classification: 3, 5, and 10 classes; multi-logloss, accuracy, balanced accuracy.
  - optional multi-output regression: 2, 8, and 32 outputs for the known host fallback.
- Feature regimes:
  - low feature count: 20 to 30.
  - wide: 200.
  - very wide stress: 1000 if memory allows.
  - categorical mix: 0%, 10%, 50% categoricals; low and high cardinality.
- Hyperparameter sweeps:
  - `split_backend`: `numpy`, `rust` if built, `cuda`.
  - `leaf_model`: `constant`, `embedded_linear`.
  - `encoder`: `identity`, `plr`, `torch_mlp` with CPU/CUDA pretraining.
  - `max_bins`: 32, 64, 256, 1024.
  - `num_leaves`: 8, 31, 127.
  - `max_leaf_emb_dim`: 16, 64, 256.
  - projection: off by choosing a low-dimensional encoder, on by forcing `max_leaf_emb_dim` below encoder output.

### Metrics To Record

- Fit time, predict time, and per-phase timings: preprocessing, binning, histogram, split scan, partition, leaf fit, eval, transform.
- Peak host memory: `psutil.Process().memory_info().rss` and optional `tracemalloc` for Python allocations.
- GPU memory: `cupy.get_default_memory_pool().used_bytes()`, CUDA runtime mem info, and `nvidia-smi`.
- GPU utilization, memory utilization, power, and temperature sampled via `nvidia-smi --query-gpu=... --format=csv`.
- CPU utilization and thread count.
- Estimated and observed data transfer volume:
  - binned upload bytes.
  - per-node rows/grad/hess upload bytes.
  - histogram copy-back bytes.
  - categorical slice copy-back bytes.
  - multi-output stack copy-back bytes.
- Quality metrics: RMSE/MAE, logloss/AUC/balanced accuracy, accuracy, multi-logloss.
- Backend parity: predictions vs NumPy at documented tolerances.
- Environment: git SHA, dirty flag, Python, NumPy, CuPy, CUDA driver/runtime, GPU model, CPU model, RAM, installed `repleafgbm_native`.

### Execution Command Ideas

Existing CUDA loop:

```bash
bash scripts/colab_gpu_test.sh --gpu T4
bash scripts/colab_gpu_test.sh --gpu L4
```

Proposed future benchmark runner:

```bash
python -m benchmarks.gpu_profile --task regression --size medium --backend numpy --out artifacts/gpu_bench/regression_medium.jsonl
python -m benchmarks.gpu_profile --task regression --size medium --backend cuda --out artifacts/gpu_bench/regression_medium.jsonl
python -m benchmarks.gpu_profile --task multiclass --n-classes 5 --size medium --backend cuda --out artifacts/gpu_bench/multiclass_5.jsonl
```

### Logs To Save

- `artifacts/gpu_bench/<date>/<gpu>/<case>.jsonl`
- `artifacts/gpu_bench/<date>/<gpu>/nvidia_smi.csv`
- `artifacts/gpu_bench/<date>/<gpu>/summary.md`
- `experiments/results/<date>-cuda-benchmark.md`
- Optional Nsight Systems trace for one representative wide run.

## Historical First Patch

### Patch Scope

1. Add `benchmarks/gpu_profile.py` and extend the Colab remote runner to execute selected benchmark cases and write JSONL plus a markdown summary.
2. Add private transfer counters to `CudaSplitBackend` for binned upload, rows upload, grad/hess upload, histogram copy-back, categorical slice copy-back, and winner copy-back.
3. The later cache experiment reduced H2D bytes but was deferred after wall-clock results did not move materially.

### Why This First

- It targeted the clearest CUDA transfer inefficiency without changing public APIs or the RepLeafGBM thesis. Later phase evidence shifted the near-term recommendation toward CPU/Rust prediction traversal.
- It gives evidence before larger Rust/CUDA work.
- It is compatible with the current Colab validation model.

### Impact Area

- `src/repleafgbm/backends/cuda_backend.py`
- `tests/test_cuda_backend.py`
- `scripts/colab_remote_test.py`
- `benchmarks/`

### Test Plan

- Existing CPU test suite.
- Existing CUDA parity tests on Colab T4.
- New transfer-counter tests using a small synthetic histogram build.
- End-to-end NumPy vs CUDA prediction allclose for regression, binary, multiclass, weighted, categorical.

### Benchmark Plan

- Re-run the existing narrow/wide T4 benchmark.
- Add before/after histogram-only timing where `binned` is already resident.
- Add measured H2D bytes before/after for `grad`/`hess` caching.
- Add one multiclass case to detect non-contiguous class-column regressions.

## Detailed Improvement Proposals

### 1. Device-cache Gradients And Hessians (deferred)

- Improvement target file/function: `src/repleafgbm/backends/cuda_backend.py::build_histograms`, `_BUILD_HIST_SRC`.
- Current problem: every node uploads `grad[rows]` and `hess[rows]` after CPU-side fancy indexing.
- Why GPU is not fully used: the device kernel already gathers `binned[row, feature]`, but gradient/Hessian values are gathered on host.
- Improvement direction: cache full contiguous `grad` and `hess` device buffers per round/tree; change kernel inputs from `g_sel/h_sel` to full `grad/hess`.
- Expected effect: less host CPU gather and lower H2D transfer; benchmarked wall-clock impact was neutral, so this is not the next speed lever.
- Implementation difficulty: low to medium.
- API break risk: none if backend-internal.
- Test direction: CUDA histogram parity, weighted gradients, multiclass class-column views, no stale cache across rounds.
- Benchmark direction: per-node transfer bytes and end-to-end narrow/wide runs.

### 2. Native Binning And Partition Follow-up

- Improvement target file/function: `core/histogram.py`, `core/splitter.py::partition`, `native/src/lib.rs`.
- Current problem: binning is still a CPU NumPy feature loop; partitioning is solved for `split_backend="rust"` but remains as the NumPy reference fallback.
- Why GPU is not fully used: CUDA starts after host binning and still consumes host row arrays.
- Improvement direction: keep `partition_rows`; consider Rust `bin_features` only if post-PR phase profiles make binning a leading phase.
- Expected effect: faster CPU baseline and CUDA end-to-end only when the remaining host orchestration is material.
- Implementation difficulty: medium.
- API break risk: none.
- Test direction: exact parity for missing, categorical, high-cardinality fallback, thresholds.
- Benchmark direction: sweep `n_features`, `max_bins`, `num_leaves`.

### 3. Compiled Prediction Path

- Improvement target file/function: `core/tree.py::Tree.apply`, `core/prediction.py`, `native/src/lib.rs`.
- Current problem: prediction loops over trees and traverses rows in Python/NumPy.
- Why GPU is not fully used: no backend participates in prediction.
- Improvement direction: implement Rust batched leaf assignment first; later fold leaf output computation into native predictor.
- Expected effect: large predict-time and eval-loop speedup, especially multiclass.
- Implementation difficulty: medium.
- API break risk: none.
- Test direction: leaf IDs and predictions identical for numeric, categorical subset, missing default-left, saved/load models.
- Benchmark direction: predict throughput vs rows, trees, classes, leaf model — measured by `benchmarks/predict_profile.py` (routing vs leaf-eval decomposition).

### 4. Multi-output Backend Scan

- Improvement target file/function: `core/splitter.py::build_histograms`, `backends/numpy_backend.py::find_best_split_multioutput`, CUDA/Rust backend interfaces.
- Current problem: multi-output copies device histograms to host and scans on CPU.
- Why GPU is not fully used: `_as_host` is required before `np.stack`.
- Improvement direction: optional backend method for multi-output hist/scan; CUDA returns device stacked hist and reduces gain across outputs on device.
- Expected effect: major speedup for vector targets and many outputs.
- Implementation difficulty: high.
- API break risk: low to medium depending on backend contract extension.
- Test direction: multi-output objective parity across squared/huber/quantile; CUDA allclose.
- Benchmark direction: `n_outputs` sweep with constant and embedded leaves.

### 5. Multiclass Batched Histogram

- Improvement target file/function: `core/multiclass.py::fit`, backend histogram API.
- Current problem: each round grows `K` trees and builds separate histograms for each class.
- Why GPU is not fully used: repeated kernel launches and repeated row/gradient/Hessian transfers per class.
- Improvement direction: backend accepts `(n_rows, K)` grad/hess and emits `(F, B, 3, K)` histograms, then scans per class.
- Expected effect: improved multiclass scaling.
- Implementation difficulty: high.
- API break risk: medium if not optional.
- Test direction: multiclass parity, label smoothing, class/sample weights.
- Benchmark direction: `K=3,5,10,20` with medium and large datasets.

### 6. Embedded Leaf Fitting Improvements

- Improvement target file/function: `core/leaf_models.py::EmbeddedLinearLeafModel.fit_leaves`, `core/multioutput.py::fit_vector_leaves`, `native::leaf_linear_stats`.
- Current problem: wide embeddings use per-leaf CPU BLAS loops; vector leaves have no Rust stats path; solves remain CPU.
- Why GPU is not fully used: `Z`, grad/hess, and leaf rows are host-resident.
- Improvement direction: first add Rust stats for vector leaves and improve CPU batching; revisit GPU batched ridge only when profiling shows leaf fitting dominates.
- Expected effect: better embedded-leaf training, especially multi-output and medium-width embeddings.
- Implementation difficulty: medium for Rust stats, high for GPU ridge.
- API break risk: none to low.
- Test direction: leaf solve parity, fallback behavior, extrapolation guard.
- Benchmark direction: `emb_dim=16,64,256`, `num_leaves=31,127`, projection on/off.

### 7. Encoder Transform And Projection Memory

- Improvement target file/function: `encoders/plr.py`, `periodic.py`, `cross.py`, `projection.py`, `data/dataset.py::get_embeddings`.
- Current problem: transforms materialize dense `float64` full matrices; random projection materializes base `Z` before projection.
- Why GPU is not fully used: torch CUDA pretraining does not keep transform or `Z` on device.
- Improvement direction: add batch transform protocol and optional `float32` embedding cache; fuse projection with base transform where practical.
- Expected effect: lower memory, less bandwidth, fewer OOMs on large data.
- Implementation difficulty: medium.
- API break risk: low if default unchanged.
- Test direction: serialization, prediction parity within tolerance, batch equivalence.
- Benchmark direction: peak memory and fit time with PLR/high-dimensional encoders.

### 8. Sparse/Dense Strategy

- Improvement target file/function: dataset and sklearn validation.
- Current problem: sparse input is rejected and all internal arrays are dense.
- Why GPU is not fully used: sparse GPU histograms require a separate data layout and algorithm.
- Improvement direction: do not bolt sparse onto dense path. Design a separate CSR/CSC dataset and backend contract when needed.
- Expected effect: future large sparse support without destabilizing dense routing.
- Implementation difficulty: high.
- API break risk: medium.
- Test direction: sparse/dense parity on simple cases.
- Benchmark direction: high-dimensional sparse classification.

### 9. Documentation Consistency

- Improvement target file/function: `docs/backend_strategy.md`, `docs/adr/0005-cuda-backend-cupy.md`, `docs/cuda.md`.
- Current problem: some backend-strategy text still describes CUDA as host split scan only, while implementation and CUDA docs include B2 GPU numeric scan.
- Why GPU is not fully used: not a runtime issue, but stale docs make future work error-prone.
- Improvement direction: update backend strategy to match Phase B2.
- Expected effect: clearer maintenance path.
- Implementation difficulty: low.
- API break risk: none.
- Test direction: docs review.
- Benchmark direction: none.
