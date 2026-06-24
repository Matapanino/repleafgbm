# GPU/CPU GBDT Performance Techniques — Transferability Survey

**Date:** 2026-06-25
**Author:** cuda-researcher
**Scope:** External GPU and CPU GBDT techniques (XGBoost, LightGBM, CatBoost, cuML/RAPIDS,
sklearn HistGBM, literature) surveyed for transferability into RepLeafGBM. Only in-thesis,
code-map-mapped hypotheses are listed. Each maps to the measured phase breakdown:

| Config | Dominant phases |
|---|---|
| WIDE emb (50k×200f, emb=200) | leaf_fit 69%, histogram 9%, eval 7%, preprocessing 5% |
| NARROW emb (50k×30f, emb=30) | preprocessing 27%, leaf_fit 22%, histogram 21%, eval 10% |

---

## H1 — Histogram pool: reuse parent histogram as a child (avoid one allocation + zero per split)

**What it is.** In leaf-wise growth the parent histogram is subtracted to get the sibling
child. LightGBM and sklearn HGBT both keep the parent in memory and reuse its slot as the
sibling's output, avoiding one `np.zeros` allocation per split. sklearn PR #27865 measured
~10% runtime improvement on histogram construction (3.9s → 3.3s over a 100-tree fit) with
no logic change — only memory management.

**Evidence strength.** Strong (merged sklearn, matches LightGBM design, 2023–2024).

**Code-map target.** `core/splitter.py` (histogram pool) + `core/tree.py` (TreeGrower
sibling-subtraction logic). The CUDA backend already keeps the histogram resident on device;
the CPU/Rust path still allocates fresh zero arrays every node.

**Expected signal.** ~5–10% reduction in histogram phase wall-clock on the Rust backend
at wide shapes (200f, 200 trees, many splits). Smaller at narrow (histogram is 21% of fit
at 30f but absolute time is low).

**Cheap local test.** Profile `benchmarks/gpu_profile.py --backend rust --size medium`
before/after pooling `np.zeros` into a reused buffer. Compare `phase_seconds.histogram`.

**Risk / parity.** Pure memory management — no arithmetic change. NumPy↔Rust parity
stays bitwise as long as the zeroing boundary is respected. The CUDA path is unaffected
(already device-resident). Risk: cyclic GC references (the sklearn PR notes this;
must be handled explicitly).

**Needs GPU?** No.

---

## H2 — uint8 bin storage: downgrade `binned` matrix from uint16 to uint8 when max_bins ≤ 255

**What it is.** sklearn HistGBM uses `uint8` for all binned feature values (max_bins
fixed at ≤255). RepLeafGBM uses `uint16` (`histogram.py`, `bin_features`), leaving
half the bits unused at the default max_bins=256. Halving the binned matrix size reduces
memory bandwidth in the histogram kernel (both CPU and GPU), improves CUDA H2D transfer
cost, and fits more of the resident matrix into GPU SMEM/L2.

**Evidence strength.** Strong (sklearn production code, LightGBM uses uint8 too, XGBoost
ELLPACK compresses to minimum bit-width). Documented: `_binning.pyx` uses `X_BINNED_DTYPE
= np.uint8` with "hence max_bins == 256".

**Code-map target.** `core/histogram.py::bin_features` (dtype=uint16 → uint8) +
`backends/cuda_backend.py` (kernel signature already declares `unsigned short*` = uint16;
must change to `unsigned char*` if uint8). Also the CUDA Phase B1 cache: binned H2D bytes
halved.

**Expected signal.** Histogram phase: ~5–15% on wide (200f, bandwidth-bound). CUDA
binned_h2d_bytes halved. Narrow: likely <5% (histogram 21% but DRAM bandwidth less
saturated on small matrix).

**Cheap local test.** Change dtype in `bin_features` to uint8 (clamping max_bins to 255),
run `pytest tests/ -q` for bitwise parity (uint16→uint8 is exact same bin values when
max_bins ≤ 255), then profile. Keep uint16 as fallback for max_bins > 255.

**Risk / parity.** Breaking change to the on-disk format and the CUDA kernel signature
(both need updating together). NumPy↔Rust parity must still hold. CUDA kernel needs a
matching `unsigned char*` parameter. A max_bins=256 user sends bin index 255 (valid for
uint8) plus missing bin 257 (overflows uint8 → must cap missing at 255 and reserve one
bin). Requires careful invariant: missing_bin = n_bins (currently up to 257) must fit in
uint8.

**Needs GPU?** No for CPU test; full CUDA kernel change needs Colab.

---

## H3 — Quantized gradient histogram: int16 accumulation in the histogram kernel

**What it is.** "Quantized Training of Gradient Boosting Decision Trees" (Microsoft Research
Asia, arXiv 2207.09682, NeurIPS 2022) shows that as few as 2–3 bits suffice for gradients
without accuracy loss, and int8/int16 histogram accumulation gives up to 2× speedup on CPU
and GPU vs float64. LightGBM GPU training quantizes gradients to int16 before histogram
accumulation. For RepLeafGBM: quantize `grad`/`hess` to int16 per-node (scale by max
absolute value), accumulate integer histograms, dequantize before split-gain computation.

**Evidence strength.** Good-to-strong (peer-reviewed NeurIPS paper, production LightGBM
GPU feature). Int16 accumulation + float64 gain computation is the production pattern.

**Code-map target.** `backends/cuda_backend.py` — replace `double` histogram atoms with
`short`/`int` atoms, rescale before gain computation. CPU path: `backends/numpy_backend.py`
+ `native/src/lib.rs` would need matching changes (integer bincount).

**Expected signal.** CUDA histogram phase: potentially 2× (memory bandwidth + SMEM occupancy
— int16 is ¼ the bytes of float64 per cell). CPU path: smaller benefit but integer SIMD
may help at wide (200f, bandwidth-bound histogram).

**Cheap local test.** Implement quantized int16 histogram in numpy_backend only (no Rust/CUDA
change), compare resulting split candidates against float64 on the existing parity test suite.
If quality is maintained (allclose on leaf values, not bitwise), the technique is sound.

**Risk / parity.** CUDA path is allclose (not bitwise) anyway, so quantization fits that
contract. CPU NumPy↔Rust parity would need both paths changed together — considerable scope.
The key accuracy risk is per-node scale mismatch: if two features have very different gradient
ranges in the same node, per-feature scaling is needed. Start with per-node global scale.
Also: Newton target uses `-g/h`; both must be quantized consistently.

**Needs GPU?** NumPy-only feasibility test is local (CPU). CUDA kernel change needs Colab.

---

## H4 — k-means binning: drop-in replacement for quantile threshold computation

**What it is.** arXiv 2505.12460 "A Case for Library-Level k-Means Binning in Histogram
Gradient-Boosted Trees" (May 2025, Labovich) shows k-means bin edges (initialized from
quantile edges, then Lloyd-iteration refined) are a pure drop-in for quantile threshold
computation. Key wins: 55% MSE reduction on highly skewed data (Brazilian Houses, skew=30),
50–90% error reduction when outliers are label-relevant, consistently large MSE gains at
low bin budgets (≤63 bins). CPU overhead: ~3.5s/feature for 10M rows (75% slower than
quantile) but one-time and cacheable.

**Evidence strength.** Good (preprint, May 2025, 18 regression + 15 classification datasets,
no regression on classification tasks).

**Code-map target.** `core/histogram.py::compute_bin_thresholds` — add `bin_method` param,
loop replaces `np.quantile` with k-means Lloyd iterations on non-NaN values. Training
cost only (thresholds are fitted once). Binning thread-pool already maps per-feature (PR
#25), so k-means per-feature overhead is automatically parallel.

**Expected signal.** Not a speed win — a quality win on skewed/low-budget datasets. No
phase time change (binning is ≤4% of fit at both wide and narrow). The experiment budget
is the real question: does a real dataset in our benchmark suite show skew >= 10?

**Cheap local test.** Implement `_thresholds_kmeans` alongside `_thresholds_quantile`, add
a `bin_method` flag (default unchanged), run on a synthetic skewed dataset (log-normal
features) and verify lower MSE.

**Risk / parity.** Not a parity-breaking change if guarded behind a flag. Thresholds
change → different bins → different trees; this is expected. No NumPy↔Rust issue (bin
assignment uses the same `searchsorted` either way after thresholds are computed). Risk:
k-means may not converge in a fixed number of iterations on degenerate features.

**Needs GPU?** No.

---

## H5 — Constant leaf vectorization: replace per-leaf loop with `np.add.reduceat`

**What it is.** `ConstantLeafModel.fit_leaves` iterates a Python list comprehension over
leaf_rows. The Newton step `−sum(g)/sum(h)` is a weighted segmented sum. `np.add.reduceat`
(or `np.bincount` with sorted segments) vectorizes this to a single C call, eliminating
the Python loop entirely. LightGBM and sklearn both vectorize constant leaf aggregation.
This is already on the gpu_roadmap.md Phase 1.2 list.

**Evidence strength.** Strong algorithmic argument; already queued in the roadmap.

**Code-map target.** `core/leaf_models.py::ConstantLeafModel.fit_leaves` — replace the
list comprehension over `leaf_rows` with `np.bincount(seg, weights=g_seg) / np.bincount(seg,
weights=h_seg)` (mirroring the existing NumPy path in `EmbeddedLinearLeafModel`).

**Expected signal.** Modest but universal: any constant-leaf run + the multiclass fallback
path. At narrow emb=30 where leaf_fit is 22% of fit, this helps if constant leaves dominate;
less impact for embedded_linear.

**Cheap local test.** Profile before/after with `benchmarks/gpu_profile.py --leaf-model
constant --size medium`.

**Risk / parity.** Arithmetic change (order of summation) → result may not be bitwise
identical to the loop if leaf_rows are not sorted by index. Use `np.argsort` on offsets to
guarantee same order. Test: `test_leaf_models.py` must stay green.

**Needs GPU?** No.

---

## H6 — Fused forest predictor: route + leaf-eval + accumulate in one native pass

**What it is.** RAPIDS FIL and cuML use reorganized tree node storage (TREE_REORG) so
multiple trees can be traversed with coalesced memory access: tree roots for all trees
stored first, then level-1 nodes, etc. This reduces cache misses during per-sample
traversal. The CPU equivalent is packing all trees into a flat SoA node array, routing
all trees for a batch of rows simultaneously, and accumulating leaf outputs without a
Python-level per-tree loop.

**Evidence strength.** Good (RAPIDS FIL production design; Medium post by Zedlewski 2019).
28× throughput over CPU in the FIL paper (GPU vs single-threaded CPU), but that includes
GPU parallelism. The CPU memory-locality argument alone is: better cache use when routing
many samples across a shallow ensemble.

**Code-map target.** `core/prediction.py::predict_raw` — the current Python for-loop over
trees is the target. `native/src/lib.rs` could implement a batched `apply_forest` that
takes the full tree list + X_raw and returns leaf indices for all trees and all rows in one
pass, using a flat node representation. Note: the rejected PR list says "forest-batched
routing (standalone) — REJECT 2026-06-24 (iter 001)" because it only removes
pyo3/marshalling overhead (4–7% overhead). The FIL pattern is stronger — it also changes
the memory access pattern for nodes — but requires a flat serialized tree format (a larger
change than standalone apply_forest).

**Expected signal.** Predict routing is 60–100% of predict time (prediction-traversal bench
verdict). FIL-style node reorg could yield 2–4× on CPU prediction for large ensembles
(≥100 trees), primarily through L2 cache locality, NOT through eliminating the pyo3 call.
The threshold is a genuine architectural change.

**Cheap local test.** Benchmark `benchmarks/predict_profile.py` with 200 trees, 50k×200f,
embedded_linear. Measure `apply_tree` contribution. If routing accounts for ≥30% of predict
and L2 misses are high (use `perf stat`), the locality argument holds.

**Risk / parity.** Requires a flat serialized tree format (breaking the current `Tree`
object list). Needs `core-reviewer` sign-off on the interface before implementation.
Leaf-evaluation must still use the per-leaf `Z[leaf_rows]` gather, which can be fused.
This is the "full forest-fused predictor (route+leaf-eval+accumulate)" held (not rejected)
in `docs/perf-notes/rejected-ideas.md`.

**Needs GPU?** No for the CPU SoA/packing change; GPU version deferred.

---

## H7 — encode_features copy avoidance: DataFrame → float64 without per-column Python loop

**What it is.** `data/preprocessing.py::encode_features` iterates feature-by-feature in
Python, calling `_get_column` (which calls `X[col].to_numpy()`) per column and writing
into a pre-allocated `out` float64 matrix. For a pure-numerical DataFrame this is slower
than `X.to_numpy(dtype=np.float64)` (a single C-level cast). The Python loop is also the
reason preprocessing is 27% of fit on narrow (30f) cases — it's proportionally large
because the numerical features are trivial but the column-wise dispatch has fixed overhead.

**Evidence strength.** Moderate — directly visible in the code; no published benchmark.
The 27% share at narrow (30f, emb=30) is measured in the project's own profiler.

**Code-map target.** `data/preprocessing.py::encode_features` — fast path for all-numerical
ndarray/DataFrame input: detect no-categorical, no-frequency-encoded case and use a single
`np.asarray(X, dtype=np.float64)` or `X.to_numpy(dtype=np.float64)` rather than the
column loop.

**Expected signal.** Preprocessing from 27% → substantially lower at narrow (the loop is
bounded by Python dispatch, not data size). At wide (200f) preprocessing is 5% of fit —
less impactful but still saves allocations.

**Cheap local test.** `timeit` comparison: current `encode_features` vs `np.asarray(X,
dtype=np.float64)` on a 50k×30 float64 DataFrame.

**Risk / parity.** Parity: for float64-only columns the result is identical. The fast path
must gate strictly on: all columns numerical, no categoricals, no freq-encoded, no
non-float dtypes. Categorical and mixed-type paths must fall through to the existing loop.
Robust dtype check needed (int columns in DataFrame must still cast to float64 with NaN
handling).

**Needs GPU?** No.

---

## H8 — Batched `np.linalg.solve`: confirm numpy uses batched LAPACK for the leaf Gram solve

**What it is.** `EmbeddedLinearLeafModel.fit_leaves` already calls `np.linalg.solve(A,
rhs[:, :, None])` on the full `(k, emb_dim, emb_dim)` stack. NumPy 1.x/2.x both broadcast
`linalg.solve` over the leading dimension using LAPACK `dgesv` in a loop. The ridge
matrices are symmetric positive-definite (SPD), but we use `dgesv` (LU) not `dpotrs`
(Cholesky). Cholesky is roughly 2× faster for SPD systems (half the FLOPs for the same
matrix size). For emb_dim in [32, 64] and k leaves per tree, this is entirely in the
batched numpy call that dominates leaf_fit.

**Evidence strength.** Moderate — standard numerical methods result (SPD → Cholesky is
2× vs LU in FLOPs). No GBDT-specific paper; applies universally.

**Code-map target.** `core/leaf_models.py::EmbeddedLinearLeafModel._solve_and_assemble` —
replace `np.linalg.solve(A, rhs[:, :, None])` with `np.linalg.cholesky(A)` + triangular
solve `scipy.linalg.cho_solve_banded` or batched `scipy.linalg.solve_triangular`. Or use
`scipy.linalg.solve(A, rhs, assume_a='pos')` which selects Cholesky automatically.

**Expected signal.** leaf_fit is 69% (wide) and 22% (narrow); the solve is a fraction of
that (Gram construction dominates at wide emb). Expected 5–15% on leaf_fit for emb ~30–64,
where the solve is a larger fraction vs the O(n·d²) Gram.

**Cheap local test.** Micro-benchmark: `scipy.linalg.solve(A_batch, rhs, assume_a='pos')`
vs `np.linalg.solve(A_batch, rhs)` for k=20 leaves, emb_dim=32/64. Time with `%timeit`.

**Risk / parity.** The Cholesky path requires `A` to be positive-definite; the ridge term
`l2·I` guarantees this for l2>0. For l2=0 or near-singular leaves, the existing
`LinAlgError` catch must apply to Cholesky too. NumPy↔Rust parity: this is the Python
fallback path (emb_dim > 64 or native unavailable), not the Rust path. The Rust path
(`leaf_linear_stats`) computes the Gram but does NOT solve — the solve is always in Python.
Parity impact: the result may differ from the current LU in low bits (Cholesky vs LU
pivoting), so this is allclose-not-bitwise for the Python wide path. The Rust path is
unchanged.

**Needs GPU?** No.

---

## H9 — Rayon + BLAS oversubscription gate: set BLAS thread count = 1 during rayon leaf fits

**What it is.** When rayon parallelizes `leaf_linear_stats` across leaves (the current
shipped path for emb ≤ 64), each leaf's Gram matrix calls into BLAS (via numpy) for the
matrix multiply. BLAS libraries (OpenBLAS/MKL) themselves spawn threads; those threads
compete with rayon's worker pool. The result is thread oversubscription. The known fix
(OpenBLAS PR #1192, NumPy parallelism design by Thomas J. Fan) is to set `OMP_NUM_THREADS=1`
(or the equivalent BLAS threadcount API) before spawning rayon workers. Our test harness
already sets `OMP_NUM_THREADS=1` to avoid a torch+lightgbm libomp deadlock — but the
production path does not enforce this.

**Evidence strength.** Strong (known OpenBLAS issue, production pattern). At
`OMP_NUM_THREADS=1` the rayon leaf-parallel path already shows 1.7–2.2× speedup; at
default BLAS threads the gain may be masked or even negative.

**Code-map target.** `native/src/lib.rs` — the `leaf_linear_stats` function launches rayon;
before the parallel section, set BLAS threads to 1 via `openblas_set_num_threads(1)` (via
FFI) or expose a Python-side toggle. Alternatively, make the Rust path call into a
`no-blas` matmul for small emb_dim (the fused pass already does this for emb≤64; the solve
path calls Python/BLAS later).

**Expected signal.** On a multi-core machine without `OMP_NUM_THREADS=1` forced, this may
un-mask the rayon leaf-parallel gain. On CI / the dev machine with `OMP_NUM_THREADS=1` the
effect is neutral (already serialized). The risk is that production users with default
environment get oversubscription.

**Cheap local test.** Run `benchmarks/gpu_profile.py --backend rust --size medium` with
`OMP_NUM_THREADS=4` (default) vs `OMP_NUM_THREADS=1`, compare leaf_fit phase_seconds.

**Risk / parity.** No correctness impact. Must not change global thread state permanently
(thread-local or scoped BLAS count change preferred). This is a runtime configuration
concern, not a kernel change.

**Needs GPU?** No.

---

## H10 — Incremental eval score update: avoid `tree.apply(Xe)` per eval set per round

**What it is.** In `booster.py::_run_boosting`, per-round eval cost is:
`Fe += lr * leaf_values.predict(tree.apply(Xe), Ze)`. The `tree.apply(Xe)` call re-routes
the full eval set through the just-grown tree. This is unavoidable for the leaf idx.
However, the eval is measured at 10% (narrow) and 7% (wide) of fit time. The `apply_tree`
native router is already used. The remaining optimization is batching the `predict` call
across eval sets (if multiple eval sets are used), or overlapping eval with the next tree's
gradient computation.

**Evidence strength.** Moderate. The incremental update avoids a full ensemble re-predict
(already implemented). The bottleneck is routing, not accumulation.

**Code-map target.** `core/booster.py::_run_boosting` eval loop (lines ~262–266) +
`core/prediction.py`. A concrete micro-improvement: use `np.add.at` or pre-allocate the
leaf-predict result buffer to avoid a temporary allocation per round.

**Expected signal.** Small. If multiple eval sets are used, batching the `predict` call
over sets could save repeated `Z[leaf_idx]` gathers. With one eval set, the bottleneck is
`tree.apply(Xe)` which is already native. Realistically <3% on total fit.

**Cheap local test.** Time eval phase with 1 vs 3 eval sets of the same size.

**Risk / parity.** Allocation-only change; parity trivially maintained.

**Needs GPU?** No.

---

## H11 — CatBoost oblivious-tree leaf-index-as-bitmask for symmetric grow_policy

**What it is.** CatBoost's symmetric/oblivious trees compute the leaf index at prediction
time via a bitmask: each tree level contributes one bit (1 = right, 0 = left), so the leaf
index = bitwise-concatenation of level conditions. This replaces a branchy per-level
conditional with a bitwise OR and an integer table lookup. On CPUs with SIMD, all rows'
conditions at one level can be evaluated in a single vectorized pass. RepLeafGBM's
symmetric grow_policy (ADR 0006) could use this at prediction time.

**Evidence strength.** Strong (CatBoost production design, documented in their inference
speedup blog post). CatBoost claims "dozens of times faster" inference vs asymmetric trees,
though that conflates the symmetric structure benefit with other factors.

**Code-map target.** `core/tree.py::Tree.apply` symmetric path only (guarded by
`grow_policy == "symmetric"`). `native/src/lib.rs::apply_tree` would need a symmetric
fast path that uses bitwise leaf-index accumulation instead of per-node branching.

**Expected signal.** Predict speedup for symmetric trees only. Not applicable to leafwise
(the default). Worth quantifying: current `apply_tree` for symmetric is O(depth × n_rows);
bitmask approach is the same O() but with lower constant (no branching, one vectorized pass
per level).

**Cheap local test.** Run `benchmarks/predict_profile.py` with `grow_policy="symmetric"`,
compare routing fraction vs `grow_policy="leafwise"`.

**Risk / parity.** Symmetric-only path, not the default. The bitmask leaf index must agree
with the existing `Tree.apply` output — full parity test required. No invariant violation
(raw features only, NaN left).

**Needs GPU?** No.

---

## H12 — CUDA node-batched split scan: amortize kernel launch across frontier nodes (designed, queued)

**What it is.** The per-node on-device scan is launch-bound (proven 2026-06-23 sweep:
threshold 0 is 3–5× slower on narrow). The roadmap's next lever is batching M frontier
nodes in one kernel call: stack M node histograms `(M, F, B, 3)`, run one kernel with grid
`M × F`, take M independent argmaxes. The design is validated in
`docs/perf-notes/research-node-batched-split-scan.md` (2026-06-24); the math is locally
proven bitwise for the NumPy reference. Target phase: split_scan is 48–85% of CUDA fit.

**Evidence strength.** Strong design (internally validated); XGBoost GPU histogram PR
#10538 also pursues batched node enumeration ("preparation for batched nodes enumeration").

**Code-map target.** `backends/cuda_backend.py::find_best_split_batched` (new method) +
`core/tree.py` grower refactor (batch the depthwise level's `_make_candidate` calls).

**Expected signal.** Primary target is multiclass-K5 (split_scan 85% of CUDA fit). Expected
2–4× on split_scan for M=32 node batches, translating to ~30–50% end-to-end CUDA speedup
on wide multiclass. See design note for full validation plan.

**Cheap local test.** None that exercises the kernel; the design is locally validated at
the NumPy level. Requires Colab T4 for the CUDA kernel speedup measurement.

**Risk / parity.** Requires grower refactor (architectural, needs core-reviewer sign-off).
Near-tied splits can flip via low-bit CuPy reductions (allclose-modulo-flips, not rtol=1e-6
— already documented as the CUDA parity contract). Not locally testable without a GPU.

**Needs GPU?** Yes (Colab T4 required for kernel iteration). This is the primary GPU-gated
hypothesis.

---

## H13 — D2H histogram copy elimination for wide nodes: keep `(F, B, 3)` entirely on device

**What it is.** In the current CudaSplitBackend the "large histogram" path (above
`_GPU_SCAN_MIN_CELLS = 32768 cells`) already stays resident on device for the numeric
scan — only 4 float64 scalars cross D2H per node (winner_d2h_bytes = 32). But for small
nodes (`n_small_scans`), the full `(F × B × 3 × 8)` bytes of the histogram cross D2H for
every node, even if they share the same feature set. For multiclass where many small nodes
exist, this becomes significant. The mitigation is a lower adaptive threshold for wide
shapes, or pre-filtering zero-gain features before the scan to reduce D2H volume.

**Evidence strength.** Moderate (derived from measured `n_small_scans` / `n_gpu_scans`
ratio in transfer_stats). No external citation; internal logic from the existing CUDA
benchmark harness.

**Code-map target.** `backends/cuda_backend.py::_resolve_scan_min_cells` — lower the
adaptive threshold for wide F (e.g., scale `_GPU_SCAN_MIN_CELLS` by `F/200`). Or: before
D2H copy in the small-scan path, filter features whose histogram is all-zero on device
(a CuPy `any` per feature row).

**Expected signal.** Reduction in `hist_d2h_bytes` for multiclass-wide; may push some
borderline nodes onto the GPU path. Effect on end-to-end time depends on D2H bandwidth vs
kernel cost at that node size.

**Cheap local test.** Sweep `REPLEAFGBM_CUDA_SCAN_MIN_CELLS` with
`benchmarks/gpu_profile.py --scan-min-cells-sweep` on the Colab T4 for multiclass-K5
200f to find where the D2H cost exceeds the kernel-launch amortization at narrower nodes.
This requires Colab.

**Risk / parity.** No correctness impact (D2H copy followed by host scan is identical to
current). The adaptive threshold is already tunable without an API change.

**Needs GPU?** Yes (Colab T4 for measurement).

---

## Guardrail Check

All 13 hypotheses pass the thesis invariants:

- No hypothesis touches embedding-based splits (splits remain on raw features).
- No hypothesis unfreezes the encoder during boosting.
- No hypothesis imports CuPy/torch/LightGBM into the native (Rust) path.
- No hypothesis changes `"auto"` to select CUDA.
- CUDA hypotheses (H12, H13) are quality-equivalent (allclose), not rtol=1e-6.
- H3 (quantized gradients) changes histogram values but not split logic; parity is
  allclose on the CUDA path and must be validated before shipping on the Rust path.

---

## Priority by Measured Headroom (CPU-local, no GPU)

| Rank | Hypothesis | Phase target | Expected gain | GPU? |
|---|---|---|---|---|
| 1 | H7 preprocessing copy avoidance | preprocessing (27% narrow) | ~15–25% narrow fit | No |
| 2 | H1 histogram pool | histogram | ~5–10% histogram | No |
| 3 | H5 constant leaf vectorization | leaf_fit (constant path) | modest, broad | No |
| 4 | H2 uint8 bin storage | histogram + H2D transfer | 5–15% histogram | No (CPU); Colab for CUDA kernel |
| 5 | H8 Cholesky solve | leaf_fit | 5–15% leaf_fit for emb≤64 | No |
| 6 | H4 k-means binning | quality only (not speed) | MSE reduction on skewed data | No |
| 7 | H6 fused forest predictor | predict | 2–4× predict (large design) | No (CPU) |
| 8 | H9 BLAS oversubscription gate | leaf_fit (multi-core) | environment-dependent | No |
| 9 | H11 symmetric bitmask predict | predict (symmetric only) | predict speedup for symmetric | No |
| 10 | H10 eval incremental | eval | <3% total fit | No |
| 11 | H12 node-batched CUDA scan | split_scan (CUDA) | 2–4× split_scan | Yes (Colab) |
| 12 | H13 D2H adaptive threshold | hist_d2h (CUDA) | D2H bytes, unclear time | Yes (Colab) |
| 13 | H3 quantized gradient histogram | histogram (CUDA) | up to 2× histogram | No (CPU feasibility); Colab for kernel |

---

## Sources

- [XGBoost GPU Support (xgboost 3.3.0)](https://xgboost.readthedocs.io/en/stable/gpu/index.html)
- [Mitchell R, Frank E. Accelerating the XGBoost algorithm using GPU computing. PeerJ CS 2017](https://xgboost.ai/2018/07/04/gpu-xgboost-update.html)
- [Quantized Training of Gradient Boosting Decision Trees — arXiv 2207.09682 (NeurIPS 2022)](https://arxiv.org/abs/2207.09682)
- [A Case for Library-Level k-Means Binning in Histogram Gradient-Boosted Trees — arXiv 2505.12460 (May 2025)](https://arxiv.org/abs/2505.12460)
- [LightGBM GPU Performance Guide](https://lightgbm.readthedocs.io/en/latest/GPU-Performance.html)
- [LightGBM Features Documentation](https://lightgbm.readthedocs.io/en/stable/Features.html)
- [CatBoost fast inference blog (CatBoost AI)](https://catboost.ai/news/best-in-class-inference-and-a-ton-of-speedups)
- [The Secret Behind CatBoost's Blazing-Fast Inference (Ichimura, Medium)](https://medium.com/@chimuichimu/the-secret-behind-catboosts-blazing-fast-inference-c6ee21ebc391)
- [sklearn HistGBM binning source — _binning.pyx](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/ensemble/_hist_gradient_boosting/_binning.pyx)
- [sklearn PR #27865 — reuse parent histograms](https://github.com/scikit-learn/scikit-learn/pull/27865)
- [sklearn PR #18242 — histogram memory reduction](https://github.com/scikit-learn/scikit-learn/pull/18242)
- [RAPIDS FIL: Prediction at 100M rows/second (Medium, Zedlewski 2019)](https://medium.com/rapids-ai/rapids-forest-inference-library-prediction-at-100-million-rows-per-second-19558890bc35)
- [RAPIDS FIL stable docs (cuml 26.06.00)](https://docs.rapids.ai/api/cuml/stable/fil/)
- [An Experimental Evaluation of Large Scale GBDT Systems — arXiv 1907.01882 (VLDB 2019)](https://arxiv.org/pdf/1907.01882)
- [Accelerating Multi-Output GBDTs with GPUs — ICPP 2024](https://dl.acm.org/doi/10.1145/3754598.3754638)
- [XGBoost GPU histogram kernel cache PR #10538](https://github.com/dmlc/xgboost/pull/10538)
- [OpenBLAS nested parallelism issue #1192](https://github.com/OpenMathLib/OpenBLAS/issues/1192)
- [Parallelism in Numerical Python Libraries — Thomas J. Fan](https://thomasjpfan.github.io/parallelism-python-libraries-design/)
