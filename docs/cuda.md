# CUDA backend (experimental)

RepLeafGBM can run its split search on an NVIDIA GPU via CuPy, selected with
`split_backend="cuda"`. Per-node histograms are built on the GPU (Phase A), the
binned matrix is uploaded once and kept resident (Phase B1), and the histogram
stays resident across a tree's nodes while the **numeric split scan runs
on-device for large histograms** (Phase B2, adaptive) — only the winning split's
scalars cross back to the host. Small per-node histograms are scanned on the host
instead (cheaper than launching many tiny GPU kernels), so narrow fits don't
regress. Multi-output (shared-routing) trees get the same Phase-B2 treatment: the
K per-output histograms stay resident as a stacked `(F, bins, 3, K)` array and the
summed-gain scan runs on-device, returning only the winning split (the same
adaptive threshold sends narrow nodes to the host). Only the categorical subset
scan (the branchy, parity-critical part) stays on the host. See
`docs/adr/0005-cuda-backend-cupy.md` for the design and
`docs/backend_strategy.md` for where it sits among the compute backends.

## Install

Requires an NVIDIA GPU + CUDA 12 driver. CuPy JIT-compiles the kernel at
runtime, so there is no build step:

```bash
pip install "repleafgbm[cuda]"   # adds cupy-cuda12x
```

## Use

```python
from repleafgbm import RepLeafRegressor

model = RepLeafRegressor(
    n_estimators=200,
    leaf_model="embedded_linear",
    split_backend="cuda",   # explicit-only; "auto" never picks the GPU
)
model.fit(X, y)
```

`split_backend="cuda"` raises a clear `ImportError` when CuPy or a usable GPU
is missing — it never silently falls back, so a typo on a GPU box is visible.
Use `"numpy"` (always available) or `"rust"` (compiled CPU kernels) otherwise.

## Performance

The binned matrix is uploaded once and cached on-device (Phase B1), so per-node
histogram building ships only the row index plus gathered grad/hess. With
Phase B2 the histogram itself stays resident — `build_histograms` returns a
device array, the grower's sibling-subtraction runs on-device, and for large
histograms the numeric gain sweep + argmax run on the GPU — so the per-node
GPU→host transfer shrinks from the full `(n_features, n_bins_max, 3)` histogram
to the winning split's scalars. Measured on a Tesla T4
(`experiments/results/2026-06-17-cuda-parity.md`):

- Histogram micro-benchmark (200k×50, 65 bins): **~52x** over NumPy.
- End-to-end `RepLeafRegressor.fit`, 50 trees, embedded_linear:
  - **wide (50k×200): ~2.1x** — the on-device numeric scan avoids the big
    per-node histogram round-trip.
  - **narrow (100k×30): ~1.5x** — the (30×257) scan is too small to beat a bulk
    copy + vectorized host scan, so the adaptive threshold keeps it on the host
    path (it matches B1 rather than regressing).

The host/GPU crossover (`_GPU_SCAN_MIN_CELLS`, default `2^15 = 32768` feature×bin
cells) can be overridden for **profiling/tuning** via the private
`REPLEAFGBM_CUDA_SCAN_MIN_CELLS` env var (`0` forces every node onto the GPU scan;
a very large value forces the host scan). It is read once per fit and is **not**
part of the public estimator API — the default is unchanged. Sweep it with
`benchmarks/gpu_profile.py --scan-min-cells-sweep` to find a per-GPU optimum; the
effective value is recorded in the benchmark's `transfer_bytes.scan_min_cells`.

With `grow_policy="depthwise"` (scalar targets) the grower scans a whole **level**
of M frontier nodes in one device call (`find_best_split_batched`) instead of one
call per node, amortizing the per-node kernel launch that otherwise dominates the
scan. This is **on by default** for the CUDA backend; set the private
`REPLEAFGBM_CUDA_BATCHED_SCAN=0` to fall back to the per-node loop as a kill switch.
The host grower path is bitwise-identical either way — on NumPy/Rust the batched
call simply loops the per-node scan — so only the device launch count changes.
Measured on a T4 (`experiments/results/2026-06-25-batched-scan-ab.md`): split_scan
**5–9x**, whole depthwise fit **1.9–3.9x**, quality-equivalent; the same
`_scan_min_cells` crossover still routes tiny frontiers to the host loop.

The default `grow_policy="leafwise"` cannot batch a whole level (it expands one
best-gain leaf at a time), but each expansion produces two children whose scans
are independent — they are batched into one device call (M=2), halving the
per-node launch count that the depthwise A/B measured at ~89% of the device
scan (Task B; split_scan was 32.2% of leafwise CUDA fit). **On by default** for
the CUDA backend; `REPLEAFGBM_CUDA_LEAFWISE_BATCH=0` falls back to per-node
scans (and `REPLEAFGBM_CUDA_BATCHED_SCAN=0` disables both). Candidate order and
heap tie-breaking are preserved exactly, so the host path stays
bitwise-identical.

**Device leaf-fit statistics (GPU leaf ridge, roadmap Phase 4.3).** After the
batched scan shipped, profiling showed CUDA fits are **leaf_fit-bound** (65–73%
of depthwise fit — `experiments/results/2026-06-25-cuda-sizing.md`), so the
per-tree leaf-fit statistics now run on the GPU too: the embedding matrix `Z`
is uploaded once per fit (identity-cached, like the binned matrix), each tree
ships its gathered grad/hess + row order, and the per-leaf weighted Gram
stacks, gradient projections, and z-range guards are computed on-device
(`CudaSplitBackend.leaf_fit_stats`), returning the same statistics tuple the
native Rust kernel produces. Centering, the ridge solve, and the adaptive LOO
gate stay on the host in float64 — byte-identical assembly code across the
native/BLAS/device paths. **On by default** for `split_backend="cuda"` scalar
fits with an adaptive work crossover (`REPLEAFGBM_CUDA_LEAF_FIT_MIN_CELLS`,
default 1e6 gathered-row×dim cells; small trees stay on the host);
`REPLEAFGBM_CUDA_LEAF_FIT=0` is the kill switch. `leaf_fit_precision=
"float32_gram"` narrows the two large device reductions to float32, mirroring
the host contract. Multiclass (pooled) and multi-output vector leaves still
fit on the host (follow-up).

The end-to-end gain is bounded because tree growth and the categorical subset
scan still run on the host. The CUDA backend helps most when the histogram or
the leaf fit dominates — many rows, **wide feature matrices / wide embeddings /
high `max_bins`**, deep/large trees.

## Parity and determinism

Unlike the NumPy⇄Rust pair (bitwise-identical histograms), the CUDA backend is
**allclose, not bitwise**: GPU `atomicAdd` summation order is not fixed, so
histogram sums differ from NumPy in the low bits and are not reproducible
run-to-run. When the split scan runs on the **host** (the adaptive default for
narrow per-node histograms) the chosen splits — and thus the trees — are
identical to the reference, only leaf values carry that float noise, and
predictions agree to `rtol=1e-6`. When the numeric scan runs **on-device** (wide
histograms, or a forced low threshold) the gains are reduced with CuPy whose low
bits also differ, so on a *near-tied* node the argmax can select a different —
but equally good — split: the trees then differ structurally and predictions
agree to float noise except on the few rows a flipped split reroutes. Those flips
are quality-neutral (the gains were tied), so model quality matches even when the
exact tree does not (the parity tests assert `rtol=1e-6` on the host-scan path and
quality-equivalence on the device-scan path). The device leaf-fit statistics carry
the same caveat one layer down: a *near-tied* adaptive LOO-gate verdict can flip
under device reduction noise, so its e2e tests also assert quality-equivalence
(|Δr²| bound), not prediction equality. If you need bitwise determinism,
use `"numpy"` or `"rust"`.

## GPU encoder pretraining (separate from `split_backend`)

The CUDA split backend above accelerates the booster's histogram. A second,
independent GPU surface is the **learned-encoder pretraining** (the optional
torch encoders: `torch_periodic`, `torch_plr`, `torch_periodic_plr`,
`torch_mlp`). Each takes a `device` knob (via `encoder_params`):

```python
RepLeafRegressor(
    encoder="torch_periodic_plr",
    encoder_params={"device": "auto"},   # "cpu" (default) | "cuda" | "auto"
)
```

`device` only affects the one-time pretraining `fit`; `transform`,
`get_state`/`set_state`, and serialization stay NumPy, so a saved model still
predicts without torch. All random draws (head init, batch permutations) use a
CPU generator and are moved onto the device afterwards, so the random stream is
device-independent and `device="cpu"` reproduces the prior pretraining
byte-for-byte. Like the split backend, GPU pretraining is **allclose, not
bitwise** (GPU reductions reorder), so CPU stays the deterministic default and
GPU is validated only on the Colab loop below. The payoff is scale-dependent:
small default nets may not beat CPU (host↔device overhead), while large `n`,
wide periodic embeddings, or a deeper `torch_mlp` benefit.

## GPU dev loop (no local GPU required)

Because macOS dev boxes and CI have no GPU, the CUDA path is built and tested
on a Google Colab GPU VM via the [Colab CLI](https://github.com/googlecolab/google-colab-cli):

```bash
uv tool install google-colab-cli      # one-time
bash scripts/colab_gpu_test.sh --gpu T4
```

This provisions a GPU VM, uploads the current working tree (including
uncommitted changes), runs `tests/test_cuda_backend.py` on the GPU plus a
histogram micro-benchmark, downloads a dated report to
`experiments/results/<date>-cuda-parity.md`, and tears the VM down. Flags:
`--gpu {T4,L4,A100}` (default `T4`), `--session NAME`, `--keep` (leave the VM
up to iterate). The driver that runs on the VM is `scripts/colab_remote_test.py`.
