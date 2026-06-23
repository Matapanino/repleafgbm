# CUDA backend (experimental)

RepLeafGBM can run its split search on an NVIDIA GPU via CuPy, selected with
`split_backend="cuda"`. Per-node histograms are built on the GPU (Phase A), the
binned matrix is uploaded once and kept resident (Phase B1), and the histogram
stays resident across a tree's nodes while the **numeric split scan runs
on-device for large histograms** (Phase B2, adaptive) — only the winning split's
scalars cross back to the host. Small per-node histograms are scanned on the host
instead (cheaper than launching many tiny GPU kernels), so narrow fits don't
regress. The categorical subset scan (the branchy, parity-critical part) and the
multi-output scan stay on the host. See `docs/adr/0005-cuda-backend-cupy.md` for
the design and `docs/backend_strategy.md` for where it sits among the compute
backends.

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

The end-to-end gain is bounded because tree growth, categorical/multi-output
scans, and leaf fitting still run on the host. GPU leaf fitting was evaluated and
deferred (leaf stats are already accelerated by the Rust `leaf_linear_stats`
kernel; ADR 0005). The CUDA backend helps most when the histogram dominates —
many rows, **wide feature matrices / high `max_bins`**, deep/large trees.

## Parity and determinism

Unlike the NumPy⇄Rust pair (bitwise-identical histograms), the CUDA backend is
**allclose, not bitwise**: GPU `atomicAdd` summation order is not fixed, so
histogram sums differ from NumPy in the low bits and are not reproducible
run-to-run. Cross-backend predictions still agree to float noise
(`rtol=1e-6`), which is what the parity tests assert. If you need bitwise
determinism, use `"numpy"` or `"rust"`.

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
