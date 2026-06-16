# CUDA backend (experimental)

RepLeafGBM can build per-node histograms on an NVIDIA GPU via CuPy, selected
with `split_backend="cuda"`. This is the Phase A GPU path: histogram
construction runs on the GPU; the split scan reuses the NumPy reference on the
host. See `docs/adr/0005-cuda-backend-cupy.md` for the design and
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
histogram building ships only the row index plus gathered grad/hess. Measured on
a Tesla T4 (`experiments/results/2026-06-17-cuda-parity.md`):

- Histogram micro-benchmark (200k×50, 65 bins): **~32x** over NumPy.
- End-to-end `RepLeafRegressor.fit` (100k×30, 50 trees, embedded_linear):
  **~1.6x**.

The end-to-end gain is smaller because tree growth, the split scan, and leaf
fitting all run on the host. GPU leaf fitting was evaluated and deferred (leaf
stats are already accelerated by the Rust `leaf_linear_stats` kernel; ADR 0005).
The CUDA backend helps most when the histogram dominates — many rows, wide
feature matrices, deep/large trees.

## Parity and determinism

Unlike the NumPy⇄Rust pair (bitwise-identical histograms), the CUDA backend is
**allclose, not bitwise**: GPU `atomicAdd` summation order is not fixed, so
histogram sums differ from NumPy in the low bits and are not reproducible
run-to-run. Cross-backend predictions still agree to float noise
(`rtol=1e-6`), which is what the parity tests assert. If you need bitwise
determinism, use `"numpy"` or `"rust"`.

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
