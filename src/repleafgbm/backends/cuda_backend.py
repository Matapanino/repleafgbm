"""CUDA implementation of the histogram kernel (optional, CuPy-based).

This is the experimental ``split_backend="cuda"`` path. It accelerates the one
hot kernel — per-node histogram construction — on an NVIDIA GPU via a
:class:`cupy.RawKernel`, and **delegates** the (small, branchy) split scan to
the NumPy reference so the categorical-subset / stable-sort / tie-break /
missing-left logic stays byte-for-byte identical to the reference backend.

Resident-data fast path (Phase B1): the binned feature matrix is constant for
the whole run, so it is uploaded to the GPU once and cached (keyed by object
identity); each node then ships only its small ``rows`` index plus the gathered
gradients/Hessians, and the kernel gathers bins on-device. This removes the
per-node host gather of ``binned[rows]`` and its upload — the dominant transfer
cost when the feature matrix is wide. The cache holds the binned matrix on the
device for the backend's lifetime (freed on GC); a typical training run reuses
it across every node of every tree.

Differences from the Rust backend (see ``docs/adr/0005-cuda-backend-cupy.md``):

* Parity is **allclose, not bitwise.** GPU histogram reduction uses
  ``atomicAdd`` on float64, whose summation order is not fixed, so sums differ
  from NumPy's ``bincount`` in the low bits. Cross-backend predictions still
  agree to float noise (``rtol=1e-6``; tested in tests/test_cuda_backend.py).
* It is **not** bitwise reproducible run-to-run for the same reason; models
  agree to float noise rather than being identical (unlike numpy/rust, where
  same seed + same backend gives an identical model).

CuPy is torch-independent, so this module honors the invariant that the
native compute path never imports torch / lightgbm / repleafgbm.external.
Install with ``pip install "repleafgbm[cuda]"`` (needs an NVIDIA GPU + driver).
"""

from __future__ import annotations

import numpy as np

from repleafgbm.backends.base import BaseSplitBackend, SplitCandidate
from repleafgbm.backends.numpy_backend import NumPySplitBackend

# CUDA C source for the histogram kernel. One thread per (selected-row, feature)
# pair accumulates (grad, hess, count) into the shared (n_features, n_bins_max, 3)
# histogram with float64 atomicAdd. atomicAdd(double) is natively supported on
# compute capability >= 6.0 (e.g. T4 is 7.5). Bins are gathered on-device from
# the resident full ``binned`` matrix via the node's ``rows``; the gradients and
# Hessians arrive pre-gathered (g_sel/h_sel) so their reads stay coalesced. The
# missing bin is just another bin index — no special-casing here; the split scan
# routes it left.
_BUILD_HIST_SRC = r"""
extern "C" __global__
void build_hist(
    const unsigned short* __restrict__ binned, // (n_rows, n_features) resident
    const long long* __restrict__ rows,        // (n_sel,)
    const double* __restrict__ g_sel,          // (n_sel,) pre-gathered grad[rows]
    const double* __restrict__ h_sel,          // (n_sel,) pre-gathered hess[rows]
    double* __restrict__ hist,                 // (n_features, n_bins_max, 3) flat
    const long long n_sel,
    const long long n_features,
    const long long n_bins_max)
{
    const long long total = n_sel * n_features;
    long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (tid >= total) {
        return;
    }
    const long long i = tid / n_features;
    const long long f = tid % n_features;
    const long long row = rows[i];
    const long long b = (long long)binned[row * n_features + f];
    const long long base = (f * n_bins_max + b) * 3;
    atomicAdd(&hist[base + 0], g_sel[i]);
    atomicAdd(&hist[base + 1], h_sel[i]);
    atomicAdd(&hist[base + 2], 1.0);
}
"""

_BLOCK = 256


class CudaSplitBackend(BaseSplitBackend):
    """GPU histogram build (CuPy) with the NumPy split scan on the host."""

    def __init__(self) -> None:
        try:
            import cupy
        except ImportError as exc:  # pragma: no cover - exercised only off-GPU
            raise ImportError(
                "The CUDA backend requires CuPy and an NVIDIA GPU. "
                'Install it with: pip install "repleafgbm[cuda]"  (CUDA 12), '
                'or use split_backend="numpy".'
            ) from exc
        try:
            n_devices = cupy.cuda.runtime.getDeviceCount()
        except Exception as exc:  # pragma: no cover - depends on driver state
            raise ImportError(
                "CuPy is installed but no usable CUDA device was found "
                f"({exc}). Use split_backend=\"numpy\" on machines without a GPU."
            ) from exc
        if n_devices < 1:  # pragma: no cover - depends on hardware
            raise ImportError(
                "CuPy is installed but reports zero CUDA devices. "
                'Use split_backend="numpy" on machines without a GPU.'
            )
        self._cp = cupy
        self._kernel = cupy.RawKernel(_BUILD_HIST_SRC, "build_hist")
        # Split scanning (numeric thresholds + categorical subsets) is reused
        # verbatim from the reference backend; the histogram it consumes is a
        # small host array, so there is nothing to gain from a GPU port in v0.
        self._cpu = NumPySplitBackend()
        # Resident binned cache, keyed by (id, shape). binned is the same object
        # for every node of every tree, so it is uploaded once and reused.
        self._binned_key: tuple[int, tuple[int, ...]] | None = None
        self._binned_d = None

    def _device_binned(self, binned: np.ndarray):
        """Return binned as a resident C-contiguous uint16 device array.

        Uploaded once and cached by object identity + shape; subsequent nodes
        (and trees) reuse it without re-transferring the full matrix.
        """
        key = (id(binned), binned.shape)
        if self._binned_key != key:
            self._binned_d = self._cp.asarray(
                np.ascontiguousarray(binned, dtype=np.uint16)
            )
            self._binned_key = key
        return self._binned_d

    def build_histograms(
        self,
        binned: np.ndarray,
        rows: np.ndarray,
        grad: np.ndarray,
        hess: np.ndarray,
        n_bins_max: int,
    ) -> np.ndarray:
        cp = self._cp
        n_features = int(binned.shape[1])
        n_bins_max = int(n_bins_max)
        n_sel = int(rows.shape[0])
        hist = np.zeros((n_features, n_bins_max, 3), dtype=np.float64)
        if n_sel == 0 or n_features == 0:
            return hist

        binned_d = self._device_binned(binned)
        # Only the node's row index + its gathered grad/hess cross to the GPU;
        # the (n_sel, F) bin slice is gathered on-device from the resident matrix.
        rows_d = cp.asarray(np.ascontiguousarray(rows, dtype=np.int64))
        g_d = cp.asarray(np.ascontiguousarray(grad[rows], dtype=np.float64))
        h_d = cp.asarray(np.ascontiguousarray(hess[rows], dtype=np.float64))
        hist_d = cp.zeros(n_features * n_bins_max * 3, dtype=cp.float64)

        total = n_sel * n_features
        grid = (total + _BLOCK - 1) // _BLOCK
        self._kernel(
            (grid,),
            (_BLOCK,),
            (
                binned_d,
                rows_d,
                g_d,
                h_d,
                hist_d,
                np.int64(n_sel),
                np.int64(n_features),
                np.int64(n_bins_max),
            ),
        )
        return cp.asnumpy(hist_d).reshape(n_features, n_bins_max, 3)

    def find_best_split(
        self,
        hist: np.ndarray,
        n_bins_per_feature: np.ndarray,
        min_samples_leaf: int,
        l2: float,
        categorical_mask: np.ndarray | None = None,
        cat_smooth: float = 10.0,
        min_data_per_group: int = 100,
        max_cat_threshold: int = 32,
    ) -> SplitCandidate | None:
        # The histogram is a small host array; reuse the reference scan so the
        # tie-break / categorical / missing-left semantics match exactly.
        return self._cpu.find_best_split(
            hist,
            n_bins_per_feature,
            min_samples_leaf,
            l2,
            categorical_mask,
            cat_smooth,
            min_data_per_group,
            max_cat_threshold,
        )
