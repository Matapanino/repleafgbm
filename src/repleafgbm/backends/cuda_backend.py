"""CUDA implementation of the split-search kernels (optional, CuPy-based).

This is the experimental ``split_backend="cuda"`` path. Per-node histogram
construction runs on an NVIDIA GPU via a :class:`cupy.RawKernel` (Phase A), and
the *numeric* split scan runs on the GPU too (Phase B2): the histogram stays
resident on the device across a tree's nodes (the grower's sibling-subtraction
``parent - child`` is CuPy arithmetic), and for large per-node histograms the
cumulative-sum gain sweep and argmax run on-device with only the winning split's
scalars crossing back. The scan is **adaptive** — small histograms are copied
back and scanned on the host, which beats launching many tiny GPU kernels (see
``_GPU_SCAN_MIN_CELLS``), so narrow fits never regress while wide fits get the
GPU win. Categorical subset splits keep the branchy stable-sort / both-end-prefix
/ tie-break logic on the host (delegated to the NumPy reference's
``_best_categorical_split``) so they stay byte-for-byte identical to the
reference backend; only the few categorical feature slices are copied back, not
the whole histogram.

Resident-data fast paths:

* Phase B1: the binned feature matrix is constant for the whole run, so it is
  uploaded to the GPU once and cached (keyed by object identity); each node
  ships only its small ``rows`` index plus the gathered gradients/Hessians, and
  the kernel gathers bins on-device — removing the per-node host gather of
  ``binned[rows]`` and its upload (the dominant transfer when the matrix is
  wide). The cache lives for the backend's lifetime (freed on GC).
* Phase B2: ``build_histograms`` *returns* the ``(n_features, n_bins_max, 3)``
  histogram as a resident CuPy array, so it is never copied to the host during a
  tree's growth — the per-node GPU→host round-trip is cut to the winning
  split's scalars.

Multi-output trees route through the NumPy multi-output scan on the host; the
CUDA backend keeps the Phase-A GPU histogram there but not B2 residency.

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
from repleafgbm.backends.numpy_backend import NumPySplitBackend, _leaf_score

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

# Adaptive split scan: a node's numeric scan runs on the GPU only when its
# histogram has at least this many (feature x bin) cells; smaller scans are
# cheaper on the host (one bulk copy + vectorized NumPy) than as many tiny GPU
# kernels with launch/sync overhead. Measured on a Tesla T4
# (experiments/results/2026-06-17-cuda-parity.md): the on-device scan is ~2.1x
# end-to-end on a 200-feature fit but ~neutral / slightly slower at 30 features,
# so the crossover sits between; 2^15 keeps narrow fits on the (faster) host path
# while capturing the wide win. Tune with a per-GPU sweep if needed.
_GPU_SCAN_MIN_CELLS = 32_768


class CudaSplitBackend(BaseSplitBackend):
    """GPU histogram build + GPU numeric split scan (CuPy); categoricals host."""

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
        # The numeric split scan runs on-device (find_best_split below); the
        # categorical subset scan stays on the host for byte-for-byte parity, so
        # we keep a reference backend to reuse ``_best_categorical_split``.
        self._cpu = NumPySplitBackend()
        # Resident binned cache, keyed by (id, shape). binned is the same object
        # for every node of every tree, so it is uploaded once and reused.
        self._binned_key: tuple[int, tuple[int, ...]] | None = None
        self._binned_d = None
        # Private transfer/work counters (profiling only; not part of the public
        # API or the BaseSplitBackend contract). They are plain integer adds at
        # each H2D/D2H boundary — negligible next to a kernel launch — and let
        # the GPU benchmark harness account for per-fit transfer volume without
        # changing any kernel or behavior. Snapshot with get_transfer_stats()
        # after a fit; reset_transfer_stats() zeroes them for a fresh window.
        self._stats: dict[str, int] = self._zero_stats()

    @staticmethod
    def _zero_stats() -> dict[str, int]:
        return {
            "binned_h2d_bytes": 0,
            "rows_h2d_bytes": 0,
            "gradhess_h2d_bytes": 0,
            "hist_d2h_bytes": 0,
            "winner_d2h_bytes": 0,
            "cat_slice_d2h_bytes": 0,
            "binned_uploads": 0,
            "n_hist_builds": 0,
            "n_small_scans": 0,
            "n_gpu_scans": 0,
            "n_cat_slices": 0,
        }

    def _bump(self, key: str, n: int) -> None:
        self._stats[key] += int(n)

    def get_transfer_stats(self) -> dict[str, int]:
        """Snapshot of the private H2D/D2H byte + work counters (a copy).

        Profiling aid only — not part of the split-backend contract. Keys cover
        binned/rows/grad-hess uploads, small-scan and categorical-slice
        copy-backs, the winning-split scalar copy-back, and per-phase call
        counts. Counters accumulate over the backend's lifetime, so a backend
        constructed fresh for one fit reports that fit's totals.
        """
        return dict(self._stats)

    def reset_transfer_stats(self) -> None:
        """Zero the private transfer/work counters."""
        self._stats = self._zero_stats()

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
            # Cache miss only: the whole matrix crosses once per (object, shape).
            # Cache hits add nothing — that is the point of the Phase B1 cache.
            self._bump("binned_h2d_bytes", 2 * int(np.prod(binned.shape)))
            self._bump("binned_uploads", 1)
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
        if n_sel == 0 or n_features == 0:
            return cp.zeros((n_features, n_bins_max, 3), dtype=cp.float64)

        binned_d = self._device_binned(binned)
        # Only the node's row index + its gathered grad/hess cross to the GPU;
        # the (n_sel, F) bin slice is gathered on-device from the resident matrix.
        rows_d = cp.asarray(np.ascontiguousarray(rows, dtype=np.int64))
        g_d = cp.asarray(np.ascontiguousarray(grad[rows], dtype=np.float64))
        h_d = cp.asarray(np.ascontiguousarray(hess[rows], dtype=np.float64))
        # The dominant per-node transfer the next optimization targets: the
        # node's int64 rows plus host-gathered grad/hess (8 + 16 bytes per row).
        self._bump("rows_h2d_bytes", 8 * n_sel)
        self._bump("gradhess_h2d_bytes", 16 * n_sel)
        self._bump("n_hist_builds", 1)
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
        # Phase B2: return the resident device histogram. The grower keeps it on
        # the GPU (its sibling-subtraction ``parent - child`` is CuPy
        # arithmetic) and find_best_split scans it on-device.
        return hist_d.reshape(n_features, n_bins_max, 3)

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
        """Adaptive numeric scan + host categorical scan.

        Large per-node histograms are scanned on the GPU (mirrors
        NumPySplitBackend; only the winning split's scalars cross back); small
        ones are copied to the host and delegated to the reference scan, which is
        faster than launching many tiny GPU kernels (see _GPU_SCAN_MIN_CELLS).

        ``hist`` is normally the resident device array from build_histograms (the
        grower's subtraction kept it on-device); a host array is also accepted
        (``cp.asarray`` no-ops on device input, uploads a host one).
        """
        cp = self._cp
        hist_d = cp.asarray(hist)
        g, h, n = hist_d[:, :, 0], hist_d[:, :, 1], hist_d[:, :, 2]
        n_features, n_bins_max = int(g.shape[0]), int(g.shape[1])

        # Adaptive: small per-node histograms scan faster on the host (a single
        # bulk copy + vectorized NumPy) than as many tiny GPU kernels; only the
        # GPU once it is large enough to amortize launch/sync overhead. See
        # _GPU_SCAN_MIN_CELLS. The histogram stays resident for the build /
        # subtraction either way; this only copies the small ones back to scan.
        if n_features * n_bins_max < _GPU_SCAN_MIN_CELLS:
            # Small histogram: one bulk copy back, host scan (see threshold note).
            self._bump("hist_d2h_bytes", 24 * n_features * n_bins_max)
            self._bump("n_small_scans", 1)
            return self._cpu.find_best_split(
                cp.asnumpy(hist_d), n_bins_per_feature, min_samples_leaf, l2,
                categorical_mask, cat_smooth, min_data_per_group, max_cat_threshold,
            )
        self._bump("n_gpu_scans", 1)

        feat_idx = cp.arange(n_features)
        nbpf_d = cp.asarray(n_bins_per_feature)

        # Per-feature bins partition the same rows, so per-feature totals all
        # equal the node totals; read them off feature 0. Kept on-device (0-d
        # arrays) so the whole scan runs without a host round-trip until the
        # single batched fetch below — per-node GPU→host syncs dominate this tiny
        # scan, so we keep them to one.
        g_total = g[0].sum()
        h_total = h[0].sum()
        n_total = n[0].sum()
        parent_score = _leaf_score(g_total, h_total, l2)

        # Missing values (bin index n_bins_per_feature[f]) always go left.
        miss_g = g[feat_idx, nbpf_d][:, None]
        miss_h = h[feat_idx, nbpf_d][:, None]
        miss_n = n[feat_idx, nbpf_d][:, None]

        # Candidate c sends non-missing bins <= c left (plus missing); invalid
        # candidates are masked below.
        left_g = cp.cumsum(g, axis=1) + miss_g
        left_h = cp.cumsum(h, axis=1) + miss_h
        left_n = cp.cumsum(n, axis=1) + miss_n
        right_n = n_total - left_n

        valid = (
            (cp.arange(n_bins_max)[None, :] <= (nbpf_d - 2)[:, None])
            & (left_n >= min_samples_leaf)
            & (right_n >= min_samples_leaf)
        )
        if categorical_mask is not None:
            # Categorical features get the host subset scan below.
            valid &= ~cp.asarray(categorical_mask)[:, None]
        gain = (
            _leaf_score(left_g, left_h, l2)
            + _leaf_score(g_total - left_g, h_total - left_h, l2)
            - parent_score
        )
        gain = cp.where(valid & cp.isfinite(gain), gain, -np.inf)

        # Device argmax (lowest flat index on ties, matching np.argmax). Pack the
        # winner's flat index + gain + child counts into one tiny array so a
        # single ``asnumpy`` brings them all back — the only sync on the numeric
        # path (the (F, B, 3) histogram never leaves the GPU).
        best_flat_d = cp.argmax(gain)
        packed = cp.asnumpy(
            cp.stack(
                [
                    best_flat_d.astype(cp.float64),
                    gain.ravel()[best_flat_d],
                    left_n.ravel()[best_flat_d],
                    right_n.ravel()[best_flat_d],
                ]
            )
        )
        # Only the winning split's 4 float64 scalars cross back on this path —
        # the (F, B, 3) histogram stayed resident. This is the wide-fit win the
        # counters quantify against the small-scan copy-back above.
        self._bump("winner_d2h_bytes", 32)
        best_flat = int(packed[0])
        best_gain = float(packed[1])
        f, c = divmod(best_flat, n_bins_max)
        best: SplitCandidate | None = None
        if best_gain > 1e-12:
            best = SplitCandidate(
                feature=int(f),
                bin=int(c),
                gain=best_gain,
                n_left=int(packed[2]),
                n_right=int(packed[3]),
            )

        # Categorical subset splits stay on the host for byte-for-byte parity
        # (stable sort / both-end prefix / tie-break); only the categorical
        # feature slices come back, not the whole histogram.
        if categorical_mask is not None and categorical_mask.any():
            g_tot, h_tot = float(g_total), float(h_total)
            n_tot, parent = float(n_total), float(parent_score)
            for f in np.flatnonzero(categorical_mask):
                # One feature's (n_bins_max, 3) slice crosses back per categorical
                # feature scanned (not the whole histogram).
                self._bump("cat_slice_d2h_bytes", 24 * n_bins_max)
                self._bump("n_cat_slices", 1)
                cand = self._cpu._best_categorical_split(
                    int(f), cp.asnumpy(hist_d[f]), int(n_bins_per_feature[f]),
                    g_tot, h_tot, n_tot, parent,
                    min_samples_leaf, l2,
                    cat_smooth, min_data_per_group, max_cat_threshold,
                )
                if cand is not None and (best is None or cand.gain > best.gain):
                    best = cand
        return best
