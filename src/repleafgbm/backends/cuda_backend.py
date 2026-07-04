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

Multi-output trees keep the K per-output histograms resident and run the
shared-routing summed-gain scan on-device too (``build_histograms_multioutput`` /
``find_best_split_multioutput``), mirroring the scalar Phase-B2 path: the K
histograms are stacked on the GPU without a per-output host round-trip, the
per-output Newton gains are summed under one shared partition, and only the
winning split's scalars cross back. The same adaptive threshold routes narrow
nodes to the host scan, and ``REPLEAFGBM_CUDA_MO_DEVICE_SCAN=0`` forces the host
stack + host scan (the pre-device behavior) as a kill switch.

Device leaf-fit statistics (GPU leaf ridge, roadmap Phase 4.3): scalar leaf
fitting — the dominant CUDA-fit phase once the scans went on-device — computes
its per-leaf weighted Gram stacks / gradient projections / z-range guards on
the GPU (``leaf_fit_stats``), with the embedding matrix uploaded once per fit
and identity-cached like the binned matrix. The method returns the exact
statistics tuple the native Rust ``leaf_linear_stats`` produces, so the leaf
model feeds it into the same host float64 assembly (centering, ridge solve,
LOO gate). Default ON with an adaptive work crossover
(``REPLEAFGBM_CUDA_LEAF_FIT_MIN_CELLS``); ``REPLEAFGBM_CUDA_LEAF_FIT=0`` is the
kill switch. The seam covers all three leaf-fit paths: scalar
(``leaf_fit_stats``), pooled multiclass (``leaf_fit_stats_mc``), and
shared-routing multi-output vector leaves (``leaf_fit_stats_vector``).

Differences from the Rust backend (see ``docs/adr/0005-cuda-backend-cupy.md``):

* Parity is **allclose, not bitwise.** GPU histogram reduction uses
  ``atomicAdd`` on float64, whose summation order is not fixed, so sums differ
  from NumPy's ``bincount`` in the low bits. When the split scan runs on the
  host (the adaptive default for narrow per-node histograms) the chosen splits
  — and thus the trees — are identical, only leaf values carry that float noise,
  and cross-backend predictions agree to ``rtol=1e-6``. When the numeric scan
  runs **on-device** (wide histograms, or a forced low threshold) the gains are
  reduced with CuPy whose low bits also differ, so a *near-tied* split can win a
  different — but equally good — candidate: the trees then differ structurally
  and predictions agree to float noise except on the few rows a flipped split
  reroutes. Those flips are quality-neutral (the gains were tied), so model
  quality matches even when the exact tree does not. All tested in
  tests/test_cuda_backend.py.
* It is **not** bitwise reproducible run-to-run for the same reason; models
  agree to float noise rather than being identical (unlike numpy/rust, where
  same seed + same backend gives an identical model).

CuPy is torch-independent, so this module honors the invariant that the
native compute path never imports torch / lightgbm / repleafgbm.external.
Install with ``pip install "repleafgbm[cuda]"`` (needs an NVIDIA GPU + driver).
"""

from __future__ import annotations

import os
import warnings

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
# while capturing the wide win. Tune with a per-GPU sweep (the env override below
# feeds benchmarks/gpu_profile.py's --scan-min-cells-sweep) if needed.
_GPU_SCAN_MIN_CELLS = 32_768

# Private env override for the adaptive-scan threshold, for profiling/tuning only
# (a per-GPU sweep) — NOT part of the public estimator API. Unset → the measured
# default above, so the default fit is byte-for-byte unchanged.
_SCAN_MIN_CELLS_ENV = "REPLEAFGBM_CUDA_SCAN_MIN_CELLS"


def _resolve_scan_min_cells() -> int:
    """Effective adaptive-scan threshold: the env override or the default.

    Reads ``REPLEAFGBM_CUDA_SCAN_MIN_CELLS``. Unset/empty → ``_GPU_SCAN_MIN_CELLS``.
    A non-integer value is ignored (warns, keeps the default); negatives clamp to
    0 (forces the on-device scan for every node). Read once per backend (in
    ``__init__``), so per-node ``find_best_split`` never touches the environment.
    """
    raw = os.environ.get(_SCAN_MIN_CELLS_ENV)
    if raw is None or not raw.strip():
        return _GPU_SCAN_MIN_CELLS
    try:
        return max(0, int(raw))
    except ValueError:
        warnings.warn(
            f"Ignoring {_SCAN_MIN_CELLS_ENV}={raw!r} (not an integer); "
            f"using default {_GPU_SCAN_MIN_CELLS}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _GPU_SCAN_MIN_CELLS


# Private kill switch for the on-device multi-output split scan (default on). Set
# REPLEAFGBM_CUDA_MO_DEVICE_SCAN to a falsy value to fall back to the host stack +
# host scan (byte-for-byte the pre-device multi-output behavior) without touching
# NumPy — a gate for a narrow-case regression. Not part of the public API. Read
# once at construction (like the scan threshold), so per-node scans never touch
# the environment.
_MO_DEVICE_SCAN_ENV = "REPLEAFGBM_CUDA_MO_DEVICE_SCAN"


def _resolve_mo_device_scan() -> bool:
    """Whether the multi-output split scan runs on-device. Default True.

    Reads ``REPLEAFGBM_CUDA_MO_DEVICE_SCAN``. Unset/empty → True (the device
    path). A falsy value (``0``/``false``/``no``/``off``, case-insensitive) → the
    host fallback. Any other value keeps the device path on.
    """
    raw = os.environ.get(_MO_DEVICE_SCAN_ENV)
    if raw is None or not raw.strip():
        return True
    return raw.strip().lower() not in ("0", "false", "no", "off")


# Private kill switch for the node-batched depthwise scan (default on). The
# depthwise grower hands each level's frontier histograms here for one batched
# device scan (amortizing the per-node launch — T4-validated at split_scan 5-9x,
# depthwise fit 1.9-3.9x, quality-equivalent). Set REPLEAFGBM_CUDA_BATCHED_SCAN to
# a falsy value to fall back to the per-node path (base default loop), i.e. the
# pre-batched behavior — a gate for a narrow-case regression. Not part of the
# public API; read once at construction.
_BATCHED_SCAN_ENV = "REPLEAFGBM_CUDA_BATCHED_SCAN"


def _resolve_batched_scan() -> bool:
    """Whether depthwise node-batched scan runs on-device. Default True.

    Reads ``REPLEAFGBM_CUDA_BATCHED_SCAN``. Unset/empty → True (the batched device
    path — one kernel launch per level instead of per node). A falsy value
    (``0``/``false``/``no``/``off``, case-insensitive) → the per-node host loop
    (the pre-batched behavior). Any other value keeps the batched path on.
    """
    raw = os.environ.get(_BATCHED_SCAN_ENV)
    if raw is None or not raw.strip():
        return True
    return raw.strip().lower() not in ("0", "false", "no", "off")


# Private kill switch for the leafwise children-pair batched scan (default on).
# The leafwise grower scans the two children of each heap expansion in one
# batched device call (M=2 — halves the per-node launch count that dominates
# the leafwise scan); set REPLEAFGBM_CUDA_LEAFWISE_BATCH to a falsy value to
# fall back to the per-node scan. Subordinate to REPLEAFGBM_CUDA_BATCHED_SCAN
# (no batched scans at all when that is off). Not part of the public API; read
# once at construction.
_LEAFWISE_BATCH_ENV = "REPLEAFGBM_CUDA_LEAFWISE_BATCH"


def _resolve_leafwise_batch() -> bool:
    """Whether the leafwise grower batches children-pair scans. Default True.

    Reads ``REPLEAFGBM_CUDA_LEAFWISE_BATCH``. Unset/empty → True. A falsy value
    (``0``/``false``/``no``/``off``, case-insensitive) → per-node scans.
    """
    raw = os.environ.get(_LEAFWISE_BATCH_ENV)
    if raw is None or not raw.strip():
        return True
    return raw.strip().lower() not in ("0", "false", "no", "off")


# Private kill switch for the device leaf-fit statistics (default on). With
# split_backend="cuda", per-tree leaf-fit statistics (the per-leaf weighted Gram
# stacks + gradient projections that dominate wide-embedding fits) are computed
# on the GPU; set REPLEAFGBM_CUDA_LEAF_FIT to a falsy value to keep leaf fitting
# entirely on the host (the pre-device behavior). Not part of the public API;
# read once at construction.
_LEAF_FIT_ENV = "REPLEAFGBM_CUDA_LEAF_FIT"

#: Minimum per-tree leaf-fit work (gathered rows × emb_dim cells) for the device
#: path; smaller trees fit faster on the host (kernel-launch + transfer overhead
#: dominates tiny GEMMs — the same adaptive-crossover principle as
#: ``_GPU_SCAN_MIN_CELLS``). T4-validated (iter 012 sweep: a 200k-cell tree is
#: ~20% faster on the host; the 1e6 default keeps it there). Override with the
#: env var below for profiling.
_GPU_LEAF_FIT_MIN_CELLS = 1_000_000

#: The multi-output *vector* path needs a higher crossover: its per-leaf device
#: work at narrow embeddings (one Gram + a (d, K) cross GEMM) is too small to
#: beat the host loop — the iter-015 T4 A/B measured a −4.7% regression at
#: 50k×emb30 (1.5M cells) while 30k×emb200 (6M cells) wins 1.26×. 4M splits the
#: two measured points. An explicit ``REPLEAFGBM_CUDA_LEAF_FIT_MIN_CELLS``
#: override applies to BOTH paths (so forcing the device for tests/profiling
#: keeps working).
_GPU_LEAF_FIT_MIN_CELLS_VECTOR = 4_000_000
_LEAF_FIT_MIN_CELLS_ENV = "REPLEAFGBM_CUDA_LEAF_FIT_MIN_CELLS"


def _resolve_leaf_fit() -> bool:
    """Whether leaf-fit statistics run on-device. Default True.

    Reads ``REPLEAFGBM_CUDA_LEAF_FIT``. Unset/empty → True. A falsy value
    (``0``/``false``/``no``/``off``, case-insensitive) → host leaf fitting.
    """
    raw = os.environ.get(_LEAF_FIT_ENV)
    if raw is None or not raw.strip():
        return True
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _resolve_leaf_fit_min_cells() -> int:
    """Effective device leaf-fit crossover: the env override or the default.

    Reads ``REPLEAFGBM_CUDA_LEAF_FIT_MIN_CELLS``. Unset/empty →
    ``_GPU_LEAF_FIT_MIN_CELLS``. A non-integer value is ignored (warns, keeps
    the default); negatives clamp to 0 (forces the device path for every tree).
    """
    raw = os.environ.get(_LEAF_FIT_MIN_CELLS_ENV)
    if raw is None or not raw.strip():
        return _GPU_LEAF_FIT_MIN_CELLS
    try:
        return max(0, int(raw))
    except ValueError:
        warnings.warn(
            f"Ignoring {_LEAF_FIT_MIN_CELLS_ENV}={raw!r} (not an integer); "
            f"using default {_GPU_LEAF_FIT_MIN_CELLS}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _GPU_LEAF_FIT_MIN_CELLS


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
        # Effective adaptive-scan threshold for this backend's lifetime, resolved
        # once here (env override or default) so the per-node find_best_split scan
        # reads an attribute, never the environment. A fresh backend per fit picks
        # up the env value active at construction (the benchmark sets it around
        # the fit). See _resolve_scan_min_cells / _GPU_SCAN_MIN_CELLS.
        self._scan_min_cells = _resolve_scan_min_cells()
        # Multi-output device-scan kill switch (default on), resolved once. Off →
        # the base host stack + host scan (pre-device behavior). See
        # _resolve_mo_device_scan / build_histograms_multioutput below.
        self._mo_device_scan = _resolve_mo_device_scan()
        # Node-batched depthwise scan (default ON, kill switch), resolved once. On →
        # the grower hands each level's frontier histograms to find_best_split_batched
        # for one device scan; the instance attr tells the grower to take that path
        # (shadows the base-class default of False). See _resolve_batched_scan.
        self._batched_scan = _resolve_batched_scan()
        self.supports_batched_scan = self._batched_scan
        # Leafwise children-pair batching (Task B): subordinate to the batched
        # scan itself; the leafwise grower checks this attr (host backends
        # default False → bitwise per-node behavior).
        self.supports_leafwise_batched_scan = (
            self._batched_scan and _resolve_leafwise_batch()
        )
        # Device leaf-fit statistics (default ON, kill switch + adaptive
        # crossover), resolved once. The leaf models discover the capability via
        # ``supports_leaf_fit`` + ``leaf_fit_min_cells`` and call
        # ``leaf_fit_stats`` (below) per tree; off → they never look at us.
        self.supports_leaf_fit = _resolve_leaf_fit()
        self.leaf_fit_min_cells = _resolve_leaf_fit_min_cells()
        # Vector (multi-output) leaves use their own, higher crossover unless
        # the env var explicitly overrides the threshold (then both follow it).
        self.leaf_fit_min_cells_vector = (
            self.leaf_fit_min_cells
            if os.environ.get(_LEAF_FIT_MIN_CELLS_ENV, "").strip()
            else _GPU_LEAF_FIT_MIN_CELLS_VECTOR
        )
        # Resident embedding cache, keyed like the binned cache: Z is the same
        # object for every tree of a fit, so it is uploaded once and reused.
        self._Z_key: tuple[int, tuple[int, ...]] | None = None
        self._Z_d = None
        # Private transfer/work counters (profiling only; not part of the public
        # API or the BaseSplitBackend contract). They are plain integer adds at
        # each H2D/D2H boundary — negligible next to a kernel launch — and let
        # the GPU benchmark harness account for per-fit transfer volume without
        # changing any kernel or behavior. Snapshot with get_transfer_stats()
        # after a fit; reset_transfer_stats() zeroes them for a fresh window.
        self._stats: dict[str, int] = self._zero_stats()

    def _zero_stats(self) -> dict[str, int]:
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
            "z_h2d_bytes": 0,
            "z_uploads": 0,
            "leaffit_h2d_bytes": 0,
            "leaffit_d2h_bytes": 0,
            "n_leaf_fits": 0,
            # Effective adaptive-scan threshold in force for this backend (env
            # override or default). Config, not a transfer counter — re-seeded
            # (not zeroed) by reset_transfer_stats so a benchmark always records
            # the threshold that produced the scan-path counts above. It does not
            # end in "_bytes", so it is excluded from byte totals downstream.
            "scan_min_cells": self._scan_min_cells,
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
        # GPU once it is large enough to amortize launch/sync overhead. The
        # threshold (self._scan_min_cells) is _GPU_SCAN_MIN_CELLS by default,
        # overridable via REPLEAFGBM_CUDA_SCAN_MIN_CELLS for profiling sweeps. The
        # histogram stays resident for the build / subtraction either way; this
        # only copies the small ones back to scan.
        if n_features * n_bins_max < self._scan_min_cells:
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

    def find_best_split_batched(
        self,
        hists,
        n_bins_per_feature: np.ndarray,
        min_samples_leaf: int,
        l2: float,
        categorical_mask: np.ndarray | None = None,
        cat_smooth: float = 10.0,
        min_data_per_group: int = 100,
        max_cat_threshold: int = 32,
    ) -> list[SplitCandidate | None]:
        """One batched device scan of a depthwise level's M node histograms.

        Vectorizes :meth:`find_best_split`'s numeric scan over a leading node (M)
        axis — the same CuPy reductions (cumsum + Newton gain + per-node argmax),
        launched once for all M nodes instead of M times (the launch-amortization
        win, since the per-node scan is launch-bound, [[gpu-cuda-bottleneck-split-scan]]).
        Stacks the (already device-resident) histograms on-device, returns only the
        M winners' scalars (32 bytes each) plus per-node categorical slices. Off the
        gate or below the cell threshold it falls back to the per-node loop. Parity
        is allclose + quality-equivalent, not bitwise — near-tied splits can flip via
        low-bit device reductions (the host depthwise tree is untouched; ADR 0005).
        """
        if not self._batched_scan or not hists:
            return super().find_best_split_batched(
                hists, n_bins_per_feature, min_samples_leaf, l2,
                categorical_mask, cat_smooth, min_data_per_group, max_cat_threshold,
            )
        cp = self._cp
        H = cp.stack([cp.asarray(h) for h in hists])  # (M, F, B, 3), on-device
        m, n_features, n_bins_max = int(H.shape[0]), int(H.shape[1]), int(H.shape[2])
        # Adaptive: a small batch scans faster per-node on the host (one bulk copy)
        # than as a launch over M*F*B cells — same crossover as the single scan.
        if m * n_features * n_bins_max < self._scan_min_cells:
            self._bump("hist_d2h_bytes", 24 * m * n_features * n_bins_max)
            self._bump("n_small_scans", m)
            host = cp.asnumpy(H)
            return [
                self._cpu.find_best_split(
                    host[i], n_bins_per_feature, min_samples_leaf, l2,
                    categorical_mask, cat_smooth, min_data_per_group, max_cat_threshold,
                )
                for i in range(m)
            ]
        self._bump("n_gpu_scans", m)

        g, h, n = H[:, :, :, 0], H[:, :, :, 1], H[:, :, :, 2]  # (M, F, B)
        feat_idx = cp.arange(n_features)
        nbpf_d = cp.asarray(n_bins_per_feature)
        # Per-node totals (same across features) off feature 0: each (M,).
        g_total = g[:, 0].sum(axis=1)
        h_total = h[:, 0].sum(axis=1)
        n_total = n[:, 0].sum(axis=1)
        parent_score = _leaf_score(g_total, h_total, l2)  # (M,)

        miss_g = g[:, feat_idx, nbpf_d][:, :, None]  # (M, F, 1)
        miss_h = h[:, feat_idx, nbpf_d][:, :, None]
        miss_n = n[:, feat_idx, nbpf_d][:, :, None]
        left_g = cp.cumsum(g, axis=2) + miss_g  # (M, F, B)
        left_h = cp.cumsum(h, axis=2) + miss_h
        left_n = cp.cumsum(n, axis=2) + miss_n
        right_n = n_total[:, None, None] - left_n

        valid = (
            (cp.arange(n_bins_max)[None, None, :] <= (nbpf_d - 2)[None, :, None])
            & (left_n >= min_samples_leaf)
            & (right_n >= min_samples_leaf)
        )
        if categorical_mask is not None:
            valid &= ~cp.asarray(categorical_mask)[None, :, None]
        gain = (
            _leaf_score(left_g, left_h, l2)
            + _leaf_score(g_total[:, None, None] - left_g,
                          h_total[:, None, None] - left_h, l2)
            - parent_score[:, None, None]
        )
        gain = cp.where(valid & cp.isfinite(gain), gain, -np.inf)  # (M, F, B)

        # Per-node argmax over (F*B) (lowest flat index on ties, like np.argmax),
        # then one batched fetch of every winner's (flat, gain, n_left, n_right).
        flat_gain = gain.reshape(m, -1)
        best_flat_d = flat_gain.argmax(axis=1)  # (M,)
        rows_m = cp.arange(m)
        packed = cp.asnumpy(
            cp.stack(
                [
                    best_flat_d.astype(cp.float64),
                    flat_gain[rows_m, best_flat_d],
                    left_n.reshape(m, -1)[rows_m, best_flat_d],
                    right_n.reshape(m, -1)[rows_m, best_flat_d],
                ],
                axis=1,
            )
        )  # (M, 4) — the only numeric-path sync
        self._bump("winner_d2h_bytes", 32 * m)

        cat_feats = (
            np.flatnonzero(categorical_mask)
            if categorical_mask is not None and categorical_mask.any()
            else np.empty(0, dtype=np.int64)
        )
        # Categorical subset scan stays on the host (parity); pull the totals +
        # the categorical feature slices once for the whole batch.
        host_hist = cp.asnumpy(H) if cat_feats.size else None
        gt = cp.asnumpy(g_total) if cat_feats.size else None
        ht = cp.asnumpy(h_total) if cat_feats.size else None
        nt = cp.asnumpy(n_total) if cat_feats.size else None
        ps = cp.asnumpy(parent_score) if cat_feats.size else None

        out: list[SplitCandidate | None] = []
        for i in range(m):
            best_flat = int(packed[i, 0])
            best_gain = float(packed[i, 1])
            f, c = divmod(best_flat, n_bins_max)
            best: SplitCandidate | None = None
            if best_gain > 1e-12:
                best = SplitCandidate(
                    feature=int(f), bin=int(c), gain=best_gain,
                    n_left=int(packed[i, 2]), n_right=int(packed[i, 3]),
                )
            for cf in cat_feats:
                self._bump("cat_slice_d2h_bytes", 24 * n_bins_max)
                self._bump("n_cat_slices", 1)
                cand = self._cpu._best_categorical_split(
                    int(cf), host_hist[i, cf], int(n_bins_per_feature[cf]),
                    float(gt[i]), float(ht[i]), float(nt[i]), float(ps[i]),
                    min_samples_leaf, l2, cat_smooth, min_data_per_group,
                    max_cat_threshold,
                )
                if cand is not None and (best is None or cand.gain > best.gain):
                    best = cand
            out.append(best)
        return out

    # ----------------------------------------------------------------- #
    # Multi-output (shared-routing) device fast paths.
    # ----------------------------------------------------------------- #
    def build_histograms_multioutput(
        self,
        binned: np.ndarray,
        rows: np.ndarray,
        grad: np.ndarray,
        hess: np.ndarray,
        n_bins_max: int,
    ) -> np.ndarray:
        """Resident ``(n_features, n_bins_max, 3, n_outputs)`` stack on the GPU.

        Each output's scalar histogram is built by :meth:`build_histograms`
        (Phase B1/B2: the binned matrix is uploaded once and reused across all
        outputs; only each output's gathered grad/hess crosses) and the K
        resident histograms are stacked on-device with ``cp.stack`` — no
        per-output host round-trip (the cost the host stack paid). The grower's
        sibling subtraction stays CuPy arithmetic on the 4-D array, as on the
        scalar path. With the device path disabled
        (``REPLEAFGBM_CUDA_MO_DEVICE_SCAN`` off) this falls back to the base host
        stack, reproducing the pre-device behavior exactly.
        """
        if not self._mo_device_scan:
            return super().build_histograms_multioutput(
                binned, rows, grad, hess, n_bins_max
            )
        per_output = [
            self.build_histograms(binned, rows, grad[:, k], hess[:, k], n_bins_max)
            for k in range(int(grad.shape[1]))
        ]
        return self._cp.stack(per_output, axis=-1)

    def find_best_split_multioutput(
        self,
        hist: np.ndarray,
        n_bins_per_feature: np.ndarray,
        min_samples_leaf: int,
        l2: float,
    ) -> SplitCandidate | None:
        """On-device summed-gain scan for a shared-routing multi-output node.

        Mirrors :func:`numpy_backend.find_best_split_multioutput` in CuPy on the
        resident ``(F, n_bins_max, 3, K)`` histogram: cumulative gradient/Hessian
        sums per (feature, output), the per-output Newton gains summed over
        outputs under one shared left/right partition, and a device argmax (lowest
        flat index on ties, matching ``np.argmax``). Only the winning split's 4
        scalars cross back — the histogram never leaves the GPU. Every feature is
        an ordered-threshold candidate (multi-output trees never produce
        categorical subset splits), so there is no host categorical fallback.
        Missing values go left.

        With the device path disabled, or for a histogram below the adaptive
        threshold (where the host scan beats the launch/sync overhead of a tiny
        K-wide kernel — mirroring the scalar crossover so narrow multi-output
        never regresses), the array is copied back once and the base host scan
        runs.
        """
        if not self._mo_device_scan:
            return super().find_best_split_multioutput(
                hist, n_bins_per_feature, min_samples_leaf, l2
            )
        cp = self._cp
        hist_d = cp.asarray(hist)
        n_features, n_bins_max = int(hist_d.shape[0]), int(hist_d.shape[1])
        n_outputs = int(hist_d.shape[3])

        # Adaptive: a small per-node histogram scans faster on the host (one bulk
        # copy + vectorized NumPy) than as a tiny K-wide GPU kernel; the same
        # crossover the scalar path uses, so narrow multi-output never regresses.
        if n_features * n_bins_max < self._scan_min_cells:
            self._bump("hist_d2h_bytes", 24 * n_features * n_bins_max * n_outputs)
            self._bump("n_small_scans", 1)
            return super().find_best_split_multioutput(
                cp.asnumpy(hist_d), n_bins_per_feature, min_samples_leaf, l2
            )
        self._bump("n_gpu_scans", 1)

        g = hist_d[:, :, 0, :]  # (F, B, K) per-output grad
        h = hist_d[:, :, 1, :]  # (F, B, K) per-output hess
        n = hist_d[:, :, 2, 0]  # (F, B) count (shared across outputs)
        feat_idx = cp.arange(n_features)
        nbpf_d = cp.asarray(n_bins_per_feature)

        # Per-feature bins partition the same rows, so per-feature totals equal
        # the node totals; read them off feature 0. Kept on-device (the only host
        # sync is the single winner fetch below).
        g_total = g[0].sum(axis=0)  # (K,)
        h_total = h[0].sum(axis=0)  # (K,)
        n_total = n[0].sum()
        parent_score = _leaf_score(g_total, h_total, l2).sum()

        # Missing values (bin index n_bins_per_feature[f]) always go left.
        miss_g = g[feat_idx, nbpf_d, :][:, None, :]  # (F, 1, K)
        miss_h = h[feat_idx, nbpf_d, :][:, None, :]
        miss_n = n[feat_idx, nbpf_d][:, None]  # (F, 1)

        left_g = cp.cumsum(g, axis=1) + miss_g  # (F, B, K)
        left_h = cp.cumsum(h, axis=1) + miss_h
        left_n = cp.cumsum(n, axis=1) + miss_n  # (F, B)
        right_n = n_total - left_n

        valid = (
            (cp.arange(n_bins_max)[None, :] <= (nbpf_d - 2)[:, None])
            & (left_n >= min_samples_leaf)
            & (right_n >= min_samples_leaf)
        )
        # Summed per-output Newton gain. No np.errstate wrapper (unlike the host
        # reference): CuPy does not emit NumPy's divide/invalid warnings, and the
        # isfinite mask below drops any inf/nan candidate identically.
        gain = (
            _leaf_score(left_g, left_h, l2)
            + _leaf_score(g_total - left_g, h_total - left_h, l2)
        ).sum(axis=2) - parent_score  # (F, B)
        gain = cp.where(valid & cp.isfinite(gain), gain, -np.inf)

        # Device argmax + pack the winner's flat index + gain + child counts into
        # one tiny array so a single ``asnumpy`` brings them back — the only sync.
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
        self._bump("winner_d2h_bytes", 32)
        best_gain = float(packed[1])
        if best_gain <= 1e-12:
            return None
        f, c = divmod(int(packed[0]), n_bins_max)
        return SplitCandidate(
            feature=int(f),
            bin=int(c),
            gain=best_gain,
            n_left=int(packed[2]),
            n_right=int(packed[3]),
        )

    # ------------------------------------------------------------------ #
    # Device leaf-fit statistics (GPU leaf ridge, roadmap Phase 4.3)
    # ------------------------------------------------------------------ #
    def _device_Z(self, Z: np.ndarray):
        """Return Z as a resident C-contiguous float64 device array.

        Uploaded once per fit and cached by object identity + shape (the
        embedding matrix is the same object for every tree), mirroring
        ``_device_binned``.
        """
        key = (id(Z), Z.shape)
        if self._Z_key != key:
            self._Z_d = self._cp.asarray(
                np.ascontiguousarray(Z, dtype=np.float64)
            )
            self._Z_key = key
            self._bump("z_h2d_bytes", 8 * int(np.prod(Z.shape)))
            self._bump("z_uploads", 1)
        return self._Z_d

    def leaf_fit_stats(
        self,
        Z: np.ndarray,
        grad: np.ndarray,
        hess: np.ndarray,
        order: np.ndarray,
        offsets: np.ndarray,
        linear: np.ndarray,
        use_f32: bool = False,
    ) -> tuple[np.ndarray, ...]:
        """One tree's leaf-fit statistics on the GPU.

        Returns the exact host tuple the native ``leaf_linear_stats`` returns —
        ``(g_sum, h_sum, s_hz, A, gz, z_min, z_max)`` with ``A`` the *uncentered*
        per-leaf weighted Gram stack ``Σ h z zᵀ`` and ``gz = Σ g z``, both over
        the ``linear`` leaves only — so the caller feeds it straight into
        ``_leafvalues_from_native_stats`` (centering, ridge solve, and the LOO
        gate stay on the host in float64, byte-identical code).

        Batching layout: the O(n) sum reductions (``g_sum``/``h_sum``/``s_hz``/
        ``gz``) are single scatter/bincount kernels over the whole tree; the
        per-leaf Gram is one cuBLAS GEMM per linear leaf — real O(n_leaf·d²)
        work per launch, unlike the launch-bound per-node scans this backend
        batches elsewhere — and ``z_min``/``z_max`` ride the same per-leaf loop
        as exact slice reductions (CuPy's float scatter_min/max round through
        float32, and guard bounds must match the host exactly). With
        ``use_f32`` the two large reductions (Gram + gradient projection)
        accumulate in float32 (the ``leaf_fit_precision="float32_gram"``
        contract); everything else stays float64.

        Parity: allclose, never bitwise (device reduction order; ADR 0005).
        """
        cp = self._cp
        Zs, seg_d, order_d = self._gather_tree(Z, order, offsets, grad.size)
        g_d = cp.asarray(np.ascontiguousarray(grad, dtype=np.float64))
        h_d = cp.asarray(np.ascontiguousarray(hess, dtype=np.float64))
        gs = g_d[order_d]
        hs = h_d[order_d]
        return self._leaf_fit_stats_core(
            Zs, gs, hs, seg_d, offsets, linear, len(offsets) - 1, use_f32
        )

    def leaf_fit_stats_mc(
        self,
        Z: np.ndarray,
        grad: np.ndarray,
        hess: np.ndarray,
        order: np.ndarray,
        offsets: np.ndarray,
        linear: np.ndarray,
        leaf_class: np.ndarray,
        use_f32: bool = False,
    ) -> tuple[np.ndarray, ...]:
        """Pooled-multiclass leaf-fit statistics on the GPU.

        Identical contract to the native ``leaf_linear_stats_mc``: the pooled
        leaf list concatenates every class's leaves, and each pooled leaf reads
        its own class's grad/hess column (``leaf_class[leaf]``). The gather
        selects per-row columns on-device; everything else is the scalar core.
        """
        cp = self._cp
        Zs, seg_d, order_d = self._gather_tree(Z, order, offsets, grad.size)
        g_d = cp.asarray(np.ascontiguousarray(grad, dtype=np.float64))
        h_d = cp.asarray(np.ascontiguousarray(hess, dtype=np.float64))
        cls_d = cp.asarray(
            np.ascontiguousarray(leaf_class, dtype=np.int64)
        )[seg_d]
        gs = g_d[order_d, cls_d]
        hs = h_d[order_d, cls_d]
        return self._leaf_fit_stats_core(
            Zs, gs, hs, seg_d, offsets, linear, len(offsets) - 1, use_f32
        )

    def leaf_fit_stats_vector(
        self,
        Z: np.ndarray,
        grad: np.ndarray,
        hess: np.ndarray,
        order: np.ndarray,
        offsets: np.ndarray,
        linear: np.ndarray,
        use_f32: bool = False,
    ) -> tuple[np.ndarray, ...]:
        """Shared-routing multi-output (vector-leaf) statistics on the GPU.

        Mirrors ``core.multioutput.fit_vector_leaves``'s math, which relies on
        the multi-output invariant that the Hessian is identical across output
        columns (squared error and the constant-Hessian robust objectives; the
        host path already uses ``hess[:, 0]`` as the shared weight ``w``). With
        ``w == h_k`` per column, the per-output Newton cross terms collapse to
        pure gradient sums: ``Σ w·z·t_kᵀ = −Σ z·g_k`` and ``Σ w·t_k = −Σ g_k``.

        Returns per-``linear``-leaf stats ``(h_sum, s_wz, M, C, t_wsum, z_min,
        z_max)``: ``h_sum (k,)`` the shared weight total, ``s_wz (k, d) =
        Σ w·z``, ``M (k, d, d)`` the uncentered weighted Gram, ``C (k, d, K) =
        Σ w·z·tᵀ = −Σ z·gᵀ``, ``t_wsum (k, K) = Σ w·t = −Σ g``, and the exact
        z-range guards. The host assembles the centered system via
        ``A = M − z̄·s_wzᵀ + l2·I`` and ``rhs = C − z̄·t_wsumᵀ`` and keeps the
        solve + LOO gate in float64 (the constant-leaf bias for every leaf is
        already computed on the host — it is O(n·K), not worth offloading).
        """
        import cupyx

        cp = self._cp
        n_leaves = len(offsets) - 1
        emb_dim = int(Z.shape[1])
        n_outputs = int(grad.shape[1])
        k = int(linear.size)
        Zs, seg_d, order_d = self._gather_tree(Z, order, offsets, grad.size)
        g_d = cp.asarray(np.ascontiguousarray(grad, dtype=np.float64))
        h_d = cp.asarray(np.ascontiguousarray(hess, dtype=np.float64))
        gs = g_d[order_d]  # (n_sel, K)
        ws = h_d[order_d, 0]  # shared per-row weight (column 0, host contract)

        g_colsum_d = cp.zeros((n_leaves, n_outputs), dtype=cp.float64)
        cupyx.scatter_add(g_colsum_d, seg_d, gs)
        h_sum_d = cp.bincount(seg_d, weights=ws, minlength=n_leaves)

        wZs = Zs * ws[:, None]
        s_wz_d = cp.zeros((n_leaves, emb_dim), dtype=cp.float64)
        cupyx.scatter_add(s_wz_d, seg_d, wZs)

        M_d = cp.empty((k, emb_dim, emb_dim), dtype=cp.float64)
        C_d = cp.empty((k, emb_dim, n_outputs), dtype=cp.float64)
        zmn_d = cp.empty((k, emb_dim), dtype=cp.float64)
        zmx_d = cp.empty((k, emb_dim), dtype=cp.float64)
        if use_f32:
            Z32 = Zs.astype(cp.float32)
            wZ32 = wZs.astype(cp.float32)
            g32 = gs.astype(cp.float32)
        for j in range(k):
            i = int(linear[j])
            sl = slice(int(offsets[i]), int(offsets[i + 1]))
            if use_f32:
                M_d[j] = (wZ32[sl].T @ Z32[sl]).astype(cp.float64)
                C_d[j] = -(Z32[sl].T @ g32[sl]).astype(cp.float64)
            else:
                M_d[j] = wZs[sl].T @ Zs[sl]
                C_d[j] = -(Zs[sl].T @ gs[sl])
            zmn_d[j] = Zs[sl].min(axis=0)
            zmx_d[j] = Zs[sl].max(axis=0)

        linear_d = cp.asarray(np.ascontiguousarray(linear, dtype=np.int64))
        out = (
            cp.asnumpy(h_sum_d[linear_d]),
            cp.asnumpy(s_wz_d[linear_d]),
            cp.asnumpy(M_d),
            cp.asnumpy(C_d),
            cp.asnumpy(-g_colsum_d[linear_d]),
            cp.asnumpy(zmn_d),
            cp.asnumpy(zmx_d),
        )
        self._bump(
            "leaffit_d2h_bytes",
            8 * k * (emb_dim * emb_dim + emb_dim * n_outputs
                     + n_outputs + 3 * emb_dim + 1),
        )
        return out

    def _gather_tree(self, Z, order, offsets, n_gradhess_values):
        """Upload/gather one tree's rows: resident Z + order + per-row leaf ids.

        ``n_gradhess_values`` is ``grad.size`` (== ``hess.size``) — rows for
        the scalar path, rows×K for the multiclass/vector paths — so the H2D
        byte counter stays accurate across all three entry points.
        """
        cp = self._cp
        Z_d = self._device_Z(Z)
        order_d = cp.asarray(np.ascontiguousarray(order, dtype=np.int64))
        # Per-row leaf ids for the scatter reductions, built host-side from the
        # (tiny) offsets array.
        sizes = np.diff(offsets)
        seg = np.repeat(np.arange(len(offsets) - 1, dtype=np.int64), sizes)
        seg_d = cp.asarray(seg)
        # order + seg (int64) and grad + hess (float64) cross per tree.
        self._bump(
            "leaffit_h2d_bytes",
            8 * (2 * order.shape[0] + 2 * n_gradhess_values),
        )
        self._bump("n_leaf_fits", 1)
        return Z_d[order_d], seg_d, order_d

    def _leaf_fit_stats_core(
        self, Zs, gs, hs, seg_d, offsets, linear, n_leaves, use_f32
    ):
        """Scalar per-leaf statistics from gathered rows (shared by the scalar
        and pooled-multiclass entry points; the gather decides which grad/hess
        column each row reads)."""
        import cupyx

        cp = self._cp
        emb_dim = int(Zs.shape[1])
        k = int(linear.size)
        hZs = Zs * hs[:, None]

        g_sum_d = cp.bincount(seg_d, weights=gs, minlength=n_leaves)
        h_sum_d = cp.bincount(seg_d, weights=hs, minlength=n_leaves)

        s_hz_d = cp.zeros((n_leaves, emb_dim), dtype=cp.float64)
        cupyx.scatter_add(s_hz_d, seg_d, hZs)
        if not use_f32:  # the f32 branch below computes its own projection
            gz_all_d = cp.zeros((n_leaves, emb_dim), dtype=cp.float64)
            cupyx.scatter_add(gz_all_d, seg_d, Zs * gs[:, None])

        # z_min/z_max are computed per linear leaf with exact slice reductions
        # in the GEMM loop below — CuPy's float scatter_min/scatter_max round
        # through float32 (measured ~5e-8 relative error on a T4), and the
        # extrapolation-guard bounds must match the host values exactly (min/
        # max of the same numbers involves no arithmetic).
        zmn_d = cp.empty((k, emb_dim), dtype=cp.float64)
        zmx_d = cp.empty((k, emb_dim), dtype=cp.float64)

        A_d = cp.empty((k, emb_dim, emb_dim), dtype=cp.float64)
        if use_f32:
            Z32 = Zs.astype(cp.float32)
            hZ32 = hZs.astype(cp.float32)
        for j in range(k):
            i = int(linear[j])
            sl = slice(int(offsets[i]), int(offsets[i + 1]))
            if use_f32:
                A_d[j] = (Z32[sl].T @ hZ32[sl]).astype(cp.float64)
            else:
                A_d[j] = Zs[sl].T @ hZs[sl]
            zmn_d[j] = Zs[sl].min(axis=0)
            zmx_d[j] = Zs[sl].max(axis=0)
        if use_f32:
            # The float32_gram contract narrows the projection too; recompute
            # gz for the linear leaves from the f32 gather in one GEMM-like
            # scatter pass (still a single kernel).
            gz32_d = cp.zeros((n_leaves, emb_dim), dtype=cp.float32)
            cupyx.scatter_add(
                gz32_d, seg_d, Z32 * gs.astype(cp.float32)[:, None]
            )
            gz_all_d = gz32_d.astype(cp.float64)

        linear_d = cp.asarray(np.ascontiguousarray(linear, dtype=np.int64))
        g_sum = cp.asnumpy(g_sum_d)
        h_sum = cp.asnumpy(h_sum_d)
        s_hz = cp.asnumpy(s_hz_d[linear_d])
        A = cp.asnumpy(A_d)
        gz = cp.asnumpy(gz_all_d[linear_d])
        zmn = cp.asnumpy(zmn_d)
        zmx = cp.asnumpy(zmx_d)
        self._bump(
            "leaffit_d2h_bytes",
            8 * (2 * n_leaves + k * (emb_dim * emb_dim + 4 * emb_dim)),
        )
        return g_sum, h_sum, s_hz, A, gz, zmn, zmx
