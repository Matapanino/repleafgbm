"""Compute backends.

The numeric kernels (histogram accumulation, split scanning) live behind
``BaseSplitBackend`` so they can be swapped without touching tree-growing
logic. Available implementations:

* ``NumPySplitBackend`` — the always-available reference.
* ``RustSplitBackend`` — optional compiled kernels (``pip install ./native``).
* ``CudaSplitBackend`` — optional GPU histogram kernel
  (``pip install "repleafgbm[cuda]"``; NVIDIA GPU required).
"""

from repleafgbm.backends.base import BaseSplitBackend, SplitCandidate
from repleafgbm.backends.cuda_backend import CudaSplitBackend
from repleafgbm.backends.numpy_backend import NumPySplitBackend
from repleafgbm.backends.rust_backend import RustSplitBackend


def make_split_backend(name: str = "auto") -> BaseSplitBackend:
    """Resolve a backend name: "numpy", "rust", "cuda", or "auto".

    "auto" uses the Rust kernels when the ``repleafgbm_native`` extension is
    installed and falls back to NumPy otherwise. The numpy/rust backends agree
    to floating-point noise (tested); same seed + same backend gives identical
    models.

    "cuda" is **explicit-only** (never selected by "auto"): it requires CuPy
    and an NVIDIA GPU, accelerates histogram construction on the GPU, and
    agrees with the reference to float noise — but, because GPU atomic-add
    summation order is not fixed, it is allclose rather than bitwise-identical
    and not reproducible run-to-run. Raises ImportError when no GPU is usable.
    """
    if name == "numpy":
        return NumPySplitBackend()
    if name == "rust":
        return RustSplitBackend()
    if name == "cuda":
        return CudaSplitBackend()
    if name == "auto":
        try:
            return RustSplitBackend()
        except ImportError:
            return NumPySplitBackend()
    raise ValueError(
        f"Unknown split_backend {name!r}; "
        "expected 'auto', 'numpy', 'rust', or 'cuda'"
    )


__all__ = [
    "BaseSplitBackend",
    "SplitCandidate",
    "NumPySplitBackend",
    "RustSplitBackend",
    "CudaSplitBackend",
    "make_split_backend",
]
