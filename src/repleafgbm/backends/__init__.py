"""Compute backends.

The numeric kernels (histogram accumulation, split scanning) live behind
``BaseSplitBackend`` so they can be swapped without touching tree-growing
logic. Available implementations:

* ``NumPySplitBackend`` — the always-available reference.
* ``RustSplitBackend`` — optional compiled kernels (``pip install ./native``).
"""

from repleafgbm.backends.base import BaseSplitBackend, SplitCandidate
from repleafgbm.backends.numpy_backend import NumPySplitBackend
from repleafgbm.backends.rust_backend import RustSplitBackend


def make_split_backend(name: str = "auto") -> BaseSplitBackend:
    """Resolve a backend name: "numpy", "rust", or "auto".

    "auto" uses the Rust kernels when the ``repleafgbm_native`` extension is
    installed and falls back to NumPy otherwise. The backends agree to
    floating-point noise (tested); same seed + same backend gives identical
    models.
    """
    if name == "numpy":
        return NumPySplitBackend()
    if name == "rust":
        return RustSplitBackend()
    if name == "auto":
        try:
            return RustSplitBackend()
        except ImportError:
            return NumPySplitBackend()
    raise ValueError(
        f"Unknown split_backend {name!r}; expected 'auto', 'numpy', or 'rust'"
    )


__all__ = [
    "BaseSplitBackend",
    "SplitCandidate",
    "NumPySplitBackend",
    "RustSplitBackend",
    "make_split_backend",
]
