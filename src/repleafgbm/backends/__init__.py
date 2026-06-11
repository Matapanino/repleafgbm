"""Compute backends.

v0 ships a single NumPy backend. The backend boundary exists so that the
numeric kernels (histogram accumulation, split scanning) can later be
reimplemented in Rust/C++/CUDA without touching tree-growing logic.
"""

from repleafgbm.backends.base import BaseSplitBackend, SplitCandidate
from repleafgbm.backends.numpy_backend import NumPySplitBackend

__all__ = ["BaseSplitBackend", "SplitCandidate", "NumPySplitBackend"]
