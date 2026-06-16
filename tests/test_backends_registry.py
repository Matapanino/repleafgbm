"""Always-run tests for the split-backend registry.

These do not require the optional Rust / CuPy backends, so they exercise the
``make_split_backend`` dispatch on every lane (including CPU-only CI and the
macOS dev box).
"""

import pytest

from repleafgbm.backends import (
    BaseSplitBackend,
    CudaSplitBackend,
    NumPySplitBackend,
    make_split_backend,
)


def test_numpy_is_always_available():
    assert isinstance(make_split_backend("numpy"), NumPySplitBackend)


def test_unknown_backend_raises_valueerror():
    with pytest.raises(ValueError, match="split_backend"):
        make_split_backend("metal")


def test_auto_never_selects_cuda():
    # "auto" resolves to rust-or-numpy; the GPU backend is explicit-only.
    backend = make_split_backend("auto")
    assert isinstance(backend, BaseSplitBackend)
    assert not isinstance(backend, CudaSplitBackend)


def test_cuda_requires_gpu_or_raises_importerror():
    """On a GPU box ``make_split_backend("cuda")`` returns a CudaSplitBackend;
    everywhere else it raises a clear ImportError (never ValueError)."""
    try:
        backend = make_split_backend("cuda")
    except ImportError:
        pass  # expected on CPU-only machines (no CuPy / no GPU)
    else:
        assert isinstance(backend, CudaSplitBackend)
