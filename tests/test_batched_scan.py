"""Node-batched split scan (`find_best_split_batched`) — host parity.

Stage 1 of the node-batched CUDA scan (docs/proposals/node-batched-split-scan.md):
the backend contract + the level-synchronous depthwise grower. The CUDA kernel is
Stage 2 (Colab-gated). Here we prove, with NO GPU, that:
  1. the default `find_best_split_batched` is byte-identical to the per-node loop
     (numpy AND rust, numeric + categorical + tie-break), and
  2. forcing the batched grower path produces a **bitwise-identical tree** to the
     per-node FIFO depthwise grower (the guard on the grower refactor).
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from repleafgbm import RepLeafClassifier, RepLeafRegressor
from repleafgbm.backends.numpy_backend import NumPySplitBackend
from repleafgbm.backends.rust_backend import RustSplitBackend

# RustSplitBackend imports fine without the native ext (the class is pure Python);
# it only raises at INSTANTIATION when repleafgbm_native is missing. Gate the rust
# parametrization on the extension actually being importable, so the no-native CI
# `test` lane skips it (mirrors tests/test_rust_backend.py's importorskip).
_BACKENDS = [NumPySplitBackend]
if importlib.util.find_spec("repleafgbm_native") is not None:
    _BACKENDS.append(RustSplitBackend)


def _node_hists(backend, *, seed=0, n_nodes=5):
    """A few valid node histograms + the scan params (one categorical feature)."""
    rng = np.random.default_rng(seed)
    n_rows, n_features, n_bins = 500, 6, 8
    nbpf = np.full(n_features, n_bins, dtype=np.int64)
    n_bins_max = n_bins + 1
    binned = rng.integers(0, n_bins, size=(n_rows, n_features)).astype(np.uint16)
    grad = rng.normal(size=n_rows)
    hess = rng.random(n_rows) + 0.5
    cat_mask = np.zeros(n_features, dtype=bool)
    cat_mask[5] = True  # exercise the categorical subset scan in the batch
    hists = [
        backend.build_histograms(
            binned, rng.choice(n_rows, size=300, replace=False), grad, hess, n_bins_max
        )
        for _ in range(n_nodes)
    ]
    return hists, nbpf, cat_mask


def _same_split(a, b):
    if a is None or b is None:
        return a is None and b is None
    if (a.feature, a.bin, a.gain, a.n_left, a.n_right) != (
        b.feature, b.bin, b.gain, b.n_left, b.n_right
    ):
        return False
    if (a.left_categories is None) != (b.left_categories is None):
        return False
    return a.left_categories is None or np.array_equal(
        a.left_categories, b.left_categories
    )


@pytest.mark.parametrize("backend_cls", _BACKENDS)
def test_batched_equals_per_node_loop(backend_cls):
    """Default find_best_split_batched == [find_best_split per node], bitwise."""
    backend = backend_cls()
    hists, nbpf, cat_mask = _node_hists(backend)
    loop = [
        backend.find_best_split(h, nbpf, 5, 1.0, cat_mask) for h in hists
    ]
    batched = backend.find_best_split_batched(hists, nbpf, 5, 1.0, cat_mask)
    assert len(batched) == len(loop)
    assert all(_same_split(a, b) for a, b in zip(batched, loop))


def test_default_supports_batched_scan_is_false():
    assert NumPySplitBackend().supports_batched_scan is False


def _trees_identical(a, b) -> bool:
    if len(a) != len(b):
        return False
    for ta, tb in zip(a, b):
        if not (
            np.array_equal(ta.feature, tb.feature)
            and np.array_equal(ta.left, tb.left)
            and np.array_equal(ta.right, tb.right)
            and np.array_equal(ta.leaf_id, tb.leaf_id)
            and np.array_equal(
                np.nan_to_num(ta.threshold, nan=0.0),
                np.nan_to_num(tb.threshold, nan=0.0),
            )
        ):
            return False
    return True


@pytest.mark.parametrize("backend_cls", _BACKENDS)
@pytest.mark.parametrize("est_data", ["regression", "binary"])
def test_depthwise_batched_grower_is_bitwise_identical(
    backend_cls, est_data, monkeypatch
):
    """The level-synchronous batched grower == the per-node FIFO depthwise tree.

    Force ``supports_batched_scan`` on the host backend so the grower takes the
    batched path (whose backend scan is the per-node loop) and assert the trees
    match the normal depthwise grower byte-for-byte.
    """
    rng = np.random.default_rng(1)
    X = rng.normal(size=(800, 8))
    backend_name = "numpy" if backend_cls is NumPySplitBackend else "rust"
    common = dict(
        grow_policy="depthwise", max_depth=4, num_leaves=31, n_estimators=8,
        leaf_model="constant", split_backend=backend_name, random_state=0,
    )
    if est_data == "regression":
        y = 2 * X[:, 0] + np.sin(X[:, 1]) + rng.normal(scale=0.1, size=800)
        make = lambda: RepLeafRegressor(**common)  # noqa: E731
    else:
        y = (X[:, 0] + X[:, 2] > 0).astype(int)
        make = lambda: RepLeafClassifier(**common)  # noqa: E731

    per_node = make().fit(X, y)
    monkeypatch.setattr(backend_cls, "supports_batched_scan", True)
    batched = make().fit(X, y)
    assert _trees_identical(per_node.booster_.trees_, batched.booster_.trees_)
    # predictions identical too (sanity on the leaf side)
    assert np.array_equal(per_node.predict(X), batched.predict(X))
