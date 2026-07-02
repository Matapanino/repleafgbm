"""Device leaf-fit statistics (CUDA leaf ridge): parity + e2e quality tests.

Skipped unless CuPy + a usable NVIDIA GPU are present (Colab dev loop,
``scripts/colab_gpu_test.sh``). Parity is **allclose, never bitwise** (device
reduction order; ADR 0005), and end-to-end assertions are *quality-equivalence*
(|Δr2| bound), because near-tied LOO-gate verdicts — like near-tied splits —
can flip under CuPy's low-bit reduction differences.
"""

import pickle

import numpy as np
import pytest

from repleafgbm import RepLeafDataset, RepLeafRegressor
from repleafgbm.backends import CudaSplitBackend
from repleafgbm.core.leaf_models import AdaptiveLeafModel, EmbeddedLinearLeafModel

cp = pytest.importorskip("cupy", reason="CuPy not installed")
try:
    if cp.cuda.runtime.getDeviceCount() < 1:  # pragma: no cover - hardware gate
        pytest.skip("no CUDA device available", allow_module_level=True)
except Exception as exc:  # pragma: no cover - driver/runtime missing
    pytest.skip(f"CUDA runtime unavailable: {exc}", allow_module_level=True)


def _leaf_inputs(n=4000, d=64, n_leaves=8, seed=0):
    rng = np.random.default_rng(seed)
    Z = np.ascontiguousarray(rng.normal(size=(n, d)))
    resid = Z @ rng.normal(size=d) + rng.normal(0, 0.2, n)
    grad, hess = -resid, np.abs(rng.normal(1.0, 0.3, n)) + 0.1
    sizes = rng.multinomial(n, rng.dirichlet(np.full(n_leaves, 0.7)))
    order = rng.permutation(n).astype(np.int64)
    offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    linear = np.flatnonzero(sizes >= max(20, d + 2)).astype(np.int64)
    return Z, grad, hess, order, offsets, linear


def _host_reference(Z, grad, hess, order, offsets, linear):
    n_leaves = len(offsets) - 1
    d = Z.shape[1]
    seg = np.repeat(np.arange(n_leaves), np.diff(offsets))
    Zs, gs, hs = Z[order], grad[order], hess[order]
    g_sum = np.bincount(seg, weights=gs, minlength=n_leaves)
    h_sum = np.bincount(seg, weights=hs, minlength=n_leaves)
    k = linear.size
    s_hz = np.empty((k, d))
    A = np.empty((k, d, d))
    gz = np.empty((k, d))
    zmn = np.empty((k, d))
    zmx = np.empty((k, d))
    hZs = Zs * hs[:, None]
    for j, i in enumerate(linear):
        sl = slice(offsets[i], offsets[i + 1])
        s_hz[j] = hZs[sl].sum(axis=0)
        A[j] = Zs[sl].T @ hZs[sl]
        gz[j] = gs[sl] @ Zs[sl]
        zmn[j] = Zs[sl].min(axis=0)
        zmx[j] = Zs[sl].max(axis=0)
    return g_sum, h_sum, s_hz, A, gz, zmn, zmx


@pytest.mark.parametrize("d", [16, 200])
def test_leaf_fit_stats_parity_f64(d):
    Z, grad, hess, order, offsets, linear = _leaf_inputs(d=d)
    backend = CudaSplitBackend()
    got = backend.leaf_fit_stats(Z, grad, hess, order, offsets, linear)
    want = _host_reference(Z, grad, hess, order, offsets, linear)
    # Device bincount/scatter_add reductions use atomics, so the summation
    # order differs from NumPy's: sums with cancellation carry ~n*eps*max|term|
    # absolute noise (~1e-12 here). 1e-9 keeps real math errors detectable
    # while not asserting a reduction order ADR 0005 explicitly disclaims.
    for g, w in zip(got, want):
        np.testing.assert_allclose(g, w, rtol=1e-9, atol=1e-9)


def test_leaf_fit_stats_parity_f32_gram():
    Z, grad, hess, order, offsets, linear = _leaf_inputs(d=200, seed=1)
    backend = CudaSplitBackend()
    got = backend.leaf_fit_stats(
        Z, grad, hess, order, offsets, linear, use_f32=True
    )
    want = _host_reference(Z, grad, hess, order, offsets, linear)
    # f32 accumulation on the two large reductions only (A and gz): loose tol.
    names = ["g_sum", "h_sum", "s_hz", "A", "gz", "z_min", "z_max"]
    for name, g, w in zip(names, got, want):
        tol = 1e-3 if name in ("A", "gz") else 1e-9
        np.testing.assert_allclose(g, w, rtol=tol, atol=tol, err_msg=name)


def test_z_uploaded_once_per_fit():
    Z, grad, hess, order, offsets, linear = _leaf_inputs()
    backend = CudaSplitBackend()
    backend.leaf_fit_stats(Z, grad, hess, order, offsets, linear)
    backend.leaf_fit_stats(Z, grad, hess, order, offsets, linear)
    stats = backend.get_transfer_stats()
    assert stats["z_uploads"] == 1  # identity-cached across trees
    assert stats["n_leaf_fits"] == 2


def _quality_data(seed=0):
    rng = np.random.default_rng(seed)
    n = 6000
    X = rng.normal(size=(n, 40))
    y = X @ rng.normal(size=40) + np.sin(2 * X[:, 0]) + rng.normal(0, 0.3, n)
    return X[:4000], y[:4000], X[4000:], y[4000:]


@pytest.mark.parametrize("leaf_model", ["embedded_linear", "adaptive"])
def test_e2e_quality_equivalence_forced_device(monkeypatch, leaf_model):
    """Forced-device leaf fit must be quality-equivalent to the host path.

    Near-tied LOO-gate verdicts can flip under device reductions (the scalar
    batched-scan gotcha), so this asserts |delta r2| < 5e-3, not prediction
    equality.
    """
    from sklearn.metrics import r2_score

    Xtr, ytr, Xte, yte = _quality_data()

    def fit_r2(device: bool):
        monkeypatch.setenv("REPLEAFGBM_CUDA_LEAF_FIT", "1" if device else "0")
        monkeypatch.setenv("REPLEAFGBM_CUDA_LEAF_FIT_MIN_CELLS", "0")
        model = RepLeafRegressor(
            n_estimators=30,
            num_leaves=15,
            leaf_model=leaf_model,
            encoder="identity",
            split_backend="cuda",
            random_state=0,
        )
        model.fit(Xtr, ytr)
        return r2_score(yte, model.predict(Xte))

    r2_host = fit_r2(device=False)
    r2_dev = fit_r2(device=True)
    assert abs(r2_dev - r2_host) < 5e-3


def test_kill_switch_disables_capability(monkeypatch):
    monkeypatch.setenv("REPLEAFGBM_CUDA_LEAF_FIT", "0")
    backend = CudaSplitBackend()
    assert backend.supports_leaf_fit is False


def test_min_cells_env_override(monkeypatch):
    monkeypatch.setenv("REPLEAFGBM_CUDA_LEAF_FIT_MIN_CELLS", "123")
    backend = CudaSplitBackend()
    assert backend.leaf_fit_min_cells == 123
    monkeypatch.setenv("REPLEAFGBM_CUDA_LEAF_FIT_MIN_CELLS", "junk")
    with pytest.warns(RuntimeWarning):
        backend2 = CudaSplitBackend()
    assert backend2.leaf_fit_min_cells > 0


def test_cuda_leaf_fit_model_stays_picklable(monkeypatch):
    monkeypatch.setenv("REPLEAFGBM_CUDA_LEAF_FIT_MIN_CELLS", "0")
    Xtr, ytr, Xte, _ = _quality_data(seed=1)
    model = RepLeafRegressor(
        n_estimators=5,
        num_leaves=8,
        leaf_model="embedded_linear",
        encoder="identity",
        split_backend="cuda",
        random_state=0,
    )
    train = RepLeafDataset(Xtr, ytr)
    model.fit(train)
    clone = pickle.loads(pickle.dumps(model))
    np.testing.assert_allclose(clone.predict(Xte), model.predict(Xte))


def test_direct_leaf_model_dispatch_on_device():
    """The seam itself, on a real device backend: EmbeddedLinear + Adaptive
    produce host-equivalent LeafValues when handed the CUDA backend."""
    Z, grad, hess, order, offsets, linear = _leaf_inputs(d=32, seed=2)
    sizes = np.diff(offsets)
    rows = [order[offsets[i]:offsets[i + 1]] for i in range(len(sizes))]
    for cls in (EmbeddedLinearLeafModel, AdaptiveLeafModel):
        model = cls(l2=1.0, min_samples_linear=20)
        lv_host = model.fit_leaves(rows, grad, hess, Z)
        backend = CudaSplitBackend()
        backend.leaf_fit_min_cells = 0
        model.fit_backend = backend
        lv_dev = model.fit_leaves(rows, grad, hess, Z)
        np.testing.assert_allclose(lv_dev.bias, lv_host.bias, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(
            lv_dev.weights, lv_host.weights, rtol=1e-6, atol=1e-8
        )
