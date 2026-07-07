"""Always-run tests for the estimator-level ``device`` macro (ADR 0007).

``device="cuda"`` only rewires existing knobs — ``split_backend`` (when left
on "auto") and a named ``torch_*`` encoder's pretraining device (when
``encoder_params`` does not pin one) — so the resolution logic is fully
testable on CPU-only lanes. GPU execution itself is covered by the Colab loop
(``scripts/colab_gpu_test.sh``), like the rest of the CUDA path.
"""

from importlib.util import find_spec

import pytest

import repleafgbm.sklearn as sklearn_module
from repleafgbm import RepLeafClassifier, RepLeafRegressor


# --------------------------------------------------------------------- #
# split_backend resolution
# --------------------------------------------------------------------- #
def test_default_device_is_cpu_and_inert():
    est = RepLeafRegressor()
    assert est.device == "cpu"
    assert est._resolved_split_backend() == "auto"


def test_device_cuda_upgrades_auto_split_backend():
    est = RepLeafRegressor(device="cuda")
    assert est._resolved_split_backend() == "cuda"


@pytest.mark.parametrize("explicit", ["numpy", "rust"])
def test_explicit_split_backend_wins_over_device(explicit):
    est = RepLeafRegressor(device="cuda", split_backend=explicit)
    assert est._resolved_split_backend() == explicit


def test_invalid_device_raises_before_any_work(regression_data):
    X_train, y_train, _, _ = regression_data
    est = RepLeafRegressor(device="gpu", n_estimators=2)
    with pytest.raises(ValueError, match="device"):
        est.fit(X_train, y_train)


@pytest.mark.skipif(
    find_spec("cupy") is not None,
    reason="CuPy installed: device='cuda' would genuinely train",
)
def test_device_cuda_without_cupy_raises_importerror(regression_data):
    X_train, y_train, _, _ = regression_data
    est = RepLeafRegressor(device="cuda", n_estimators=2)
    with pytest.raises(ImportError, match="(?i)cupy|gpu"):
        est.fit(X_train, y_train)


def test_classifier_shares_the_macro():
    est = RepLeafClassifier(device="cuda")
    assert est._resolved_split_backend() == "cuda"


# --------------------------------------------------------------------- #
# torch encoder device injection
# --------------------------------------------------------------------- #
@pytest.fixture
def capture_make_encoder(monkeypatch):
    """Record the kwargs the estimator passes to ``make_encoder`` and return
    a CPU-safe identity encoder so fit() completes without torch or a GPU."""
    real = sklearn_module.make_encoder
    calls: dict = {}

    def wrapper(name, **kwargs):
        calls["name"] = name
        calls["kwargs"] = dict(kwargs)
        return real("identity")

    monkeypatch.setattr(sklearn_module, "make_encoder", wrapper)
    return calls


def _fit_cpu(regression_data, **params):
    X_train, y_train, _, _ = regression_data
    est = RepLeafRegressor(
        n_estimators=2,
        leaf_model="embedded_linear",
        split_backend="numpy",  # explicit CPU split path, no CuPy needed
        **params,
    )
    est.fit(X_train, y_train)
    return est


def test_device_cuda_injects_torch_encoder_device(
    regression_data, capture_make_encoder
):
    _fit_cpu(regression_data, device="cuda", encoder="torch_plr")
    assert capture_make_encoder["name"] == "torch_plr"
    assert capture_make_encoder["kwargs"]["device"] == "cuda"


def test_encoder_params_device_wins_over_macro(
    regression_data, capture_make_encoder
):
    _fit_cpu(
        regression_data,
        device="cuda",
        encoder="torch_plr",
        encoder_params={"device": "cpu"},
    )
    assert capture_make_encoder["kwargs"]["device"] == "cpu"


def test_device_cpu_injects_nothing(regression_data, capture_make_encoder):
    _fit_cpu(regression_data, encoder="torch_plr")
    assert "device" not in capture_make_encoder["kwargs"]


def test_non_torch_encoder_never_gets_device(
    regression_data, capture_make_encoder
):
    # plr does not accept a device kwarg; the macro must not inject one.
    _fit_cpu(regression_data, device="cuda", encoder="plr")
    assert "device" not in capture_make_encoder["kwargs"]


# --------------------------------------------------------------------- #
# persistence
# --------------------------------------------------------------------- #
def test_device_survives_save_load_round_trip(regression_data, tmp_path):
    est = _fit_cpu(regression_data, device="cuda", encoder="identity")
    est.save_model(tmp_path / "model")
    loaded = RepLeafRegressor.load_model(tmp_path / "model")
    assert loaded.get_params()["device"] == "cuda"
