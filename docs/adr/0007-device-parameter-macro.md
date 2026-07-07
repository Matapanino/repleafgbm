# ADR 0007: `device` parameter as a thin macro over existing GPU knobs

- Status: **accepted — implemented** (2026-07-07). `device ∈ {"cpu", "cuda"}`
  on `RepLeafRegressor` / `RepLeafClassifier`; resolution logic covered by
  CPU-only tests (`tests/test_device_param.py`); GPU execution validated on
  the Colab T4 loop before merge.
- Date: 2026-07-07
- Depends on: ADR 0005 (CUDA backend via CuPy), the `fit_backend` leaf-fit
  seam, `encoders/torch_encoders.py` device-aware pretraining.

## Context

By v1.10 every GPU-relevant capability had shipped as its own opt-in knob:

- `split_backend="cuda"` — GPU histograms, node-batched split scan, and (via
  the booster's automatic `leaf_model.fit_backend` wiring) device leaf-fit
  statistics.
- `encoder_params={"device": "cuda"}` — GPU pretraining for the learned
  `torch_*` encoders.

The roadmap's v3 line "GPU training (`device="cuda"`)" therefore no longer
names missing compute — it names a missing *front door*. Users coming from
XGBoost (`device="cuda"`) / LightGBM (`device_type="gpu"`) expect one switch,
not two knobs on different objects.

## Decision

Add `device: str = "cpu"` to the shared estimator `__init__` as a **thin
macro** that only rewires existing knobs at fit time:

1. `device="cuda"` resolves `split_backend="auto"` → `"cuda"`.
2. `device="cuda"` fills in `device="cuda"` for a **named** `torch_*` encoder
   when `encoder_params` does not set one.

Rules that bound the macro:

- **Explicit wins.** A user-set `split_backend` or `encoder_params["device"]`
  is never overridden; `device="cuda", split_backend="numpy"` is a valid
  combination (CPU splits + GPU encoder pretraining). Encoder *instances* are
  user-constructed and never mutated.
- **No `"auto"` device.** GPU use stays an explicit opt-in everywhere
  (consistent with `split_backend="auto"` never selecting CUDA): the GPU path
  is allclose-not-bitwise vs the CPU backends, so silent hardware-dependent
  selection would break same-seed reproducibility across machines.
- **Hard failure off-GPU.** When the macro selects the CUDA backend and
  CuPy/GPU are missing, fit raises the existing clear `ImportError` — no
  silent fallback.
- **Prediction is unaffected** (CPU, native `apply_tree` router).
- Values outside `{"cpu", "cuda"}` raise `ValueError` at fit, per sklearn
  validate-in-fit convention; the parameter is stored verbatim in `__init__`.

## Consequences

- One-switch GPU UX with zero new execution paths: nothing new to
  parity-test beyond the resolution logic, which is CPU-testable.
- Two knobs can now express the same configuration; docs/cuda.md presents
  `device="cuda"` as the front door and `split_backend` as the underlying
  knob.
- Router-extraction estimators do not expose `device` (their signature has no
  `split_backend` either); the shared encoder-building path treats a missing
  attribute as `"cpu"`.
- Serialization: `device` rides `get_params()` into the model config; loading
  filters config keys by signature in both directions, so old↔new model
  directories stay compatible with no format bump.
- Scale-out (`multi_gpu`, `distributed_strategy`, out-of-core) remains
  **deferred pending demand evidence** — see docs/roadmap.md v3; this ADR
  deliberately claims only the single-GPU front door.
