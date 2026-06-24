# Backend Strategy

RepLeafGBM has **two distinct backend axes**. Conflating them causes design
mistakes, so they are named explicitly.

## Axis 1: compute backend (how kernels run)

The numeric kernels behind `backends/BaseSplitBackend` (histogram
accumulation, split scanning; later leaf fitting and prediction). The
contract is deliberately narrow — integer bin matrices, float
gradient/Hessian buffers, no Python objects — so it can be reimplemented
natively:

```text
backends/
  numpy_backend.py    # reference, always available
  rust_backend.py     # implemented (Phase 10): optional compiled extension
  cuda_backend.py     # implemented (experimental): GPU histogram via CuPy
```

The Rust kernels live in `native/` (pyo3 + maturin; `pip install ./native`)
and are selected automatically via `split_backend="auto"` when installed.
They mirror the NumPy backend's tie-breaking and accumulation-order
semantics: histograms are bitwise-identical and end-to-end predictions agree
to float noise (tested). Subsequent native kernels cover `leaf_linear_stats`
for embeddings up to 64 dims, scalar fused linear prediction, and
`partition_rows` for numeric and categorical row routing. The PR #30
`partition_rows` kernel is index-identical to the NumPy reference and reduced
medium/large multiclass Rust fit by 9-12%. Tree-growing logic
(`core/tree.py`) must never depend on backend internals.

The experimental CUDA backend (`split_backend="cuda"`, ADR 0005) is a third
compute backend: a CuPy `RawKernel` builds histograms on an NVIDIA GPU, caches
the binned matrix on device, and runs the large numeric split scan on device
with an adaptive host fallback for small scans. Categorical and multi-output
scans still use host logic. It is **explicit-only**
("auto" never selects it) and its parity is **allclose, not bitwise** — GPU
atomic-add summation order is not fixed, so histograms agree to float noise
(`rtol=1e-6` end-to-end) rather than being bitwise-identical, and are not
reproducible run-to-run. Because the dev box (macOS) and CI have no GPU, it is
validated through the Colab dev loop (`scripts/colab_gpu_test.sh`); see
`docs/cuda.md`.

## Axis 2: GBM backend (which boosting engine builds the routing)

The long-term goal is **model diversity**, not engineering convenience.
Different GBDT libraries carry different inductive biases:

- **LightGBM** — histogram training, leaf-wise growth, strong CPU
  performance, `linear_tree` support (closest relative of RepLeafGBM).
- **XGBoost** — DMatrix ecosystem, robust custom-objective support,
  multi-output / vector-leaf extensions.
- **CatBoost** — ordered boosting, strong native categorical handling,
  GPU/multi-GPU support. Deep integration is expected to be hardest; it is
  deliberately last.

Planned API (not implemented):

```python
model = RepLeafClassifier(
    gbm_backend="native",    # "native" | "lightgbm" | "xgboost" | "catboost"
    backend_mode="native",   # "native" | "external_model" | "router_extraction"
)
```

### Modes

**native** — RepLeafGBM's own tree construction, leaf fitting, save/load,
and (later) native acceleration. The only mode in v0.

**external_model** — train the external library as an independent base
model; use its predictions and/or leaf indices as features for
stacking/ensembling with RepLeafGBM models. **Implemented for LightGBM**
(v0.2, `repleafgbm.external`):

```python
from repleafgbm.external import (
    LightGBMExternalModel,   # base model + score / leaf-index extraction
    oof_predictions,         # generic K-fold OOF utility (no lgb dependency)
    augment_features,        # original features + score/leaf columns
)
```

The integration honors the guardrails: nothing in the native path imports
`repleafgbm.external`; lightgbm is an optional extra
(`pip install "repleafgbm[external]"`) checked at call time; the stacking
recipe (OOF scores for train rows) lives in `examples/stacking_lightgbm.py`.
`XGBoostExternalModel` (Phase 19) and `CatBoostExternalModel` (Phase 21)
provide the same duck-typed contract for XGBoost and CatBoost (dependencies
checked at call time; custom objectives via `xgb_params`, CatBoost-native
categoricals via `cb_params(cat_features=...)`), closing v0.3.

**router_extraction** — train a tree ensemble with the external library,
freeze its routing, and fit RepLeaf-style embedded leaf models on the fixed
routes. **Implemented for LightGBM regression and binary classification**
(Phases 3-4, ADR 0002), with replay-stage early stopping:
`extract_routes` maps LightGBM trees into native `Tree` arrays (per-node
missing direction included; exact prediction reproduction tested) and
`RouterExtractionRegressor` refits leaves by sequential replay, preserving
the stage-wise additive structure. Saved models use the native format and
do not depend on LightGBM. Experiments show embedded leaves improve
LightGBM's own routes by 2-12% RMSE
(experiments/results/router_extraction.md).

**hybrid ensemble** — same encoder family, different tree-building
algorithms, combined to maximize diversity. Builds on external_model
utilities (OOF predictions, stacking helpers).

## Guardrails

- Backend abstraction must not blur the v0 focus: native, small, readable,
  testable.
- No wrapper-only product: RepLeafGBM is not a thin shim over LightGBM.
- External integrations are optional dependencies; the native path never
  imports them.
- Any GBM backend must produce the same serialized ensemble representation
  (routing trees + leaf params) so prediction and serialization stay unified.
