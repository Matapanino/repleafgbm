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
  numpy_backend.py    # v0, implemented
  cpp_backend.py      # planned (v2)
  rust_backend.py     # planned (v2)
  cuda_backend.py     # planned (v3)
```

v0 rule: NumPy only. Tree-growing logic (`core/tree.py`) must never depend on
backend internals.

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
XGBoost/CatBoost external models are still planned (v0.3).

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
