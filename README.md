# RepLeafGBM

![CI](https://github.com/Matapanino/repleafgbm/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)

**Representation-enhanced Leaf Gradient Boosting Machine** — gradient
boosting that routes on raw features and predicts with small linear models
over learned representations inside each leaf.

> RepLeafGBM is not a neural network inside a tree, nor a tree over
> embeddings. It is a boosted ensemble of raw-feature routers with
> representation-conditioned local predictors.

⚠️ **This is experimental research software.** APIs, file formats, and
behavior will change without notice. Do not use it in production.

**Highlights from the experiment log** (synthetic data; see
[docs/audit_v0.md](docs/audit_v0.md) and `experiments/results/`):

- Embedded-linear leaves beat constant leaves, LightGBM, and sklearn
  HistGradientBoosting on signals with smooth structure inside regimes.
- Refitting *LightGBM's own* routes with representation-conditioned leaves
  (router_extraction) improves LightGBM by 2-12% RMSE — isolating the leaf
  contribution from split quality.

## Motivation

GBDTs dominate tabular ML because axis-aligned splits on raw features handle
discontinuities, interactions, and messy data extremely well. Tabular deep
learning has shown that numerical feature embeddings (PLR, periodic) capture
smooth nonlinear structure that constant-leaf trees approximate only with
many splits.

RepLeafGBM combines both with a deliberately asymmetric design:

- **Routing** — every split is on a raw feature, exactly like a normal GBDT.
  Trees do what trees are good at: finding discontinuous boundaries and
  partitioning the space into local regions.
- **Leaf output** — each leaf holds a small ridge-regularized linear model
  over a learned representation `z_theta(x)` instead of a constant. The
  embedding does what it is good at: smooth interpolation within a local
  region.

```text
f_t(x) = b_{t, l_t(x_raw)} + w_{t, l_t(x_raw)}^T z_theta(x)
F_T(x) = F_0(x) + sum_t eta * f_t(x)
```

## What makes it different

- Embeddings are never used for splitting (interpretable raw-feature routing,
  no histogram blow-up, no curse of dimensionality in split search).
- The encoder is frozen during boosting in v0, preserving the stage-wise
  additive structure of gradient boosting.
- Leaf models fall back to constants when leaves are too small, so the model
  degrades gracefully toward a classic GBDT.

## Minimal example

```python
import numpy as np
from repleafgbm import RepLeafRegressor
from repleafgbm.data import RepLeafDataset

rng = np.random.default_rng(0)
X = rng.normal(size=(500, 4))
y = np.where(X[:, 0] > 0, 3.0, -2.0) + 2.0 * X[:, 1] + rng.normal(0, 0.1, 500)

model = RepLeafRegressor(
    n_estimators=50,
    learning_rate=0.1,
    num_leaves=8,
    leaf_model="embedded_linear",   # or "constant", "raw_linear"
    encoder="plr",                  # or "identity"
    max_leaf_emb_dim=16,
    random_state=42,
)
model.fit(X, y)
pred = model.predict(X)

model.save_model("repleaf_model")
loaded = RepLeafRegressor.load_model("repleaf_model")
```

pandas DataFrames with categorical columns are supported through the dataset
API:

```python
train_data = RepLeafDataset(df_train, y_train, categorical_features=["city"])
model.fit(train_data, eval_set=[RepLeafDataset(df_valid, y_valid,
                                               metadata=train_data.metadata)])
```

## Current status (v0)

Implemented:

- Native NumPy backend: histogram-based split search with sibling-histogram
  subtraction, leaf-wise tree growth — plus optional Rust kernels
  (`pip install ./native`, auto-detected; ~5.8x faster constant-leaf
  training, parity-tested against the NumPy reference)
- `leaf_model`: `"constant"`, `"embedded_linear"`, `"raw_linear"`
- Encoders: `"identity"`, `"plr"` (simplified piecewise-linear + linear
  term), `"periodic"` (PBLD-style frozen sinusoidal features), `"cross"`
  (residual-correlated pairwise products), and learned `"torch_periodic"` /
  `"torch_plr"` / `"torch_mlp"` (optional `[torch]` extra; pretrained on
  the initial residual then frozen — torch is needed only at fit time).
  `identity` is the evidence-backed default on real tabular data; the
  others are specialists for known smooth/oscillatory or interaction
  structure (see docs for guidance). Random projection down to
  `max_leaf_emb_dim` as an emergency cap
- Regression (squared error, plus `objective="huber"` / `"quantile"` /
  `"poisson"` — parameterized instances like `Quantile(alpha=0.9)` work
  too), binary classification (logistic), and multiclass classification
  (softmax, one tree per class per round — automatic at 3+ classes)
- Multi-output regression via shared-routing **vector leaves** (pass a 2-D
  `y`; one tree per round whose leaves emit a vector — squared-error only),
  and `label_smoothing` for classification
- Early stopping (`early_stopping_rounds`, `best_iteration_`, prediction at
  the best iteration) and eval metrics: rmse, mae, logloss, multi_logloss,
  auc, accuracy, or any user-supplied callable (`repleafgbm.make_metric`)
- Feature importance (`feature_importances_`, gain or split count)
- `RepLeafDataset` with pandas/categorical support (native subset splits,
  `pandas.Categorical` order fidelity, opt-in frequency encoding) and
  embedding caching
- Directory-based `save_model` / `load_model` with schema validation and a
  human-readable `summary.txt` (`model.summary()`)
- `repleafgbm.external`: LightGBM, XGBoost, and CatBoost as external base
  models (scores + leaf indices, optional native early stopping), generic
  OOF utility, stacking feature builders, and `RouterExtractionRegressor` /
  `RouterExtractionClassifier` — LightGBM routing with RepLeaf leaf models
  refit on the frozen routes, with replay-stage early stopping
  (`pip install "repleafgbm[external]"`)
- pytest suite, runnable examples, and an `experiments/` research scaffold

Not implemented (see [docs/roadmap.md](docs/roadmap.md)): encoder updates
during boosting, GPU/distributed training.

## Installation (development)

```bash
git clone <repo-url> && cd repleafgbm
pip install -e ".[dev]"
```

Or without installing, run everything from the repo root with
`PYTHONPATH=src`.

## Running tests and examples

```bash
bash scripts/check.sh               # lint + tests + all examples
python -m pytest tests/ -q          # PYTHONPATH=src if not installed
python examples/regression_basic.py
python examples/binary_classification_basic.py
python examples/multiclass_classification_basic.py
python examples/dataset_api_basic.py
```

## Benchmarks

Small synthetic benchmarks track progress across development (they are not
performance claims):

```bash
python benchmarks/benchmark_synthetic_regression.py [--quick]
python benchmarks/benchmark_synthetic_binary.py [--quick]
```

LightGBM / XGBoost / CatBoost are included automatically when installed
(`pip install -e ".[bench]"`). The latest snapshot and analysis live in
[docs/audit_v0.md](docs/audit_v0.md).

## Documentation

- [docs/design.md](docs/design.md) — architecture and responsibilities
- [docs/math.md](docs/math.md) — boosting formulation and leaf fitting
- [docs/roadmap.md](docs/roadmap.md) — implemented vs planned
- [docs/backend_strategy.md](docs/backend_strategy.md) — multi-backend plans
- [docs/serialization.md](docs/serialization.md) — model format
- [docs/dataset_and_memory.md](docs/dataset_and_memory.md) — memory strategy
- [docs/categorical_features.md](docs/categorical_features.md) — categorical policy
- [docs/audit_v0.md](docs/audit_v0.md) — Phase 0.5 audit, fixes, benchmark snapshot

## License

MIT
