# RepLeafGBM

![CI](https://github.com/Matapanino/repleafgbm/actions/workflows/ci.yml/badge.svg)
![PyPI](https://img.shields.io/pypi/v/repleafgbm.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)

**Representation-enhanced Leaf Gradient Boosting Machine** — gradient
boosting that routes on raw features and predicts with small linear models
over learned representations inside each leaf.

> RepLeafGBM is not a neural network inside a tree, nor a tree over
> embeddings. It is a boosted ensemble of raw-feature routers with
> representation-conditioned local predictors.

## Paper

📄 **[RepLeafGBM: Gradient Boosting with Raw-Feature Routing and
Representation-Conditioned Leaves](docs/paper/repleafgbm-algorithm.pdf)** — Masaya
Kawamata ([PDF](docs/paper/repleafgbm-algorithm.pdf), [LaTeX
source](docs/paper/repleafgbm-algorithm.tex)).

A technical report covering the method, the frozen-encoder correctness argument,
and an honest empirical study: a fair same-HPO-budget leaderboard on the Grinsztajn
tabular suites (RepLeafGBM is competitive mid-pack), plus niche studies where the
representation-conditioned leaf and scale-consistent robust multi-output objectives
provide a defensible edge. All numbers are reproducible with the benchmark and
experiment harness in this repository.

> **Preprint draft** — not yet peer-reviewed or posted to an external repository.

## Project status & development notes

- **GPU acceleration is an active work in progress.** The CUDA split backend
  (`split_backend="cuda"`) accelerates wide / multi-output histograms, but
  performance is still being optimized and is validated only on a limited set of
  GPUs (e.g. NVIDIA T4). For most CPU workloads the Rust backend
  (`repleafgbm-native`) is the faster, more mature path — treat GPU speed as
  evolving, not final.
- **Built with Claude Code.** RepLeafGBM's implementation and architecture were
  developed with heavy use of [Claude Code](https://claude.com/claude-code)
  (both coding and architecture design).

## How it works

GBDTs dominate tabular ML because axis-aligned splits on raw features handle
discontinuities, interactions, and messy data extremely well. Tabular deep
learning has shown that numerical feature embeddings (PLR, periodic) capture
smooth nonlinear structure that constant-leaf trees approximate only with many
splits. RepLeafGBM combines both with a deliberately **asymmetric** design:

- **Routing** — every split is on a raw feature, exactly like a normal GBDT.
  Trees find the discontinuous boundaries and partition the space into local
  regions.
- **Leaf output** — each leaf holds a small ridge-regularized **linear model
  over a learned representation** `z_theta(x)` instead of a constant. The
  embedding does the smooth interpolation within each region.

```text
f_t(x) = b + w^T z_theta(x)            # one tree: route on x_raw, predict in the leaf
F_T(x) = F_0(x) + sum_t eta * f_t(x)   # boosted sum
```

Three properties fall out of this design:

- Embeddings are **never** used for splitting → interpretable raw-feature
  routing, no split-histogram blow-up, no curse of dimensionality in the search.
- The encoder is **frozen** during boosting (v0), preserving the stage-wise
  additive structure of gradient boosting.
- Leaf models **fall back to a constant** when a leaf is too small, so the model
  degrades gracefully toward a classic GBDT.

See [docs/math.md](docs/math.md) for the full formulation and leaf fitting.

## Installation

```bash
pip install repleafgbm                 # core (numpy, pandas, scikit-learn)
pip install repleafgbm-native          # + optional Rust split/leaf kernels (auto-detected)
pip install "repleafgbm[external]"     # + LightGBM external_model / router_extraction
pip install "repleafgbm[bench]"        # + XGBoost / CatBoost for benchmarks
pip install "repleafgbm[torch]"        # + learned torch encoders
```

`repleafgbm` ships type information (PEP 561). `repleafgbm-native` is a separate
package of prebuilt Linux/macOS/Windows wheels; once installed the Rust backend
is selected automatically (`split_backend="auto"`), giving ~5.8x faster
constant-leaf training while staying parity-tested against the NumPy reference.

## Quickstart

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
    encoder="plr",                  # or "identity", "periodic", "cross"
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

## API overview

The public API is scikit-learn compatible (`fit` / `predict` / `predict_proba`,
`get_params` / `set_params`).

**Main classes**

| Class | Use |
|---|---|
| `RepLeafRegressor` | Regression (squared error, plus `huber` / `quantile` / `poisson`); multi-output via vector leaves. |
| `RepLeafClassifier` | Binary (logistic) and multiclass (softmax) — chosen automatically from the labels. Has `predict_proba`. |
| `RepLeafDataset` | pandas / categorical inputs, eval sets, and embedding caching. |

**Key parameters** (constructor; same for both estimators)

| Parameter | Default | Meaning |
|---|---|---|
| `n_estimators` | `100` | Boosting rounds (trees). |
| `learning_rate` | `0.1` | Shrinkage per round. |
| `num_leaves` | `31` | Max leaves per tree (leaf-wise growth). |
| `min_samples_leaf` | `20` | Minimum rows per leaf. |
| `leaf_model` | `"embedded_linear"` | Leaf predictor (see below). |
| `encoder` | `"identity"` | Representation `z_theta(x)` (see below). |
| `max_leaf_emb_dim` | `64` | Cap on embedding dimension (random projection above it). |
| `l2_leaf` | `1.0` | Ridge penalty for leaf models. |
| `early_stopping_rounds` | `None` | Stop when an `eval_set` metric plateaus. |
| `random_state` | `42` | Seed; same seed ⇒ same model. |

**`leaf_model`**

| Value | Leaf predicts |
|---|---|
| `"constant"` | A constant (classic GBDT leaf). |
| `"embedded_linear"` | Ridge linear model over `z_theta(x)` (default). |
| `"raw_linear"` | Ridge linear model over the raw features. |
| `"adaptive"` | Per-leaf weighted-LOO gate: keeps the `embedded_linear` leaf only where it beats a constant, else falls back. |

**`encoder`**

| Value | Representation |
|---|---|
| `"identity"` | Standardized raw features. Evidence-backed default on real tabular data. |
| `"plr"` | Piecewise-linear + linear term. |
| `"periodic"` | Frozen sinusoidal (PBLD-style) features. |
| `"cross"` | Residual-correlated pairwise products (interactions). |
| `"torch_periodic"` / `"torch_plr"` / `"torch_periodic_plr"` / `"torch_mlp"` | Learned encoders (`[torch]` extra; supervised-pretrained on the initial residual then frozen — torch is needed only at fit time, not at predict). |

API stability follows [Semantic Versioning](https://semver.org) from 1.0.0;
exactly what is covered vs. experimental is in
[docs/adr/0003-api-stability.md](docs/adr/0003-api-stability.md).

## Features

| Area | Support |
|---|---|
| Backends | NumPy reference (histogram split search, leaf-wise growth) + optional Rust kernels (`repleafgbm-native`, ~5.8x faster, parity-tested). |
| Tasks | Regression, binary & multiclass classification, multi-output regression (vector leaves). |
| Objectives | Squared error, `huber`, `quantile`, `poisson`, logistic, softmax (parameterized instances like `Quantile(alpha=0.9)` too). |
| Leaf models | `constant`, `embedded_linear`, `raw_linear`, `adaptive`. |
| Encoders | `identity`, `plr`, `periodic`, `cross`, learned `torch_*`. |
| Training | Early stopping, eval metrics (rmse, mae, logloss, multi_logloss, auc, accuracy, or a custom callable via `make_metric`), feature importances, sample weights, `class_weight`, `label_smoothing`. |
| Data | `RepLeafDataset` with pandas/categorical (native subset splits) and embedding caching. |
| Persistence | Directory-based `save_model` / `load_model` with schema validation and a human-readable `summary()`. |
| External | LightGBM / XGBoost / CatBoost as base models, OOF + stacking utilities, and `RouterExtraction{Regressor,Classifier}` (`[external]` extra). |

Full implemented-vs-planned status (and what is intentionally **not** done yet —
encoder updates during boosting, GPU / distributed training) is in
[docs/roadmap.md](docs/roadmap.md).

## Benchmarks

These small benchmarks track development progress; they are not performance
claims. Two reproducible snapshots — a real-data leaderboard and a controlled
synthetic signal.

**Real data — OpenML** mean rank across 9 standard datasets, 3 seeds, a 60/20/20
split with early stopping. Rank is among **all 11 models** in the suite (lower is
better); primary metric is RMSE (regression) / logloss (classification):

| Model | Regression (4) | Classification (5) |
|---|---|---|
| LightGBM | 2.50 | 5.00 |
| CatBoost | 3.00 | 3.40 |
| XGBoost | 4.25 | 6.40 |
| RepLeaf (adaptive) | 4.50 | 4.20 |
| RepLeaf (constant) | 4.50 | 7.60 |
| RepLeaf (adaptive_insample) | 5.00 | 5.40 |
| RepLeaf (embedded_linear) | 5.25 | 4.80 |
| RepLeaf (embedded + plr) | 8.50 | 6.80 |
| HistGradientBoosting | 8.50 | 9.80 |
| RepLeaf (embedded + torch_mlp) | 9.75 | 5.60 |
| RepLeaf (embedded + torch_periodic_plr) | 10.25 | 7.00 |

The tuned external GBMs lead on real tabular data. Among the RepLeaf arms the
**adaptive** leaf — a per-leaf weighted-LOO gate between a constant and an
embedded-linear leaf — is the most robust (2nd overall on classification, behind
only CatBoost; never worse than `constant`). Its leverage-free ablation
`adaptive_insample` ranks below it, and higher-dimensional representations
(`plr`, learned `torch_*`) do **not** help here: on real data a constant or
adaptively-gated leaf is the honest choice.

**Synthetic signal — regression RMSE** (mean of 3 seeds, n=10k, 20 features). The
leaf embeddings pay off on smoother structure: an embedded-linear leaf over
standardized raw features (`identity`) edges out a constant leaf and LightGBM
(CatBoost still leads):

| Model | RMSE |
|---|---|
| CatBoost | 0.395 |
| RepLeaf (embedded_linear, identity) | 0.407 |
| HistGradientBoosting | 0.411 |
| RepLeaf (constant) | 0.412 |
| LightGBM | 0.413 |

The higher-dimensional `plr` / `torch_*` encoders are random-projected at the
default `max_leaf_emb_dim=64` on this 20-feature signal and don't gain over
`identity` (raise the cap to fit them directly). Separately, refitting
LightGBM's own routes with representation-conditioned leaves
(`router_extraction`) improves it by 2–12% RMSE.

Reproduce with `python benchmarks/openml_suite.py --learned-encoders` and
`python benchmarks/benchmark_synthetic_regression.py`; full real-data numbers in
[experiments/results/openml_benchmark.md](experiments/results/openml_benchmark.md).

## Development

```bash
git clone https://github.com/Matapanino/repleafgbm.git && cd repleafgbm
pip install -e ".[dev]"
bash scripts/check.sh               # lint + tests + all examples
python -m pytest tests/ -q          # PYTHONPATH=src if not installed
python examples/regression_basic.py
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution workflow.

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
