# CLAUDE.md — RepLeafGBM development rules

## Project summary

RepLeafGBM is a research-oriented tabular ML library that combines raw-feature
tree routing with representation-conditioned leaf models. It is a boosted
ensemble of raw-feature routers with representation-conditioned local
predictors — not a neural network inside a tree, and not a tree over
embeddings.

## Core architectural rule

- Tree routing (splits) uses **raw features only**.
- Leaf prediction may use learned representations `z_theta(x)`.
- The encoder is **frozen** in v0: it is fitted once before boosting.
- Do **not** update encoder parameters during boosting in v0 — past trees'
  leaf outputs depend on `z_theta(x)`, so updating theta breaks the
  stage-wise additive assumption (see docs/math.md).
- Do **not** split on embedding dimensions in v0.

## v0 priority

1. Correctness
2. Readability
3. Minimal native NumPy backend
4. Regression first (binary classification second)
5. Constant leaf and embedded linear leaf
6. Dataset abstraction (`RepLeafDataset`)
7. Save/load (directory format)
8. Tests and docs

## Avoid

- No premature CUDA implementation
- No premature multi-GPU implementation
- No premature distributed framework
- No wrapper-only implementation around LightGBM/XGBoost/CatBoost
- No giant monolithic class
- No notebook-only code
- No hidden global state
- No dataset-specific hacks
- No over-engineered abstraction that obscures the core idea

## Code map

- `src/repleafgbm/data/` — `RepLeafDataset`, `FeatureMetadata`, preprocessing.
  Owns data + metadata; lazily computes/caches embeddings.
- `src/repleafgbm/encoders/` — `BaseEncoder` and implementations
  (identity, simple PLR with linear term, PBLD-style periodic,
  random-projection wrapper, and learned torch_periodic/torch_plr in
  `torch_encoders.py`). All frozen during boosting; torch encoders import
  torch only inside fit (transform/serialization are NumPy — keep it that
  way), and the native path must never import torch.
- `src/repleafgbm/core/` — objectives, metrics, histogram binning, splitter,
  tree grower, leaf models, booster, prediction, serialization.
- `src/repleafgbm/backends/` — split-search kernels behind `BaseSplitBackend`:
  `NumPySplitBackend` (reference) and `RustSplitBackend` (optional compiled
  extension in `native/`, pyo3/maturin). `native/` also provides the fused
  `leaf_linear_stats` helper used by `core/leaf_models.py` for narrow
  embeddings. NumPy and Rust paths must stay parity-tested (bitwise
  histograms, allclose leaf fits/predictions); change them together.
- `src/repleafgbm/sklearn.py`, `regressor.py`, `classifier.py` — the
  sklearn-compatible public API that glues dataset + encoder + booster.
- `src/repleafgbm/external/` — external_model mode (LightGBM base model,
  OOF, stacking features). Optional dependency; the native path must never
  import it.
- `benchmarks/` — synthetic comparison harness (sklearn + optional external
  GBMs). `experiments/` — research scripts; each writes a markdown report to
  `experiments/results/` and its conclusions feed docs/roadmap.md.

Conventions baked into the core (keep them consistent):

- Leaf fitting uses Newton targets `t = -g/h` with weights `h` everywhere.
- Missing values (NaN) always route **left** in *native training*; `Tree`
  carries per-node `missing_left` so extracted external routes (LightGBM
  `default_left`) are represented exactly.
- Categorical features are ordinal-encoded to float (unseen categories →
  NaN) and routed with native gradient-sorted **subset splits**
  (`Tree.left_categories`); ordered-threshold fallback above `max_bins`
  categories.
- `random_state` flows through `utils.random.check_random_state`; never call
  global NumPy random functions.

## Coding style

- Use type hints on public functions and classes.
- Prefer small modules over large ones.
- Prefer explicit interfaces; use `abc.ABC` or `typing.Protocol` where useful.
- Keep core algorithms readable — a new contributor should be able to follow
  the boosting loop in `core/booster.py` top to bottom.
- Add tests for new behavior.
- Keep examples small and runnable in seconds.
- Use deterministic `random_state`; same seed ⇒ same model.
- Docstrings on every public class; meaningful error messages that say what
  the user should do.

## Testing

- One command for everything: `bash scripts/check.sh` (lint + tests + examples).
- Use pytest: `PYTHONPATH=src python3 -m pytest tests/ -q`
  (or `pip install -e .` once and drop the PYTHONPATH).
- Lint with `ruff check src tests examples benchmarks` (config in pyproject).
- Run the test suite after any implementation change.
- Keep synthetic datasets small (hundreds of rows) and seeded.
- Required coverage for new model behavior: fit/predict, save/load
  round-trip, encoder transform shapes, constant-fallback for small leaves.

## Documentation

- Update docs/ when architecture changes; ADRs in docs/adr/ for decisions.
- Keep docs/roadmap.md honest: clearly distinguish implemented features from
  future plans.
- Document mathematical assumptions in docs/math.md.
- Document known limitations explicitly.

## Long-term goals (do not implement prematurely; keep designs compatible)

- Public OSS library quality (CI, contribution guide, benchmarks)
- sklearn-compatible API (already in place; preserve it)
- LightGBM/XGBoost/CatBoost backend diversity (docs/backend_strategy.md)
- Rust/C++/CUDA native backend behind `backends/`
- GPU / multi-GPU / distributed training
- Benchmark suite
