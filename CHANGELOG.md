# Changelog

All notable changes to RepLeafGBM are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); from 1.0.0 the project
adheres to [Semantic Versioning](https://semver.org) for the public API defined
in [docs/adr/0003-api-stability.md](docs/adr/0003-api-stability.md).

## [1.1.0] - 2026-06-16

Imbalanced-classification support and an explicit loss/metric separation. All
additions are opt-in and backwards-compatible — defaults reproduce prior
behavior, and the model format is unchanged.

### Added
- **`sample_weight`** on `fit` (regressor and classifier, including multiclass
  and multi-output): per-row weights scale each row's gradient/Hessian and the
  init score (`core.booster.weight_grad_hess`). The Newton leaf target `-g/h`
  is invariant, so weighting reweights split gains and leaf magnitudes without
  distorting per-row targets. Weighting happens upstream of the histogram, so
  the NumPy/Rust split backends and their parity are untouched. `RepLeafDataset`
  gains an optional `sample_weight`.
- **`class_weight`** estimator parameter (classifier only): `None`,
  `"balanced"`, or a `{label: weight}` dict, expanded to per-row weights via
  sklearn `compute_sample_weight` and composed multiplicatively with
  `sample_weight`. Serialized with the model config.
- **`balanced_accuracy`** eval metric (mean per-class recall, greater-is-better;
  matches `sklearn.metrics.balanced_accuracy_score`) for monitoring/early
  stopping on imbalanced targets. `get_metric` is now exported.
- **Capability layer** (`_supports_sample_weight`): estimators that cannot
  reweight rows — frozen-route replay (`RouterExtraction*`) — drop weights with
  a `UserWarning` instead of raising. Documented fallback: train the plain loss,
  early-stop on a built-in metric, and compute balanced accuracy externally.
- **Docs**: `docs/weighting_and_metrics.md` (usage + the loss / early-stopping /
  report-metric / regularizer separation), ADR 0004, and the weighted-Newton
  derivation in `docs/math.md`. `label_smoothing` is clarified as a regularizer,
  not a class rebalancer.

## [1.0.2] - 2026-06-15

OSS-quality hardening; no public API or model-format changes.

### Added
- **PEP 561 typing marker** (`py.typed`): the shipped package now advertises its
  inline type hints, so type checkers (mypy, pyright) resolve `repleafgbm`'s
  public API.
- **Cross-platform native wheels**: the optional Rust extension
  `repleafgbm-native` is now built and published as Linux/macOS/Windows wheels
  for CPython 3.10-3.12 (`.github/workflows/publish-native.yml`, maturin +
  OIDC trusted publishing). `pip install repleafgbm-native` now gives PyPI users
  the Rust split/leaf kernels (auto-detected) instead of a NumPy-only fallback.
- **Coverage gate**: the test suite runs under `pytest-cov` on the Linux/3.12
  lane with a `fail_under` floor (`pytest-cov` added to the `dev` extra;
  `[tool.coverage]` configured in `pyproject.toml`).

### Changed
- **Cross-platform CI**: the `test` and `rust-backend` jobs now run on
  ubuntu/macos/windows (`OMP_NUM_THREADS=1` to avoid the torch+lightgbm libomp
  deadlock; `shell: bash` for uniform scripting). scikit-learn floor/1.6 pins
  stay on Linux; macOS/Windows smoke-test the latest stack.
- `CONTRIBUTING.md` documents the deprecation cycle (summarizing ADR 0003).

## [1.0.1] - 2026-06-15

### Fixed
- **Compatibility with scikit-learn >= 1.6** (1.0.0 broke on modern sklearn):
  estimators now implement `__sklearn_tags__` in addition to `_more_tags`, and
  array validation selects `ensure_all_finite` vs the removed `force_all_finite`
  keyword depending on the installed version. Verified against scikit-learn 1.9.
- `predict` on an array with the wrong number of features now raises the
  standard "X has N features, but ... is expecting M features" message.

### Changed
- The `check_estimator` compliance battery (`tests/test_sklearn_compat.py`)
  runs on scikit-learn >= 1.6 (where the modern contract is stable) and is
  skipped on older installs; the hand-written compatibility tests still run.
- CI/publish workflows opt into the Node.js 24 action runtime.

## [1.0.0] - 2026-06-15

First stable release. The API, on-disk model format, and registered
encoder/objective/metric names are now covered by SemVer.

### Added — models & training
- `RepLeafRegressor` / `RepLeafClassifier`: gradient boosting that routes on
  raw features and predicts with ridge-regularized linear models over a frozen
  representation inside each leaf (constant / `embedded_linear` / `raw_linear`).
- Regression objectives: squared error, `Huber`, `Quantile`, `PoissonRegression`.
- Binary and multiclass (softmax) classification; `label_smoothing`.
- Multi-output regression with shared-routing vector leaves.
- Encoders (frozen): `identity`, `plr`, `periodic`, `cross`, and optional
  learned `torch_periodic` / `torch_plr` / `torch_mlp` (pretrained then frozen;
  torch only needed at fit time).
- Early stopping (`early_stopping_rounds`, `best_iteration_`/`best_score_`),
  metrics (RMSE, MAE, AUC, accuracy, logloss, multi_logloss) and
  user-supplied metrics via `make_metric`.
- Per-leaf extrapolation guard (z clipped to the leaf's training support).

### Added — data, backends & integrations
- `RepLeafDataset` with pandas/categorical support (ordinal codes, native
  gradient-sorted subset splits, frequency encoding, embedding cache).
- Optional Rust split/leaf kernels (`split_backend="auto"`, parity-tested).
- `external_model` mode for LightGBM, XGBoost, CatBoost (OOF + stacking
  helpers); `router_extraction` mode (LightGBM regression/binary).
- Directory-based save/load (`format_version` 6, full read ladder from v1).

### Added — quality & docs (1.0 work)
- Full scikit-learn `check_estimator` compliance (Phase 24).
- ADR 0003 (API stability/versioning) and `docs/api_freeze.md`.
- OpenML benchmark suite: 9-dataset reproducible leaderboard vs
  LightGBM/XGBoost/CatBoost/HistGB (`benchmarks/openml_suite.py`, Phase 25).
- API reference generation (`scripts/build_docs.sh`, pdoc) + CI docs job.

### Changed
- `_check_is_fitted` now raises `sklearn.exceptions.NotFittedError`.
- Array inputs are validated with `check_array`/`check_X_y` (NaN allowed —
  it routes left; sparse/complex/inf/empty/1-D rejected with clear messages).
- README: the "APIs will change without notice" warning is replaced by the
  SemVer stability policy.

### Fixed
- DataFrames with non-string column labels (e.g. an integer `RangeIndex`)
  raised `KeyError`; columns are now matched by stringified label.

For the full development history (Phases 0–25), see
[docs/roadmap.md](docs/roadmap.md).
