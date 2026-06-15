# ADR 0003 — API stability and versioning policy (v1.0)

Status: accepted (2026-06-15, Phase 24)

## Context

RepLeafGBM has been research software through Phase 23: the README warned that
"APIs, file formats, and behavior will change without notice." For a 1.0
release we need a contract users can depend on, and a clear statement of what
that contract does and does not cover.

This ADR defines the public API surface, the versioning policy, and the
deprecation process.

## Decision

### Semantic versioning

From package version **1.0.0** the project follows [SemVer](https://semver.org):

- **MAJOR** — backwards-incompatible changes to the public API (below).
- **MINOR** — backwards-compatible additions (new parameters with defaults,
  new estimators, new encoders, new serialization format versions that still
  read old models).
- **PATCH** — backwards-compatible bug fixes.

Note: the roadmap's `v0/v1/v1.5/v2/v3` labels are **capability tiers**, not
package versions. Package versioning is independent and starts its stable line
at 1.0.0.

### Public API (covered by SemVer)

- Exported symbols in `repleafgbm.__all__`: `RepLeafRegressor`,
  `RepLeafClassifier`, `RepLeafDataset`, `make_metric`, `Huber`, `Quantile`,
  `PoissonRegression`, `__version__`.
- The constructor hyperparameters of `RepLeafRegressor` / `RepLeafClassifier`
  (names, defaults, semantics) — i.e. `BaseRepLeafModel.__init__`.
- The fitted attributes documented as public: `feature_importances_`,
  `feature_names_in_`, `n_features_in_`, `best_iteration_`, `best_score_`,
  `classes_`, `n_classes_`, `evals_result_`, and the methods `fit`, `predict`,
  `predict_proba`, `score`, `get_feature_importance`, `summary`, `save_model`,
  `load_model`.
- The registered string names for encoders (`identity`, `plr`, `periodic`,
  `cross`, optional `torch_*`), objectives (`squared_error`, `huber`,
  `quantile`, `poisson`), and metrics (`rmse`, `mae`, `logloss`, `auc`,
  `accuracy`, `multi_logloss`).
- The **on-disk serialization format** (`docs/serialization.md`): a model
  saved by version X loads in every version ≥ X within the same major.
  Format changes bump `format_version` and ship a migration test; older
  readers are never silently broken.
- scikit-learn estimator-contract conformance (`check_estimator`), enforced by
  `tests/test_sklearn_compat.py`, so the estimators stay usable in pipelines,
  `clone`, cross-validation, and grid search.

### Not covered (may change in any release)

- `repleafgbm.external` (LightGBM/XGBoost/CatBoost integration), the
  `router_extraction` estimators, and `repleafgbm.backends` /
  `repleafgbm_native` internals — documented as **experimental**; changes are
  noted in the changelog but not gated by SemVer.
- Anything under `repleafgbm.core` (objectives/metrics base classes excepted
  where re-exported), private names (leading underscore), and the exact text
  of error/warning messages.
- Experiment scripts (`experiments/`) and benchmarks (`benchmarks/`).

### Deprecation process

A public API element is removed only after at least one MINOR release in which
using it emits a `DeprecationWarning` naming the replacement. Removal happens
in the next MAJOR.

## Consequences

- The README instability warning is replaced by this policy (Phase 26).
- New estimator parameters must have defaults that preserve existing behavior.
- Serialization changes follow the existing additive-format precedent
  (v1→v6 all readable; see `docs/serialization.md`).
- PyPI publication (Phase 27) is unblocked: the API is now a contract.
