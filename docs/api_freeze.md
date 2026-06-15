# API Freeze Report (Phase 24, v1.0)

Date: 2026-06-15. Scope: stabilize the public API ahead of the 1.0 release —
classify the public surface, achieve scikit-learn estimator-contract
compliance, and lock the policy. The contract itself lives in
`docs/adr/0003-api-stability.md`; this report records what shipped.

## scikit-learn `check_estimator` compliance

`docs/audit_v0.md` flagged `check_estimator` conformance as partial (input
validation corner cases, 1-feature/1-sample edges). It is now **complete** for
both estimators, enforced by `tests/test_sklearn_compat.py`
(`parametrize_with_checks`, 86 checks green).

Changes that got there:

- **Array input validation** (`BaseRepLeafModel._validate_array_X` /
  `_validate_array_Xy` in `src/repleafgbm/sklearn.py`): plain array-likes pass
  through `check_array` / `check_X_y` (rejects sparse, complex, infinite,
  empty, and 1-D inputs with the standard messages). DataFrame and
  `RepLeafDataset` inputs keep their own categorical-aware path. **NaN is
  allowed** (it routes left) via `force_all_finite="allow-nan"`, so the
  finiteness checks verify inf-rejection only.
- **Estimator tags** (`_more_tags`): `allow_nan=True`, `requires_y=True`.
- **`NotFittedError`** replaces the old `RuntimeError` from `_check_is_fitted`
  (the correct sklearn type; tests updated). The external-model classes keep
  their own `RuntimeError` (separate hierarchy, not sklearn estimators).
- **Target handling**: numeric-target validation for the regressor; a
  single-column 2-D `y` is raveled with a `DataConversionWarning` while a
  genuine `(n, k>1)` multi-output target is preserved (Phase 22 vector
  leaves). The classifier rejects continuous targets via `type_of_target`
  ("Unknown label type").
- **Bug fixed**: DataFrames with non-string column labels (e.g. an integer
  `RangeIndex`) raised `KeyError` because metadata stores names stringified;
  `_get_column` now matches columns by their stringified label
  (`src/repleafgbm/data/preprocessing.py`).

## Public surface (see ADR 0003 for the contract)

Covered by SemVer: `repleafgbm.__all__`, the estimator constructor
hyperparameters, the documented fitted attributes/methods, the registered
encoder/objective/metric names, the serialization format, and
`check_estimator` conformance.

Experimental (not SemVer-gated): `repleafgbm.external`, the `router_extraction`
estimators, `backends` / `repleafgbm_native` internals, `core` internals,
private names, and exact message text.

## Serialization

Format is at `format_version = 6` with the full v1→v6 read ladder tested
(`docs/serialization.md`, `tests/test_serialization.py`). Declared **stable**:
within a major version a model saved by version X loads in every version ≥ X;
format changes bump the version and ship a migration test.

## Deliberately deferred (documented limitations, not blockers)

- **Numeric ndarray predict for categorical models**: a *numeric* ndarray
  cannot carry category labels (its float codes never match the training string
  category maps), so passing one to a model with declared categoricals now
  **raises `ValueError`** instead of silently routing every row through the
  missing branch (Phase 28b). DataFrames and `RepLeafDataset` (built with the
  model's metadata) remain the supported paths for categorical data.
- **`min_samples_linear`** stays hardwired to `2 * min_samples_leaf`. Exposing
  it is a backwards-compatible MINOR addition if a use case appears.
- Compact (non-JSON) serialization remains a future option; the directory
  format is the stable one.

## Verification

- `tests/test_sklearn_compat.py` — 86 `check_estimator` checks pass.
- Full suite: `OMP_NUM_THREADS=1 PYTHONPATH=src python3 -m pytest tests/ -q`
  → 301 passed. `ruff check src tests` clean.
