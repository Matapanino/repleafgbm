# Changelog

All notable changes to RepLeafGBM are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); from 1.0.0 the project
adheres to [Semantic Versioning](https://semver.org) for the public API defined
in [docs/adr/0003-api-stability.md](docs/adr/0003-api-stability.md).

## [1.5.0] - 2026-06-17

Multi-output robust objectives and opt-in GPU encoder pretraining. Both are
backward-compatible additions — the defaults (squared-error multi-output,
CPU pretraining, `identity` encoder), the leaf math, the model format, and
NumPy/Rust/CUDA parity are unchanged; existing models load and predict
bit-for-bit identically.

### Added
- **Multi-output Huber and quantile losses.** `RepLeafRegressor` with a 2-D `y`
  now accepts `objective="huber"`/`"quantile"` (or instances like
  `Quantile(alpha=0.9)`) in addition to squared error, closing the Phase 22
  "squared-error only" limitation. Because these losses keep a constant Hessian
  (`h = 1`), the shared-Gram vector-leaf solve and the multi-output split scan
  are reused unchanged — only the gradient (clipped / pinball residual) and the
  per-output init score (median / alpha-quantile) differ (docs/math.md).
  `objective="poisson"` stays rejected for multi-output (non-constant Hessian).
  New `MultiOutputHuber` / `MultiOutputQuantile` objectives; serialization
  (format v6, unchanged) reconstructs the loss by name on load.
- **Opt-in GPU pretraining for learned encoders.** The torch encoders
  (`torch_periodic`, `torch_plr`, `torch_periodic_plr`, `torch_mlp`) take a
  `device` knob (`"cpu"` default, `"cuda"`, `"auto"`) via `encoder_params`.
  Only the one-time pretraining `fit` uses the device; `transform`/serialization
  stay NumPy (no torch at predict). All random draws use a CPU generator, so the
  stream is device-independent and `device="cpu"` reproduces prior pretraining
  **byte-for-byte**; GPU is allclose-only (CPU stays the deterministic default),
  validated on the Colab loop (docs/cuda.md).
- **`experiments/multioutput_real_and_robust.py`** — real multi-output
  validation (OpenML energy-efficiency) plus a robustness study. Under 8%
  heavy-tailed contamination of the training targets, huber (real RMSE 2.22,
  r² 0.94) and quantile (3.87) decisively beat squared error (12.49, r² −0.77);
  RepLeaf multi-output beats a per-output LightGBM reference (1.32 vs 1.71) and
  the `(n, K)` vector-pretraining before/after gap is seed noise on real data.

### Changed
- Multi-output robust-loss support documented in the `RepLeafRegressor`
  docstring, `docs/roadmap.md` (Phase 31), and `docs/math.md` (the
  constant-Hessian family); NumPy↔Rust parity extended to the new objectives.

## [1.4.0] - 2026-06-17

Vector `(n, K)` encoder pretraining target: learned encoders now pretrain
*supervised* on multiclass and multi-output targets, closing the scalar-target
limitation noted in 1.3.0. Opt-in and backwards-compatible — the default encoder
(`identity`), the leaf math, the model format, and NumPy/Rust/CUDA parity are
unchanged; scalar (regression / binary / single-output) pretraining is
**bit-for-bit identical**.

### Changed
- **Learned-encoder pretraining generalized to an `(n, K)` target.** For 3+
  classes and multi-output regression, `_pretrain_target` now returns the
  negative-gradient residual *matrix* at the initial score (`onehot -
  softmax(F0)` for multiclass, `Y - mean` for multi-output) and the throwaway
  pretraining head emits `K` outputs (per-output standardization, loss averaged
  over rows and outputs; docs/math.md). Previously these encoders fit
  *unsupervised* because the residual is a matrix. torch is still needed only at
  fit time — `transform` and serialization stay NumPy, and saved models load and
  predict without torch.
  - Consequence: a `torch_*` encoder chosen for a multiclass / multi-output
    model now **requires torch at fit** (it pretrains) rather than silently
    fitting unsupervised.
  - Scalar targets are unchanged: `weight=None` / `K=1` reproduces the prior
    pretraining bit-for-bit.

### Added
- **`experiments/vector_target_pretraining.py`** — before/after study isolating
  the vector-target change on real OpenML multiclass (`wine`, `vehicle`) and a
  synthetic multi-output target (seeds ≥ 5, mean ± std). On the
  encoder-favorable multi-output target it is a clear win (`torch_periodic_plr`
  RMSE 0.51 → 0.41, best overall); on the small real multiclass sets the gain is
  within seed noise.
- `benchmarks/trainable_embeddings.py` summary now reports **mean ± std** over
  seeds.

### Notes
- No default changed; `repleafgbm-native` is unchanged (no Rust changes).

## [1.3.0] - 2026-06-17

Trainable-embeddings track: a new learned encoder and weighted encoder
pretraining. Opt-in and backwards-compatible — the default encoder
(`identity`), the leaf math, the model format, and NumPy/Rust/CUDA parity are
all unchanged.

### Added
- **`torch_periodic_plr` encoder** (optional `[torch]` extra): the full rtdl
  *PeriodicEmbeddings* (Gorishniy et al. 2022) — learned periodic features
  followed by a per-feature linear map + ReLU, supervised-pretrained on the
  initial Newton residual and then frozen to NumPy. torch is needed only at fit
  time; `transform` and serialization stay NumPy. Complements `torch_plr`
  (= rtdl *PiecewiseLinearEmbeddings*) and `torch_periodic` (= lite
  *PeriodicEmbeddings*).
- **Trainable-embeddings benchmark harness**:
  `benchmarks/trainable_embeddings.py` (CPU `--quick` + full run) and a Colab
  driver (`scripts/colab_trainable_embeddings.{py,sh}`) comparing the encoder
  families and optional external GBMs across regression/binary/multiclass.

### Changed
- **Learned-encoder pretraining now honors `sample_weight`/`class_weight`**: the
  pretraining target is taken at the *weighted* initial score and the
  pretraining loss is per-row weighted, so a weighted fit with a `torch_*`
  encoder now produces a (correctly) different frozen encoder than before.
  Unweighted fits are **bit-for-bit unchanged**; the fixed encoders
  (`identity`/`plr`/`periodic`/`cross`) and the model format are unaffected.

### Notes
- Learned-encoder pretraining stays **scalar-target only**, so multiclass /
  multi-output encoders fit *unsupervised* (the Newton residual is a matrix
  there). A vector-target pretraining path is on the roadmap (docs/roadmap.md).

## [1.2.0] - 2026-06-17

Experimental CUDA split backend (`split_backend="cuda"`). Opt-in and
backwards-compatible — the default backend (`"auto"`: Rust→NumPy) and the model
format are unchanged, and `"auto"` never selects the GPU.

### Added
- **CUDA split backend** (`split_backend="cuda"`, optional `[cuda]` extra,
  CuPy-based): per-node histogram construction on an NVIDIA GPU via a
  `cupy.RawKernel` (Phase A), with the binned matrix uploaded once and cached
  on-device (Phase B1). Raises a clear `ImportError` when CuPy or a usable GPU
  is missing — it never silently falls back. See `docs/cuda.md` and ADR 0005.
- **Resident histograms + adaptive GPU numeric split scan** (Phase B2):
  `build_histograms` returns the histogram resident on the GPU (the tree
  grower's sibling-subtraction `parent - child` runs on-device), and the numeric
  gain sweep + argmax run on-device for large per-node histograms — small
  histograms fall back to the host reference scan, so narrow fits do not regress
  (`_GPU_SCAN_MIN_CELLS`). Categorical subset splits and multi-output scans stay
  on the host. Measured on a Tesla T4: ~52x histogram micro-benchmark, ~2.1x
  end-to-end on a wide fit (50k×200), ~1.5x on narrow (100k×30).

### Notes
- Parity for the CUDA path is **allclose, not bitwise** (GPU atomic-add /
  reduction ordering is not fixed) and not reproducible run-to-run; use
  `"numpy"` or `"rust"` for bitwise determinism. The NumPy⇄Rust pair is
  unchanged (still bitwise-identical histograms).
- GPU validation runs through the Colab dev loop (`scripts/colab_gpu_test.sh`),
  not CI — CI and macOS skip the CUDA tests.

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
