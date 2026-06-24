# Prediction-traversal benchmark: routing vs leaf-eval

Date: 2026-06-24
Change context: post-PR #30 (`fcd7c9d`), native `partition_rows` (0.2.0) merged.
Measurement-only; no core/native change, no version bump.

## Goal

The post-PR #30 phase refresh
(`experiments/results/2026-06-24-post-pr30-phase-refresh.md`) showed prediction
traversal is now the next visible CPU cost — multiclass K=5 already spends
`predict 1.153s` in-fit and `predict_seconds 1.222s` at test time — because
`core/prediction.py` loops over every tree calling `Tree.apply` (a NumPy
level-synchronous router) and multiclass stores `n_rounds × n_classes` trees.
`docs/gpu_audit.md` names a Rust `Tree.apply`/`apply_forest` as the next
low-risk native target and prescribes its prerequisite: "Benchmark: predict time
vs `n_trees`, `n_rows`, `n_classes`".

This note builds that prerequisite. The new harness
`benchmarks/predict_profile.py` decomposes a fitted model's `predict` into

    routing   = Σ over predicting trees of  Tree.apply(X_raw)
    leaf_eval = Σ over predicting trees of  LeafValues.predict(leaf_idx, Z)

so `routing_share = routing / predict` is the **ceiling** a future Rust
`apply_forest` could remove (it makes routing ~free; leaf-eval is already native
`predict_linear` for embedded-linear leaves). The decomposition reuses the
public `Tree.apply` / `LeafValues.predict` on the fitted estimator and is
validated per case against `booster.predict_raw` (`parity_max_abs_diff`). The
split is **backend-independent** — routing is pure NumPy and leaf-eval uses the
native `predict_linear` regardless of `split_backend`; only *fit* differs — so
`--backend rust` here only speeds the harness's own fits.

## Method

Medium preset (100k train rows, 100 features), `encoder=identity`,
`max_leaf_emb_dim=64`, best-of-3 wall time per timed region, Rust backend (fits
only). Sweep: `{regression, binary, multiclass K=5} × {constant,
embedded_linear} × n_estimators {50, 200} × test rows {10k, 50k, 200k}`, plus one
categorical/missing worst case (regression, embedded_linear; first 8 columns
bucketized to `category` with 5% NaN, forcing `Tree.left_categories` subset
splits and the per-level `np.unique` loop in `Tree.apply`).

```bash
env OMP_NUM_THREADS=1 RAYON_NUM_THREADS=8 \
  python3 -m benchmarks.predict_profile --backend rust --size medium \
  --sweep-trees 50 200 --sweep-rows 10000 50000 200000 --repeats 3 \
  --out artifacts/predict_bench/std/cases.jsonl
```

Note: for multiclass, `n_estimators` counts rounds; predicting trees =
`rounds × n_classes` (so 50/200 rounds → 250/1000 trees at K=5).

Environment: Python 3.11.1, NumPy 1.23.5, scikit-learn 1.2.0,
`repleafgbm-native` 0.2.0, macOS arm64, `git_sha=fcd7c9d` (`git_dirty=true` from
this PR's untracked files). Identity encoder on 100 features exceeds
`max_leaf_emb_dim=64`, so a random projection to 64 dims is applied — this only
affects accuracy (quality stays sane: reg r²≈0.99, binary auc≈0.998), not the
routing/leaf-eval timing split, which is what this note is about. Full matrix:
`artifacts/predict_bench/std/{cases.jsonl,summary.md}` (39 cases).

## Results

Decomposition is faithful: `parity_max_abs_diff ≤ 3e-14` everywhere (multiclass
exactly `0.0`), so `routing + leaf_eval (+ accumulation)` is the real
`booster.predict_raw` path.

**Headline — `n_estimators=200`, 200k test rows (the stress point):**

| task | leaf_model | predict s | routing s | leaf_eval s | routing % | ceiling if routing→0 |
|---|---|---:|---:|---:|---:|---:|
| regression | constant | 5.16 | 5.26 | 0.06 | ~100† | router-bound |
| regression | embedded_linear | 5.88 | 5.48 | 0.60 | 93.2 | ~14.7× |
| binary | constant | 6.45 | 6.31 | 0.05 | 97.8 | router-bound |
| binary | embedded_linear | 7.74 | 6.35 | 0.59 | 82.1 | ~5.6× |
| multiclass K=5 | constant | 27.72 | 25.83 | 0.55 | 93.2 | ~14.7× |
| multiclass K=5 | embedded_linear | 31.00 | 27.06 | 5.42 | 87.3 | ~7.9× |
| regression (categorical) | embedded_linear | 7.18 | 6.57 | 0.63 | 91.6 | ~11.8× |

† constant-leaf `leaf_eval ≈ 0` (a pure bias gather), so routing ≈ predict and
the share sits at ~100% ± a few % of best-of-3 noise; the robust signal is the
embedded-linear shares and the absolute routing seconds.

**Routing share grows with rows** (`n_estimators=200`, embedded_linear) — routing
(`Tree.apply`) scales ~linearly in rows while the native `predict_linear`
leaf-eval scales sub-linearly, so the Python router becomes the bottleneck
exactly where predict is most expensive:

| test rows | regression | binary | multiclass K=5 |
|---|---:|---:|---:|
| 10k | 60.3% | 68.5% | 64.7% |
| 50k | 74.7% | 72.0% | 74.5% |
| 200k | 93.2% | 82.1% | 87.3% |

**Categorical/missing worst case:** the per-level `np.unique` categorical loop in
`Tree.apply` makes routing ~20% slower in absolute terms than the equivalent
numeric routing (6.57s vs 5.48s at 200k/200est, same leaf model), so categorical
routing is an additional tax on top of an already routing-bound predict.

## Interpretation

- **Routing dominates predict** — 60–100% of it, and the part that scales worst.
  Constant-leaf models are essentially router-bound (~100%); embedded-linear
  spends 60–93% in routing with the remainder in the already-native leaf-eval.
- **Multiclass is the worst absolute cost** as `docs/gpu_audit.md` predicted:
  K=5 × rounds means routing is 26–27s of a 28–31s predict at 1000 trees / 200k
  rows. This is also the in-fit eval cost (early stopping routes eval rows
  through each new tree on the same Python path).
- **Verdict: GO** for a Rust `Tree.apply`/`apply_forest` predictor as the next
  (separate, evidence-gated) native PR. The routing share is the ceiling: a
  compiled batched traversal over the flat tree arrays that makes routing
  negligible would cut predict by ~2.5–10×+ at the 200k/200-est setting (largest
  for constant leaves and multiclass), and more as rows grow. leaf-eval is
  already native (`predict_linear`), so the predictor should target **routing
  first** (numeric + categorical-subset + missing-left), optionally folding leaf
  output in later. A real router won't make routing free, so realized speedups
  will be a fraction of the `1/(1−share)` ceilings above — but even a 3–5×
  routing kernel is a large end-to-end predict and eval win.
- **Guardrails for that PR** (from `docs/gpu_audit.md`): exact leaf-id parity for
  numeric/categorical-subset/missing-left, prediction parity, NumPy⇄Rust bitwise
  parity (both paths change together), serialization unchanged. This benchmark is
  the before/after harness.

## Scope note

This PR ships measurement only (`benchmarks/predict_profile.py` + this note +
smoke test + docs). No core/native change, no README speed-claim edit (the
evidence is single-machine and routing-vs-leaf-eval only), no version bump.
