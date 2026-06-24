# Native tree routing (`apply_tree`): routing leaf-id acceleration

Date: 2026-06-24
Change context: follow-up to the post-PR #30 prediction-traversal benchmark
(`experiments/results/2026-06-24-prediction-traversal-bench.md`), whose verdict
was GO for a Rust `Tree.apply`/`apply_forest` router. This PR ships the first
piece — a compiled single-tree router — and **only** that (no fused leaf-value
prediction, no version bump, no public API change).

## What changed

- `native/src/lib.rs`: new `apply_tree` — one rayon pass of independent per-row
  root-to-leaf descents over the flat tree arrays. Override precedence matches
  `Tree.apply` exactly: NaN follows `missing_left`; else a categorical node
  (non-empty CSR slice) tests subset membership (`np.isin`, exact code
  equality); else `x <= threshold`. Numeric, `missing_left` (incl. external
  `default_left=False`), and categorical-subset splits are all supported.
- `core/tree.py`: `Tree.apply` builds a per-node categorical CSR and calls
  `repleafgbm_native.apply_tree` when the symbol is present, else the NumPy
  level-synchronous reference, now factored out as `Tree._apply_numpy`.
  Feature-detected via `hasattr` (graceful fallback for an absent/older
  extension), so no native version bump is required.

## Correctness (exact leaf-id parity)

Routing is integer leaf-id assignment, so parity is **exact**
(`assert_array_equal`), not allclose. `tests/test_tree_routing_native.py` checks
native `apply` == NumPy `_apply_numpy` == an independent scalar oracle across:
numeric thresholds, NaN→`missing_left`, `missing_left=False` (external routes),
categorical subset (incl. singleton sets), mixed numeric/categorical depths,
root-only/empty/all-left/all-right/singleton edges, the >16384-row rayon branch,
and fitted regression/multiclass/categorical trees. The older/absent-extension
fallback is covered by monkeypatching the module global. Because `Tree.apply`
now defaults to the native router whenever the extension is built, the **entire**
prediction/serialization suite (`scripts/check.sh`) also runs against it — all
green.

## Method (before/after A/B)

Same-process A/B: `benchmarks/predict_profile.py` run with the native router on
(`apply_tree`), then again with the module global forced to the NumPy router
(`tree._native = None`). Identical seeded trees and test matrices, so the
routing/predict deltas isolate the kernel. Medium preset (100k train rows, 100
features), Rust split backend (fits only), `n_estimators=100`, best-of-2,
`{regression, binary, multiclass K=5} × {constant, embedded_linear}` ×
`{50k, 200k}` test rows + the categorical/missing case.

```bash
# native (apply_tree) and forced-NumPy passes, identical args:
OMP_NUM_THREADS=1 RAYON_NUM_THREADS=8 python3 -m benchmarks.predict_profile \
  --backend rust --size medium --tasks regression binary multiclass \
  --leaf-models constant embedded_linear --sweep-trees 100 \
  --sweep-rows 50000 200000 --n-classes 5 --repeats 2 --out <path>
```

Environment: Python 3.11.1, NumPy 1.23.5, `repleafgbm-native` 0.2.0 (+`apply_tree`),
macOS arm64, `OMP_NUM_THREADS=1 RAYON_NUM_THREADS=8`.

## Results

| case | rows | route np[s] | route rs[s] | **route×** | pred np[s] | pred rs[s] | **pred×** |
|---|---:|---:|---:|---:|---:|---:|---:|
| regression const | 50k | 0.548 | 0.066 | 8.3× | 0.583 | 0.073 | 8.0× |
| regression const | 200k | 3.103 | 0.314 | 9.9× | 3.188 | 0.349 | 9.1× |
| regression emb | 50k | 0.532 | 0.069 | 7.8× | 0.816 | 0.229 | 3.6× |
| regression emb | 200k | 3.128 | 0.333 | 9.4× | 3.477 | 0.635 | 5.5× |
| binary const | 200k | 3.014 | 0.306 | 9.8× | 3.195 | 0.332 | 9.6× |
| binary emb | 200k | 3.008 | 0.313 | 9.6× | 3.349 | 0.614 | 5.5× |
| multiclass K=5 const | 50k | 2.582 | 0.306 | 8.4× | 3.115 | 0.367 | 8.5× |
| multiclass K=5 const | 200k | 12.742 | 1.424 | 8.9× | 12.866 | 1.595 | 8.1× |
| multiclass K=5 emb | 50k | 2.492 | 0.314 | 7.9× | 3.520 | 1.065 | 3.3× |
| multiclass K=5 emb | 200k | 13.041 | 1.428 | 9.1× | 14.894 | 3.355 | 4.4× |
| regression categorical | 50k | 1.128 | 0.106 | 10.7× | 1.364 | 0.257 | 5.3× |
| regression categorical | 200k | 4.939 | 0.381 | **13.0×** | 5.325 | 0.747 | 7.1× |

(`const` = constant leaf, `emb` = embedded_linear, K=5 → 500 predicting trees.)

## Interpretation

- **Routing kernel: 7.8–13.0× faster.** Categorical is the biggest win (10.7–13.0×)
  because the NumPy path's per-level `np.unique` loop over categorical nodes was
  its worst case; the native per-row descent replaces it with a small linear
  membership scan.
- **End-to-end predict: 3.3–9.6× faster.** For constant leaves predict ≈ routing,
  so the predict speedup tracks routing (8.0–9.6×). For embedded_linear the
  already-native `predict_linear` leaf-eval is unchanged, so it now dominates the
  residual and Amdahl caps the predict speedup at 3.3–7.1× — exactly the split the
  prediction-traversal benchmark predicted (routing was 60–100% of predict). In
  absolute terms the worst cases shrink sharply: multiclass-emb 200k
  14.9s→3.4s, categorical 200k 5.3s→0.7s, multiclass-const 200k 12.9s→1.6s.
- This also speeds the early-stopping eval loop (it routes eval rows through each
  new tree on the same `Tree.apply` path).

## Verdict: SHIP

Exact leaf-id parity (integer routing, `assert_array_equal`; full suite green
with the native router as default) **and** a material speedup on every benchmarked
case (routing ~8–13×, predict ~3–10×). Decision criteria met.

Deliberately **not** in this PR (kept scoped to routing leaf-id acceleration):
forest-level batched traversal (one native call over all trees, removing the
remaining Python per-tree loop) and fused leaf-output computation. Those are the
natural next steps; the embedded_linear Amdahl gap above is the evidence for the
leaf-output fusion follow-up.
