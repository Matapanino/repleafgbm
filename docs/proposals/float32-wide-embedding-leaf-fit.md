# Proposal: float32 Gram for wide-embedding leaf-fit (`leaf_fit_precision`)

- **Status:** Approved design (locked via plan + AskUserQuestion + core-reviewer
  sign-off). Formalizes the overnight-loop iter-002 HOLD into a spec.
- **Date:** 2026-06-25
- **Author:** CUDA/perf overnight loop (driver), reviewed by `core-reviewer`.
- **Type:** Performance optimization (opt-in leaf-fit precision knob), **not** a
  default change and **not** an algorithm change.
- **Thesis check:** PASS — splits stay raw-feature-only, leaves stay
  representation-conditioned, encoder stays frozen, no embedding-dim splits, no
  `format_version` bump, no native rebuild. Touches only the host NumPy BLAS
  leaf-fit fallback; the Rust/CUDA paths are untouched.
- **Companion evidence:** `docs/perf-notes/experiment-log.md` (iter 002),
  `artifacts/gpu_bench/exp2_reconfirm/`, `scratchpad/f32_leaf_ceiling.py`.

---

## 1. Motivation

For **wide embeddings** (`emb_dim > 64`), fitting the per-leaf ridge model is the
single dominant cost of training. The per-leaf weighted Gram matrices are
accumulated in float64 BLAS; float64 is required for the *solve* (cancellation in
the centered normal equations), but the Gram **accumulation** does not need it.
Doing the two large reductions in float32 (≈2× SIMD throughput) while keeping the
solve in float64 recovers most of that cost at a numerically negligible price.

## 2. Evidence (iter 002, re-confirmed on the current branch)

- Phase probe (rust, `OMP_NUM_THREADS=1`, regression, 50k×200f, identity
  encoder ⇒ emb=200 > 64 ⇒ BLAS Gram path, 50 trees): **`leaf_fit` = 69.6%** of
  fit (7.91s / 11.37s). Narrow (emb=30) → `leaf_fit` 21.6%, spread across phases —
  no lever (native rayon path).
- Isolated ceiling (`f32_leaf_ceiling.py`, emb=200, float32 Gram + float64 solve):
  **1.94× single-thread / 1.64× multi-thread** on leaf_fit (−39…−48%), weight
  deviation vs float64 **rel ~1.2e-6**, robust to BLAS threading (multi ≈ single —
  the per-leaf GEMMs are small, so threads add little; the win is precision, not
  parallelism).
- Projected whole-fit: ~25–30% faster on the wide-emb case; **zero** effect on
  narrow/constant/default-width models.

## 3. Affected code path

`src/repleafgbm/core/leaf_models.py:287-308` — the per-leaf NumPy BLAS Gram
fallback inside `EmbeddedLinearLeafModel.fit_leaves`, reached **only** when
`_native is None` **or** `emb_dim > _NATIVE_STATS_MAX_DIM (=64)`. The narrow
native path (`:264-274`, `leaf_linear_stats` rayon) is **not touched**, so the
NumPy↔Rust **bitwise** parity surface is unaffected.

## 4. Current vs proposed behavior

Current (float64), per linear leaf `j`:
```
A[j]   = Zl.T @ hZ                  # (emb,emb) weighted Gram   <- dominant cost
A[j]  -= np.outer(z_mean[j], s_hz)  # centering (cancellation-prone)
rhs[j] = -(g_seg[sl] @ Zl) - t_mean[j] * s_hz
... batched np.linalg.solve in float64 (_solve_and_assemble)
```
Proposed (`leaf_fit_precision="float32_gram"`): compute **only** the two
reductions `Zl.T @ hZ` and `g_seg[sl] @ Zl` in float32 (cast `Z_seg`/`hZ_seg`/
`g_seg` views to float32 for those products), then store back into the **float64**
`A`/`rhs` containers. Everything else — centering subtractions, `z_mean`,
`t_mean`, `z_min/z_max`, `bias`, and the entire `_solve_and_assemble` (diagonal
`+l2`, `np.linalg.solve`, the singular one-by-one fallback, the constant-fallback
gate) — stays **float64**. Narrow (native) path unchanged.

## 5. Why the default must remain float64

float64 is the **reproducible, parity** path: same seed ⇒ same model, and the
NumPy↔Rust bitwise parity tests live here. float32 Gram is allclose-not-bitwise
and can, in rare near-singular leaves, flip the constant-fallback `keep` decision
(`leaf_models.py:380-390`) — a structural model difference (same risk class as the
CUDA near-tied-split flip). That trade is acceptable **only as an explicit opt-in**.

## 6. Public API

Add a constructor hyperparameter to `RepLeafRegressor` / `RepLeafClassifier`:

```
leaf_fit_precision: str = "float64"     # | "float32_gram"
```

Threaded `sklearn.py.__init__` → `make_leaf_model(...)` (`leaf_models.py:694`) →
`EmbeddedLinearLeafModel.__init__` → used in `fit_leaves`. Additive,
backward-compatible, **MINOR** SemVer (per ADR 0003 / `docs/api_freeze.md`).
Stored as a plain attribute (sklearn clone/get_params safe). Inert for
`leaf_model="constant"` (no Gram); affects only `embedded_linear`/`raw_linear`/
`adaptive` **and only when `emb_dim > 64`**. Validated in `fit` (invalid value →
`ValueError` listing the choices). Docstring states the cross-platform
reproducibility caveat.

## 7. Alternative considered (rejected): internal env var

`REPLEAFGBM_LEAF_FIT_PRECISION` would avoid the public surface, but it is **hidden
global state** (CLAUDE.md) that silently changes the **default CPU path** numerics
and makes a fitted model irreproducible from its constructor alone. The existing
`REPLEAFGBM_CUDA_*` env flags are **not** a precedent: they are private
kill-switches for an explicitly-selected, CI-untested GPU backend (ADR 0005), not
a knob on the path every CPU user exercises. Rejected by `core-reviewer`.

## 8. Numerical tolerance policy

- Default float64 path: **bitwise** NumPy↔Rust parity, unchanged.
- `float32_gram`: **allclose, not bitwise.** The acceptance assertion is on
  **leaf predictions / model RMSE (quality-equivalence)**, not raw weights, to
  cover the rare constant-fallback flip. Tolerance: `rtol≈1e-5` on predictions +
  an RMSE-equivalence check. (Measured weight deviation rel ~1.2e-6; predictions
  are tighter still in the common case.)

## 9. Test plan

New `tests/test_leaf_fit_precision.py` (seeded, hundreds of rows): (a) float32 vs
float64 on a **wide-emb** synthetic — predictions allclose `rtol≈1e-5` + RMSE
equivalence; (b) the singular-leaf one-by-one fallback (`:370-378`) still works
under float32; (c) default is float64 and **byte-identical** to today; (d)
`constant` leaf ignores the knob; (e) invalid value → `ValueError`. Untouched and
green: `tests/test_rust_backend.py` (bitwise), `tests/test_sklearn_compat.py`
(`parametrize_with_checks` auto-covers the new param), full suite + ruff.

## 10. Benchmark plan

Through the C-extended orchestrator (median+spread+signal), interleaved:
```
bash scripts/perf_loop.sh --mode ab --task regression --variant-a rust --variant-b rust \
  --n-features 200 --max-leaf-emb-dim 256 --leaf-model embedded_linear --n-estimators 50 \
  --precision-a float64 --precision-b float32_gram --reps 5
# narrow control (expect flat): --n-features 30 --max-leaf-emb-dim 64 ...
```

## 11. Acceptance criteria

Wide-emb: median fit speedup ≥ **1.10** (conservative floor; expect 1.25–1.35×),
`rel_spread < 0.10` both variants, `|r2_B − r2_A| < 0.005`, signal = B wins ≥4/5 ∧
|median Δ| > 1σ. Narrow control: < 1.05× (confirms wide-only). All tests green.

## 12. Rejection criteria

Quality regression beyond `|Δr2| ≥ 0.005`; predictions outside `rtol 1e-5`;
default float64 path no longer bitwise; narrow control shows a real change;
speedup < 1.10 after the harness change.

## 13. Backward / serialization compatibility

Backward-compatible: new param defaults to current behavior. **No serialization
change** — `LeafValues` always stores float64 `bias/weights/z_min/z_max`
regardless of accumulation dtype; `format_version` does not move; old models load
unchanged and a float32-fit model saves/loads like any other.

## 14. Failure modes

- float32 leaks into the solve/centering → cancellation error. Guard: narrowing
  confined to the two reductions; allclose+RMSE test catches leakage.
- Cross-platform BLAS/SIMD low-bit differences → a near-singular leaf flips the
  constant-fallback gate → structurally different model. Acceptable because opt-in;
  documented in the param docstring. Never enabled by default.

## 15. Rollback

Remove the param (no serialized-format dependency). Because the default path is
untouched and bitwise, a revert is a pure no-op for every existing model.
