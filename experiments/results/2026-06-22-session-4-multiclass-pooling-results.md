# Perf Session 4 results — multiclass leaf-pooling + fused predict kernel

Date: 2026-06-22
Branch: `perf/s4-multiclass-histogram` (off `main` @ #25 `ad2d418`)
Machine: darwin arm64, 8 logical / 4 perf cores. `OMP_NUM_THREADS=1 RAYON_NUM_THREADS=8`.
Config: `--backend rust --task multiclass --n-classes 5 --max-leaf-emb-dim 64`, n_estimators=100.
Reps: 5, `--seed 0`. See `2026-06-22-session-4-baseline-and-A-earlykill.md` for how A
(histogram) and B (leaf glue→native) were measured dead and the real lever found.

## What shipped
Two output-preserving optimizations, both behind the existing native gate (emb_dim ≤ 64):

1. **leaf_fit cross-K pooling** (`native/src/lib.rs::leaf_linear_stats_mc`,
   `core/leaf_models.py::fit_leaves_multiclass`, `core/multiclass.py` grow-all-then-fit).
   Real softmax trees put >50% of a class's rows in one leaf, so per-class rayon
   leaf-parallelism is Amdahl-capped near ~2x. Pooling all K trees' leaves into one
   parallel pass dilutes any one giant leaf. **Bitwise-identical** to per-class fitting
   (same per-leaf row order + per-system solve) — a pure scheduling change.
2. **Fused `predict_linear` kernel** (`native/src/lib.rs`, wired in
   `core/leaf_models.py::LeafValues.predict`). Replaces `bias[leaf_idx] +
   einsum("ij,ij->i", Z, weights[leaf_idx])` — whose `weights[leaf_idx]` materializes a
   256 MB gather — with one rayon pass over rows reading the L1-resident per-leaf tables.
   Drives the multiclass training `eval` F-update (and prediction). allclose to einsum
   (3.5e-15); rows independent ⇒ serial==parallel bitwise.

## Results (fit wall-clock, 5 reps)

| scale | baseline med | treat med | Δ median | save / σ | leaf_fit | eval | non-target localization |
|---|---|---|---|---|---|---|---|
| medium | 37.60 | 31.29 | **−16.8%** | 10.8σ, t=14.7 | −24% | −83% | **flat** (histogram −0%, partition 0%) |
| large (raw) | 251.75 | 217.05 | −13.8% | 7.7σ, t=9.5 | −27% | −85% | drifted up (histogram +6%, partition +7%) |
| large (thermal-matched A/B) | 249.37 | 205.91 | **−17.4%** | 17.6σ, t=35.7 | −32% | −85% | **flat** (histogram +0%, partition +3%) |

**Quality is bitwise-identical** at both scales (medium multi_logloss 0.27483026454,
accuracy 0.89232; large 0.22878579135, 0.93034 — match baseline to 14 digits), confirming
the model is unchanged beyond float-noise (pooling is bitwise; predict kernel is allclose
and does not perturb quality at 4+ digits).

### The large raw run was thermal-contaminated
The raw large treat run was the 3rd consecutive `--size large` benchmark of the day; its
**untouched** phases rose (histogram +6%, partition +7%, split_scan +11%) vs the cool
baseline run 1.5 h earlier — a clear thermal-drift signal that deflates the measured fit
Δ. Two drift-robust reads both clear 15%:
- **Target-phase attribution:** leaf_fit (72.02→52.30 = −19.7 s) + eval (26.84→3.97 =
  −22.9 s) = **42.6 s saved = 16.9% of baseline fit**, independent of non-target drift.
- **Thermal-matched interleaved A/B** (baseline/treat reps alternated in one run via a
  temporary `REPLEAF_S4_OFF` runtime toggle, since removed): baseline 249.37 s vs treat
  205.91 s median = **−17.4%** (17.6σ, t=35.7), with non-target phases now **flat**
  (histogram +0%, partition +3%) — drift cancelled, confirming the 16.9% attribution.

Medium has no such drift (non-target phases flat) and is the clean ≥15% proof on its own.

## Verdict — PASS (≥15% on both scales)
- medium fit **−16.8%** (clean, flat localization, 10.8σ); large fit **−17.4%**
  (thermal-matched, flat localization, 17.6σ). Both clear the ≥15% ship-bar decisively.
- Model output preserved: quality bitwise-identical; pooling is bitwise, predict kernel
  allclose (3.5e-15) within the leaf-predict contract.
- Recommend keeping both optimizations. **No defaults/flags changed** — always-on behind
  the existing native gate (emb_dim ≤ 64); NumPy fallback unchanged. Native crate stays
  0.1.0. Bonus: prediction (`predict_seconds`) also ~2–3× faster from the shared kernel.

## Parity / correctness
- `tests/test_leaf_models.py::test_multiclass_pooled_matches_per_class`: pooled == per-class
  **bitwise** (+ NumPy fallback allclose).
- `tests/test_leaf_models.py::test_predict_linear_native_matches_numpy`: native predict ==
  einsum allclose, clip/no-clip × serial/parallel branches.
- `tests/test_leaf_models.py::test_predict_linear_serial_parallel_bitwise`: the native
  serial and rayon branches are **bitwise**-identical on the same rows (added per review,
  pins the thread-count-independence / determinism property).
- Full suite (after fold-in) green via `OMP_NUM_THREADS=1 bash scripts/check.sh`
  ("All checks passed": ruff + pytest + all examples); rust bitwise-histogram + end-to-end
  backend-agreement tests unchanged and green.

## core-reviewer sign-off
**approve, no blocking issues** — output-preservation bitwise-verified at the kernel
boundary for both new kernels; determinism, optional-dep isolation, SemVer/serialization
back-compat all confirmed. Folded in the one substantive non-blocking item (serial-vs-
parallel bitwise regression test) + the `k`→`off` readability nit. One non-blocking note
**for the PR body**: the fused `predict_linear` kernel changes the prediction path for **all
scalar embedded-linear leaves** (regression + binary + multiclass), not only multiclass —
covered by the existing extrapolation-guard / regressor / classifier suites (green); a
welcome side-benefit is faster prediction across all single-output tasks.
