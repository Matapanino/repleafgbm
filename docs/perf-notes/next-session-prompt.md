# Next-session prompt (hand to Claude Code)

Paste the block below into a fresh Claude Code session on this repo.

---

You are continuing the RepLeafGBM CUDA/perf overnight loop. The previous session
(iter 008–010) shipped + GPU-validated two wins, **released them as repleafgbm 1.9.0
(merged to `main` + published to PyPI)**, and — crucially — **measured the next two
levers**. This session BUILDS them.

**Start in plan mode.** First read the source-of-truth ledgers, then confirm the top
priorities with `experiment-strategist`, then present a plan, then implement.

## Branch & state
- **1.9.0 is shipped** (PR #35 squash-merged to `main`; tagged `v1.9.0`; on PyPI as
  `repleafgbm==1.9.0`, `repleafgbm-native` unchanged at 0.3.0). The old branch
  `perf/cuda-overnight-loop-20260624` is squash-merged — **start a FRESH branch off
  `main`** (e.g. `perf/cuda-leafwise-leaffit-<date>`); do NOT reuse it.
- `git status` clean except untracked `docs/gpu-research/` (leave it). `.claude/` +
  `artifacts/` gitignored.

## Read first (the complete record)
- `docs/perf-notes/experiment-log.md` — **iter 001–010** verdicts (newest at bottom:
  iter 008 default-on, iter 009 E15, iter 010 HOLD, + the GPU bottleneck-shift /
  Task-B sizing block).
- `docs/perf-notes/reflections.md`, `experiment-backlog.md` (see "Session 2 results"),
  `rejected-ideas.md`, `harness-log.md`.
- `docs/perf-notes/research-node-batched-split-scan.md`, `docs/proposals/*.md`.
- `CHANGELOG.md` `[1.9.0]` for exactly what shipped.

## Already SHIPPED in 1.9.0 (do NOT redo)
- `leaf_fit_precision="float64"|"float32_gram"` (opt-in) — float32 the two wide-emb
  (emb>128) Gram/projection reductions for **scalar AND multi-output vector** leaves.
- Node-batched CUDA depthwise split scan, **default ON** (`REPLEAFGBM_CUDA_BATCHED_SCAN=0`
  kill switch). Native leaf-fit gate **64→128**. Device-resident multi-output CUDA scan.

## Implementation tasks (confirm/re-order with experiment-strategist)
1. **Task-B — leafwise frontier-batch (the headline GPU lever; it is now MEASURED, not
   speculative).** Last session sized it on a T4: a **leafwise + cuda** fit spends
   **32.2% of fit in split_scan** (leafwise scans per-node, unbatched), and the
   depthwise A/B showed the per-node device scan is ~89% launch overhead → an M=2
   leafwise batch (halve the launch count) projects **~1.8× scan → ~14% whole-fit
   ceiling**. Leafwise is the DEFAULT grow_policy, so this is the broad win.
   - **Stage it exactly like `_grow_depthwise_batched`:** prove the host frontier-batch
     **bitwise** vs the heap pop-order FIRST (M=2: pop one best-gain node, batch-scan
     its 2 children) — **reuse `_make_candidates_batched`** (`core/tree.py:389`), which
     already preserves the tie-break `counter` order. Then the device path is the proven
     CuPy M-axis lift of `find_best_split` (no RawKernel). Add a `_grow_leafwise_batched`
     guarded by `grad.ndim==1 ∧ supports_batched_scan` (like the depthwise dispatch at
     `tree.py:281`). `core-reviewer` on the grower refactor (readable hot path), then
     Colab A/B (leafwise default, embedded_linear, wide-200f).
2. **E02 — CUDA leaf_fit (the bottleneck SHIFTED here).** Post-batched-scan, leaf_fit is
   **65–73% of depthwise CUDA fit** (scan fell to 7–8%). leaf_fit is host-side, so this
   is the same target E01/E03/E15 attacked. Options (cheap-test-first):
   (a) native-rust wide-emb Gram (extend `leaf_linear_stats` past the 128 gate,
   optionally float32 — could beat NumPy float32 BLAS + cut the per-leaf Python loop;
   big Rust+parity surface, allclose for f32); (b) GPU leaf-fit (larger). Micro-bench
   native-vs-BLAS at emb 160/256 before committing. **Top non-GPU prize.**
3. **Backlog (cheap-test-first, local):** E14 float32 `predict_linear`, E17 float32
   embedding cache. Reject/hold per the criteria. (iter-010 batched histogram is HOLD —
   histogram is only 2.7–3.2% of fit; don't revisit. E22 mc-histogram similarly low.)
4. **Package** the session's wins → 1.10.0 (bump `repleafgbm`; bump `repleafgbm-native`
   too **iff** the Rust path changed — E02 would), CHANGELOG/docs, `qa-verifier` +
   `core-reviewer`, single PR to `main`. **NOTE the release-infra gotcha (now fixed):**
   both publish workflows have `skip-existing: true`, so a repleafgbm-only release no
   longer fails on the unchanged native version. **Do not push/merge/tag/publish without
   asking.**

## Loop discipline
One hypothesis → cheap evidence FIRST → implement only on **+3% median over ≥5 reps with
green tests + parity** → record in `experiment-log.md` (continue at **iter 011**) +
≤15-line GEPA reflection. Harness change every 5 product iters (separate commit,
re-baseline). Commit each accepted change separately.

## Safety / invariants
- No push/merge/tag/publish without asking; no new deps; no encoder unfreeze; no
  embedding-dim splits.
- NumPy↔Rust parity stays **bitwise** on default paths (`tests/test_rust_backend.py` —
  and **gate any RustSplitBackend-parametrized test on `find_spec("repleafgbm_native")`**,
  or the no-native CI `test` lane fails, see [[booster-picklability-rust-ci]]). CUDA is
  **allclose + quality-equivalence** (assert quality, NOT rtol=1e-6 — near-tied splits flip).
- Default changes get a deliberate decision; new public API params are human-gated.

## GPU validation (billable Colab T4; creds persist)
- `bash scripts/colab_gpu_test.sh --gpu T4 --keep` → parity + matrix → reports to
  `experiments/results/`. With `--keep`: `colab exec -s rlgbm-gpu --timeout 1800 -f
  scripts/colab_batched_ab.py` (depthwise A/B) and `scripts/colab_sizing.py` (phase-share
  sizer — extend for a **leafwise** A/B), `colab download …`, then ALWAYS
  `colab stop -s rlgbm-gpu`. Batch all GPU work into ONE `--keep` session.

## Commands
- `OMP_NUM_THREADS=1 PYTHONPATH=src python3 -m pytest tests/ -q` (OMP=1 avoids the
  torch+lightgbm libomp deadlock + single-threads BLAS so wide-emb leaf_fit shares are
  real-world-representative). `ruff check src tests benchmarks scripts`. `bash scripts/check.sh`.
- `bash scripts/perf_loop.sh --mode ab …` / `python -m benchmarks.cuda_overnight_loop
  --mode ab …` (median+spread+signal; `--task multioutput --precision-a/-b`,
  `--n-features/--max-leaf-emb-dim/--leaf-model/--n-estimators/--n-train/--n-test`).
- Parity: `tests/test_rust_backend.py` (bitwise), `tests/test_batched_scan.py`,
  `tests/test_cuda_backend.py` (GPU-only, runs on Colab).

## Subagents (fleet in `.claude/agents/`)
`experiment-strategist` (prioritize), `cuda-researcher` (external ideas), `perf-profiler`
(run+measure), `native-optimizer` (Rust/backends/CUDA impl — owns Task-B + E02),
`qa-verifier` (green-gate), `core-reviewer` (architecture/SemVer/parity sign-off),
`results-analyst` (run verdict). Route CUDA/Rust impl to `native-optimizer`; gate every
source change with `qa-verifier` + `core-reviewer`.

Begin: read the ledgers, run `experiment-strategist`, and present a plan (ExitPlanMode)
before implementing.
