# Next-session prompt (hand to Claude Code)

Paste the block below into a fresh Claude Code session on this repo.

---

You are continuing the RepLeafGBM CUDA/perf overnight loop. The previous session
shipped and **GPU-validated** several wins; this session moves into the next round
of implementation, driven by what those experiments found.

**Start in plan mode.** First read the source-of-truth ledgers, then confirm the
top priorities with `experiment-strategist`, then present a plan, then implement.

## Branch & state
- Work on branch `perf/cuda-overnight-loop-20260624` (do NOT touch `main`). It has
  ~17 commits ahead of the v1.8.0 release base. `git status` should be clean except
  the untracked `docs/gpu-research/` (leave it).
- `.claude/` and `artifacts/` are gitignored (agent defs + bench outputs live there
  but aren't committed).

## Read first (the complete record of last session's experiments)
- `docs/perf-notes/experiment-log.md` — iter 001–007 verdicts (what shipped/rejected/held).
- `docs/perf-notes/reflections.md` — GEPA reflections + the meta-lessons.
- `docs/perf-notes/experiment-backlog.md` — the ~30-hypothesis backlog + status.
- `docs/perf-notes/rejected-ideas.md` — settled dead ends (do NOT re-litigate).
- `docs/perf-notes/research-node-batched-split-scan.md` + `docs/proposals/*.md`.
- `docs/perf-notes/harness-log.md` — harness version + the BLAS-thread caveat.

## Already SHIPPED + validated last session (do NOT redo)
1. **float32 opt-in leaf-fit** — `leaf_fit_precision="float64"|"float32_gram"`
   (public param, default float64). 1.18× wide-emb fit; applies only to the
   **emb>128** BLAS path now (see #2). allclose-not-bitwise on the opt-in path.
2. **native leaf-fit gate 64→128** (`_NATIVE_STATS_MAX_DIM`). Default, float64,
   quality-identical; **1.65× fit @emb=128**. (Re-measuring a "tuned" gate was the
   night's biggest lever — keep that instinct.)
3. **node-batched CUDA split scan** — `find_best_split_batched` +
   `_grow_depthwise_batched`, opt-in `REPLEAFGBM_CUDA_BATCHED_SCAN` (default OFF).
   Colab T4: split_scan **5–9×**, whole depthwise fit **1.9–3.9×**, quality
   identical, parity 35/35. The CUDA path is a plain CuPy M-axis lift of
   `find_best_split` (NO RawKernel). The host grower is bitwise-identical to FIFO.

## Implementation tasks (confirm/re-order with experiment-strategist)
1. **Flip `REPLEAFGBM_CUDA_BATCHED_SCAN` default → ON** for cuda+depthwise. It is
   already T4-validated (3.9×, quality-identical) and the MO device-scan precedent
   (`REPLEAFGBM_CUDA_MO_DEVICE_SCAN`) defaults on; the CUDA path is already
   allclose. Update `_resolve_batched_scan` + the test that asserts default-off
   (`tests/test_cuda_backend.py::test_batched_scan_gate_off_loops_per_node` — keep a
   kill-switch test) and `tests/test_batched_scan.py::test_default_supports_batched_scan_is_false`.
   This is a **default-behavior change** → core-reviewer sign-off + a Colab re-validation.
2. **Extend node-batching to leafwise** (the DEFAULT grow_policy → the broad win).
   Leafwise pops one best-gain node at a time, so there is no natural level; design
   the frontier-batch with `core-reviewer` first (it is harder than depthwise — the
   priority-queue ordering must be preserved bitwise on the host). Then implement +
   Colab A/B. This is the highest-value extension; treat its host parity exactly
   like `_grow_depthwise_batched` (bitwise vs the per-node FIFO).
3. **Batched `build_histograms`** — build a level's M node histograms in one device
   call to feed the batched scan (cuts launch overhead on the build side too).
4. **Remaining backlog levers (cheap-test-first):** E14 float32 `predict_linear`,
   E15 float32 multi-output (vector) leaves, E17 float32 embedding cache, E02
   native-rust wide-emb float32 Gram (emb>128). Reject/hold per the criteria.
5. **Package for merge:** the validated wins are independent — split into reviewable
   PRs to `main` (e.g. (a) float32 param + gate 64→128 leaf-fit; (b) node-batched
   scan). Bump versions per `release-infra-v102` memory (repleafgbm + native if the
   Rust path changed — it did NOT this round, so likely just repleafgbm MINOR),
   update CHANGELOG/docs, run `qa-verifier` (green) + `core-reviewer` (sign-off)
   before any commit you intend to merge. **Do not push/merge/tag/publish without asking.**

## Loop discipline (keep it)
Per iteration: one hypothesis → cheap evidence FIRST → implement only on **+3%
median over ≥5 reps with green tests + parity held** → record the verdict in
`experiment-log.md` (continue iter numbering at 008) + a ≤15-line GEPA reflection.
Harness change every 5 product iters (separate commit, re-baseline). Commit each
accepted change separately on the branch.

## Safety / invariants (non-negotiable)
- No push/merge/tag/PyPI publish; no new dependencies; no encoder unfreeze; no
  splitting on embedding dims.
- NumPy↔Rust parity stays **bitwise** on default paths; CUDA is **allclose +
  quality-equivalence** (assert quality, NOT rtol=1e-6 — near-tied splits flip).
- A new public API param is human-gated (the `leaf_fit_precision` param was already
  approved; anything new asks first). Default changes get a deliberate decision.
- Keep CuPy/torch/lightgbm out of the native (Rust) path.

## GPU validation (billable Colab T4; creds persist)
- Parity + benchmarks: `bash scripts/colab_gpu_test.sh --gpu T4 [--keep]` runs
  `tests/test_cuda_backend.py` + the matrix on a T4 and downloads reports to
  `experiments/results/`. With `--keep` the VM stays up.
- Batched A/B: after `--keep`, `colab exec -s rlgbm-gpu --timeout 1800 -f
  scripts/colab_batched_ab.py` then `colab download -s rlgbm-gpu
  /content/batched_ab_report.md experiments/results/<date>-...md`, then **`colab
  stop -s rlgbm-gpu`** (always tear the VM down — it is billable).
- The Colab loop is fire-and-forget (upload→run→download); batch related GPU work
  into one `--keep` session.

## Commands
- Tests: `OMP_NUM_THREADS=1 PYTHONPATH=src python3 -m pytest tests/ -q`
  (`OMP_NUM_THREADS=1` avoids the torch+lightgbm libomp deadlock; it also makes
  BLAS single-threaded, so wide-emb leaf_fit shares look larger than multi-threaded
  — `env.threads`/`env.blas` are now logged per bench row).
- Lint: `ruff check src tests benchmarks`. Full gate: `bash scripts/check.sh`.
- Local perf A/B: `bash scripts/perf_loop.sh --mode ab ...` (median+spread+signal;
  supports `--n-features/--max-leaf-emb-dim/--leaf-model/--n-estimators/--precision-a/-b`).
- Parity: `tests/test_rust_backend.py` (bitwise), `tests/test_batched_scan.py`,
  `tests/test_cuda_backend.py` (GPU-only, runs on Colab).

## Subagents available (fleet in `.claude/agents/`)
`experiment-strategist` (prioritize), `cuda-researcher` (external ideas),
`perf-profiler` (run+measure, returns numbers), `native-optimizer` (Rust/backends/
CUDA impl), `qa-verifier` (green-gate), `core-reviewer` (architecture/SemVer/parity
sign-off), `results-analyst` (verdict on a run). Route CUDA/Rust impl to
native-optimizer; gate every source change with qa-verifier + core-reviewer.

Begin: read the ledgers, run `experiment-strategist`, and present a plan
(ExitPlanMode) before implementing.
