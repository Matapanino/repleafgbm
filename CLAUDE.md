# CLAUDE.md — RepLeafGBM development rules

## Project summary

RepLeafGBM is a research-oriented tabular ML library that combines raw-feature
tree routing with representation-conditioned leaf models. It is a boosted
ensemble of raw-feature routers with representation-conditioned local
predictors — not a neural network inside a tree, and not a tree over
embeddings.

## Core architectural rule

- Tree routing (splits) uses **raw features only**.
- Leaf prediction may use learned representations `z_theta(x)`.
- The encoder is **frozen** in v0: it is fitted once before boosting.
- Do **not** update encoder parameters during boosting in v0 — past trees'
  leaf outputs depend on `z_theta(x)`, so updating theta breaks the
  stage-wise additive assumption (see docs/math.md).
- Do **not** split on embedding dimensions in v0.

## v0 priority

1. Correctness
2. Readability
3. Minimal native NumPy backend
4. Regression first (binary classification second)
5. Constant leaf and embedded linear leaf
6. Dataset abstraction (`RepLeafDataset`)
7. Save/load (directory format)
8. Tests and docs

## Avoid

- No premature CUDA implementation
- No premature multi-GPU implementation
- No premature distributed framework
- No wrapper-only implementation around LightGBM/XGBoost/CatBoost
- No giant monolithic class
- No notebook-only code
- No hidden global state
- No dataset-specific hacks
- No over-engineered abstraction that obscures the core idea

## Code map

- `src/repleafgbm/data/` — `RepLeafDataset`, `FeatureMetadata`, preprocessing.
  Owns data + metadata; lazily computes/caches embeddings.
- `src/repleafgbm/encoders/` — `BaseEncoder` and implementations
  (identity, simple PLR with linear term, PBLD-style periodic,
  random-projection wrapper, and learned torch_periodic/torch_plr in
  `torch_encoders.py`). All frozen during boosting; torch encoders import
  torch only inside fit (transform/serialization are NumPy — keep it that
  way), and the native path must never import torch.
- `src/repleafgbm/core/` — objectives, metrics, histogram binning, splitter,
  tree grower, leaf models, booster, prediction, serialization.
- `src/repleafgbm/backends/` — split-search kernels behind `BaseSplitBackend`:
  `NumPySplitBackend` (reference), `RustSplitBackend` (optional compiled
  extension in `native/`, pyo3/maturin), and `CudaSplitBackend` (optional GPU
  histogram + resident-histogram numeric split scan via CuPy,
  `split_backend="cuda"`; multi-output scan is device-resident too, only the
  categorical subset scan stays on host; ADR 0005, docs/cuda.md). `native/`
  also provides the fused `leaf_linear_stats` helper used by
  `core/leaf_models.py` for narrow embeddings. NumPy and Rust paths must stay
  parity-tested with **bitwise** histograms (allclose leaf fits/predictions);
  change them together. The CUDA path is parity-tested at **allclose, not
  bitwise** (GPU atomic-add reordering) and only on a GPU via the Colab dev
  loop (`scripts/colab_gpu_test.sh`) — CI/macOS skip it; "auto" never selects
  it. Keep CuPy out of the native (Rust) path; like torch/lightgbm it is an
  optional dependency.
- `src/repleafgbm/sklearn.py`, `regressor.py`, `classifier.py` — the
  sklearn-compatible public API that glues dataset + encoder + booster.
- `src/repleafgbm/external/` — external_model mode (LightGBM base model,
  OOF, stacking features). Optional dependency; the native path must never
  import it.
- `benchmarks/` — synthetic comparison harness (sklearn + optional external
  GBMs). `experiments/` — research scripts; each writes a markdown report to
  `experiments/results/` and its conclusions feed docs/roadmap.md.

Conventions baked into the core (keep them consistent):

- Leaf fitting uses Newton targets `t = -g/h` with weights `h` everywhere.
- Missing values (NaN) always route **left** in *native training*; `Tree`
  carries per-node `missing_left` so extracted external routes (LightGBM
  `default_left`) are represented exactly.
- Categorical features are ordinal-encoded to float (unseen categories →
  NaN) and routed with native gradient-sorted **subset splits**
  (`Tree.left_categories`); ordered-threshold fallback above `max_bins`
  categories.
- `random_state` flows through `utils.random.check_random_state`; never call
  global NumPy random functions.

## Coding style

- Use type hints on public functions and classes.
- Prefer small modules over large ones.
- Prefer explicit interfaces; use `abc.ABC` or `typing.Protocol` where useful.
- Keep core algorithms readable — a new contributor should be able to follow
  the boosting loop in `core/booster.py` top to bottom.
- Add tests for new behavior.
- Keep examples small and runnable in seconds.
- Use deterministic `random_state`; same seed ⇒ same model.
- Docstrings on every public class; meaningful error messages that say what
  the user should do.

## Testing

- One command for everything: `bash scripts/check.sh` (lint + tests + examples).
- Use pytest: `PYTHONPATH=src python3 -m pytest tests/ -q`
  (or `pip install -e .` once and drop the PYTHONPATH).
- Lint with `ruff check src tests examples benchmarks` (config in pyproject).
- Run the test suite after any implementation change.
- Keep synthetic datasets small (hundreds of rows) and seeded.
- Required coverage for new model behavior: fit/predict, save/load
  round-trip, encoder transform shapes, constant-fallback for small leaves.

## Documentation

- Update docs/ when architecture changes; ADRs in docs/adr/ for decisions.
- Keep docs/roadmap.md honest: clearly distinguish implemented features from
  future plans.
- Document mathematical assumptions in docs/math.md.
- Document known limitations explicitly.

## Long-term goals (do not implement prematurely; keep designs compatible)

- Public OSS library quality (CI, contribution guide, benchmarks)
- sklearn-compatible API (already in place; preserve it)
- LightGBM/XGBoost/CatBoost backend diversity (docs/backend_strategy.md)
- Rust/C++/CUDA native backend behind `backends/`
- GPU / multi-GPU / distributed training
- Benchmark suite

## Subagents

A specialized, context-isolated subagent fleet lives in `.claude/agents/`. Each agent
is scoped to a RepLeafGBM task (not a generic job title), carries the core invariants in
its own prompt (so it starts correct from cold), and owns a single write surface. Models
are tiered: Opus for judgment/design, Sonnet for execution.

| agent | model | owns / when to invoke |
|---|---|---|
| `literature-scout` | sonnet | External papers/methods/competitor techniques → dated note in `docs/research/`. Invoke before proposing/implementing something new. |
| `experiment-runner` | sonnet | Runs `experiments/*.py` + `benchmarks/*.py` (multi-seed, detached OK), ensures the report lands in `experiments/results/`. Invoke to *run* a study. Does not interpret. |
| `results-analyst` | opus | Interprets `experiments/results/*` into an evidence-backed verdict (keep/change a default, null result, follow-up). Invoke after a run finishes. |
| `research-proposer` | opus | Designs a new encoder/objective/metric/experiment as a spec in `docs/proposals/` (with code-map touch points + validating experiment). Invoke to plan a direction. |
| `core-reviewer` | opus | Architectural + correctness review of a diff against the invariants, ADRs, and test adequacy. Invoke before committing/merging source. |
| `native-optimizer` | opus | Rust/`native/` + `backends/` performance; keeps NumPy⇄Rust parity green (both paths change together). Invoke for kernel/perf work or parity failures. |
| `qa-verifier` | sonnet | Green-gate: runs `scripts/check.sh`/ruff/pytest with `OMP_NUM_THREADS=1`, fixes only mechanical (lint/import/format) failures, escalates the rest. Invoke as the pre-commit gate. |
| `agent-architect` | opus | Meta-agent that owns the fleet: creates/edits agents, audits overlap/utilization, proposes merges/splits, keeps this section in sync. Edit agent files through it, not by hand. |
| `cuda-researcher` | sonnet | GPU-perf scout (XGBoost/LightGBM/CatBoost GPU, cuML/cuDF/CuPy/Numba/RAPIDS, torch/TF kernels) → transferable speedup hypotheses in `docs/perf-notes/research-*.md`. Invoke before a GPU push. GPU-scoped twin of `literature-scout`. |
| `perf-profiler` | sonnet | Runs `benchmarks/gpu_profile.py`/`predict_profile.py`/`cuda_overnight_loop.py` (`REPLEAFGBM_PROFILE=1`, `get_transfer_stats`), keeps logs in-context, returns compact per-phase seconds + median/spread (>=5 reps) + bottleneck verdict. No edits. |
| `harness-optimizer` | sonnet | Improves the measurement harness only (`benchmarks/*`, `scripts/*`, `benchmarks/results/*`, `docs/perf-notes/harness-log.md`) — reproducibility, schema, baselines; never edits `src/`/`native/`; separate commit; never deletes cases. |
| `experiment-strategist` | opus | Reads the perf-note ledgers + `docs/gpu_roadmap.md`, runs a GEPA-style reflection, returns EXACTLY 3 prioritized next-experiment hypotheses (rationale + expected signal + implementer). No implementation/runs. |

Hybrid routing for the perf loop: CUDA/Rust implementation → **native-optimizer**
(CUDA-scoped); regression/sign-off → **qa-verifier** + **core-reviewer**; run verdicts →
**results-analyst**; the `docs/perf-notes/` experiment/reflection ledger is owned by the
loop driver.

### Invocation examples
- "Use `experiment-runner` to run the OpenML suite (`--quick`), then `results-analyst` to give the verdict."
- "Have `literature-scout` survey periodic feature encoders, then `research-proposer` turn the strongest idea into a spec."
- "Run `native-optimizer` to add rayon to the histogram kernel; it must keep `test_rust_backend.py` parity green."
- "Before commit: `qa-verifier` for green, then `core-reviewer` for sign-off."
- "Ask `agent-architect` to audit the fleet for overlap."

### Recommended workflows
- **Research loop:** `literature-scout` → `research-proposer` → (implement) → `experiment-runner` → `results-analyst` → `core-reviewer` → `qa-verifier`.
- **Perf loop:** `native-optimizer` → `qa-verifier` (parity + suite) → `core-reviewer`.
- **Perf/overnight loop:** `cuda-researcher` → `experiment-strategist` → `native-optimizer`/`harness-optimizer` → `perf-profiler` → `qa-verifier` → `core-reviewer` (+ `results-analyst` on GPU-pass verdicts).
- **Ship loop:** `qa-verifier` (green) + `core-reviewer` (sign-off) before any commit.

### Multi-agent / parallel patterns
- Launch `experiment-runner` on a long detached run **in parallel** with `literature-scout` — they share no state.
- Fan out independent benchmarks across multiple `experiment-runner` invocations, then have one `results-analyst` aggregate the reports.
- Keep `core-reviewer` (reads/judges) and `qa-verifier` (runs/fixes) as a cost split: cheap Sonnet establishes green, Opus reasons about the diff and trusts that report.

### Best practices
- Keep each agent's scope tight; pass file paths, not pasted dumps.
- Let `agent-architect` own all agent-file edits (single source of truth).
- Always end a source change with `qa-verifier` + `core-reviewer`.
- Change a model-behavior default only with a `results-analyst`-backed report.

### Anti-patterns
- Generic role-based agents ("backend engineer"); editing agent files by hand instead of via `agent-architect`.
- Touching the Rust path without moving NumPy too / without parity tests.
- Importing torch/lightgbm/`external/` into the native path.
- Splitting on embedding dims or unfreezing the encoder during boosting (breaks the thesis).
