# ADR 0006: Tree growth policies (leafwise / depthwise / symmetric)

- Status: **accepted — implemented** (2026-06-23). All three policies fit/predict
  across `{constant, embedded_linear}` leaves and `{regression, binary,
  multiclass}` (plus `depthwise` for multi-output); NumPy⇄Rust end-to-end parity
  is green for every policy; the default stays `leafwise`.
- Date: 2026-06-23
- Depends on: ADR 0001 (core architecture / thesis), docs/math.md (Newton gain,
  routing), the `BaseSplitBackend` contract + sibling-subtraction trick.
- Research: docs/research/2026-06-22-tree-growth-policies.md (CatBoost oblivious
  trees, XGBoost grow_policy), docs/proposals/2026-06-22-grow-policy.md (spec).

## Context

RepLeafGBM grew every tree **leaf-wise** (best-gain-first heap, `num_leaves`
control). Mature GBDTs expose two more strategies with materially different
regularization/inference behavior: XGBoost's **depthwise** (level-order to
`max_depth`) and CatBoost's **symmetric/oblivious** trees (every node at a level
shares one split; complete `2**depth` tree; strong implicit regularization; fast
inference). This ADR adds a `grow_policy` hyperparameter offering all three.

This is an **algorithm-feature** addition — a standard, expected knob and a
comparison-research surface — **not** a speed optimization. It must preserve the
core thesis completely: **raw-feature routing only, representation-conditioned
leaves only, encoder frozen during boosting, never split on embedding
dimensions.** A symmetric tree still chooses its shared per-level split on raw
features; leaves remain `constant`/`embedded_linear` and are fit by the existing,
growth-policy-orthogonal `fit_leaves` / `fit_leaves_multiclass` /
`fit_vector_leaves`.

## Decisions

1. **Symmetric trees expand into the existing flat `Tree`** (not a compact
   oblivious format). All depth-`d` nodes are written as ordinary internal nodes
   sharing one `(feature, threshold)`. `Tree.apply`, serialization
   (`to_dict`/`from_dict`), prediction, and feature importance are **unchanged**,
   so `grow_policy` adds **no `format_version` bump**: the version a model writes
   is decided by ensemble type, not policy (single-output regression still writes
   **v3**; multiclass v5; multi-output v6; the `FORMAT_VERSION` constant stays 6).
   This gives up compact bitwise-indexed inference, but v0 prioritizes
   correctness and zero serialization risk; a compact format is a possible
   follow-up.

2. **The symmetric per-level split scan is one host-side NumPy function**
   (`backends/numpy_backend.find_best_level_split`), shared by all compute
   backends via thin `Splitter` wrappers — *not* a new `BaseSplitBackend` kernel.
   Histograms are bitwise-identical NumPy⇄Rust, so feeding them through one host
   scan makes end-to-end parity automatic with **no native rebuild** and no
   widening of the parity surface. Device-resident (CUDA) histograms are pulled
   to host with the existing `_as_host` helper.

3. **The symmetric objective sums per-node gains, not histograms.** Newton gain
   `G²/(H+λ)` is nonlinear, so the shared split maximizes `Σ_j gain_j(f,b)` over
   the level's nodes (docs/math.md). A candidate must satisfy `min_samples_leaf`
   at *every* node (invalid anywhere ⇒ summed gain −∞), so growth is all-or-none
   per level and the tree stays complete. Tie-break is the lowest `(feature, bin)`
   index, identical to `find_best_split`, keeping growth deterministic.

4. **`depthwise` and `symmetric` require `max_depth >= 1`** (clear `ValueError`
   otherwise). `leafwise` is unchanged (`num_leaves` primary, `max_depth` optional
   cap). `symmetric` ignores `num_leaves` (leaves = `2**depth`); `depthwise` keeps
   `num_leaves` as an optional secondary cap (raise it for a full depth-`d` tree).

5. **v0 scope.** `symmetric` supports numeric/ordered splits and scalar targets
   (regression, binary, and one-tree-per-class multiclass). Categorical features
   route as ordered thresholds (no gradient-sorted subset splits); multi-output
   targets raise `NotImplementedError`, enforced centrally in
   `TreeGrower._grow_symmetric` (`grad.ndim > 1`) — a single source of truth that
   fires on the first `grow()` call, before any tree is committed, rather than a
   duplicated booster-level guard. `depthwise` reuses `find_best_split` (and its
   multi-output dispatch), so it covers **all** task types day one.

## Implementation

- `core/tree.py`: `TreeGrower` gains `grow_policy` and dispatches to
  `_grow_leafwise` / `_grow_depthwise` / `_grow_symmetric`. Shared scaffolding
  (`_NodeStore`, `_commit_split`, `_child_hists`, `_expand`, `_finalize`,
  `_make_candidate`) keeps each policy short and the leaf-wise path
  **byte-identical** to before (depthwise swaps the gain-heap for a FIFO `deque`;
  symmetric drives the level scan). No heavy Strategy hierarchy.
- `backends/numpy_backend.py`: extracted `_numeric_split_table` (the shared
  numeric scan now feeding both `find_best_split` — a pure refactor — and the
  level scan), added `find_best_level_split` and `split_at`.
- `core/splitter.py`: host-side wrappers `find_best_level_split` / `split_at`.
- Plumbing: `grow_policy` on `BoosterParams`, threaded by the three boosters and
  the sklearn estimators; it serializes automatically via `get_params()`.

## Consequences

- A standard, expected knob plus a comparison-research surface, with the thesis
  fully intact and no format/parity regression.
- Symmetric inference is *not* faster than leaf-wise here (it uses the general
  `Tree.apply`); the compact oblivious representation that would make it so is
  deferred.
- Known limitations (also in docs/roadmap.md): symmetric is numeric/scalar-only
  in v0; categorical-subset and multi-output symmetric, and a compact oblivious
  storage/inference path, are follow-ups. CUDA + symmetric is plumbed (host scan
  via `_as_host`) but unvalidated on GPU in this milestone.
- The default remains `leafwise`. The synthetic multi-seed study
  (`experiments/results/2026-06-23-grow-policy-verdict.md`) confirmed keeping it:
  symmetric's broad wins there are an oblivious-friendly-design artifact (planted
  low-order axis-aligned structure, no real data, leaf-wise capacity plausibly
  handicapped), decisive only on clean/large piecewise + multiclass. A real-data
  comparison is the required gate before any default change.
