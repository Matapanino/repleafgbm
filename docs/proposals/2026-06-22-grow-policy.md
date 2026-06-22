# Proposal: Tree growth policies (`grow_policy`: leafwise / depthwise / symmetric)

- **Status:** Approved design (locked via plan + AskUserQuestion). This document
  formalizes the approved plan
  (`~/.claude/plans/repleafgbm-tree-growth-policy-shimmying-gadget.md`) into the
  proposal/ADR spec format. It does not redesign it.
- **Date:** 2026-06-22
- **Author:** research-proposer
- **Type:** Algorithm-feature addition (new growth knob + comparison-research surface),
  **not** a performance optimization.
- **Thesis check:** PASS — no thesis/architecture violation found (see
  [Guardrail / thesis check](#guardrail--thesis-check)). Splits stay raw-feature-only,
  leaves stay representation-conditioned, encoder stays frozen, no embedding-dim splits,
  no `format_version` bump, no native rebuild.
- **Companion ADR:** `docs/adr/<next>-tree-growth-policies.md` (decisions A–D; draft to be
  written alongside implementation).
- **Source of truth:** the approved plan above. Where this spec and the plan disagree, the
  plan wins.

---

## 1. Problem

RepLeafGBM grows every tree **leaf-wise** (best-gain-first heap, `num_leaves` control,
sibling subtraction) in `core/tree.py::TreeGrower.grow`. That is the only growth strategy
available. Mature GBDT libraries expose two more, with materially different
regularization / inference behavior:

- **depthwise** (XGBoost `grow_policy=depthwise`): level-order expansion to `max_depth`; a
  familiar, balanced baseline.
- **symmetric / oblivious** (CatBoost): every node at a given level uses the *same*
  `(feature, threshold)`; a depth-`d` tree is complete with `2^d` leaves. The forced
  uniformity is a strong implicit regularizer (helps on small / noisy data).

Two gaps follow: (a) RepLeafGBM cannot answer "how does growth policy interact with
representation-conditioned leaves?", which is exactly the comparison-research surface this
library exists to study; (b) practitioners who expect `grow_policy` as a standard knob do
not find it. This addition closes both **without touching the thesis**.

**Outcome.** `grow_policy ∈ {"leafwise"(default), "depthwise", "symmetric"}` across
`{constant, embedded_linear}` leaves and `{regression, binary, multiclass}` (+ depthwise
for multi-output), with parity / determinism / round-trip guarantees intact, plus a
multi-seed experiment telling us *where* each policy helps. **Default stays `leafwise`**
unless a `results-analyst` report justifies otherwise.

---

## 2. Hypothesis (why it could help *this* architecture)

The thesis fixes routing to raw features and pushes capacity into the leaf model
`z_theta(x)`. Growth policy changes the *partition* the leaves are fit on:

- **symmetric** imposes a low-complexity, balanced partition (one `(feature, threshold)`
  per level, `2^depth` leaves). On small / noisy / low-signal tabular data the leaf-wise
  heap can chase spurious high-gain raw-feature splits and starve siblings; the oblivious
  constraint regularizes the *routing* while the representation-conditioned leaf still
  supplies local capacity. Hypothesis: **symmetric is the best regularizer on small/noisy
  problems**, and its forced balance pairs well with linear leaves (every leaf sees a
  comparable, non-degenerate row count, reducing constant-fallback churn).
- **depthwise** gives a balanced baseline whose depth is the single complexity knob — a
  clean control against which to attribute any leaf-wise win to *adaptivity* rather than
  to depth.
- **leafwise** (unchanged) remains expected-strongest on larger / structured data where
  adaptive deep splits on the most informative raw features pay off.

This is a *structure* lever (the routing partition) orthogonal to the leaf model, so it
exercises a degree of freedom trees+frozen-encoder currently expose only as leaf-wise.

---

## 3. Verified objective functions

Captured from CatBoost (arXiv 1706.09516), XGBoost params / issue #9123, NODE
(arXiv 1909.06312), apxml; see the literature-scout note
`docs/research/<date>-tree-growth-policies.md`.

### 3.1 Newton gain (existing, unchanged)

Per the existing scan (`backends/numpy_backend.py::_leaf_score`, `docs/math.md`):

```text
leaf_score(G, H) = G^2 / (H + λ)
gain(split)      = leaf_score(G_L, H_L) + leaf_score(G_R, H_R) - leaf_score(G, H)
```

with `min_samples_leaf` enforced on both children and missing values fixed left.

### 3.2 Symmetric / oblivious objective (the new math)

At each level, all current frontier nodes must share **one** `(feature, bin)`. The chosen
candidate **maximizes the SUM of per-node Newton gains** across all level nodes:

```text
for a candidate (f, b) valid at every level node n:
    level_gain(f, b) = Σ_n  gain_n(f, b)
                     = Σ_n [ leaf_score(G_{n,L}, H_{n,L})
                           + leaf_score(G_{n,R}, H_{n,R})
                           - leaf_score(G_n,    H_n) ]
choose argmax_{(f,b)} level_gain(f, b)   (lowest-index tie-break), apply to every node.
```

**Critical correctness point (verified): gain is nonlinear in the histogram, so you sum
*gains*, not histograms.** `leaf_score(G,H)=G²/(H+λ)` is not additive across nodes, hence
pooling the level's histograms and splitting the pooled histogram would optimize the wrong
objective. The per-node gain tables are computed independently and *then* summed.

- **Validity is global / all-or-none per level.** A candidate `(f,b)` is admissible only
  if it satisfies `min_samples_leaf` on *both* children at *every* level node. A candidate
  invalid at any node is dropped (gain → −∞). This keeps the tree **complete**: `2^depth`
  leaves, leaf index = depth-bit pattern.
- **Stop conditions:** stop a level if not every frontier node is splittable
  (`_can_split`), or if no globally-valid candidate has summed gain `> 1e-12` (matching the
  existing scan's positive-gain floor), or `depth == max_depth`. Remaining frontier nodes
  become leaves via the shared finalize.
- **Why per-level split is node-independent at storage time:** `threshold_value(feature,
  bin)` uses per-feature *global* quantiles (node-independent), and v0 symmetric splits are
  numeric so `left_categories=None`. Therefore expanding the shared `(feature, bin)` into
  the flat `Tree` yields the *identical* `(feature, threshold)` on every level-`d` node —
  this is what makes the general-`Tree` expansion (decision A) exact.

### 3.3 depthwise vs lossguide (= leafwise)

depthwise expands level-order (FIFO) to `max_depth`, pruning nodes with no positive-gain /
`min_samples_leaf`-valid split. lossguide is exactly RepLeafGBM's current leaf-wise growth.
Both reuse the **existing** `find_best_split` (which already dispatches to
`find_best_split_multioutput` for 4-D histograms), so depthwise covers
regression / binary / multiclass / multi-output with no special-casing.

---

## 4. Confirmed design decisions (asked & answered — mirror of the plan)

| # | Decision | Choice |
|---|---|---|
| **A** | Symmetric tree storage | **Expand into the existing flat-array `Tree`** (all level-`d` nodes share one `(feature, threshold)`; `2^d` leaves). `apply()`, serialization, prediction, feature-importance unchanged. `FORMAT_VERSION` stays **6**. (Compact oblivious format + bitwise inference = post-v0.) |
| **B** | Symmetric split scan | **One host-side NumPy function**, shared by both backends. `build_histograms` stays bitwise-identical NumPy⇄Rust, so end-to-end parity is automatic; **no native rebuild**. |
| **C** | Symmetric v0 scope | **Numeric (ordered-threshold) splits + scalar targets** (regression / binary / multiclass via K independent trees). Categorical features handled as **ordered thresholds** (no gradient-sorted subset splits). **symmetric + multi-output → `NotImplementedError`.** depthwise supports **all** tasks day one. |
| **D** | `max_depth` contract | depthwise & symmetric **require `max_depth ≥ 1`** (else `ValueError`). leafwise unchanged (`num_leaves` primary, `max_depth` optional cap). symmetric **ignores** `num_leaves` (leaves = `2^depth`); depthwise keeps `num_leaves` as an **optional secondary cap** (documented). |

---

## 5. Design

### 5.1 Hyperparameter plumbing (mechanical — mirrors `num_leaves`)

- `core/booster.py::BoosterParams` — add `grow_policy: str = "leafwise"`.
- `core/booster.py::Booster.fit`, `core/multiclass.py::MulticlassBooster.fit`,
  `core/multioutput.py::MultiOutputBooster.fit` — each currently constructs
  `TreeGrower(splitter, num_leaves=p.num_leaves, max_depth=p.max_depth)`; add
  `grow_policy=p.grow_policy`.
- `core/multioutput.py::MultiOutputBooster.fit` — **fail fast** before the grow loop: if
  `p.grow_policy == "symmetric"`, raise
  `NotImplementedError("grow_policy='symmetric' does not support multi-output regression in v0; use 'leafwise' or 'depthwise'.")`.
- `sklearn.py::BaseRepLeafModel.__init__` — add `grow_policy: str = "leafwise"` arg +
  `self.grow_policy = grow_policy`; in `BaseRepLeafModel.fit` add
  `grow_policy=self.grow_policy` to the `BoosterParams(...)` build (currently
  `sklearn.py:274–287`, which does **not** pass it today).
- `regressor.py`, `classifier.py` — **no change required.** `RepLeafRegressor` and
  `RepLeafClassifier` do not define their own `__init__` (they inherit
  `BaseRepLeafModel.__init__`), and `_make_booster` only constructs the booster from the
  already-built `BoosterParams`. So adding the arg to the base `__init__` + threading it
  into `BoosterParams` is sufficient for both. (Plan §1 lists regressor/classifier in the
  7-step pattern; verified against the current code, the single-base edit covers them — no
  per-subclass `__init__` exists to touch.)
- `get_params()` / save / load — **automatic.** sklearn introspects `__init__`, so
  `grow_policy` lands in `model_config.json` and round-trips with no `format_version` bump.

### 5.2 `TreeGrower` refactor — shared scaffolding, then 3 policies (`core/tree.py`)

Add `grow_policy: str = "leafwise"` to `TreeGrower.__init__`; **validate there** (single
source of truth): `grow_policy` must be in the allowed set (else `ValueError`), and
`max_depth >= 1` when `grow_policy in {"depthwise", "symmetric"}` (else `ValueError`, clear
message). `grow(grad, hess)` becomes a thin dispatcher to `_grow_leafwise` /
`_grow_depthwise` / `_grow_symmetric`, each returning `(Tree, list[np.ndarray])` exactly as
today.

Extract the parts every policy shares (keeps `core/booster.py`'s loop readable — CLAUDE.md;
line numbers reference the current `core/tree.py`):

- `_new_store() -> dict` → the growable node lists
  (`feature/threshold/left/right/gain/left_cats`) + `node_rows={0: all_rows}`
  (current ~171–174).
- `_apply_split(store, node_index, split, rows_l, rows_r) -> (li, ri)` → the existing
  "append two child slots; set this node's `feature`; set `left_categories` (via
  `split.left_categories.astype(float64)`) **or** `threshold` (via
  `splitter.threshold_value(split)`); set `left/right/gain`; update `node_rows`" block
  (current ~186–209). Returns the two child indices.
- `_finalize(store) -> (Tree, list[np.ndarray])` → the leaf-id-by-sorted-node assignment +
  `Tree(...)` construction (current ~231–252). `missing_left` stays all-True (v0 rule);
  `left_categories` is `None` unless some node set one.
- `_make_candidate(node_index, rows, depth, hist) -> _GrowCandidate | None` → the current
  `_push_candidate` body but **returns** the candidate (or `None` if
  `splitter.find_best_split(hist) is None`); the caller chooses heap vs FIFO.
- `_child_hists(grad, hess, parent_hist, rows_l, rows_r, child_depth) -> list[tuple]` → the
  sibling-subtraction block (build the smaller child's histogram directly, derive the
  larger by subtraction; current ~211–225). Returns only children that pass `_can_split`,
  each as `(node_index_placeholder_or_rows, hist, depth)` — exact tuple shape chosen by the
  implementer to feed both the heap and the FIFO paths; it must reuse the existing
  smaller-build/subtract rule unchanged.

**`_grow_leafwise`** — the current algorithm verbatim, re-expressed via the helpers: heap of
`_make_candidate` results, pop best gain, `_apply_split`, push splittable children via
`_child_hists` + `_make_candidate`, stop at `num_leaves`. **Behavior must be byte-identical
to today** — the existing suite must stay green with no numeric drift (regression guard).

**`_grow_depthwise`** — replace the priority heap with a **FIFO `collections.deque`**
(level-order). Pop front, `_apply_split`, enqueue splittable children. Stop when the frontier
empties (every leaf hit `max_depth` or had no valid split) **or** the optional `num_leaves`
cap is reached. Reuses `splitter.find_best_split`, which already dispatches to
`find_best_split_multioutput` for 4-D histograms ⇒ depthwise covers
**regression / binary / multiclass / multi-output** with zero special-casing.

**`_grow_symmetric`** — the core new work (scalar only). Reference algorithm (from the plan;
pseudocode, names match the new helpers below):

```text
if grad.ndim > 1:
    raise NotImplementedError(...)          # symmetric is scalar-only in v0
root_hist = splitter.build_histograms(all_rows, grad, hess)
level = [(node_index=0, rows=all_rows, hist=root_hist)]
for depth in range(max_depth):
    if not all(_can_split(rows, depth) for _, rows, _ in level):
        break                               # all-or-none: any unsplittable node stops the level
    fb = splitter.find_best_level_split([h for _, _, h in level])    # NEW (host-side)
    if fb is None:
        break                               # no globally-valid (f,b) with positive summed gain
    feature, bin = fb
    next_level = []
    for node_index, rows, hist in level:
        split = splitter.split_at(hist, feature, bin)                # NEW per-node SplitCandidate @ fixed (f,b)
        rows_l, rows_r = splitter.partition(rows, split)
        li, ri = _apply_split(store, node_index, split, rows_l, rows_r)
        # same smaller-build/subtract rule as leaf-wise (sibling subtraction):
        hist_l, hist_r = <build smaller child, subtract for the larger>
        next_level += [(li, rows_l, hist_l), (ri, rows_r, hist_r)]
    level = next_level
return _finalize(store)                      # remaining `level` nodes (+ early-stop) become leaves
```

All level-`d` nodes get the **identical** `(feature, threshold)` because
`threshold_value(feature, bin)` is node-independent (per-feature global quantiles) and
`left_categories=None` (numeric). Missing values route left (consistent v0 rule).
Determinism comes from the `argmax` row-major lowest-index tie-break inside
`find_best_level_split`. The sibling-subtraction in the per-node loop must reuse the same
smaller-build/subtract logic as leaf-wise (factor it into `_child_hists` or an inline twin)
so symmetric inherits the exact histogram arithmetic — keeping NumPy⇄Rust parity automatic.

### 5.3 New host-side split functions (`backends/numpy_backend.py` + thin `Splitter` wrappers)

Per decision B these are **plain host-side NumPy module functions**, *not* methods on the
`BaseSplitBackend` interface (Rust / CUDA untouched). Histograms arrive as host arrays for
numpy/rust; defensively `_as_host(...)` each input (the helper already lives in
`core/splitter.py`) so a CUDA-resident hist also works, though CUDA-symmetric is left
**unvalidated** this milestone.

1. **Extract** `_numeric_split_table(hist, n_bins_per_feature, min_samples_leaf, l2) ->
   (gain[F,B], valid[F,B])` from the existing numeric scan in
   `NumPySplitBackend.find_best_split` (the cumsum + missing-left + `_leaf_score` math,
   current `numpy_backend.py:67–95`). **Refactor `find_best_split` to call it** (then do
   `argmax` + the categorical subset scan exactly as today) ⇒ single source of truth.
   - This is a **pure refactor** (bitwise-identical numbers) but it touches the
     parity-critical reference path, so it is **explicitly gated** on the existing
     NumPy⇄Rust parity tests + `core-reviewer` sign-off.
   - **Categorical exclusion stays inside `find_best_split` only**: the
     `valid &= ~categorical_mask[:, None]` line is applied *after* `_numeric_split_table`
     returns, so the extracted helper itself treats every feature as ordered/numeric. This
     is precisely the "categorical ordered-threshold fallback" the symmetric path needs.
2. `find_best_level_split(hists, n_bins_per_feature, min_samples_leaf, l2) ->
   tuple[int, int] | None`:
   - for each node histogram call `_numeric_split_table` (**all features treated as
     ordered/numeric** — the categorical fallback for symmetric v0);
   - `global_valid = AND over nodes of per-node valid`  → `(F, B)` bool;
   - `summed = Σ_node gain[node]` → `(F, B)`;
   - `summed[~global_valid] = -inf`; `best_flat = argmax(summed)` (row-major ⇒ lowest
     feature, then lowest bin — same tie-break as `find_best_split`);
   - return `None` if no globally-valid candidate or best summed gain `<= 1e-12`, else
     `divmod(best_flat, n_bins_max)` as `(feature, bin)`.
   - **`min_samples_leaf` is enforced by global invalidation** — a `(f,b)` invalid at *any*
     node is dropped, so the tree stays complete / all-or-none per level.
3. `split_at(hist, feature, bin, n_bins_per_feature, min_samples_leaf, l2) ->
   SplitCandidate`:
   - per-node `n_left` = counts in bins `0..bin` + the missing bin (missing-left), `n_right
     = n_total - n_left`, node `gain` via the `_leaf_score` reduction at the fixed `(f,b)`;
     `bin=bin`, `left_categories=None`. Feeds the existing `partition` / `_apply_split`.
4. `Splitter` gets **thin wrappers** that supply its stored
   `n_bins_per_feature` / `min_samples_leaf` / `l2`:
   - `Splitter.find_best_level_split(self, hists) -> tuple[int, int] | None` (wrap under the
     `timed(self._profiler, "split_scan")` block, matching `find_best_split`);
   - `Splitter.split_at(self, hist, feature, bin) -> SplitCandidate`.
   These are the only two new public Splitter methods; `_grow_symmetric` calls them.

### 5.4 Serialization (`core/serialization.py` — no change)

Nothing changes structurally. `grow_policy` rides in `model_config.json` via `get_params()`;
the expanded symmetric tree uses the unchanged `Tree.to_dict` / `Tree.from_dict`;
`FORMAT_VERSION` stays **6**. Round-trip is covered by tests (§7), including an assertion
that `model_config.json["format_version"] == 6`.

---

## 6. Code-map touch points (summary)

| File / symbol | Change |
|---|---|
| `core/booster.py::BoosterParams` | add field `grow_policy: str = "leafwise"` |
| `core/booster.py::Booster.fit` | pass `grow_policy=p.grow_policy` to `TreeGrower(...)` |
| `core/multiclass.py::MulticlassBooster.fit` | pass `grow_policy=p.grow_policy` to `TreeGrower(...)` |
| `core/multioutput.py::MultiOutputBooster.fit` | pass `grow_policy`; **raise `NotImplementedError` for `symmetric`** (fail fast) |
| `core/tree.py::TreeGrower.__init__` | add + validate `grow_policy` and the `max_depth>=1` contract |
| `core/tree.py::TreeGrower.grow` | becomes dispatcher → `_grow_leafwise/_grow_depthwise/_grow_symmetric` |
| `core/tree.py` (new) | `_new_store`, `_apply_split`, `_finalize`, `_make_candidate`, `_child_hists`, `_grow_leafwise`, `_grow_depthwise`, `_grow_symmetric` |
| `backends/numpy_backend.py` | extract `_numeric_split_table`; refactor `find_best_split` to use it; add `find_best_level_split`, `split_at` (module functions) |
| `core/splitter.py::Splitter` | add thin wrappers `find_best_level_split`, `split_at` |
| `sklearn.py::BaseRepLeafModel.__init__` / `.fit` | add `grow_policy` arg + `self.grow_policy`; thread into `BoosterParams(...)` |
| `regressor.py`, `classifier.py` | **no change** (inherit base `__init__`; `_make_booster` is policy-agnostic) |
| `core/serialization.py` | **no change** (`FORMAT_VERSION` stays 6) |
| `backends/rust_backend.py`, `native/`, `backends/cuda_backend.py` | **no change** (host-side scan; no maturin rebuild; crate version not bumped) |

---

## 7. Guardrail / thesis check

Independent sanity-check against the project invariants. **Result: PASS — no violation.**

| Invariant | Status | Why |
|---|---|---|
| **Raw-feature routing only** | ✅ | `find_best_level_split` / `split_at` / `find_best_split` all scan `Splitter.binned`, which is built only from raw features (categoricals ordinal-encoded; `core/splitter.py:79–97`). The symmetric *shared* split is selected over the same raw histograms. **No embedding dim enters routing.** |
| **Leaf-only representation** | ✅ | Leaf fitting is untouched: `fit_leaves` / `fit_leaves_multiclass` / `fit_vector_leaves` consume `Z` exactly as today, including session-#26 cross-K multiclass pooling. Growth policy only changes the row partition handed to them. |
| **Encoder frozen during boosting** | ✅ | No code path updates θ mid-boost; the `freeze_encoder=False` guard in `sklearn.fit` (`sklearn.py:239–243`) is unchanged. Embeddings are fetched once before the grow loop. |
| **No embedding-dim splits** | ✅ | `_numeric_split_table` indexes `(feature, bin)` over raw features only; `_grow_symmetric` raises on `grad.ndim > 1` and never touches `Z`. |
| **Newton-target leaf fitting + extrapolation guard** | ✅ | Unchanged. Leaf models still fit `t=-g/h` weighted by `h`; linear leaves keep their per-leaf `z_min/z_max` clip. Symmetric's complete/balanced partition tends to *reduce* tiny-leaf constant fallbacks, not bypass the guard. |
| **No wrapper-only-around-LightGBM/XGBoost** | ✅ | This is native-path growth; nothing imports external. |
| **Determinism** (`check_random_state`; same seed ⇒ same model) | ✅ | The new scan's `argmax` uses the row-major **lowest-index tie-break** identical to `find_best_split` / `find_best_split_multioutput`. depthwise FIFO order and per-node `split_at` are deterministic. No RNG introduced. |
| **NumPy⇄Rust parity (bitwise histograms)** | ✅ | `build_histograms` is reused unchanged (bitwise NumPy⇄Rust). The new scans are **host-side and shared by both backends** (decision B), so end-to-end parity is automatic — and *allclose* across backends by construction (single code path). The `find_best_split` refactor is bitwise-identical and gated on existing parity tests. **No dual kernel; no native rebuild.** |
| **SemVer / serialization read-ladder** | ✅ | `grow_policy` is an additive, defaulted param; `leafwise` default = current behavior. Expanded symmetric trees serialize through the unchanged `Tree.to_dict/from_dict`; `FORMAT_VERSION` stays **6**, so old models still read and no new format is introduced (minor-version feature, not a format change). |
| **Optional-deps isolation** | ✅ | Native path imports no torch/lightgbm/cupy/external; crate version not bumped. |

**Documented v0 limitations** (must land in ADR + roadmap, honestly): symmetric is
numeric ordered-threshold + scalar only (categorical-subset and multi-output deferred);
storage is the *expanded* flat `Tree`, not a compact oblivious format (compact format + fast
bitwise inference is a post-v0 follow-up); CUDA-symmetric is unvalidated this milestone.

Conclusion: the design is a faithful, thesis-preserving extension. Proceed to implementation.

---

## 8. Required tests

New `tests/test_grow_policy.py` (+ extend `tests/test_rust_backend.py`), matching existing
styles (`np.testing.assert_allclose`; structure read via `model.booster_.trees_`):

- **Functional matrix.** Each `grow_policy ∈ {leafwise, depthwise, symmetric}` ×
  `{constant, embedded_linear}` × `{regression, binary, multiclass}` **fits & predicts**
  finite outputs on small seeded synthetic data. **depthwise also covers multi-output**
  (2-D `y`). **symmetric + multi-output ⇒ `pytest.raises(NotImplementedError)`.**
- **Determinism.** Same seed + same policy ⇒ `assert_allclose` over two independent fits'
  predictions (each task × policy).
- **Save/load round-trip.** `save_model`/`load_model` ⇒ predictions `allclose` **and**
  `loaded.get_params()["grow_policy"]` preserved; assert
  `model_config.json["format_version"] == 6`.
- **Backend parity.** Per policy, numpy vs rust end-to-end
  `assert_allclose(rtol=1e-6, atol=1e-8)` on predictions (extend
  `tests/test_rust_backend.py`). Covers leafwise/depthwise/symmetric.
- **Structure — depthwise.** Walk each grown `Tree`: no leaf deeper than `max_depth`
  (BFS depth check); tree is reasonably balanced (sanity, not exact).
- **Structure — symmetric.** A helper walks the expanded `Tree`, groups internal nodes by
  depth, and asserts **all nodes at a depth share the same `(feature, threshold)`**;
  `n_leaves == 2 ** achieved_depth` (complete); an independent **bit-pattern oblivious
  routing reproduces `Tree.apply` on a sample** (cross-check the expansion).
- **Encoder transform shapes.** With `leaf_model="embedded_linear"`, assert the embedding
  matrix shape `(n_rows, emb_dim)` feeding each policy is unchanged (guards that growth
  policy does not perturb the representation path).
- **Constant-fallback for small leaves.** With `embedded_linear` + small/noisy data +
  small `max_depth`/`num_leaves`, confirm leaves below the linear-fit threshold fall back
  to constant (finite predictions, no solve error) under each policy — exercising
  `fit_vector_leaves` / `fit_leaves` guards on the new partitions.
- **Validation errors.** invalid `grow_policy` (e.g. `"oblivious"`) ⇒ `ValueError`;
  depthwise/symmetric with `max_depth=-1` (and `max_depth=0`) ⇒ `ValueError` (decision D).
- **leafwise regression guard.** The existing suite stays green with **no numeric drift**
  from the `TreeGrower` / `find_best_split` refactor (leaf-wise predictions byte-identical
  to pre-change). This is the single most important test: it certifies the refactor is
  behavior-preserving.
- **Optional.** A small `examples/grow_policy.py` (or extend an existing example) showing
  `grow_policy="symmetric"` / `"depthwise"`, runnable in seconds (CLAUDE.md examples rule).

---

## 9. Validating experiment

After green + `core-reviewer` sign-off, hand to **experiment-runner** →
**results-analyst**.

- **Comparison:** `leafwise` vs `depthwise` vs `symmetric`.
- **Data:** synthetic (`benchmarks/`) spanning small/noisy → larger/structured, **plus** a
  few OpenML / real datasets (`experiments/openml_suite`).
- **Tasks × leaves:** `{regression, binary, multiclass}` × `{constant, embedded_linear}`.
  (Multi-output is depthwise/leafwise only.)
- **Seeds:** multiple (≥5), report mean ± std; treat single-seed deltas as noise (memory:
  reg/binary gains are often seed-noise — require multi-seed before any claim).
- **Fair budget:** for depthwise/symmetric set `max_depth` (and lift `num_leaves` for
  depthwise) so the three policies have comparable leaf budgets; otherwise the comparison
  penalizes the constrained policies unfairly.
- **Metric:** task-appropriate (RMSE / logloss / accuracy / multiclass logloss); the
  primary read is per-dataset rank of the three policies.
- **Hypothesis / expected effect:** symmetric **regularizes on small/noisy** (best or tied
  there, especially with `embedded_linear`); depthwise is a **balanced baseline**; leafwise
  is **strongest on larger/structured**. **Default expected to remain `leafwise`.**
- **Verdict (results-analyst):** *where* each policy helps + whether the default should
  change (default change only with an evidence-backed report, per CLAUDE.md).
- **Honesty rule (CLAUDE.md / memory):** if no new policy beats leafwise anywhere, **still
  ship them** as standard knobs / comparison-research value and **say so plainly** — do not
  inflate numbers. A null result is a publishable result here.

---

## 10. Risks

- **Refactor drift (highest-blast-radius).** Extracting `_numeric_split_table` and
  re-expressing leaf-wise growth via helpers touches the parity-critical reference scan and
  the most-exercised growth path. *Mitigation:* the leafwise-regression guard (byte-identical
  predictions) + existing NumPy⇄Rust parity tests + mandatory `core-reviewer` on the
  refactor. Land the pure refactor and confirm green **before** adding new policies.
- **Symmetric objective mistake (sum gains, not histograms).** Easy to "optimize" by pooling
  histograms — which is the wrong objective. *Mitigation:* §3.2 states the math explicitly;
  the structure test asserts per-level-identical split and `2^depth` completeness, and the
  bit-pattern cross-check guards the expansion.
- **All-or-none stalling.** Global `min_samples_leaf` invalidation can stop symmetric growth
  early on imbalanced partitions (a feature, not a bug — it preserves completeness).
  *Mitigation:* document in ADR + roadmap; the experiment uses a fair `max_depth` so this is
  visible, not silent.
- **CUDA-symmetric unvalidated.** Host-side scan accepts `_as_host` device hists but is not
  validated on GPU this milestone. *Mitigation:* documented limitation; CUDA path untouched
  and "auto" never selects it on CI/macOS.
- **`num_leaves` semantics confusion.** symmetric ignores `num_leaves` (leaves = `2^depth`);
  depthwise treats it as an optional secondary cap. *Mitigation:* docstrings + roadmap note
  + validation-error tests pin the contract (decision D).

---

## 11. Recommendation

**Proceed.** The design is a faithful, thesis-preserving algorithm-feature addition with a
contained blast radius (host-side scan, general-`Tree` expansion, no format bump, no native
rebuild) and a clear validating experiment. Sequence the work as: (1) land the
behavior-preserving `TreeGrower` + `find_best_split` refactor and confirm the leafwise
regression guard + parity tests are green; (2) add `_grow_depthwise` (reuses existing scan,
covers all tasks); (3) add `_grow_symmetric` + the two host-side scan functions; (4) plumb
`grow_policy` + docs + tests; (5) `qa-verifier` green gate; (6) `core-reviewer` sign-off
(thesis / parity / refactor / determinism / serialization); (7) experiment-runner →
results-analyst for the use-case verdict. **Keep `leafwise` the default absent contrary
evidence.** Implementation is host-side only ⇒ **no maturin build**.

---

## 12. References

- Plan (source of truth): `~/.claude/plans/repleafgbm-tree-growth-policy-shimmying-gadget.md`
- Math: `docs/math.md` (Newton gain, multi-output summed gain, stage-wise assumption)
- Code: `core/tree.py`, `backends/numpy_backend.py`, `core/splitter.py`, `core/booster.py`,
  `core/multiclass.py`, `core/multioutput.py`, `sklearn.py`, `core/serialization.py`
- Literature (to be captured by literature-scout): CatBoost (arXiv 1706.09516), XGBoost
  params / issue #9123, NODE (arXiv 1909.06312), apxml.
- Companion ADR (to be drafted): `docs/adr/<next>-tree-growth-policies.md`
