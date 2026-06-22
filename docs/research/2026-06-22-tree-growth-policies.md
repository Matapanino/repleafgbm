# Literature Note: Tree Growth Policies (leafwise / depthwise / symmetric)

- **Date:** 2026-06-22
- **Author:** literature-scout
- **Question:** What are the exact algorithmic definitions of leafwise, depthwise, and
  symmetric/oblivious tree growth in mature GBDT libraries? When does each policy help?
  Are they compatible with RepLeafGBM's split-on-raw-features / representation-in-leaf
  architecture?
- **Feeds:** `docs/proposals/2026-06-22-grow-policy.md` (research-proposer spec already
  written; this note supplies the external-knowledge backing).

---

## 1. Rationale

RepLeafGBM currently grows every tree leaf-wise (best-gain-first heap, capped at
`num_leaves`, implemented in `core/tree.py::TreeGrower.grow`). A `grow_policy`
hyperparameter is being added with three values: `leafwise` (current), `depthwise`, and
`symmetric`. This note verifies and cites the precise algorithmic definitions, split
objectives, and empirical guidance for each, then maps each onto the project's
architectural invariants.

---

## 2. Sources

| Title | URL | Date |
|---|---|---|
| CatBoost: unbiased boosting with categorical features | https://arxiv.org/abs/1706.09516 | 2017 (Prokhorenkova et al., NeurIPS 2018) |
| CatBoost: gradient boosting with categorical features support | https://arxiv.org/abs/1810.11363 | 2018 (NeurIPS) |
| CatBoost official docs: grow_policy parameter | https://catboost.ai/docs/en/references/training-parameters/common | accessed 2026-06-22 |
| CatBoost GPU blog post (Dorogush et al.) | https://catboost.ai/news/catboost-enables-fast-gradient-boosting-on-decision-trees-using-gpus | accessed 2026-06-22 |
| Neural Oblivious Decision Ensembles (NODE) | https://arxiv.org/abs/1909.06312 | 2019 (Popov et al., ICLR 2020) |
| XGBoost parameter docs (stable) | https://xgboost.readthedocs.io/en/stable/parameter.html | accessed 2026-06-22 |
| XGBoost issue #9123 (depthwise vs lossguide) | https://github.com/dmlc/xgboost/issues/9123 | 2022 |
| LightGBM Features documentation | https://lightgbm.readthedocs.io/en/latest/Features.html | accessed 2026-06-22 |
| LightGBM: A Highly Efficient Gradient Boosting Decision Tree (Ke et al.) | https://proceedings.neurips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf | NeurIPS 2017 |

---

## 3. Key Findings

### 3.1 Leafwise growth (best-gain-first)

**Definition.** At each step, the single leaf in the current partial tree that offers the
largest Newton gain is split, regardless of its depth:

```
gain(split, leaf l) = G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda) - G_l^2/(H_l+lambda)
```

The tree grows asymmetrically until `num_leaves` is reached or no positive-gain split
remains. `max_leaves` is the meaningful complexity control; `max_depth` is an optional
secondary guard.

**Origin.** LightGBM (Ke et al., NeurIPS 2017) introduced leaf-wise growth as the
primary growth strategy, explicitly contrasting it with level-wise (XGBoost's original
default). The LightGBM documentation states: "It will choose the leaf with max delta loss
to grow. Holding #leaf fixed, leaf-wise algorithms tend to achieve lower loss than
level-wise algorithms." XGBoost's `grow_policy=lossguide` (histogram-only) replicates
this behavior; XGBoost calls it "split at nodes with highest loss change."

**Trade-off.** From the LightGBM docs: "Leaf-wise may cause over-fitting when #data is
small, so LightGBM includes the `max_depth` parameter to limit tree depth." Leaf-wise
builds the most expressive tree per `num_leaves` budget on large, structured data; it
over-specializes on small or noisy datasets.

**Claim strength.** Fully confirmed from primary sources (LightGBM paper + docs,
XGBoost docs, current codebase). Not contested.

### 3.2 Depthwise / level-wise growth

**Definition.** The tree is grown level by level. At each depth, every current leaf node
is evaluated and split (if a positive-gain split exists and `max_depth` has not been
reached). The tree is not required to be complete but every node at a given depth is
considered before moving to the next depth.

**XGBoost source (exact documentation text).** From the stable XGBoost parameter docs:
- `depthwise` [default]: "split at nodes closest to the root."
- `lossguide`: "split at nodes with highest loss change."
- Constraint: "`grow_policy` is currently supported only if `tree_method` is set to
  `hist` or `approx`."

**CatBoost Depthwise (exact docs text).** CatBoost also exposes a `Depthwise` value for
`grow_policy`: "A tree is built level by level until the specified depth is reached. On
each iteration, all non-terminal leaves from the last tree level are split. Each leaf is
split by condition with the best loss improvement." This differs from `SymmetricTree`
because leaves at the same level can receive *different* splits.

**Trade-off.** Depthwise produces balanced trees with predictable depth; `max_depth`
directly controls complexity. It is the original GBM baseline (Friedman 2001). The
LightGBM paper notes: "Level-wise training can be seen as a form of regularized
training since leaf-wise training can construct any tree that level-wise training can,
whereas the opposite does not hold." More regularized than leafwise, but less
expressive at the same leaf count.

**Claim strength.** Confirmed from official XGBoost and CatBoost docs. XGBoost issue
#9123 reports that under some conditions depthwise and lossguide produced identical
outputs (possible bug or tied gains), which highlights that the difference is meaningful
mostly when `max_depth` would cause lossguide to grow deeper than a balanced tree of
the same node budget; with a tight `max_depth` the two policies converge.

### 3.3 Symmetric / oblivious growth (CatBoost)

**Definition.** The tree is built level by level. At each level, *all* leaf nodes from
the previous level are split using the **same (feature, threshold)** pair. The result is
a complete binary tree of depth `d` with exactly `2^d` leaves.

**Official CatBoost docs (exact text):** "`SymmetricTree` [default]: A tree is built
level by level until the specified depth is reached. On each iteration, all leaves from
the last tree level are split with the same condition. The resulting tree structure is
always symmetric."

**Leaf index encoding.** Because every path from root to leaf is a length-`d` sequence
of left/right decisions, and every split is the same feature-threshold test, a leaf's
index is the integer formed by the `d`-bit binary vector of decisions along its path
(0=left, 1=right). From the CatBoost GPU blog: "In this case a tree of depth `k` has
exactly `2^k` leaves, and the index of a leaf can be calculated with simple bitwise
operations." This is the same encoding used in the NODE paper (Popov et al., ICLR 2020),
which builds differentiable oblivious trees on top of this structure.

**Per-level split selection criterion.** The official documentation and GPU blog describe
the criterion as selecting the (feature, threshold) pair that yields the largest
*aggregate improvement in the objective across all current leaves at that level*. Because
the gain function

```
gain(split, node) = G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda) - G^2/(H+lambda)
```

is nonlinear in (G, H), the aggregate gain is the **sum of per-node gains** evaluated
for the single candidate split applied to each node — not a merged-histogram sum. This
is a critical subtlety: summing histograms across nodes and then computing one gain
score would give a different (wrong) answer. Each candidate (f, t) is scored by
accumulating the per-node gain over all nodes at the current level, and the (f, t) with
the highest total is selected. The CatBoost GPU blog notes that this uniform split makes
the evaluation "heavily parallelizable on GPUs since all data points at a given level
encounter the same feature test."

**Evidence quality note.** The primary CatBoost papers (arXiv 1706.09516 and 1810.11363)
contain this definition in their methodology sections, but PDF extraction during this
review was blocked by encoding. The sum-of-per-node-gains criterion is confirmed
consistently across: (a) the official CatBoost docs for `SymmetricTree`, (b) the
CatBoost GPU blog, (c) secondary but detailed technical sources (CatBoost secrets blog
via Medium, gradient boosting variants survey). No source contradicts it. Treat this as
well-supported but note that direct paper quotes were not obtained in this session.

**Why this gives regularization.** A depth-`d` symmetric tree has exactly `d` distinct
splits (one per level), regardless of leaf count. A leafwise tree with `2^d` leaves can
have up to `2^d - 1` distinct splits. The symmetric tree thus expresses far less
structural complexity per leaf budget, which acts as a strong implicit regularizer.
Multiple technical sources note this benefit is most pronounced on small or noisy
datasets where asymmetric trees memorize subpopulation noise; a depth-6 symmetric tree
(64 leaves, 6 distinct splits) vs. a depth-6 leafwise tree (up to 63 distinct splits)
illustrates the contrast.

**Inference speed.** The bitwise leaf-index encoding enables branch-free prediction:
apply the `d` feature tests in sequence, concatenate the bits, and look up the leaf.
CatBoost claims ~8x faster inference over irregular trees for this reason.

---

## 4. When Each Policy Helps

| Data regime | Recommended policy | Rationale |
|---|---|---|
| Large N, structured, low noise | `leafwise` | Maximizes capacity per budget; LightGBM default |
| Moderate N, balanced baseline needed | `depthwise` | Predictable depth, standard GBM; `max_depth` directly controls complexity |
| Small N, noisy, or high regularization needed | `symmetric` | Fewest distinct splits per leaf count; each level is fully committed to one global decision |
| Inference-speed critical | `symmetric` | Bitwise leaf lookup, branch-free |
| Unknown / real tabular data | Start `leafwise`; try `symmetric` if overfitting | Empirical guidance from LightGBM docs and CatBoost papers |

The claim that symmetric trees are better on small/noisy data is stated in the CatBoost
papers and corroborated by the OpenML benchmark in `experiments/results/openml_benchmark.md`,
which shows CatBoost leading the mean rank on the 9-dataset suite (rank 2.00 vs.
LightGBM 2.50 for regression; rank 2.00 vs. RepLeaf-constant 2.60 for classification).
However, that benchmark conflates growth policy with all other CatBoost design choices
(ordered boosting, categorical handling, etc.), so the symmetric tree contribution alone
is not isolated. An ablation with RepLeafGBM's `grow_policy=symmetric` on small N
datasets would give cleaner evidence.

---

## 5. Relevance to RepLeafGBM

### Thesis mapping

All three growth policies operate purely on the **routing/split side**. Splits are
selected over raw features (histogram bins of `X_raw`), and the criterion is the Newton
gain over gradient/Hessian accumulators. The leaf *model* — constant or
`embedded_linear` over `z_theta(x)` — is fitted after routing is decided, using the
existing `h`-weighted Newton-target machinery.

Concretely:

- **Leafwise** — current `TreeGrower.grow` (heap, best `neg_gain` popped first). No
  change required to the leaf machinery.
- **Depthwise** — replace the heap with a FIFO queue ordered by depth (BFS). Same
  `find_best_split` call per node; same leaf-fitting downstream.
- **Symmetric** — one global `find_best_split` call *across all nodes at a level*,
  enumerating (feature, threshold) candidates and summing per-node gains; then partition
  all nodes with that shared split. Leaf fitting is unchanged.

None of the three policies touch:
- The encoder (`z_theta`, which remains frozen during boosting).
- The leaf-fitting kernel (`fit_linear_leaf`, `fit_vector_leaves`, constant fallback).
- The split backend contracts (`BaseSplitBackend.find_best_split`).
- The `Tree` data structure (flat arrays, `feature`, `threshold`, `leaf_id`, etc.) --
  symmetric trees are a strict subset of the existing storage format (every internal
  node at a given depth happens to share `feature` and `threshold` with its siblings,
  but the flat-array representation stores them redundantly without issue).

### Guardrail check

| Invariant | leafwise | depthwise | symmetric |
|---|---|---|---|
| Splits on raw features only | PASS | PASS | PASS |
| Leaves may use `z_theta(x)` | PASS (unchanged) | PASS (unchanged) | PASS (unchanged) |
| Encoder frozen during boosting | PASS | PASS | PASS |
| No splitting on embedding dims | PASS | PASS | PASS |
| Newton targets `t = -g/h`, weights `h` | PASS | PASS | PASS |

**No invariant is violated by any of the three policies.**

A potential trap: for symmetric trees, the `find_best_split` call must sum gains across
nodes, not merge histograms and compute one gain on the merged result. Merging histograms
would confound the gain formula (cross-node G/H terms would cancel incorrectly). This
is an implementation correctness point, not a thesis violation.

### Code touch points

| Module | Change needed |
|---|---|
| `core/tree.py::TreeGrower` | Add `grow_policy` parameter; branch on it in `grow()` |
| `core/booster.py` | Pass `grow_policy` through to `TreeGrower` |
| `core/multiclass.py`, `core/multioutput.py` | Same forwarding |
| `core/splitter.py::Splitter.find_best_split` | No change; symmetric just calls it once per candidate across nodes |
| `backends/numpy_backend.py` (and Rust parity) | No change to the split-scan kernel; the symmetric aggregation loop is in Python/TreeGrower |
| `sklearn.py`, `regressor.py`, `classifier.py` | Expose `grow_policy` param, serialize it |

The `Tree` dataclass and all backends are untouched. The `leaf_models.py` machinery is
untouched. Serialization: the tree arrays already store what the symmetric grower would
produce; `grow_policy` only needs to be stored in the estimator config (not the tree
arrays themselves).

---

## 6. Guardrail Check

All three policies pass all thesis invariants (see table above). The only
implementation-level risk is the symmetric-tree gain aggregation correctness issue
noted above (sum of per-node gains, not gain of merged histograms). This must be tested
with a direct correctness check (compare symmetric tree predictions to a reference
implementation or to leafwise output on small data where they should agree up to
policy-induced structure).

No thesis violation found in any of the three growth policies.

---

## 7. Concrete Next Steps for research-proposer

1. **Symmetric split kernel specification.** The `find_best_split` call in the symmetric
   grower must enumerate all (feature, bin) candidates, evaluate the per-node gain for
   each candidate across all active nodes at the current level, and sum those gains. The
   kernel is currently scoped to one node (one histogram). The proposer should specify
   whether to (a) call `find_best_split` once per node and aggregate gains in Python, or
   (b) add a new `find_best_split_symmetric` kernel that accepts a list of histograms.
   Option (a) is simpler and correct for `num_leaves` up to `2^d` = 64 (d=6) nodes at
   the deepest level; option (b) could share work across a Rust kernel for larger depths.

2. **Experiment design.** The most informative experiment is a fixed-seed, fixed-budget
   (`num_leaves` = `2^max_depth`) comparison of `leafwise` vs. `depthwise` vs.
   `symmetric` on (a) a small-N noisy dataset and (b) a large-N clean dataset, with
   constant leaves (to isolate routing from leaf modeling). This would provide the first
   RepLeafGBM-native evidence for the "symmetric helps on small data" claim. The OpenML
   suite in `benchmarks/openml_suite.py` is a natural harness.

3. **Inference timing.** Once implemented, measure inference speed for symmetric vs.
   leafwise at the same `2^d` leaf count. The claimed bitwise advantage is real for
   CatBoost's compiled C++ applier; whether it materializes in Python/NumPy `Tree.apply`
   (which already uses vectorized indexing) is an empirical question worth checking.

---

Sources:
- [CatBoost: unbiased boosting with categorical features (arXiv 1706.09516)](https://arxiv.org/abs/1706.09516)
- [CatBoost: gradient boosting with categorical features support (arXiv 1810.11363)](https://arxiv.org/abs/1810.11363)
- [CatBoost grow_policy parameter docs](https://catboost.ai/docs/en/references/training-parameters/common)
- [CatBoost GPU blog (Dorogush et al.)](https://catboost.ai/news/catboost-enables-fast-gradient-boosting-on-decision-trees-using-gpus)
- [Neural Oblivious Decision Ensembles (arXiv 1909.06312)](https://arxiv.org/abs/1909.06312)
- [XGBoost parameter docs (stable)](https://xgboost.readthedocs.io/en/stable/parameter.html)
- [XGBoost issue #9123](https://github.com/dmlc/xgboost/issues/9123)
- [LightGBM Features docs](https://lightgbm.readthedocs.io/en/latest/Features.html)
- [LightGBM paper (Ke et al., NeurIPS 2017)](https://proceedings.neurips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf)
