"""Decision tree: leaf-wise growth on raw features, flat-array storage.

The tree only routes: it stores split features/thresholds and leaf ids.
Leaf *values* live in :mod:`repleafgbm.core.leaf_models`, because in
RepLeafGBM a leaf is a small model over the representation, not a constant.
"""

from __future__ import annotations

import heapq
import itertools
from collections import deque
from dataclasses import dataclass, field

import numpy as np

try:  # Optional compiled router (native/); Tree.apply uses it when present.
    import repleafgbm_native as _native
except ImportError:  # pragma: no cover - depends on optional extension
    _native = None

from repleafgbm.backends.base import SplitCandidate
from repleafgbm.core.splitter import Splitter


@dataclass
class Tree:
    """Routing tree stored as flat arrays (node 0 is the root).

    For internal node i: rows with ``x[feature[i]] <= threshold[i]`` go to
    ``left[i]``, the rest to ``right[i]``; missing values follow
    ``missing_left[i]``. For leaf node i, ``leaf_id[i]`` is its index into
    the per-tree leaf parameter arrays.

    Natively grown trees always set ``missing_left=True`` (the v0 training
    convention). The per-node field exists so extracted external routes
    (e.g. LightGBM's learned ``default_left``) can be represented exactly.
    """

    feature: np.ndarray  # int32, -1 for leaves
    threshold: np.ndarray  # float64, NaN for leaves
    left: np.ndarray  # int32, -1 for leaves
    right: np.ndarray  # int32, -1 for leaves
    leaf_id: np.ndarray  # int32, -1 for internal nodes
    missing_left: np.ndarray  # bool, True routes NaN to left[i]
    gain: np.ndarray  # float64, split gain (feature importance); 0.0 for leaves
    #: Per-node categorical subset splits: ``left_categories[i]`` is the
    #: float64 array of ordinal codes routed left at node i, or None for
    #: numerical threshold nodes (and leaves). Codes not in the set —
    #: including categories never seen at fit time after NaN-mapping — go
    #: right; missing values follow ``missing_left``. None for trees without
    #: any categorical split.
    left_categories: list | None = None

    @property
    def n_leaves(self) -> int:
        return int((self.leaf_id >= 0).sum())

    def apply(self, X_raw: np.ndarray) -> np.ndarray:
        """Route rows to leaves; returns leaf ids of shape (n_rows,).

        Uses the compiled ``repleafgbm_native.apply_tree`` router when the
        optional extension is available — a single pass of independent per-row
        root-to-leaf descents, which the post-PR #30 prediction-traversal
        benchmark showed is 60-100% of predict and the part that scales worst.
        Falls back to the NumPy level-synchronous reference :meth:`_apply_numpy`
        (also used when an older extension predates ``apply_tree``). Both return
        index-identical leaf ids — asserted in tests/test_tree_routing_native.py.
        """
        if _native is not None and hasattr(_native, "apply_tree"):
            cat_offsets, cat_values = self._cat_csr()
            return _native.apply_tree(
                np.ascontiguousarray(X_raw, dtype=np.float64),
                np.ascontiguousarray(self.feature, dtype=np.int32),
                np.ascontiguousarray(self.threshold, dtype=np.float64),
                np.ascontiguousarray(self.left, dtype=np.int32),
                np.ascontiguousarray(self.right, dtype=np.int32),
                np.ascontiguousarray(self.leaf_id, dtype=np.int32),
                np.ascontiguousarray(self.missing_left, dtype=bool),
                cat_offsets,
                cat_values,
            )
        return self._apply_numpy(X_raw)

    def _cat_csr(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-node left-category codes as a CSR for the native router.

        Returns ``(cat_offsets, cat_values)``: ``cat_offsets`` is
        ``(n_nodes + 1,)`` int64 and ``cat_values[cat_offsets[i]:cat_offsets[i+1]]``
        are node ``i``'s left-category codes (float64) — empty for numeric nodes
        and leaves, so the native router treats a non-empty slice as the
        categorical-membership branch. A tree with no categorical split yields
        all-zero offsets and an empty value array (the common, cheap path).
        """
        n_nodes = self.feature.shape[0]
        if self.left_categories is None:
            return (
                np.zeros(n_nodes + 1, dtype=np.int64),
                np.empty(0, dtype=np.float64),
            )
        sizes = np.fromiter(
            (0 if c is None else c.shape[0] for c in self.left_categories),
            dtype=np.int64,
            count=n_nodes,
        )
        cat_offsets = np.empty(n_nodes + 1, dtype=np.int64)
        cat_offsets[0] = 0
        np.cumsum(sizes, out=cat_offsets[1:])
        parts = [c for c in self.left_categories if c is not None]
        cat_values = (np.concatenate(parts) if parts else np.empty(0)).astype(
            np.float64
        )
        return cat_offsets, np.ascontiguousarray(cat_values)

    def _apply_numpy(self, X_raw: np.ndarray) -> np.ndarray:
        """NumPy level-synchronous router (fallback + parity reference).

        At each depth it advances every still-active row by one node, handling
        numeric thresholds, categorical subset membership, and the missing-value
        direction (NaN follows ``missing_left``). This is the bitwise reference
        the native :func:`apply_tree` is parity-tested against.
        """
        n = X_raw.shape[0]
        node = np.zeros(n, dtype=np.int64)
        active = self.leaf_id[node] < 0
        while active.any():
            active_rows = np.flatnonzero(active)
            idx = node[active_rows]
            x = X_raw[active_rows, self.feature[idx]]
            go_left = x <= self.threshold[idx]
            if self.left_categories is not None:
                # Membership test per categorical node at this level.
                for node_index in np.unique(idx):
                    cats = self.left_categories[node_index]
                    if cats is not None:
                        at_node = idx == node_index
                        go_left[at_node] = np.isin(x[at_node], cats)
            go_left = np.where(np.isnan(x), self.missing_left[idx], go_left)
            node[active_rows] = np.where(go_left, self.left[idx], self.right[idx])
            active = self.leaf_id[node] < 0
        return self.leaf_id[node].astype(np.int64)

    def to_dict(self) -> dict:
        return {
            "feature": self.feature.tolist(),
            "threshold": [None if np.isnan(t) else float(t) for t in self.threshold],
            "left": self.left.tolist(),
            "right": self.right.tolist(),
            "leaf_id": self.leaf_id.tolist(),
            "missing_left": self.missing_left.tolist(),
            "gain": self.gain.tolist(),
            "left_categories": (
                None
                if self.left_categories is None
                else [None if c is None else c.tolist() for c in self.left_categories]
            ),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Tree:
        n_nodes = len(d["feature"])
        return cls(
            feature=np.asarray(d["feature"], dtype=np.int32),
            threshold=np.asarray(
                [np.nan if t is None else t for t in d["threshold"]], dtype=np.float64
            ),
            left=np.asarray(d["left"], dtype=np.int32),
            right=np.asarray(d["right"], dtype=np.int32),
            leaf_id=np.asarray(d["leaf_id"], dtype=np.int32),
            # Format version 1 predates missing_left; its trees were grown
            # with the NaN-left convention, so all-True is the exact default.
            missing_left=np.asarray(
                d.get("missing_left", [True] * n_nodes), dtype=bool
            ),
            # gain is optional on read (older files): importances degrade to
            # zeros rather than failing to load.
            gain=np.asarray(d.get("gain", [0.0] * n_nodes), dtype=np.float64),
            # Absent in formats < 3 (no categorical splits existed).
            left_categories=(
                None
                if d.get("left_categories") is None
                else [
                    None if c is None else np.asarray(c, dtype=np.float64)
                    for c in d["left_categories"]
                ]
            ),
        )


@dataclass(order=True)
class _GrowCandidate:
    """Heap entry: a leaf that could be split, prioritized by gain.

    Carries the node's histogram so children can reuse it: the smaller child
    is accumulated directly and the larger one obtained by subtraction. The
    histogram object is backend-defined (a NumPy array, or a device-resident
    CuPy array for the CUDA backend) and used only via subtraction/indexing.
    """

    neg_gain: float
    tiebreak: int
    node_index: int = field(compare=False)
    rows: np.ndarray = field(compare=False)
    depth: int = field(compare=False)
    split: SplitCandidate = field(compare=False)
    hist: np.ndarray = field(compare=False)


@dataclass
class _NodeStore:
    """Growable node arrays for one tree (node 0 is the root).

    All three growth policies populate this same flat structure — they differ
    only in *which* node they split next and in what order — then hand it to
    :meth:`TreeGrower._finalize`, which assigns leaf ids and builds the ``Tree``.
    """

    feature: list[int]
    threshold: list[float]
    left: list[int]
    right: list[int]
    gain: list[float]
    left_cats: list[np.ndarray | None]
    node_rows: dict[int, np.ndarray]


class TreeGrower:
    """Grows one tree under a configurable ``grow_policy`` on raw features.

    * ``"leafwise"`` (default): best-gain-first growth like LightGBM, where
      ``num_leaves`` is the natural complexity control and ``max_depth``
      (-1 = unlimited) additionally caps depth. Unchanged from earlier versions.
    * ``"depthwise"``: level-order (breadth-first) growth to ``max_depth``, like
      XGBoost's ``grow_policy=depthwise``; ``num_leaves`` still applies as an
      optional secondary cap.
    * ``"symmetric"``: CatBoost-style oblivious trees — every node at a level
      shares one ``(feature, threshold)`` chosen to maximize the *summed*
      per-node gain, giving a complete tree with up to ``2**max_depth`` leaves
      and strong implicit regularization. Numeric/ordered splits and scalar
      targets only in v0 (see :meth:`_grow_symmetric`).

    ``depthwise`` and ``symmetric`` require ``max_depth >= 1``. Routing always
    uses raw features only; leaf modeling is orthogonal (see leaf_models).
    """

    _POLICIES = ("leafwise", "depthwise", "symmetric")

    def __init__(
        self,
        splitter: Splitter,
        num_leaves: int = 31,
        max_depth: int = -1,
        grow_policy: str = "leafwise",
    ) -> None:
        if num_leaves < 2:
            raise ValueError(f"num_leaves must be >= 2, got {num_leaves}")
        if grow_policy not in self._POLICIES:
            raise ValueError(
                f"grow_policy must be one of {self._POLICIES}, got {grow_policy!r}"
            )
        if grow_policy in ("depthwise", "symmetric") and max_depth < 1:
            raise ValueError(
                f"grow_policy={grow_policy!r} requires max_depth >= 1 (a finite "
                f"depth bounds the tree); got max_depth={max_depth}. Set a positive "
                "max_depth, or use grow_policy='leafwise'."
            )
        self.splitter = splitter
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.grow_policy = grow_policy

    def grow(
        self, grad: np.ndarray, hess: np.ndarray
    ) -> tuple[Tree, list[np.ndarray]]:
        """Grow a tree on the full training set; returns (tree, rows-per-leaf)."""
        if self.grow_policy == "depthwise":
            # A device backend with the batched-scan gate on scans each level's
            # frontier in one kernel launch; the level-synchronous grower is
            # bitwise-identical to the per-node FIFO on any backend whose batched
            # scan equals the loop (scalar targets only — multi-output keeps FIFO).
            if grad.ndim == 1 and getattr(
                self.splitter.backend, "supports_batched_scan", False
            ):
                return self._grow_depthwise_batched(grad, hess)
            return self._grow_depthwise(grad, hess)
        if self.grow_policy == "symmetric":
            return self._grow_symmetric(grad, hess)
        return self._grow_leafwise(grad, hess)

    # ------------------------------------------------------------------ #
    # Growth policies
    # ------------------------------------------------------------------ #
    def _grow_leafwise(
        self, grad: np.ndarray, hess: np.ndarray
    ) -> tuple[Tree, list[np.ndarray]]:
        """Best-gain-first growth (the historical default).

        A max-heap on split gain pops the most promising leaf each step until
        ``num_leaves`` is reached.
        """
        store = self._new_store()
        counter = itertools.count()  # deterministic heap tie-break
        heap: list[_GrowCandidate] = []
        root_rows = store.node_rows[0]
        if self._can_split(root_rows, depth=0):
            root_hist = self.splitter.build_histograms(root_rows, grad, hess)
            root = self._make_candidate(counter, 0, root_rows, 0, root_hist)
            if root is not None:
                heapq.heappush(heap, root)

        n_leaves = 1
        while heap and n_leaves < self.num_leaves:
            cand = heapq.heappop(heap)
            n_leaves += 1
            for node_index, rows, depth, hist in self._expand(store, grad, hess, cand):
                child = self._make_candidate(counter, node_index, rows, depth, hist)
                if child is not None:
                    heapq.heappush(heap, child)
        return self._finalize(store)

    def _grow_depthwise(
        self, grad: np.ndarray, hess: np.ndarray
    ) -> tuple[Tree, list[np.ndarray]]:
        """Level-order growth to ``max_depth`` (XGBoost ``grow_policy=depthwise``).

        A FIFO queue expands nodes breadth-first, so the tree fills level by
        level; ``_can_split`` stops it at ``max_depth``. ``num_leaves`` remains an
        optional secondary cap (raise it to allow a full depth-``d`` tree).
        """
        store = self._new_store()
        counter = itertools.count()  # deterministic tie-break inside _GrowCandidate
        frontier: deque[_GrowCandidate] = deque()
        root_rows = store.node_rows[0]
        if self._can_split(root_rows, depth=0):
            root_hist = self.splitter.build_histograms(root_rows, grad, hess)
            root = self._make_candidate(counter, 0, root_rows, 0, root_hist)
            if root is not None:
                frontier.append(root)

        n_leaves = 1
        while frontier and n_leaves < self.num_leaves:
            cand = frontier.popleft()  # FIFO -> breadth-first / level order
            n_leaves += 1
            for node_index, rows, depth, hist in self._expand(store, grad, hess, cand):
                child = self._make_candidate(counter, node_index, rows, depth, hist)
                if child is not None:
                    frontier.append(child)
        return self._finalize(store)

    def _grow_depthwise_batched(
        self, grad: np.ndarray, hess: np.ndarray
    ) -> tuple[Tree, list[np.ndarray]]:
        """Depthwise growth that scans each level's frontier in ONE backend call.

        Processes the tree level-synchronously: expand the current level's
        candidates left-to-right (committing splits + building child histograms in
        the exact order the per-node FIFO would pop them), then scan all the
        level's children in a single :meth:`Splitter.find_best_split_batched` call.
        On a device backend this batches M nodes into one kernel launch; on the
        host the batched scan loops the per-node scan, so the produced tree is
        **bitwise-identical** to :meth:`_grow_depthwise` (asserted in
        tests/test_grow_policy.py). Used only when the backend opts in
        (``supports_batched_scan``) for scalar targets.
        """
        store = self._new_store()
        counter = itertools.count()  # same tie-break order as the FIFO path
        root_rows = store.node_rows[0]
        if not self._can_split(root_rows, depth=0):
            return self._finalize(store)
        root_hist = self.splitter.build_histograms(root_rows, grad, hess)
        root = self._make_candidate(counter, 0, root_rows, 0, root_hist)
        if root is None:
            return self._finalize(store)

        level = [root]
        n_leaves = 1
        while level and n_leaves < self.num_leaves:
            children: list[tuple[int, np.ndarray, int, np.ndarray]] = []
            for cand in level:
                if n_leaves >= self.num_leaves:
                    break  # FIFO stops popping at the cap, left-to-right
                n_leaves += 1
                children.extend(self._expand(store, grad, hess, cand))
            if not children:
                break
            level = self._make_candidates_batched(counter, children)
        return self._finalize(store)

    def _make_candidates_batched(
        self,
        counter,
        children: list[tuple[int, np.ndarray, int, np.ndarray]],
    ) -> list[_GrowCandidate]:
        """One batched split scan over the level's children → next-level candidates.

        Preserves the per-node path's order exactly: the tie-break ``counter`` is
        advanced only for children that yield a split, in input order, so the
        ``_GrowCandidate.tiebreak`` values match :meth:`_make_candidate`.
        """
        splits = self.splitter.find_best_split_batched([h for _, _, _, h in children])
        out: list[_GrowCandidate] = []
        for (node_index, rows, depth, hist), split in zip(children, splits):
            if split is None:
                continue
            out.append(
                _GrowCandidate(
                    neg_gain=-split.gain,
                    tiebreak=next(counter),
                    node_index=node_index,
                    rows=rows,
                    depth=depth,
                    split=split,
                    hist=hist,
                )
            )
        return out

    def _grow_symmetric(
        self, grad: np.ndarray, hess: np.ndarray
    ) -> tuple[Tree, list[np.ndarray]]:
        """CatBoost-style oblivious growth: one shared split per level.

        Each level picks the single ``(feature, bin)`` maximizing the summed
        per-node gain (host-side :meth:`Splitter.find_best_level_split`) and
        applies it to *every* node, doubling the level. A candidate must be valid
        at all nodes, so growth is all-or-none per level and the tree stays
        complete (up to ``2**max_depth`` leaves). v0 limitations: scalar targets
        only (multi-output raises), and numeric/ordered-threshold splits only
        (categorical features route as ordered thresholds, no subset splits).
        """
        if grad.ndim > 1:
            raise NotImplementedError(
                "grow_policy='symmetric' does not support multi-output / vector "
                "targets in v0; use grow_policy='leafwise' or 'depthwise'."
            )
        store = self._new_store()
        root_rows = store.node_rows[0]
        if not self._can_split(root_rows, depth=0):
            return self._finalize(store)  # root-only tree
        root_hist = self.splitter.build_histograms(root_rows, grad, hess)
        # One level at a time: (node_index, rows, histogram) sharing one split.
        level = [(0, root_rows, root_hist)]
        for depth in range(self.max_depth):
            if not level or not all(
                self._can_split(rows, depth) for _, rows, _ in level
            ):
                break
            choice = self.splitter.find_best_level_split([h for _, _, h in level])
            if choice is None:
                break  # no globally-valid split improves the level
            feature, bin_ = choice
            build_children = depth + 1 < self.max_depth
            next_level: list[tuple[int, np.ndarray, np.ndarray]] = []
            for node_index, rows, hist in level:
                split = self.splitter.split_at(hist, feature, bin_)
                rows_l, rows_r = self.splitter.partition(rows, split)
                li, ri = self._commit_split(store, node_index, split, rows_l, rows_r)
                if build_children:
                    hist_l, hist_r = self._child_hists(
                        grad, hess, hist, rows_l, rows_r, can_l=True, can_r=True
                    )
                    next_level.append((li, rows_l, hist_l))
                    next_level.append((ri, rows_r, hist_r))
            level = next_level
        return self._finalize(store)

    # ------------------------------------------------------------------ #
    # Shared scaffolding
    # ------------------------------------------------------------------ #
    def _new_store(self) -> _NodeStore:
        """Fresh node store holding just the root (all rows)."""
        n_rows = self.splitter.binned.shape[0]
        all_rows = np.arange(n_rows, dtype=np.int64)
        return _NodeStore(
            feature=[-1],
            threshold=[np.nan],
            left=[-1],
            right=[-1],
            gain=[0.0],
            left_cats=[None],
            node_rows={0: all_rows},
        )

    def _commit_split(
        self,
        store: _NodeStore,
        node_index: int,
        split: SplitCandidate,
        rows_l: np.ndarray,
        rows_r: np.ndarray,
    ) -> tuple[int, int]:
        """Turn ``node_index`` into an internal node; append its two leaf slots.

        Returns the new (left, right) child node indices. Identical bookkeeping
        for every policy: set the split feature/threshold (or categorical subset),
        the gain, and the child row partitions.
        """
        li, ri = len(store.feature), len(store.feature) + 1
        for _ in range(2):
            store.feature.append(-1)
            store.threshold.append(np.nan)
            store.left.append(-1)
            store.right.append(-1)
            store.gain.append(0.0)
            store.left_cats.append(None)
        store.feature[node_index] = split.feature
        if split.left_categories is not None:
            # Categorical subset split: codes compared by membership, stored as
            # float64 to match the raw feature matrix.
            store.left_cats[node_index] = split.left_categories.astype(np.float64)
        else:
            store.threshold[node_index] = self.splitter.threshold_value(split)
        store.left[node_index] = li
        store.right[node_index] = ri
        store.gain[node_index] = split.gain
        del store.node_rows[node_index]
        store.node_rows[li] = rows_l
        store.node_rows[ri] = rows_r
        return li, ri

    def _child_hists(
        self,
        grad: np.ndarray,
        hess: np.ndarray,
        parent_hist: np.ndarray,
        rows_l: np.ndarray,
        rows_r: np.ndarray,
        can_l: bool,
        can_r: bool,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Child histograms via sibling subtraction.

        Build the smaller child directly and derive the larger from the parent
        (``parent - child == sibling``). A child whose ``can_*`` flag is False is
        not needed downstream and returned as None (its histogram is not built).
        """
        hist_l = hist_r = None
        if can_l or can_r:
            if rows_l.shape[0] <= rows_r.shape[0]:
                hist_l = self.splitter.build_histograms(rows_l, grad, hess)
                if can_r:
                    hist_r = parent_hist - hist_l
            else:
                hist_r = self.splitter.build_histograms(rows_r, grad, hess)
                if can_l:
                    hist_l = parent_hist - hist_r
        return hist_l, hist_r

    def _expand(
        self,
        store: _NodeStore,
        grad: np.ndarray,
        hess: np.ndarray,
        cand: _GrowCandidate,
    ) -> list[tuple[int, np.ndarray, int, np.ndarray]]:
        """Split ``cand``'s node and return its splittable children.

        Shared by leaf-wise and depth-wise growth (they differ only in the order
        children are revisited). Each returned tuple is
        ``(node_index, rows, depth, histogram)`` ready for :meth:`_make_candidate`.
        """
        rows_l, rows_r = self.splitter.partition(cand.rows, cand.split)
        li, ri = self._commit_split(store, cand.node_index, cand.split, rows_l, rows_r)
        child_depth = cand.depth + 1
        can_l = self._can_split(rows_l, child_depth)
        can_r = self._can_split(rows_r, child_depth)
        hist_l, hist_r = self._child_hists(
            grad, hess, cand.hist, rows_l, rows_r, can_l, can_r
        )
        children: list[tuple[int, np.ndarray, int, np.ndarray]] = []
        if can_l:
            children.append((li, rows_l, child_depth, hist_l))
        if can_r:
            children.append((ri, rows_r, child_depth, hist_r))
        return children

    def _finalize(self, store: _NodeStore) -> tuple[Tree, list[np.ndarray]]:
        """Assign leaf ids to the remaining nodes and build the ``Tree``."""
        n_nodes = len(store.feature)
        leaf_id = np.full(n_nodes, -1, dtype=np.int32)
        leaf_rows: list[np.ndarray] = []
        for node_index in sorted(store.node_rows):
            leaf_id[node_index] = len(leaf_rows)
            leaf_rows.append(store.node_rows[node_index])

        tree = Tree(
            feature=np.asarray(store.feature, dtype=np.int32),
            threshold=np.asarray(store.threshold, dtype=np.float64),
            left=np.asarray(store.left, dtype=np.int32),
            right=np.asarray(store.right, dtype=np.int32),
            leaf_id=leaf_id,
            # Native training always routes missing values left (v0 rule).
            missing_left=np.ones(n_nodes, dtype=bool),
            gain=np.asarray(store.gain, dtype=np.float64),
            left_categories=(
                store.left_cats
                if any(c is not None for c in store.left_cats)
                else None
            ),
        )
        return tree, leaf_rows

    def _can_split(self, rows: np.ndarray, depth: int) -> bool:
        if self.max_depth >= 0 and depth >= self.max_depth:
            return False
        return rows.shape[0] >= 2 * self.splitter.min_samples_leaf

    def _make_candidate(
        self,
        counter,
        node_index: int,
        rows: np.ndarray,
        depth: int,
        hist: np.ndarray,
    ) -> _GrowCandidate | None:
        """Best split for a node as a heap/queue entry, or None if it can't split."""
        split = self.splitter.find_best_split(hist)
        if split is None:
            return None
        return _GrowCandidate(
            neg_gain=-split.gain,
            tiebreak=next(counter),
            node_index=node_index,
            rows=rows,
            depth=depth,
            split=split,
            hist=hist,
        )
