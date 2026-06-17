"""Decision tree: leaf-wise growth on raw features, flat-array storage.

The tree only routes: it stores split features/thresholds and leaf ids.
Leaf *values* live in :mod:`repleafgbm.core.leaf_models`, because in
RepLeafGBM a leaf is a small model over the representation, not a constant.
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field

import numpy as np

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
        """Route rows to leaves; returns leaf ids of shape (n_rows,)."""
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


class TreeGrower:
    """Grows one tree leaf-wise (best-gain-first), like LightGBM.

    Leaf-wise growth makes ``num_leaves`` the natural complexity control;
    ``max_depth`` (-1 = unlimited) additionally caps depth.
    """

    def __init__(
        self,
        splitter: Splitter,
        num_leaves: int = 31,
        max_depth: int = -1,
    ) -> None:
        if num_leaves < 2:
            raise ValueError(f"num_leaves must be >= 2, got {num_leaves}")
        self.splitter = splitter
        self.num_leaves = num_leaves
        self.max_depth = max_depth

    def grow(
        self, grad: np.ndarray, hess: np.ndarray
    ) -> tuple[Tree, list[np.ndarray]]:
        """Grow a tree on the full training set; returns (tree, rows-per-leaf)."""
        n_rows = self.splitter.binned.shape[0]
        all_rows = np.arange(n_rows, dtype=np.int64)

        # Growable node store; flattened to arrays at the end.
        feature: list[int] = [-1]
        threshold: list[float] = [np.nan]
        left: list[int] = [-1]
        right: list[int] = [-1]
        gain: list[float] = [0.0]
        left_cats: list[np.ndarray | None] = [None]
        node_rows: dict[int, np.ndarray] = {0: all_rows}

        counter = itertools.count()  # deterministic heap tie-break
        heap: list[_GrowCandidate] = []
        if self._can_split(all_rows, depth=0):
            root_hist = self.splitter.build_histograms(all_rows, grad, hess)
            self._push_candidate(heap, counter, 0, all_rows, depth=0, hist=root_hist)

        n_leaves = 1
        while heap and n_leaves < self.num_leaves:
            cand = heapq.heappop(heap)
            rows_l, rows_r = self.splitter.partition(cand.rows, cand.split)
            li, ri = len(feature), len(feature) + 1
            for _ in range(2):
                feature.append(-1)
                threshold.append(np.nan)
                left.append(-1)
                right.append(-1)
                gain.append(0.0)
                left_cats.append(None)
            feature[cand.node_index] = cand.split.feature
            if cand.split.left_categories is not None:
                # Categorical subset split: codes compared by membership,
                # stored as float64 to match the raw feature matrix.
                left_cats[cand.node_index] = cand.split.left_categories.astype(
                    np.float64
                )
            else:
                threshold[cand.node_index] = self.splitter.threshold_value(cand.split)
            left[cand.node_index] = li
            right[cand.node_index] = ri
            gain[cand.node_index] = cand.split.gain
            del node_rows[cand.node_index]
            node_rows[li] = rows_l
            node_rows[ri] = rows_r
            n_leaves += 1

            # Sibling subtraction: accumulate the smaller child's histogram,
            # derive the larger one from the parent's.
            child_depth = cand.depth + 1
            can_l = self._can_split(rows_l, child_depth)
            can_r = self._can_split(rows_r, child_depth)
            hist_l = hist_r = None
            if can_l or can_r:
                if rows_l.shape[0] <= rows_r.shape[0]:
                    hist_l = self.splitter.build_histograms(rows_l, grad, hess)
                    if can_r:
                        hist_r = cand.hist - hist_l
                else:
                    hist_r = self.splitter.build_histograms(rows_r, grad, hess)
                    if can_l:
                        hist_l = cand.hist - hist_r
            if can_l:
                self._push_candidate(heap, counter, li, rows_l, child_depth, hist_l)
            if can_r:
                self._push_candidate(heap, counter, ri, rows_r, child_depth, hist_r)

        # Remaining entries in node_rows are the final leaves.
        n_nodes = len(feature)
        leaf_id = np.full(n_nodes, -1, dtype=np.int32)
        leaf_rows: list[np.ndarray] = []
        for node_index in sorted(node_rows):
            leaf_id[node_index] = len(leaf_rows)
            leaf_rows.append(node_rows[node_index])

        tree = Tree(
            feature=np.asarray(feature, dtype=np.int32),
            threshold=np.asarray(threshold, dtype=np.float64),
            left=np.asarray(left, dtype=np.int32),
            right=np.asarray(right, dtype=np.int32),
            leaf_id=leaf_id,
            # Native training always routes missing values left (v0 rule).
            missing_left=np.ones(n_nodes, dtype=bool),
            gain=np.asarray(gain, dtype=np.float64),
            left_categories=(
                left_cats if any(c is not None for c in left_cats) else None
            ),
        )
        return tree, leaf_rows

    def _can_split(self, rows: np.ndarray, depth: int) -> bool:
        if self.max_depth >= 0 and depth >= self.max_depth:
            return False
        return rows.shape[0] >= 2 * self.splitter.min_samples_leaf

    def _push_candidate(
        self,
        heap: list[_GrowCandidate],
        counter,
        node_index: int,
        rows: np.ndarray,
        depth: int,
        hist: np.ndarray,
    ) -> None:
        split = self.splitter.find_best_split(hist)
        if split is None:
            return
        heapq.heappush(
            heap,
            _GrowCandidate(
                neg_gain=-split.gain,
                tiebreak=next(counter),
                node_index=node_index,
                rows=rows,
                depth=depth,
                split=split,
                hist=hist,
            ),
        )
