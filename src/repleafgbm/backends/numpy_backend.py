"""NumPy implementation of histogram-based split search.

Both kernels are fully vectorized across features: histogram accumulation is
three ``bincount`` calls over a flattened (row, feature) index, and the split
scan is a cumulative-sum sweep over the whole (feature, bin) grid.
"""

from __future__ import annotations

import numpy as np

from repleafgbm.backends.base import BaseSplitBackend, SplitCandidate


class NumPySplitBackend(BaseSplitBackend):
    """Reference split-search kernels: vectorized, still readable."""

    def build_histograms(
        self,
        binned: np.ndarray,
        rows: np.ndarray,
        grad: np.ndarray,
        hess: np.ndarray,
        n_bins_max: int,
    ) -> np.ndarray:
        sub = binned[rows].astype(np.int64)  # (n, F)
        n, n_features = sub.shape
        # Flatten to one index space: feature f occupies [f*B, (f+1)*B).
        sub += np.arange(n_features, dtype=np.int64) * n_bins_max
        flat = sub.ravel()  # row-major: row 0 all features, row 1, ...
        size = n_features * n_bins_max

        g_rep = np.repeat(grad[rows], n_features)
        h_rep = np.repeat(hess[rows], n_features)
        hist = np.empty((n_features, n_bins_max, 3), dtype=np.float64)
        hist[:, :, 0] = np.bincount(flat, weights=g_rep, minlength=size).reshape(
            n_features, n_bins_max
        )
        hist[:, :, 1] = np.bincount(flat, weights=h_rep, minlength=size).reshape(
            n_features, n_bins_max
        )
        hist[:, :, 2] = np.bincount(flat, minlength=size).reshape(n_features, n_bins_max)
        return hist

    def find_best_split(
        self,
        hist: np.ndarray,
        n_bins_per_feature: np.ndarray,
        min_samples_leaf: int,
        l2: float,
        categorical_mask: np.ndarray | None = None,
        cat_smooth: float = 10.0,
        min_data_per_group: int = 100,
        max_cat_threshold: int = 32,
    ) -> SplitCandidate | None:
        g, h, n = hist[:, :, 0], hist[:, :, 1], hist[:, :, 2]
        n_features, n_bins_max = g.shape
        feat_idx = np.arange(n_features)

        # Every feature's bins partition the same rows, so per-feature totals
        # all equal the node totals; read them off feature 0.
        g_total = float(g[0].sum())
        h_total = float(h[0].sum())
        n_total = float(n[0].sum())
        parent_score = _leaf_score(g_total, h_total, l2)

        # Missing values (bin index n_bins_per_feature[f]) always go left.
        miss_g = g[feat_idx, n_bins_per_feature][:, None]
        miss_h = h[feat_idx, n_bins_per_feature][:, None]
        miss_n = n[feat_idx, n_bins_per_feature][:, None]

        # Candidate c sends non-missing bins <= c left. Cumsums up to
        # c < n_bins_per_feature[f] never include the missing bin, which sits
        # at a higher index; invalid candidates are masked below.
        left_g = np.cumsum(g, axis=1) + miss_g
        left_h = np.cumsum(h, axis=1) + miss_h
        left_n = np.cumsum(n, axis=1) + miss_n
        right_n = n_total - left_n

        valid = (
            (np.arange(n_bins_max)[None, :] <= (n_bins_per_feature - 2)[:, None])
            & (left_n >= min_samples_leaf)
            & (right_n >= min_samples_leaf)
        )
        if categorical_mask is not None:
            # Categorical features get the subset scan below, not the
            # ordered-threshold scan.
            valid &= ~categorical_mask[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            gain = (
                _leaf_score(left_g, left_h, l2)
                + _leaf_score(g_total - left_g, h_total - left_h, l2)
                - parent_score
            )
        gain = np.where(valid & np.isfinite(gain), gain, -np.inf)

        best_flat = int(np.argmax(gain))  # deterministic tie-break: lowest index
        best_gain = float(gain.flat[best_flat])
        best: SplitCandidate | None = None
        if best_gain > 1e-12:
            f, c = divmod(best_flat, n_bins_max)
            best = SplitCandidate(
                feature=int(f),
                bin=int(c),
                gain=best_gain,
                n_left=int(left_n[f, c]),
                n_right=int(right_n[f, c]),
            )

        if categorical_mask is not None and categorical_mask.any():
            for f in np.flatnonzero(categorical_mask):
                cand = self._best_categorical_split(
                    int(f), hist[f], int(n_bins_per_feature[f]),
                    g_total, h_total, n_total, parent_score,
                    min_samples_leaf, l2,
                    cat_smooth, min_data_per_group, max_cat_threshold,
                )
                if cand is not None and (best is None or cand.gain > best.gain):
                    best = cand
        return best

    def _best_categorical_split(
        self,
        feature: int,
        hist_f: np.ndarray,
        n_bins: int,
        g_total: float,
        h_total: float,
        n_total: float,
        parent_score: float,
        min_samples_leaf: int,
        l2: float,
        cat_smooth: float,
        min_data_per_group: int,
        max_cat_threshold: int,
    ) -> SplitCandidate | None:
        """Gradient-sorted subset scan for one categorical feature.

        Categories present in the node are sorted by their smoothed Newton
        direction ``sum_g / (sum_h + cat_smooth)``; the optimal binary
        partition is then a prefix of that order (the classic LightGBM
        trick), scanned from both ends so a small subset on *either* side
        fits under ``max_cat_threshold``. Missing values always join the
        left side. High-cardinality overfitting guards (LightGBM-default
        values; see experiments/results/real_data_validation.md Phase 8b):

        * ``min_data_per_group``: categories with fewer node rows are not
          eligible for the left subset (they implicitly go right),
        * ``max_cat_threshold``: at most this many categories on the left,
        * ``cat_smooth``: keeps rare categories' noisy gradients from
          grabbing the extreme sort positions.
        """
        g, h, n = hist_f[:n_bins, 0], hist_f[:n_bins, 1], hist_f[:n_bins, 2]
        miss_g, miss_h, miss_n = hist_f[n_bins]
        present = np.flatnonzero(n >= max(min_data_per_group, 1))
        if present.size < 2:
            return None
        order = present[
            np.argsort(g[present] / (h[present] + cat_smooth), kind="stable")
        ]

        best: SplitCandidate | None = None
        for direction in (order, order[::-1]):
            limit = min(direction.size - 1, max_cat_threshold)
            left_g = np.cumsum(g[direction[:limit]]) + miss_g
            left_h = np.cumsum(h[direction[:limit]]) + miss_h
            left_n = np.cumsum(n[direction[:limit]]) + miss_n
            right_n = n_total - left_n
            valid = (left_n >= min_samples_leaf) & (right_n >= min_samples_leaf)
            with np.errstate(divide="ignore", invalid="ignore"):
                gain = (
                    _leaf_score(left_g, left_h, l2)
                    + _leaf_score(g_total - left_g, h_total - left_h, l2)
                    - parent_score
                )
            gain = np.where(valid & np.isfinite(gain), gain, -np.inf)
            c = int(np.argmax(gain))
            if not np.isfinite(gain[c]) or gain[c] <= 1e-12:
                continue
            if best is None or gain[c] > best.gain:
                best = SplitCandidate(
                    feature=feature,
                    bin=-1,
                    gain=float(gain[c]),
                    n_left=int(left_n[c]),
                    n_right=int(right_n[c]),
                    left_categories=np.sort(direction[: c + 1]).astype(np.int64),
                )
        return best


def _leaf_score(g, h, l2: float):
    """Newton objective reduction of a leaf: G^2 / (H + l2) (factor 1/2 dropped)."""
    return g * g / (h + l2)


def find_best_split_multioutput(
    hist: np.ndarray,
    n_bins_per_feature: np.ndarray,
    min_samples_leaf: int,
    l2: float,
) -> SplitCandidate | None:
    """Best numerical split for a shared-routing multi-output node.

    ``hist`` is the stacked layout ``(n_features, n_bins_max, 3, n_outputs)``
    (the scalar 3-channel histogram built once per output). Routing is shared
    across outputs, so the split gain is the per-output Newton gain summed over
    outputs: ``sum_k G_k^2 / (H_k + l2)``. Missing values always go left,
    exactly as in the scalar :meth:`NumPySplitBackend.find_best_split`.

    Categorical subset splits are not produced here (multi-output trees route
    categoricals as ordered thresholds); this scan handles every feature as an
    ordered-threshold candidate.
    """
    g, h, n = hist[:, :, 0, :], hist[:, :, 1, :], hist[:, :, 2, 0]
    n_features, n_bins_max, n_outputs = g.shape
    feat_idx = np.arange(n_features)

    # Per-output totals (same across features); read off feature 0.
    g_total = g[0].sum(axis=0)  # (K,)
    h_total = h[0].sum(axis=0)  # (K,)
    n_total = float(n[0].sum())
    parent_score = float(_leaf_score(g_total, h_total, l2).sum())

    miss_g = g[feat_idx, n_bins_per_feature, :][:, None, :]  # (F, 1, K)
    miss_h = h[feat_idx, n_bins_per_feature, :][:, None, :]
    miss_n = n[feat_idx, n_bins_per_feature][:, None]  # (F, 1)

    left_g = np.cumsum(g, axis=1) + miss_g  # (F, B, K)
    left_h = np.cumsum(h, axis=1) + miss_h
    left_n = np.cumsum(n, axis=1) + miss_n  # (F, B)
    right_n = n_total - left_n

    valid = (
        (np.arange(n_bins_max)[None, :] <= (n_bins_per_feature - 2)[:, None])
        & (left_n >= min_samples_leaf)
        & (right_n >= min_samples_leaf)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = (
            _leaf_score(left_g, left_h, l2)
            + _leaf_score(g_total - left_g, h_total - left_h, l2)
        ).sum(axis=2) - parent_score  # (F, B)
    gain = np.where(valid & np.isfinite(gain), gain, -np.inf)

    best_flat = int(np.argmax(gain))  # deterministic tie-break: lowest index
    best_gain = float(gain.flat[best_flat])
    if best_gain <= 1e-12:
        return None
    f, c = divmod(best_flat, n_bins_max)
    return SplitCandidate(
        feature=int(f),
        bin=int(c),
        gain=best_gain,
        n_left=int(left_n[f, c]),
        n_right=int(right_n[f, c]),
    )
