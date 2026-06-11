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
        with np.errstate(divide="ignore", invalid="ignore"):
            gain = (
                _leaf_score(left_g, left_h, l2)
                + _leaf_score(g_total - left_g, h_total - left_h, l2)
                - parent_score
            )
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


def _leaf_score(g, h, l2: float):
    """Newton objective reduction of a leaf: G^2 / (H + l2) (factor 1/2 dropped)."""
    return g * g / (h + l2)
