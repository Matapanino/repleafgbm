"""Leaf models: per-leaf predictors fitted on boosting statistics.

This is the core RepLeafGBM idea. A leaf is not a constant; it may be a small
ridge-regularized linear model over the representation Z:

    f_t(x) = b_{leaf} + w_{leaf}^T z_theta(x)

All leaf models are fitted to the Newton targets ``t_i = -g_i / h_i`` with
weights ``h_i``, which makes the same code path exact for squared error
(h = 1, t = residual) and a Newton approximation for other losses.

Overfitting guards (docs/design.md section "Leaf model variants"):

* ridge penalty ``l2`` on weights (never on the intercept),
* fallback to a constant leaf when the leaf has too few samples relative to
  the embedding dimension.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

try:  # Optional compiled fast path for fused per-leaf statistics (native/).
    import repleafgbm_native as _native_module

    _native = getattr(_native_module, "leaf_linear_stats", None) and _native_module
except ImportError:  # pragma: no cover - depends on optional extension
    _native = None

#: Above this embedding width the BLAS-based NumPy path beats the native fused
#: pass, so wider embeddings fall back to BLAS. The native helper is rayon
#: leaf-parallel (parallelizing across leaves — the right axis for the small
#: per-leaf Gram matrices, which thread poorly inside BLAS). The 32→64 move
#: (2026-06-19) only validated the default emb=64; a 2026-06-25 crossover sweep
#: (8-core arm64) showed native still wins to ~200 single-thread and ~128
#: multi-threaded BLAS (crossover ~256), output allclose ~1e-14, so the gate is
#: raised to a conservative 128 — well below the measured crossover and a win
#: under both threading regimes (docs/perf-notes/experiment-log.md iter 005).
_NATIVE_STATS_MAX_DIM = 128

#: Valid ``leaf_fit_precision`` values. ``"float64"`` (default) is the
#: bitwise-parity path; ``"float32_gram"`` accumulates only the wide-embedding
#: (emb>128) per-leaf Gram + gradient projection in float32 (≈2x faster) while the
#: solve stays float64 — opt-in, allclose-not-bitwise
#: (docs/proposals/float32-wide-embedding-leaf-fit.md).
_LEAF_FIT_PRECISIONS = ("float64", "float32_gram")

#: Per-tree leaf parameters: output = bias[leaf] + Z @ weights[leaf].
#: For constant leaves, the weight row is all zeros (weights may have zero
#: columns when the whole tree is constant-leaf).


@dataclass
class LeafValues:
    #: Per-leaf intercept: (n_leaves,) for scalar leaves, or (n_leaves,
    #: n_outputs) for vector leaves (multi-output regression).
    bias: np.ndarray
    #: Per-leaf linear weights over the representation Z: (n_leaves, emb_dim)
    #: for scalar leaves, or (n_leaves, emb_dim, n_outputs) for vector leaves.
    #: ``emb_dim`` may be 0 (constant leaves).
    weights: np.ndarray
    #: Per-leaf embedding clip bounds, (n_leaves, emb_dim) — shared across
    #: outputs for vector leaves. At prediction time Z is clipped to the range
    #: the leaf was fitted on, so beyond its training support a leaf
    #: extrapolates as a constant — the guard that prevents linear blow-ups on
    #: feature-space outliers (Phase 6/7,
    #: experiments/results/real_data_validation.md). None disables clipping
    #: (models saved before this guard existed).
    z_min: np.ndarray | None = None
    z_max: np.ndarray | None = None

    def predict(
        self, leaf_idx: np.ndarray, Z: np.ndarray | None, clip: bool = True
    ) -> np.ndarray:
        """Leaf outputs for routed rows.

        Returns (n_rows,) for scalar leaves or (n_rows, n_outputs) for vector
        leaves. ``clip=False`` skips the extrapolation guard; it is only valid
        for the rows the leaves were fitted on (the booster's training-score
        update), where clipping to the leaf's own min/max is exactly the
        identity.
        """
        if (
            self.weights.ndim == 2
            and self.weights.shape[1] > 0
            and _native is not None
            and hasattr(_native, "predict_linear")
        ):
            # Fused native gather+dot (Session 4): replaces the bias gather +
            # einsum (which materializes an (n_rows, d) weight gather) — the
            # dominant multiclass training-eval cost and a prediction speedup.
            # Vector (multi-output) and constant leaves fall through to NumPy.
            assert Z is not None, "embedding matrix required for linear leaves"
            do_clip = clip and self.z_min is not None
            zmn = self.z_min if self.z_min is not None else self.weights
            zmx = self.z_max if self.z_max is not None else self.weights
            return _native.predict_linear(
                np.ascontiguousarray(leaf_idx, dtype=np.int64),
                np.ascontiguousarray(Z, dtype=np.float64),
                np.ascontiguousarray(self.bias, dtype=np.float64),
                np.ascontiguousarray(self.weights, dtype=np.float64),
                np.ascontiguousarray(zmn, dtype=np.float64),
                np.ascontiguousarray(zmx, dtype=np.float64),
                do_clip,
            )

        out = self.bias[leaf_idx]
        if self.weights.shape[1] > 0:
            assert Z is not None, "embedding matrix required for linear leaves"
            if clip and self.z_min is not None:
                Z = np.clip(Z, self.z_min[leaf_idx], self.z_max[leaf_idx])
            if self.weights.ndim == 3:  # vector leaves: (n_leaves, emb, K)
                out = out + np.einsum("ij,ijk->ik", Z, self.weights[leaf_idx])
            else:
                out = out + np.einsum("ij,ij->i", Z, self.weights[leaf_idx])
        return out

    @property
    def emb_dim(self) -> int:
        return int(self.weights.shape[1])

    @property
    def n_outputs(self) -> int:
        """Number of outputs (1 for scalar leaves)."""
        return 1 if self.bias.ndim == 1 else int(self.bias.shape[1])


class BaseLeafModel(ABC):
    """Fits per-leaf parameters from gradients/Hessians and embeddings."""

    name: str = "base"

    #: Whether this leaf model consumes the embedding matrix Z.
    uses_embeddings: bool = False

    @abstractmethod
    def fit_leaves(
        self,
        leaf_rows: list[np.ndarray],
        grad: np.ndarray,
        hess: np.ndarray,
        Z: np.ndarray | None,
    ) -> LeafValues:
        """Fit parameters for every leaf of one tree."""

    def fit_leaves_multiclass(
        self,
        leaf_rows_per_class: list[list[np.ndarray]],
        grad: np.ndarray,
        hess: np.ndarray,
        Z: np.ndarray | None,
    ) -> list[LeafValues]:
        """Fit the K per-class trees' leaves for one boosting round.

        Default: independent per-class fits. ``grad``/``hess`` are the
        ``(n_rows, n_classes)`` matrices; ``leaf_rows_per_class[k]`` holds class
        k's leaf row-sets. :class:`EmbeddedLinearLeafModel` overrides this to pool
        all classes into a single native pass (Session 4).
        """
        return [
            self.fit_leaves(leaf_rows_per_class[k], grad[:, k], hess[:, k], Z)
            for k in range(len(leaf_rows_per_class))
        ]


class ConstantLeafModel(BaseLeafModel):
    """Classic GBDT leaf: a single Newton step value per leaf."""

    name = "constant"
    uses_embeddings = False

    def __init__(self, l2: float = 1.0) -> None:
        self.l2 = l2

    def fit_leaves(
        self,
        leaf_rows: list[np.ndarray],
        grad: np.ndarray,
        hess: np.ndarray,
        Z: np.ndarray | None = None,
    ) -> LeafValues:
        bias = np.array(
            [_newton_constant(grad[r], hess[r], self.l2) for r in leaf_rows],
            dtype=np.float64,
        )
        return LeafValues(bias=bias, weights=np.zeros((len(leaf_rows), 0)))


class EmbeddedLinearLeafModel(BaseLeafModel):
    """Ridge-regularized linear model over the representation Z in each leaf.

    Args:
        l2: Ridge penalty on the weight vector (intercept unpenalized).
        min_samples_linear: Minimum leaf size to fit a linear model; smaller
            leaves fall back to a constant. The effective minimum is
            ``max(min_samples_linear, emb_dim + 2)`` so the weighted normal
            equations stay well-posed.
    """

    name = "embedded_linear"
    uses_embeddings = True

    def __init__(
        self,
        l2: float = 1.0,
        min_samples_linear: int = 10,
        leaf_fit_precision: str = "float64",
    ) -> None:
        if leaf_fit_precision not in _LEAF_FIT_PRECISIONS:
            raise ValueError(
                f"leaf_fit_precision must be one of {_LEAF_FIT_PRECISIONS}, "
                f"got {leaf_fit_precision!r}"
            )
        self.l2 = l2
        self.min_samples_linear = min_samples_linear
        self.leaf_fit_precision = leaf_fit_precision

    def _make_gate(
        self,
        Z: np.ndarray,
        grad: np.ndarray,
        hess: np.ndarray,
        order: np.ndarray,
        offsets: np.ndarray,
        leaf_class: np.ndarray | None,
    ) -> _LeafGate | None:
        """Build the per-leaf generalization gate, or ``None`` when gating is off.

        :class:`EmbeddedLinearLeafModel` has no ``leaf_gate_margin``, so this
        returns ``None`` and every finite linear fit is kept (behavior
        unchanged). :class:`AdaptiveLeafModel` sets the attribute and gets a real
        gate. ``grad``/``hess`` are 1-D (scalar/native paths) or
        ``(n_rows, n_classes)`` with ``leaf_class`` set (pooled-multiclass path).
        """
        margin = getattr(self, "leaf_gate_margin", None)
        if margin is None:
            return None
        insample = getattr(self, "leaf_gate", "loo") == "insample"
        return _LeafGate(Z, grad, hess, order, offsets, leaf_class, margin, insample)

    def fit_leaves(
        self,
        leaf_rows: list[np.ndarray],
        grad: np.ndarray,
        hess: np.ndarray,
        Z: np.ndarray | None,
    ) -> LeafValues:
        """Fit all leaves of one tree with batched normal equations.

        Per-leaf work is reduced to two BLAS products (the weighted Gram
        matrix and the gradient projection); all systems are then solved in
        one batched ``np.linalg.solve`` call. Uses the uncentered identities

            sum_i h (z - z_mean)(z - z_mean)^T = M - S_h z_mean z_mean^T
            sum_i h (z - z_mean)(t - t_mean)   = -G_z + S_g z_mean

        with M = sum h z z^T, G_z = sum g z (since h t = -g). This is
        algebraically identical to the centered reference implementation
        (`_fit_weighted_ridge`, kept for parity testing) and numerically
        safe here because encoders produce standardized/bounded embeddings;
        degenerate directions are damped by the ridge term as before.
        """
        if Z is None:
            raise ValueError("EmbeddedLinearLeafModel requires an embedding matrix Z")
        emb_dim = Z.shape[1]
        n_leaves = len(leaf_rows)
        weights = np.zeros((n_leaves, emb_dim), dtype=np.float64)
        # Extrapolation guard: bounds stay infinite (no-op) for constant
        # leaves, whose weights are zero anyway.
        z_min = np.full((n_leaves, emb_dim), -np.inf, dtype=np.float64)
        z_max = np.full((n_leaves, emb_dim), np.inf, dtype=np.float64)

        # Gather everything once in leaf order; per-leaf data is then a
        # contiguous view (no per-leaf fancy indexing).
        sizes = np.array([r.shape[0] for r in leaf_rows], dtype=np.int64)
        order = np.concatenate(leaf_rows) if leaf_rows else np.empty(0, np.int64)
        offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        min_n = max(self.min_samples_linear, emb_dim + 2)
        linear = np.flatnonzero(sizes >= min_n)
        k = linear.size

        if k and _native is not None and emb_dim <= _NATIVE_STATS_MAX_DIM:
            # Fused single-pass statistics in Rust (narrow embeddings only:
            # for wide ones the BLAS Gram below wins).
            g_sum, h_sum, s_hz, A, gz, zmn, zmx = _native.leaf_linear_stats(
                np.ascontiguousarray(Z), grad, hess, order, offsets,
                linear.astype(np.int64),
            )
            return self._leafvalues_from_native_stats(
                g_sum, h_sum, s_hz, A, gz, zmn, zmx, linear, n_leaves, emb_dim,
                gate=self._make_gate(Z, grad, hess, order, offsets, None),
            )

        # NumPy fallback: per-leaf BLAS Gram (native unavailable, or embeddings
        # too wide for the fused pass).
        seg = np.repeat(np.arange(n_leaves), sizes)
        g_seg = grad[order]
        h_seg = hess[order]
        g_sum = np.bincount(seg, weights=g_seg, minlength=n_leaves)
        h_sum = np.bincount(seg, weights=h_seg, minlength=n_leaves)
        bias = -g_sum / (h_sum + self.l2)
        if k == 0:
            return LeafValues(bias=bias, weights=weights, z_min=z_min, z_max=z_max)

        Z_seg = Z[order]
        hZ_seg = Z_seg * h_seg[:, None]
        A = np.empty((k, emb_dim, emb_dim), dtype=np.float64)
        rhs = np.empty((k, emb_dim), dtype=np.float64)
        z_mean = np.empty((k, emb_dim), dtype=np.float64)
        t_mean = np.empty(k, dtype=np.float64)
        # Opt-in: accumulate ONLY the two large reductions (the weighted Gram and
        # the gradient projection) in float32 (~2x SIMD throughput); the float64
        # ``A``/``rhs`` containers upcast on assignment, and the cancellation-prone
        # centering + the solve stay float64. The float64 default branch below is
        # byte-identical to before (bitwise NumPy<->Rust parity must hold).
        use_f32 = self.leaf_fit_precision == "float32_gram"
        if use_f32:
            Z32, hZ32, g32 = (Z_seg.astype(np.float32),
                              hZ_seg.astype(np.float32), g_seg.astype(np.float32))
        for j, i in enumerate(linear):
            sl = slice(offsets[i], offsets[i + 1])
            Zl = Z_seg[sl]
            hZ = hZ_seg[sl]
            s_hz = hZ.sum(axis=0)
            z_mean[j] = s_hz / h_sum[i]
            t_mean[j] = -g_sum[i] / h_sum[i]
            if use_f32:
                A[j] = Z32[sl].T @ hZ32[sl]
            else:
                A[j] = Zl.T @ hZ
            A[j] -= np.outer(z_mean[j], s_hz)
            if use_f32:
                rhs[j] = -(g32[sl] @ Z32[sl]) - t_mean[j] * s_hz
            else:
                rhs[j] = -(g_seg[sl] @ Zl) - t_mean[j] * s_hz
            z_min[i] = Zl.min(axis=0)
            z_max[i] = Zl.max(axis=0)
        return self._solve_and_assemble(
            A, rhs, bias, weights, z_mean, t_mean, z_min, z_max, linear, emb_dim,
            gate=self._make_gate(Z, grad, hess, order, offsets, None),
        )

    def _leafvalues_from_native_stats(
        self,
        g_sum: np.ndarray,
        h_sum: np.ndarray,
        s_hz: np.ndarray,
        A: np.ndarray,
        gz: np.ndarray,
        zmn: np.ndarray,
        zmx: np.ndarray,
        linear: np.ndarray,
        n_leaves: int,
        emb_dim: int,
        gate: _LeafGate | None = None,
    ) -> LeafValues:
        """Assemble :class:`LeafValues` from the fused native statistics.

        Shared by the scalar native path (:func:`leaf_linear_stats`) and the
        pooled-multiclass path (:func:`leaf_linear_stats_mc`); the centering
        identities mirror the :meth:`fit_leaves` docstring. ``linear`` indexes
        the (possibly pooled) leaves that received a linear fit.
        """
        weights = np.zeros((n_leaves, emb_dim), dtype=np.float64)
        z_min = np.full((n_leaves, emb_dim), -np.inf, dtype=np.float64)
        z_max = np.full((n_leaves, emb_dim), np.inf, dtype=np.float64)
        bias = -g_sum / (h_sum + self.l2)
        if linear.size == 0:
            return LeafValues(bias=bias, weights=weights, z_min=z_min, z_max=z_max)
        z_mean = s_hz / h_sum[linear][:, None]
        t_mean = -g_sum[linear] / h_sum[linear]
        A = A - z_mean[:, :, None] * s_hz[:, None, :]
        rhs = -gz - t_mean[:, None] * s_hz
        z_min[linear] = zmn
        z_max[linear] = zmx
        return self._solve_and_assemble(
            A, rhs, bias, weights, z_mean, t_mean, z_min, z_max, linear, emb_dim,
            gate=gate,
        )

    def _solve_and_assemble(
        self,
        A: np.ndarray,
        rhs: np.ndarray,
        bias: np.ndarray,
        weights: np.ndarray,
        z_mean: np.ndarray,
        t_mean: np.ndarray,
        z_min: np.ndarray,
        z_max: np.ndarray,
        linear: np.ndarray,
        emb_dim: int,
        gate: _LeafGate | None = None,
    ) -> LeafValues:
        """Batched ridge solve + per-leaf assembly shared by every fit path."""
        k = linear.size
        A[:, np.arange(emb_dim), np.arange(emb_dim)] += self.l2
        try:
            # rhs as (k, d, 1): NumPy 2.0 treats a 2-D b as a matrix, not a
            # stack of vectors, so the explicit trailing axis is required for
            # batched vector solves on both NumPy 1.x and 2.x.
            w = np.linalg.solve(A, rhs[:, :, None])[:, :, 0]
        except np.linalg.LinAlgError:
            # Rare: some leaf's system is exactly singular. Solve one by one
            # so only the degenerate leaves fall back to constants.
            w = np.zeros((k, emb_dim), dtype=np.float64)
            for j in range(k):
                try:
                    w[j] = np.linalg.solve(A[j], rhs[j])
                except np.linalg.LinAlgError:
                    w[j] = np.nan  # handled by the finite check below

        ok = np.isfinite(w).all(axis=1)
        # The adaptive gate (if any) demotes finite linear fits that fail the
        # weighted-LOO generalization test; ``A`` is the regularized matrix M.
        keep = ok if gate is None else gate.keep_mask(linear, ok, A, w, z_mean, t_mean)
        for j, i in enumerate(linear):
            if keep[j]:
                weights[i] = w[j]
                bias[i] = t_mean[j] - w[j] @ z_mean[j]
            else:  # constant fallback: Newton bias kept, guard disabled
                z_min[i] = -np.inf
                z_max[i] = np.inf
        return LeafValues(bias=bias, weights=weights, z_min=z_min, z_max=z_max)

    def fit_leaves_multiclass(
        self,
        leaf_rows_per_class: list[list[np.ndarray]],
        grad: np.ndarray,
        hess: np.ndarray,
        Z: np.ndarray | None,
    ) -> list[LeafValues]:
        """Fit all K class trees' leaves in one pooled native pass.

        A single class tree routinely puts >50% of its rows in one leaf, so
        fitting each class separately caps rayon leaf-parallelism near ~2x.
        Pooling every class's leaves into one ``leaf_linear_stats_mc`` call
        dilutes any one giant leaf to a small fraction of the total work, so the
        scheduler keeps all cores busy. Each pooled leaf accumulates its own rows
        in order (reading its class's grad/hess column), so the result is
        bitwise-identical to per-class fitting — only the schedule changes.
        Falls back to independent per-class fits when the native pooled helper is
        unavailable or the embedding is too wide for the fused pass.
        """
        if Z is None:
            raise ValueError("EmbeddedLinearLeafModel requires an embedding matrix Z")
        emb_dim = Z.shape[1]
        n_classes = len(leaf_rows_per_class)
        if (
            _native is None
            or not hasattr(_native, "leaf_linear_stats_mc")
            or emb_dim > _NATIVE_STATS_MAX_DIM
        ):
            return [
                self.fit_leaves(
                    leaf_rows_per_class[k], grad[:, k], hess[:, k], Z
                )
                for k in range(n_classes)
            ]

        # Pool every class's leaves into one global leaf list (class 0's leaves,
        # then class 1's, ...); leaf_class[l] selects leaf l's grad/hess column.
        n_leaves_per_class = [len(lr) for lr in leaf_rows_per_class]
        all_leaves = [r for lr in leaf_rows_per_class for r in lr]
        total_leaves = len(all_leaves)
        sizes = np.array([r.shape[0] for r in all_leaves], dtype=np.int64)
        order = np.concatenate(all_leaves) if all_leaves else np.empty(0, np.int64)
        offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        leaf_class = np.repeat(np.arange(n_classes), n_leaves_per_class).astype(
            np.int64
        )
        min_n = max(self.min_samples_linear, emb_dim + 2)
        linear = np.flatnonzero(sizes >= min_n)
        g_sum, h_sum, s_hz, A, gz, zmn, zmx = _native.leaf_linear_stats_mc(
            np.ascontiguousarray(Z),
            np.ascontiguousarray(grad),
            np.ascontiguousarray(hess),
            order,
            offsets,
            linear.astype(np.int64),
            leaf_class,
        )
        pooled = self._leafvalues_from_native_stats(
            g_sum, h_sum, s_hz, A, gz, zmn, zmx, linear, total_leaves, emb_dim,
            gate=self._make_gate(Z, grad, hess, order, offsets, leaf_class),
        )
        # Split the pooled per-leaf parameters back into per-class LeafValues.
        out: list[LeafValues] = []
        start = 0
        for nl in n_leaves_per_class:
            sl = slice(start, start + nl)
            out.append(
                LeafValues(
                    bias=pooled.bias[sl].copy(),
                    weights=pooled.weights[sl].copy(),
                    z_min=pooled.z_min[sl].copy(),
                    z_max=pooled.z_max[sl].copy(),
                )
            )
            start += nl
        return out


class AdaptiveLeafModel(EmbeddedLinearLeafModel):
    """Embedded-linear leaves with a per-leaf generalization gate.

    Fits the same ridge-regularized linear leaf as
    :class:`EmbeddedLinearLeafModel`, then keeps each leaf's linear model only if
    it beats a plain constant leaf in weighted leave-one-out (LOO) error;
    otherwise the leaf falls back to a constant (its weight row is zeroed and the
    Newton bias kept — exactly the existing singular-solve fallback). This takes
    the embedded-linear gain on leaves where it generalizes and demotes the
    noise-absorbing leaves that degrade binary tasks
    (experiments/results/binary_leaf_gain.md): the constant-vs-linear choice is
    made per leaf instead of globally.

    The gate runs *after* the existing ``min_samples_linear`` size pre-filter, so
    sub-threshold leaves are already constant before it is consulted, and it
    consumes only host-side statistics — the native (Rust) stats path is
    unchanged, and a gated-to-constant leaf serializes/predicts like any other
    constant leaf (no format change).

    The leave-one-out test assumes a ridge-regularized leaf (``l2 > 0``); with
    ``l2 = 0`` a rank-deficient leaf takes the existing singular-solve constant
    fallback before the gate is consulted, so the gate stays well-posed without a
    special case.

    Args:
        l2: Ridge penalty on the weight vector (intercept unpenalized).
        min_samples_linear: Size pre-filter (as in
            :class:`EmbeddedLinearLeafModel`).
        leaf_gate_margin: A leaf keeps its linear fit only if
            ``E_lin < (1 - leaf_gate_margin) * E_const`` in weighted LOO. ``0``
            keeps any non-worse linear fit; larger values demand a larger
            held-out improvement. Conservative toward linear (small default) so
            regression gains are preserved. Must be ``>= 0``.
        leaf_gate: ``"loo"`` (default; leverage-corrected held-out error) or
            ``"insample"`` (drops the leverage correction — a deliberately weak
            baseline used to show the LOO correction earns its keep).
    """

    name = "adaptive"
    uses_embeddings = True

    def __init__(
        self,
        l2: float = 1.0,
        min_samples_linear: int = 10,
        leaf_gate_margin: float = 0.01,
        leaf_gate: str = "loo",
        leaf_fit_precision: str = "float64",
    ) -> None:
        super().__init__(l2=l2, min_samples_linear=min_samples_linear,
                         leaf_fit_precision=leaf_fit_precision)
        if leaf_gate_margin < 0:
            raise ValueError(f"leaf_gate_margin must be >= 0, got {leaf_gate_margin}")
        if leaf_gate not in ("loo", "insample"):
            raise ValueError(
                f"leaf_gate must be 'loo' or 'insample', got {leaf_gate!r}"
            )
        self.leaf_gate_margin = leaf_gate_margin
        self.leaf_gate = leaf_gate


def _newton_constant(g: np.ndarray, h: np.ndarray, l2: float) -> float:
    """Optimal constant leaf value: -sum(g) / (sum(h) + l2)."""
    return float(-g.sum() / (h.sum() + l2))


def _fit_weighted_ridge(
    Z: np.ndarray, g: np.ndarray, h: np.ndarray, l2: float
) -> tuple[float, np.ndarray | None]:
    """Solve the leaf's second-order objective for an affine model.

    Minimizes sum_i h_i * (b + w.z_i - t_i)^2 + l2 * ||w||^2 with t_i = -g_i/h_i,
    which equals the Newton objective up to a constant (docs/math.md).
    Centering Z and t by their h-weighted means decouples the (unpenalized)
    intercept from the weights.
    """
    h_sum = h.sum()
    t = -g / h
    z_mean = (h[:, None] * Z).sum(axis=0) / h_sum
    t_mean = float((h * t).sum() / h_sum)
    Zc = Z - z_mean
    tc = t - t_mean

    d = Z.shape[1]
    A = (Zc * h[:, None]).T @ Zc + l2 * np.eye(d)
    rhs = (Zc * h[:, None]).T @ tc
    try:
        w = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        return 0.0, None
    if not np.all(np.isfinite(w)):
        return 0.0, None
    b = t_mean - float(w @ z_mean)
    return b, w


#: Per-row LOO leverages are clamped to ``1 - _LOO_LEVERAGE_CAP`` so the
#: leave-one-out division ``r / (1 - H)`` stays finite for high-leverage rows: a
#: degenerate leverage then yields a large-but-finite error, making the leaf fail
#: the gate and fall back to constant — the safe direction.
_LOO_LEVERAGE_CAP = 1e-6


def _loo_leverages(
    z_tilde: np.ndarray, weights: np.ndarray, M: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted-ridge leave-one-out leverages for one leaf.

    ``weights`` are the per-row Hessian weights ``h_i``; ``M = A_c + l2 I`` is the
    centered matrix the ridge weights solve against; ``z_tilde`` are the
    h-weighted-centered embeddings. Returns ``(H, H0)`` clamped to
    ``<= 1 - _LOO_LEVERAGE_CAP``::

        H_i  = h_i (z_tilde_i^T M^-1 z_tilde_i) + h_i / sum_h   (linear-model hat)
        H0_i = h_i / sum_h                                      (constant-only hat)

    The ``h_i / sum_h`` term is the unpenalized intercept's leverage (centering is
    a rank-1 h-weighted-mean smoother).
    """
    h_sum = weights.sum()
    sol = np.linalg.solve(M, z_tilde.T)  # (d, n): M^-1 z_tilde_i for every row
    quad = np.einsum("ij,ji->i", z_tilde, sol)  # z_tilde_i^T M^-1 z_tilde_i
    cap = 1.0 - _LOO_LEVERAGE_CAP
    H = np.minimum(weights * quad + weights / h_sum, cap)
    H0 = np.minimum(weights / h_sum, cap)
    return H, H0


def _loo_keep_linear(
    Zl: np.ndarray,
    gl: np.ndarray,
    hl: np.ndarray,
    w: np.ndarray,
    z_mean: np.ndarray,
    t_mean: float,
    M: np.ndarray,
    margin: float,
    insample: bool,
) -> bool:
    """Whether a leaf's linear fit beats the constant fit in weighted LOO.

    Compares the linear model's weighted leave-one-out error against the
    constant-only (intercept) weighted LOO error and returns ``True`` (keep the
    linear fit) iff ``E_lin < (1 - margin) * E_const``. ``insample=True`` drops
    the leverage correction (a deliberately weak baseline gate). The comparison
    is strict, so an exact tie keeps the constant — the conservative default.
    """
    z_tilde = Zl - z_mean
    t = -gl / hl
    H, H0 = _loo_leverages(z_tilde, hl, M)
    resid = z_tilde @ w + (t_mean - t)  # fitted f_i - t_i
    if insample:
        e_lin = float(np.sum(hl * resid * resid))
    else:
        e_lin = float(np.sum(hl * (resid / (1.0 - H)) ** 2))
    # Constant baseline: the intercept-only weighted-LOO error. (A demoted leaf
    # actually ships the l2-regularized Newton bias -g_sum/(h_sum+l2); the
    # O(l2/h_sum) gap only makes the constant look marginally better, i.e. it
    # biases the comparison toward demotion — the safe direction.)
    resid0 = t_mean - t
    e_const = float(np.sum(hl * (resid0 / (1.0 - H0)) ** 2))
    return e_lin < (1.0 - margin) * e_const


@dataclass
class _LeafGate:
    """Per-leaf generalization gate context for :class:`AdaptiveLeafModel`.

    Carries the per-row data needed to evaluate the weighted-ridge LOO gate after
    the batched ridge solve. ``grad``/``hess`` are 1-D (scalar/native paths) or
    ``(n_rows, n_classes)`` with ``leaf_class`` set (pooled-multiclass path), in
    which case leaf ``i`` reads column ``leaf_class[i]``.
    """

    Z: np.ndarray
    grad: np.ndarray
    hess: np.ndarray
    order: np.ndarray
    offsets: np.ndarray
    leaf_class: np.ndarray | None
    margin: float
    insample: bool

    def keep_mask(
        self,
        linear: np.ndarray,
        ok: np.ndarray,
        M_batch: np.ndarray,
        w: np.ndarray,
        z_mean: np.ndarray,
        t_mean: np.ndarray,
    ) -> np.ndarray:
        """Refine the finite-solve mask ``ok`` with the per-leaf LOO verdict.

        ``M_batch[j]`` is the regularized matrix ``A_c + l2 I`` already solved for
        leaf ``linear[j]``; only leaves with ``ok[j]`` (a finite linear fit) are
        evaluated.
        """
        keep = ok.copy()
        for j, i in enumerate(linear):
            if not ok[j]:
                continue
            rows = self.order[self.offsets[i] : self.offsets[i + 1]]
            Zl = self.Z[rows]
            if self.leaf_class is None:
                gl = self.grad[rows]
                hl = self.hess[rows]
            else:
                c = self.leaf_class[i]
                gl = self.grad[rows, c]
                hl = self.hess[rows, c]
            keep[j] = _loo_keep_linear(
                Zl,
                gl,
                hl,
                w[j],
                z_mean[j],
                float(t_mean[j]),
                M_batch[j],
                self.margin,
                self.insample,
            )
        return keep


def make_leaf_model(
    name: str,
    l2: float,
    min_samples_linear: int,
    leaf_gate_margin: float = 0.01,
    leaf_gate: str = "loo",
    leaf_fit_precision: str = "float64",
) -> BaseLeafModel:
    """Factory for the ``leaf_model`` parameter.

    ``raw_linear`` reuses the embedded-linear machinery; the model wrapper is
    responsible for supplying standardized raw numerical features as Z.
    ``adaptive`` adds a per-leaf LOO gate on top of the embedded-linear fit
    (``leaf_gate_margin``/``leaf_gate`` are ignored by the other models).
    ``leaf_fit_precision`` only affects the wide-embedding (emb>128) BLAS leaf-fit
    of the linear models (and multi-output vector leaves); it is inert for ``constant``.
    """
    if leaf_fit_precision not in _LEAF_FIT_PRECISIONS:
        raise ValueError(
            f"leaf_fit_precision must be one of {_LEAF_FIT_PRECISIONS}, "
            f"got {leaf_fit_precision!r}"
        )
    if name == "constant":
        return ConstantLeafModel(l2=l2)
    if name in ("embedded_linear", "raw_linear"):
        return EmbeddedLinearLeafModel(
            l2=l2, min_samples_linear=min_samples_linear,
            leaf_fit_precision=leaf_fit_precision,
        )
    if name == "adaptive":
        return AdaptiveLeafModel(
            l2=l2,
            min_samples_linear=min_samples_linear,
            leaf_gate_margin=leaf_gate_margin,
            leaf_gate=leaf_gate,
            leaf_fit_precision=leaf_fit_precision,
        )
    raise ValueError(
        f"Unknown leaf_model {name!r}. Available: 'constant', 'embedded_linear', "
        "'raw_linear', 'adaptive'"
    )
