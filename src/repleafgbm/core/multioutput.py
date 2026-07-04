"""Multi-output regression: shared-routing trees with vector-valued leaves.

Kept separate from :mod:`repleafgbm.core.booster` so the scalar boosting loop
stays readable, mirroring how :mod:`repleafgbm.core.multiclass` lifts the loop
to a score *matrix*. The crucial difference from multiclass is the routing:
multiclass grows one tree per class per round, whereas multi-output regression
grows **one shared tree per round** whose leaves emit an ``(n_outputs,)``
vector. Routing (splits on raw features) is therefore shared across outputs;
the split gain is the per-output Newton gain summed over outputs
(``core.splitter`` / ``backends.numpy_backend.find_best_split_multioutput``).

The encoder stays frozen; every output reuses the same embedding matrix Z.
Vector leaves are fitted on constant-Hessian statistics (``h = 1`` for squared
error, Huber, and quantile alike), so the embedded-linear leaf's centered Gram
matrix is shared across outputs and the per-leaf system is one factorization
with ``n_outputs`` right-hand sides (docs/math.md). Only the gradient and the
init score change between objectives.
"""

from __future__ import annotations

import numpy as np

from repleafgbm.backends import make_split_backend
from repleafgbm.backends.base import BaseSplitBackend
from repleafgbm.core.booster import BoosterParams, weight_grad_hess
from repleafgbm.core.leaf_models import BaseLeafModel, LeafValues, _loo_leverages
from repleafgbm.core.metrics import BaseMetric
from repleafgbm.core.objectives import MultiOutputObjective
from repleafgbm.core.prediction import predict_raw_multioutput
from repleafgbm.core.profiling import PhaseProfiler, timed
from repleafgbm.core.splitter import Splitter
from repleafgbm.core.tree import Tree, TreeGrower
from repleafgbm.core.verbose import EvalLogger
from repleafgbm.data import RepLeafDataset
from repleafgbm.encoders.base import BaseEncoder


def fit_vector_leaves(
    leaf_model: BaseLeafModel,
    leaf_rows: list[np.ndarray],
    grad: np.ndarray,
    hess: np.ndarray,
    Z: np.ndarray | None,
    l2: float,
) -> LeafValues:
    """Fit vector-valued leaves for one shared-routing tree.

    ``grad``/``hess`` are ``(n_rows, n_outputs)``. The result's ``bias`` is
    ``(n_leaves, n_outputs)`` and, for embedded-linear leaves, ``weights`` is
    ``(n_leaves, emb_dim, n_outputs)``. Constant and embedded-linear leaves are
    the vector analogues of :mod:`repleafgbm.core.leaf_models`; the same
    overfitting guards apply (ridge ``l2``, ``min_samples_linear`` fallback to
    a constant vector, per-leaf extrapolation clip bounds shared across
    outputs).
    """
    n_leaves = len(leaf_rows)
    n_outputs = grad.shape[1]
    bias = np.empty((n_leaves, n_outputs), dtype=np.float64)
    for i, rows in enumerate(leaf_rows):
        g = grad[rows].sum(axis=0)
        h = hess[rows].sum(axis=0)
        bias[i] = -g / (h + l2)

    if not leaf_model.uses_embeddings:
        return LeafValues(bias=bias, weights=np.zeros((n_leaves, 0, n_outputs)))

    if Z is None:
        raise ValueError("embedded-linear vector leaves require an embedding matrix Z")
    emb_dim = Z.shape[1]
    weights = np.zeros((n_leaves, emb_dim, n_outputs), dtype=np.float64)
    z_min = np.full((n_leaves, emb_dim), -np.inf, dtype=np.float64)
    z_max = np.full((n_leaves, emb_dim), np.inf, dtype=np.float64)
    min_samples_linear = getattr(leaf_model, "min_samples_linear", 10)
    min_n = max(min_samples_linear, emb_dim + 2)
    # Per-leaf LOO gate (AdaptiveLeafModel): one verdict per leaf, summed over
    # outputs (the leverage is shared because the Gram/row-weights are shared).
    # ``None`` for constant/embedded_linear leaves leaves the gate off.
    gate_margin = getattr(leaf_model, "leaf_gate_margin", None)
    gate_insample = getattr(leaf_model, "leaf_gate", "loo") == "insample"
    # Device leaf-fit statistics (split_backend="cuda"): the boosting loop
    # hands its backend to the leaf model (transient fit_backend, as in the
    # scalar path). The device computes the per-leaf weighted Gram / cross /
    # weight sums; centering, the K-column ridge solve, and the LOO gate stay
    # host float64 (allclose parity, ADR 0005).
    backend = getattr(leaf_model, "fit_backend", None)
    use_device = (
        backend is not None
        and getattr(backend, "supports_leaf_fit", False)
        and hasattr(backend, "leaf_fit_stats_vector")
    )
    # Opt-in float32 leaf-fit (mirrors the scalar EmbeddedLinearLeafModel.fit_leaves
    # branch): accumulate ONLY the two large per-leaf reductions — the weighted Gram
    # and the target projection — in float32 (~1.3x on those reductions, ~5.5% whole
    # wide-emb multi-output fit), while the centering,
    # the float64 solve, and the LOO-gate leverage stay float64. The default float64
    # path below is byte-identical; the float32 path is allclose (~1e-5), NOT bitwise
    # (near-tied LOO-gate decisions can flip) — quality-equivalent, opt-in only.
    use_f32 = getattr(leaf_model, "leaf_fit_precision", "float64") == "float32_gram"

    if use_device:
        sizes = np.array([r.shape[0] for r in leaf_rows], dtype=np.int64)
        order = (np.concatenate(leaf_rows) if leaf_rows
                 else np.empty(0, np.int64)).astype(np.int64)
        if order.shape[0] * emb_dim >= backend.leaf_fit_min_cells:
            offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
            linear = np.flatnonzero(sizes >= min_n).astype(np.int64)
            h_sum_l, s_wz, M, C, t_wsum, zmn, zmx = backend.leaf_fit_stats_vector(
                Z, grad, hess, order, offsets, linear, use_f32=use_f32
            )
            eye = l2 * np.eye(emb_dim)
            for j, i in enumerate(linear):
                z_mean = s_wz[j] / h_sum_l[j]
                A = M[j] - np.outer(z_mean, s_wz[j]) + eye
                rhs = C[j] - np.outer(z_mean, t_wsum[j])
                t_mean = t_wsum[j] / h_sum_l[j]
                try:
                    W = np.linalg.solve(A, rhs)
                except np.linalg.LinAlgError:
                    continue
                if not np.all(np.isfinite(W)):
                    continue
                if gate_margin is not None:
                    # The LOO gate needs the centered rows; recompute on host
                    # (adaptive-only, same verdict math as the host loop).
                    rows = leaf_rows[i]
                    Zl = Z[rows]
                    Zc = Zl - z_mean
                    w = hess[rows][:, 0]
                    t = -grad[rows] / hess[rows]
                    H, H0 = _loo_leverages(Zc, w, A)
                    resid = Zc @ W + (t_mean[None, :] - t)
                    if gate_insample:
                        e_lin = float(np.sum(w[:, None] * resid * resid))
                    else:
                        e_lin = float(np.sum(
                            w[:, None] * (resid / (1.0 - H)[:, None]) ** 2
                        ))
                    resid0 = t_mean[None, :] - t
                    e_const = float(np.sum(
                        w[:, None] * (resid0 / (1.0 - H0)[:, None]) ** 2
                    ))
                    if e_lin >= (1.0 - gate_margin) * e_const:
                        continue
                weights[i] = W
                bias[i] = t_mean - W.T @ z_mean
                z_min[i] = zmn[j]
                z_max[i] = zmx[j]
            return LeafValues(bias=bias, weights=weights, z_min=z_min, z_max=z_max)

    for i, rows in enumerate(leaf_rows):
        if rows.shape[0] < min_n:
            continue  # constant fallback (bias already set, guard disabled)
        Zl = Z[rows]
        g = grad[rows]  # (n_l, K)
        h = hess[rows]  # (n_l, K)
        # Squared-error multi-output: the Hessian is identical across outputs
        # (sample weights, when used, scale every column equally), so one
        # weighted Gram (from output 0's weights) serves all outputs.
        w = h[:, 0]
        h_sum = w.sum()
        z_mean = (w[:, None] * Zl).sum(axis=0) / h_sum
        Zc = Zl - z_mean
        t = -g / h  # Newton targets per output, (n_l, K)
        t_mean = (w[:, None] * t).sum(axis=0) / h_sum  # (K,)
        tc = t - t_mean
        wZc = Zc * w[:, None]
        if use_f32:
            wZc32, Zc32, tc32 = (wZc.astype(np.float32),
                                 Zc.astype(np.float32), tc.astype(np.float32))
            A = (wZc32.T @ Zc32).astype(np.float64) + l2 * np.eye(emb_dim)
            rhs = (wZc32.T @ tc32).astype(np.float64)  # (emb, K)
        else:
            A = wZc.T @ Zc + l2 * np.eye(emb_dim)
            rhs = wZc.T @ tc  # (emb, K)
        try:
            W = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            continue
        if not np.all(np.isfinite(W)):
            continue
        if gate_margin is not None:
            # Weighted-LOO gate, summed over outputs with the shared leverage.
            H, H0 = _loo_leverages(Zc, w, A)
            resid = Zc @ W + (t_mean[None, :] - t)  # (n_l, K): fitted - target
            if gate_insample:
                e_lin = float(np.sum(w[:, None] * resid * resid))
            else:
                e_lin = float(np.sum(w[:, None] * (resid / (1.0 - H)[:, None]) ** 2))
            resid0 = t_mean[None, :] - t
            e_const = float(np.sum(w[:, None] * (resid0 / (1.0 - H0)[:, None]) ** 2))
            if e_lin >= (1.0 - gate_margin) * e_const:
                continue  # gate keeps the constant fallback for this leaf
        weights[i] = W
        bias[i] = t_mean - W.T @ z_mean
        z_min[i] = Zl.min(axis=0)
        z_max[i] = Zl.max(axis=0)

    return LeafValues(bias=bias, weights=weights, z_min=z_min, z_max=z_max)


class MultiOutputBooster:
    """Trains and stores the shared-routing vector-leaf ensemble.

    The target ``y`` is an ``(n_rows, n_outputs)`` matrix. Construction,
    fitting, and prediction mirror :class:`~repleafgbm.core.booster.Booster`
    lifted to a score matrix, but with a single tree per round.
    """

    def __init__(self, params: BoosterParams, objective: MultiOutputObjective) -> None:
        self.params = params
        self.objective = objective
        #: Per-output init scores (column means), shape (n_outputs,).
        self.init_score_: np.ndarray = np.zeros(objective.n_outputs)
        self.trees_: list[Tree] = []
        self.leaf_values_: list[LeafValues] = []
        self.evals_result_: dict[str, dict[str, list[float]]] = {}
        #: Best number of trees found by early stopping (None when unused).
        self.best_iteration_: int | None = None
        self.best_score_: float | None = None
        #: Split backend from the last ``fit`` (runtime-only introspection
        #: handle, never serialized); see :class:`Booster.split_backend_`.
        self.split_backend_: BaseSplitBackend | None = None
        #: Per-output target standardization (identity by default); robust
        #: objectives (huber/quantile) are fit in standardized space and
        #: ``predict_raw``/eval un-standardize. Shapes broadcast over (n, K).
        #: See docs/proposals/robust-target-standardization.md.
        self.target_loc_: float | np.ndarray = 0.0
        self.target_scale_: float | np.ndarray = 1.0

    def __getstate__(self) -> dict:
        # Drop the runtime split-backend handle so the model stays picklable;
        # see :meth:`Booster.__getstate__`.
        return {**self.__dict__, "split_backend_": None}

    @property
    def n_outputs(self) -> int:
        return self.objective.n_outputs

    @property
    def n_trees(self) -> int:
        return len(self.trees_)

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def fit(
        self,
        dataset: RepLeafDataset,
        encoder: BaseEncoder | None,
        leaf_model: BaseLeafModel,
        eval_sets: list[tuple[str, RepLeafDataset]] | None = None,
        eval_metric: BaseMetric | None = None,
        profiler: PhaseProfiler | None = None,
    ) -> MultiOutputBooster:
        """Grow ``n_estimators`` shared-routing vector-leaf trees."""
        if dataset.y is None:
            raise ValueError("Training dataset must contain a target (y)")
        y = np.asarray(dataset.y, dtype=np.float64)
        if y.ndim != 2:
            raise ValueError(
                f"MultiOutputBooster expects a 2-D target (n_rows, n_outputs), "
                f"got shape {y.shape}"
            )
        w = dataset.sample_weight
        if leaf_model.uses_embeddings:
            with timed(profiler, "encoder"):
                Z = dataset.get_embeddings(encoder)
        else:
            Z = None

        p = self.params
        if p.early_stopping_rounds is not None and not eval_sets:
            raise ValueError(
                "early_stopping_rounds requires at least one eval_set to monitor"
            )
        splitter = Splitter(
            dataset.get_raw_features(),
            max_bins=p.max_bins,
            min_samples_leaf=p.min_samples_leaf,
            l2=p.l2_leaf,
            backend=make_split_backend(p.split_backend),
            categorical_indices=dataset.metadata.categorical_indices,
            cat_smooth=p.cat_smooth,
            min_data_per_group=p.min_data_per_group,
            max_cat_threshold=p.max_cat_threshold,
            profiler=profiler,
        )
        self.split_backend_ = splitter.backend
        # Transient device leaf-fit handle, as in Booster.fit (the leaf model
        # is a fit-local; reset before returning).
        leaf_model.fit_backend = splitter.backend
        try:
            grower = TreeGrower(
                splitter,
                num_leaves=p.num_leaves,
                max_depth=p.max_depth,
                grow_policy=p.grow_policy,
            )

            self.init_score_ = self.objective.init_score(y, weight=w)
            F = np.tile(self.init_score_, (y.shape[0], 1))

            evals: list[tuple[str, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]] = []
            if eval_sets:
                for name, ds in eval_sets:
                    if ds.y is None:
                        raise ValueError(f"eval_set {name!r} must contain a target (y)")
                    Ze = ds.get_embeddings(encoder) if leaf_model.uses_embeddings else None
                    Fe = np.tile(self.init_score_, (ds.n_rows, 1))
                    ye = np.asarray(ds.y, dtype=np.float64)
                    evals.append((name, ds.get_raw_features(), ye, Ze, Fe))
                self.evals_result_ = {name: {eval_metric.name: []} for name, *_ in evals}

            leaf_idx = np.empty(y.shape[0], dtype=np.int64)
            best_score: float | None = None
            rounds_since_best = 0
            logger = EvalLogger(p.verbose)
            for it in range(p.n_estimators):
                grad, hess = self.objective.grad_hess(y, F)
                grad, hess = weight_grad_hess(grad, hess, w)
                tree, leaf_rows = grower.grow(grad, hess)
                with timed(profiler, "leaf_fit"):
                    leaf_values = fit_vector_leaves(
                        leaf_model, leaf_rows, grad, hess, Z, p.l2_leaf
                    )
                self.trees_.append(tree)
                self.leaf_values_.append(leaf_values)

                with timed(profiler, "eval"):
                    for i, rows in enumerate(leaf_rows):
                        leaf_idx[rows] = i
                    # clip=False is exact on training rows (see Booster).
                    F += p.learning_rate * leaf_values.predict(leaf_idx, Z, clip=False)

                if evals and eval_metric is not None:
                    with timed(profiler, "eval"):
                        for name, Xe, ye, Ze, Fe in evals:
                            Fe += p.learning_rate * leaf_values.predict(tree.apply(Xe), Ze)
                            # eval_set y is raw-scale; un-standardize (identity default).
                            pred = self.objective.transform(
                            self.target_loc_ + self.target_scale_ * Fe
                        )
                            self.evals_result_[name][eval_metric.name].append(
                                eval_metric(ye, pred)
                            )
                    logger.log_round(it + 1, self.evals_result_)
                    if p.early_stopping_rounds is not None:
                        score = self.evals_result_[evals[0][0]][eval_metric.name][-1]
                        improved = best_score is None or (
                            score < best_score if eval_metric.minimize else score > best_score
                        )
                        if improved:
                            best_score = score
                            self.best_iteration_ = self.n_trees
                            self.best_score_ = score
                            rounds_since_best = 0
                        else:
                            rounds_since_best += 1
                            if rounds_since_best >= p.early_stopping_rounds:
                                logger.log_early_stop(
                                    self.best_iteration_, self.evals_result_
                                )
                                break
            return self
        finally:
            leaf_model.fit_backend = None

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #
    def predict_raw(
        self, X_raw: np.ndarray, Z: np.ndarray | None, n_trees: int | None = None
    ) -> np.ndarray:
        """Raw score matrix (n_rows, n_outputs); best iteration by default."""
        if n_trees is None:
            n_trees = self.best_iteration_  # None -> all trees
        raw = predict_raw_multioutput(
            self.trees_,
            self.leaf_values_,
            self.init_score_,
            self.params.learning_rate,
            X_raw,
            Z,
            n_trees=n_trees,
        )
        # Identity by default; per-output un-standardization for robust losses.
        return self.target_loc_ + self.target_scale_ * raw

    def feature_importance(
        self, n_features: int, importance_type: str = "gain"
    ) -> np.ndarray:
        """Per-feature importance over the predicting trees."""
        if importance_type not in ("gain", "split"):
            raise ValueError(
                f"importance_type must be 'gain' or 'split', got {importance_type!r}"
            )
        importance = np.zeros(n_features, dtype=np.float64)
        n_trees = self.best_iteration_ or self.n_trees
        for tree in self.trees_[:n_trees]:
            internal = tree.feature >= 0
            weights = tree.gain[internal] if importance_type == "gain" else 1.0
            np.add.at(importance, tree.feature[internal], weights)
        return importance
