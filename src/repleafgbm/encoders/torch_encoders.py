"""Learned encoders (optional PyTorch extra) — Phases 13 and 16.

Motivated by two experiments: frozen random frequencies do not deliver
PBLD's benefit (experiments/results/encoder_variants.md), and binary
embedded-leaf gains route through better representations rather than
reweighting (experiments/results/binary_leaf_gain.md). Phase 16 adds the
interaction-aware ``torch_mlp`` after Phases 14/14b showed the *per-feature*
learned encoders find nothing on real data that the router doesn't.

Design contract:

* PyTorch is needed **only during fit**: training learns parameters, which
  are then frozen into the NumPy arrays of the frozen base classes. The
  ``transform`` path, ``get_state``/``set_state``, and serialization are the
  inherited NumPy implementations — saved models load and predict without
  torch installed.
* Pretraining is supervised: the model wrapper passes the Newton residual at
  the initial score as ``y`` (fit -> freeze before boosting, so the v0
  frozen-encoder rule and the stage-wise analysis in docs/math.md hold
  unchanged).
* The native path never imports this module's heavy dependency: torch is
  imported lazily inside ``fit`` with an actionable error message.
"""

from __future__ import annotations

import numpy as np

from repleafgbm.encoders.base import BaseEncoder
from repleafgbm.encoders.periodic import PeriodicEncoder
from repleafgbm.encoders.plr import SimplePLREncoder


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "This encoder needs PyTorch for its pretraining step. Install it "
            'with: pip install torch  (or pip install "repleafgbm[torch]"). '
            "Already-fitted/saved models predict without torch."
        ) from exc
    return torch


def _pretrain(torch, params: list, embed_fn, X: np.ndarray, target: np.ndarray,
              out_dim: int, n_epochs: int, lr: float, batch_size: int,
              seed: int, weight_decay: float = 0.0, val_fraction: float = 0.0,
              patience: int = 0) -> int:
    """Shared loop: regress a linear head on the embedding onto the target.

    Trains ``params`` (encoder parameters) jointly with a throwaway linear
    head by minimizing MSE (AdamW); deterministic given ``seed``. CPU only.

    Phase 14b regularization: a ``val_fraction`` split of the pretraining
    data is held out, training stops once its loss has not improved for
    ``patience`` epochs, and the best-epoch encoder parameters are restored
    — the guard against the real-data overfitting documented in
    experiments/results/real_data_validation.md (Phase 14). Returns the
    number of epochs actually run.
    """
    gen = torch.Generator().manual_seed(seed)
    X_t = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    t_t = torch.from_numpy(np.ascontiguousarray(target, dtype=np.float32))
    t_t = (t_t - t_t.mean()) / (t_t.std() + 1e-12)

    n = X_t.shape[0]
    perm0 = torch.randperm(n, generator=gen)
    n_val = int(n * val_fraction) if patience > 0 else 0
    use_es = n_val >= 16
    val_idx = perm0[:n_val] if use_es else None
    train_idx = perm0[n_val:] if use_es else perm0

    head = torch.nn.Linear(out_dim, 1)
    with torch.no_grad():
        torch.nn.init.normal_(head.weight, std=0.01, generator=gen)
        torch.nn.init.zeros_(head.bias)
    opt = torch.optim.AdamW(list(params) + list(head.parameters()), lr=lr,
                            weight_decay=weight_decay)

    best_val = float("inf")
    best_state: list | None = None
    rounds_since_best = 0
    epochs_used = 0
    n_train = train_idx.shape[0]
    for _ in range(n_epochs):
        order = train_idx[torch.randperm(n_train, generator=gen)]
        for start in range(0, n_train, batch_size):
            idx = order[start:start + batch_size]
            pred = head(embed_fn(X_t[idx])).squeeze(-1)
            loss = torch.mean((pred - t_t[idx]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
        epochs_used += 1
        if use_es:
            with torch.no_grad():
                val_pred = head(embed_fn(X_t[val_idx])).squeeze(-1)
                val_loss = float(torch.mean((val_pred - t_t[val_idx]) ** 2))
            if val_loss < best_val - 1e-7:
                best_val = val_loss
                rounds_since_best = 0
                best_state = [p.detach().clone() for p in params]
            else:
                rounds_since_best += 1
                if rounds_since_best >= patience:
                    break
    if use_es and best_state is not None:
        with torch.no_grad():
            for p, b in zip(params, best_state):
                p.copy_(b)
    return epochs_used


class TorchPeriodicEncoder(PeriodicEncoder):
    """PBLD-style periodic encoder with *learned* frequencies and phases.

    Identical output structure to :class:`PeriodicEncoder` (sinusoidal
    components plus a linear term per feature), but ``fit`` initializes
    frequencies/phases exactly like the frozen version and then trains them
    on the supervised pretraining target before freezing. Falls back to the
    frozen initialization when no target is provided.

    Extra args over PeriodicEncoder: n_epochs, lr, batch_size, plus the
    Phase 14b pretraining-regularization knobs weight_decay (AdamW),
    val_fraction, and patience (validation early stopping with best-epoch
    restore; conservative defaults on).
    """

    name = "torch_periodic"

    def __init__(
        self,
        n_frequencies: int = 4,
        frequency_scale: float = 1.0,
        add_linear: bool = True,
        n_epochs: int = 30,
        lr: float = 0.01,
        batch_size: int = 256,
        weight_decay: float = 1e-3,
        val_fraction: float = 0.15,
        patience: int = 5,
        random_state: int | None = 0,
    ) -> None:
        super().__init__(n_frequencies=n_frequencies, frequency_scale=frequency_scale,
                         add_linear=add_linear, random_state=random_state)
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.val_fraction = val_fraction
        self.patience = patience
        #: Epochs actually run by the last fit (early stopping diagnostic).
        self.pretrain_epochs_used_: int | None = None

    def fit(self, X_num: np.ndarray, y: np.ndarray | None = None) -> TorchPeriodicEncoder:
        super().fit(X_num)  # frozen-random initialization + standardization
        if y is None:
            return self
        torch = _require_torch()
        torch.manual_seed(self.random_state or 0)

        x_std = (np.where(np.isnan(X_num), self.mean_, X_num) - self.mean_) / self.scale_
        freq = torch.nn.Parameter(torch.from_numpy(self.frequencies_.astype(np.float32)))
        phase = torch.nn.Parameter(torch.from_numpy(self.phases_.astype(np.float32)))
        n_features = self.frequencies_.shape[0]

        def embed(xb):  # (b, F) -> (b, F * (k + add_linear)), matching transform
            sines = torch.sin(2.0 * np.pi * (xb.unsqueeze(-1) * freq + phase))
            parts = [sines]
            if self.add_linear:
                parts.append(xb.unsqueeze(-1))
            return torch.cat(parts, dim=-1).reshape(xb.shape[0], -1)

        self.pretrain_epochs_used_ = _pretrain(
            torch, [freq, phase], embed, x_std, y,
            out_dim=n_features * (self.n_frequencies + int(self.add_linear)),
            n_epochs=self.n_epochs, lr=self.lr, batch_size=self.batch_size,
            seed=self.random_state or 0, weight_decay=self.weight_decay,
            val_fraction=self.val_fraction, patience=self.patience)

        self.frequencies_ = freq.detach().numpy().astype(np.float64)
        self.phases_ = phase.detach().numpy().astype(np.float64)
        return self

    def get_config(self) -> dict:
        return {
            **super().get_config(),
            "n_epochs": self.n_epochs,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "weight_decay": self.weight_decay,
            "val_fraction": self.val_fraction,
            "patience": self.patience,
        }


class TorchMLPEncoder(BaseEncoder):
    """Interaction-aware learned encoder: a small MLP over all numericals.

    Phase 16. Unlike ``torch_periodic`` / ``torch_plr`` (independent
    per-feature transforms), the MLP mixes features, so the representation
    can carry cross-feature structure into the linear leaves — the one
    encoder hypothesis Phases 14/14b left open. Architecture:

        z = [x_std, relu(W2 @ relu(W1 @ x_std + b1) + b2)]

    with the standardized features appended (``add_linear``) so a leaf is
    never worse-equipped than with ``identity``. Pretrained on the supervised
    target with the Phase 14b regularization (AdamW weight decay, validation
    early stopping with best-epoch restore), then frozen to NumPy arrays:
    transform and serialization never need torch. Without a target the
    seeded random initialization is frozen as-is.
    """

    name = "torch_mlp"

    def __init__(
        self,
        hidden_dim: int = 64,
        out_dim: int = 16,
        add_linear: bool = True,
        n_epochs: int = 30,
        lr: float = 0.01,
        batch_size: int = 256,
        weight_decay: float = 1e-3,
        val_fraction: float = 0.15,
        patience: int = 5,
        random_state: int | None = 0,
    ) -> None:
        if hidden_dim < 1 or out_dim < 1:
            raise ValueError(
                f"hidden_dim and out_dim must be >= 1, got {hidden_dim}, {out_dim}"
            )
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.add_linear = add_linear
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.val_fraction = val_fraction
        self.patience = patience
        self.random_state = random_state
        #: Epochs actually run by the last fit (early stopping diagnostic).
        self.pretrain_epochs_used_: int | None = None
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.w1_: np.ndarray | None = None  # (F, hidden_dim)
        self.b1_: np.ndarray | None = None  # (hidden_dim,)
        self.w2_: np.ndarray | None = None  # (hidden_dim, out_dim)
        self.b2_: np.ndarray | None = None  # (out_dim,)

    def fit(self, X_num: np.ndarray, y: np.ndarray | None = None) -> TorchMLPEncoder:
        from repleafgbm.utils.random import check_random_state

        X_num = np.asarray(X_num, dtype=np.float64)
        self.mean_ = np.nanmean(X_num, axis=0)
        std = np.nanstd(X_num, axis=0)
        self.scale_ = np.where(std > 0, std, 1.0)

        n_features = X_num.shape[1]
        rng = check_random_state(self.random_state)
        self.w1_ = rng.normal(0.0, 1.0 / np.sqrt(n_features), (n_features, self.hidden_dim))
        self.b1_ = np.zeros(self.hidden_dim)
        self.w2_ = rng.normal(0.0, 1.0 / np.sqrt(self.hidden_dim), (self.hidden_dim, self.out_dim))
        self.b2_ = np.zeros(self.out_dim)
        if y is None:
            return self

        torch = _require_torch()
        torch.manual_seed(self.random_state or 0)
        x_std = self._standardize(X_num)
        params = [
            torch.nn.Parameter(torch.from_numpy(a.astype(np.float32)))
            for a in (self.w1_, self.b1_, self.w2_, self.b2_)
        ]
        w1, b1, w2, b2 = params

        def embed(xb):  # (b, F) -> (b, F * add_linear + out_dim), = transform
            h = torch.relu(xb @ w1 + b1)
            z = torch.relu(h @ w2 + b2)
            return torch.cat([xb, z], dim=1) if self.add_linear else z

        self.pretrain_epochs_used_ = _pretrain(
            torch, params, embed, x_std, y,
            out_dim=n_features * int(self.add_linear) + self.out_dim,
            n_epochs=self.n_epochs, lr=self.lr, batch_size=self.batch_size,
            seed=self.random_state or 0, weight_decay=self.weight_decay,
            val_fraction=self.val_fraction, patience=self.patience)
        self.w1_, self.b1_, self.w2_, self.b2_ = (
            p.detach().numpy().astype(np.float64) for p in params
        )
        return self

    def _standardize(self, X_num: np.ndarray) -> np.ndarray:
        Z = np.where(np.isnan(X_num), self.mean_, X_num)
        return (Z - self.mean_) / self.scale_

    def transform(self, X_num: np.ndarray) -> np.ndarray:
        self._check_fitted("w1_")
        X_num = np.asarray(X_num, dtype=np.float64)
        x_std = self._standardize(X_num)
        h = np.maximum(x_std @ self.w1_ + self.b1_, 0.0)
        z = np.maximum(h @ self.w2_ + self.b2_, 0.0)
        return np.concatenate([x_std, z], axis=1) if self.add_linear else z

    @property
    def output_dim(self) -> int:
        self._check_fitted("w1_")
        return int(self.w1_.shape[0] * int(self.add_linear) + self.out_dim)

    def get_config(self) -> dict:
        return {
            "hidden_dim": self.hidden_dim,
            "out_dim": self.out_dim,
            "add_linear": self.add_linear,
            "n_epochs": self.n_epochs,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "weight_decay": self.weight_decay,
            "val_fraction": self.val_fraction,
            "patience": self.patience,
            "random_state": self.random_state,
        }

    def get_state(self) -> dict[str, np.ndarray]:
        self._check_fitted("w1_")
        return {
            "mean": self.mean_,
            "scale": self.scale_,
            "w1": self.w1_,
            "b1": self.b1_,
            "w2": self.w2_,
            "b2": self.b2_,
        }

    def set_state(self, state: dict[str, np.ndarray]) -> None:
        self.mean_ = np.asarray(state["mean"], dtype=np.float64)
        self.scale_ = np.asarray(state["scale"], dtype=np.float64)
        self.w1_ = np.asarray(state["w1"], dtype=np.float64)
        self.b1_ = np.asarray(state["b1"], dtype=np.float64)
        self.w2_ = np.asarray(state["w2"], dtype=np.float64)
        self.b2_ = np.asarray(state["b2"], dtype=np.float64)


class TorchPLREncoder(SimplePLREncoder):
    """PLR with a learned per-feature projection (Gorishniy et al. 2022).

    The quantile piecewise-linear basis (plus linear slot) of
    :class:`SimplePLREncoder` is followed by a learned per-feature linear
    map to ``n_outputs`` dims and a ReLU — the "LR" part of PLR, trained on
    the supervised pretraining target and then frozen. Output dimension:
    ``n_features * n_outputs``. Without a target the projection stays at its
    random initialization.
    """

    name = "torch_plr"

    def __init__(
        self,
        n_bins: int = 4,
        n_outputs: int = 4,
        n_epochs: int = 30,
        lr: float = 0.01,
        batch_size: int = 256,
        weight_decay: float = 1e-3,
        val_fraction: float = 0.15,
        patience: int = 5,
        random_state: int | None = 0,
    ) -> None:
        super().__init__(n_bins=n_bins, add_linear=True)
        if n_outputs < 1:
            raise ValueError(f"n_outputs must be >= 1, got {n_outputs}")
        self.n_outputs = n_outputs
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.val_fraction = val_fraction
        self.patience = patience
        self.random_state = random_state
        #: Epochs actually run by the last fit (early stopping diagnostic).
        self.pretrain_epochs_used_: int | None = None
        self.proj_weight_: np.ndarray | None = None  # (F, n_bins+1, n_outputs)
        self.proj_bias_: np.ndarray | None = None  # (F, n_outputs)

    def _basis(self, X_num: np.ndarray) -> np.ndarray:
        """(n, F, n_bins + 1) piecewise-linear basis with linear slot."""
        flat = super().transform(X_num)
        n_features = self.edges_.shape[0]
        return flat.reshape(X_num.shape[0], n_features, self.n_bins + 1)

    def fit(self, X_num: np.ndarray, y: np.ndarray | None = None) -> TorchPLREncoder:
        super().fit(X_num)  # quantile edges + standardization stats
        from repleafgbm.utils.random import check_random_state

        n_features = self.edges_.shape[0]
        d_in = self.n_bins + 1
        rng = check_random_state(self.random_state)
        self.proj_weight_ = rng.normal(
            0.0, 1.0 / np.sqrt(d_in), size=(n_features, d_in, self.n_outputs)
        )
        self.proj_bias_ = np.zeros((n_features, self.n_outputs))
        if y is None:
            return self

        torch = _require_torch()
        torch.manual_seed(self.random_state or 0)
        basis = self._basis(X_num).astype(np.float32)  # (n, F, d_in)
        weight = torch.nn.Parameter(torch.from_numpy(self.proj_weight_.astype(np.float32)))
        bias_p = torch.nn.Parameter(torch.from_numpy(self.proj_bias_.astype(np.float32)))

        def embed(bb):  # (b, F, d_in) -> (b, F * n_outputs)
            out = torch.relu(torch.einsum("bfi,fio->bfo", bb, weight) + bias_p)
            return out.reshape(bb.shape[0], -1)

        self.pretrain_epochs_used_ = _pretrain(
            torch, [weight, bias_p], embed, basis, y,
            out_dim=n_features * self.n_outputs,
            n_epochs=self.n_epochs, lr=self.lr, batch_size=self.batch_size,
            seed=self.random_state or 0, weight_decay=self.weight_decay,
            val_fraction=self.val_fraction, patience=self.patience)
        self.proj_weight_ = weight.detach().numpy().astype(np.float64)
        self.proj_bias_ = bias_p.detach().numpy().astype(np.float64)
        return self

    def transform(self, X_num: np.ndarray) -> np.ndarray:
        self._check_fitted("proj_weight_")
        basis = self._basis(X_num)  # NumPy only
        out = np.einsum("bfi,fio->bfo", basis, self.proj_weight_) + self.proj_bias_
        np.maximum(out, 0.0, out=out)
        return out.reshape(X_num.shape[0], -1)

    @property
    def output_dim(self) -> int:
        self._check_fitted("proj_weight_")
        return int(self.proj_weight_.shape[0] * self.n_outputs)

    def get_config(self) -> dict:
        return {
            "n_bins": self.n_bins,
            "n_outputs": self.n_outputs,
            "n_epochs": self.n_epochs,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "weight_decay": self.weight_decay,
            "val_fraction": self.val_fraction,
            "patience": self.patience,
            "random_state": self.random_state,
        }

    def get_state(self) -> dict[str, np.ndarray]:
        self._check_fitted("proj_weight_")
        return {
            **super().get_state(),
            "proj_weight": self.proj_weight_,
            "proj_bias": self.proj_bias_,
        }

    def set_state(self, state: dict[str, np.ndarray]) -> None:
        super().set_state(state)
        self.proj_weight_ = np.asarray(state["proj_weight"], dtype=np.float64)
        self.proj_bias_ = np.asarray(state["proj_bias"], dtype=np.float64)
