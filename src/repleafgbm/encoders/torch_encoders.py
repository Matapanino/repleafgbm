"""Learned encoders (optional PyTorch extra) — Phase 13.

Motivated by two experiments: frozen random frequencies do not deliver
PBLD's benefit (experiments/results/encoder_variants.md), and binary
embedded-leaf gains route through better representations rather than
reweighting (experiments/results/binary_leaf_gain.md).

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
              seed: int) -> None:
    """Shared loop: regress a linear head on the embedding onto the target.

    Trains ``params`` (encoder parameters) jointly with a throwaway linear
    head by minimizing MSE; deterministic given ``seed``. CPU only (v0).
    """
    gen = torch.Generator().manual_seed(seed)
    X_t = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    t_t = torch.from_numpy(np.ascontiguousarray(target, dtype=np.float32))
    t_t = (t_t - t_t.mean()) / (t_t.std() + 1e-12)

    head = torch.nn.Linear(out_dim, 1)
    with torch.no_grad():
        torch.nn.init.normal_(head.weight, std=0.01, generator=gen)
        torch.nn.init.zeros_(head.bias)
    opt = torch.optim.Adam(list(params) + list(head.parameters()), lr=lr)

    n = X_t.shape[0]
    for _ in range(n_epochs):
        perm = torch.randperm(n, generator=gen)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            pred = head(embed_fn(X_t[idx])).squeeze(-1)
            loss = torch.mean((pred - t_t[idx]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()


class TorchPeriodicEncoder(PeriodicEncoder):
    """PBLD-style periodic encoder with *learned* frequencies and phases.

    Identical output structure to :class:`PeriodicEncoder` (sinusoidal
    components plus a linear term per feature), but ``fit`` initializes
    frequencies/phases exactly like the frozen version and then trains them
    on the supervised pretraining target before freezing. Falls back to the
    frozen initialization when no target is provided.

    Extra args over PeriodicEncoder: n_epochs, lr, batch_size.
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
        random_state: int | None = 0,
    ) -> None:
        super().__init__(n_frequencies=n_frequencies, frequency_scale=frequency_scale,
                         add_linear=add_linear, random_state=random_state)
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size

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

        _pretrain(torch, [freq, phase], embed, x_std, y,
                  out_dim=n_features * (self.n_frequencies + int(self.add_linear)),
                  n_epochs=self.n_epochs, lr=self.lr,
                  batch_size=self.batch_size, seed=self.random_state or 0)

        self.frequencies_ = freq.detach().numpy().astype(np.float64)
        self.phases_ = phase.detach().numpy().astype(np.float64)
        return self

    def get_config(self) -> dict:
        return {
            **super().get_config(),
            "n_epochs": self.n_epochs,
            "lr": self.lr,
            "batch_size": self.batch_size,
        }


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
        random_state: int | None = 0,
    ) -> None:
        super().__init__(n_bins=n_bins, add_linear=True)
        if n_outputs < 1:
            raise ValueError(f"n_outputs must be >= 1, got {n_outputs}")
        self.n_outputs = n_outputs
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = random_state
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

        _pretrain(torch, [weight, bias_p], embed, basis, y,
                  out_dim=n_features * self.n_outputs,
                  n_epochs=self.n_epochs, lr=self.lr,
                  batch_size=self.batch_size, seed=self.random_state or 0)
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
