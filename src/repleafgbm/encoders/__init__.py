"""Encoders: numerical feature -> representation z_theta(x).

Two families implement :class:`~repleafgbm.encoders.base.BaseEncoder`:

* **Fixed** (pure NumPy): ``identity``, ``plr``, ``periodic``, ``cross``,
  ``target_encoding`` — fitted once and frozen.
* **Pretrained-then-frozen** (optional ``[torch]`` extra): ``torch_periodic``,
  ``torch_plr``, ``torch_periodic_plr``, ``torch_mlp`` — parameters are
  supervised-pretrained on the initial Newton residual and then frozen into
  NumPy arrays. They are **not** joint-trainable during boosting; torch is
  needed only at fit time, while ``transform`` and serialization stay NumPy.

The encoder is frozen for all of boosting (see docs/math.md); joint/alternating
encoder training is a roadmap item, not implemented here. New encoders register
in ``_ENCODER_REGISTRY`` below. The experimental space bank of 1.13.0 (``pca``,
``quantile``, ``spline``, ``rff``, ``row_stats``, ``prototype``) was removed in
1.14.0 (docs/adr/0013-remove-space-encoders.md); naming one raises a removal
error instead of the generic unknown-encoder error.
"""

from __future__ import annotations

import inspect

from repleafgbm.encoders.base import BaseEncoder
from repleafgbm.encoders.bundle import ExpertBundleEncoder, LeafExpert
from repleafgbm.encoders.cross import CrossInteractionEncoder
from repleafgbm.encoders.identity import IdentityEncoder
from repleafgbm.encoders.periodic import PeriodicEncoder
from repleafgbm.encoders.plr import SimplePLREncoder
from repleafgbm.encoders.projection import RandomProjectionEncoder
from repleafgbm.encoders.subset import ColumnSubsetEncoder
from repleafgbm.encoders.target_encoding import TargetEncodingEncoder
from repleafgbm.encoders.target_reduction import TargetCorrelationEncoder
from repleafgbm.encoders.torch_encoders import (
    TorchMLPEncoder,
    TorchPeriodicEncoder,
    TorchPeriodicPLREncoder,
    TorchPLREncoder,
)

_ENCODER_REGISTRY: dict[str, type[BaseEncoder]] = {
    "identity": IdentityEncoder,
    "plr": SimplePLREncoder,
    "periodic": PeriodicEncoder,
    "cross": CrossInteractionEncoder,
    "target_encoding": TargetEncodingEncoder,
    "column_subset": ColumnSubsetEncoder,
    "expert_bundle": ExpertBundleEncoder,
    # Learned encoders: torch needed only at fit time (see torch_encoders).
    "torch_periodic": TorchPeriodicEncoder,
    "torch_plr": TorchPLREncoder,
    "torch_periodic_plr": TorchPeriodicPLREncoder,
    "torch_mlp": TorchMLPEncoder,
}

#: Registry names removed in 1.14.0 (ADR 0013). Naming one, directly or through
#: a saved model, raises :class:`_RemovedEncoderError` instead of the generic
#: unknown-encoder error.
_REMOVED_ENCODERS = frozenset({"pca", "quantile", "spline", "rff", "row_stats", "prototype"})


class _RemovedEncoderError(ValueError):
    """An encoder name that repleafgbm 1.13.x registered and 1.14.0 removed."""

    def __init__(self, name: str) -> None:
        self.encoder_name = name
        super().__init__(
            f"Encoder {name!r} was removed in repleafgbm 1.14.0 "
            "(docs/adr/0013-remove-space-encoders.md). Choose one of the "
            f"remaining encoders: {sorted(_ENCODER_REGISTRY)}"
        )


def make_encoder(name: str, _default_random_state: int | None = None, **kwargs) -> BaseEncoder:
    """Instantiate a registered encoder by name (e.g. encoder="plr").

    ``_default_random_state`` seeds encoders that sample parameters (e.g.
    "periodic") when the caller did not pass ``random_state`` explicitly;
    encoders without that argument ignore it.
    """
    if name in _REMOVED_ENCODERS:
        raise _RemovedEncoderError(name)
    if name not in _ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown encoder {name!r}. Available encoders: {sorted(_ENCODER_REGISTRY)}"
        )
    if name == ColumnSubsetEncoder.name:
        base = encoder_from_config(
            kwargs.pop("base_name"),
            kwargs.pop("base_config", {}),
            _default_random_state=_default_random_state,
        )
        return ColumnSubsetEncoder(base, **kwargs)
    if name == ExpertBundleEncoder.name:
        children = kwargs.pop("children")
        encoders = [
            encoder_from_config(
                child["name"],
                child["config"],
                _default_random_state=_default_random_state,
            )
            for child in children
        ]
        return ExpertBundleEncoder(encoders=encoders, **kwargs)
    cls = _ENCODER_REGISTRY[name]
    if (
        _default_random_state is not None
        and "random_state" not in kwargs
        and "random_state" in inspect.signature(cls.__init__).parameters
    ):
        kwargs["random_state"] = _default_random_state
    return cls(**kwargs)


def encoder_from_config(
    name: str, config: dict, _default_random_state: int | None = None
) -> BaseEncoder:
    """Rebuild an encoder (possibly projection-wrapped) from serialized config."""
    if name == RandomProjectionEncoder.name:
        base = encoder_from_config(
            config["base_name"],
            config["base_config"],
            _default_random_state=_default_random_state,
        )
        return RandomProjectionEncoder(
            base, out_dim=config["out_dim"], random_state=config["random_state"]
        )
    if name == TargetCorrelationEncoder.name:
        base = encoder_from_config(
            config["base_name"],
            config["base_config"],
            _default_random_state=_default_random_state,
        )
        return TargetCorrelationEncoder(base, out_dim=config["out_dim"])
    return make_encoder(name, _default_random_state=_default_random_state, **config)


__all__ = [
    "BaseEncoder",
    "IdentityEncoder",
    "SimplePLREncoder",
    "PeriodicEncoder",
    "CrossInteractionEncoder",
    "TargetEncodingEncoder",
    "TorchPeriodicEncoder",
    "TorchPLREncoder",
    "TorchPeriodicPLREncoder",
    "TorchMLPEncoder",
    "RandomProjectionEncoder",
    "TargetCorrelationEncoder",
    "ColumnSubsetEncoder",
    "ExpertBundleEncoder",
    "LeafExpert",
    "make_encoder",
    "encoder_from_config",
]
