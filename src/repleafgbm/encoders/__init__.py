"""Encoders: numerical feature -> representation z_theta(x).

Two families implement :class:`~repleafgbm.encoders.base.BaseEncoder`:

* **Fixed** (pure NumPy): ``identity``, ``plr``, ``periodic``, ``cross`` —
  fitted once (statistics / seeded parameters / pair selection) and frozen.
* **Pretrained-then-frozen** (optional ``[torch]`` extra): ``torch_periodic``,
  ``torch_plr``, ``torch_periodic_plr``, ``torch_mlp`` — parameters are
  supervised-pretrained on the initial Newton residual and then frozen into
  NumPy arrays. They are **not** joint-trainable during boosting; torch is
  needed only at fit time, while ``transform`` and serialization stay NumPy.

The encoder is frozen for all of boosting (see docs/math.md); joint/alternating
encoder training is a roadmap item, not implemented here. New encoders register
in ``_ENCODER_REGISTRY`` below.
"""

from __future__ import annotations

import inspect

from repleafgbm.encoders.base import BaseEncoder
from repleafgbm.encoders.cross import CrossInteractionEncoder
from repleafgbm.encoders.identity import IdentityEncoder
from repleafgbm.encoders.periodic import PeriodicEncoder
from repleafgbm.encoders.plr import SimplePLREncoder
from repleafgbm.encoders.projection import RandomProjectionEncoder
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
    # Learned encoders: torch needed only at fit time (see torch_encoders).
    "torch_periodic": TorchPeriodicEncoder,
    "torch_plr": TorchPLREncoder,
    "torch_periodic_plr": TorchPeriodicPLREncoder,
    "torch_mlp": TorchMLPEncoder,
}


def make_encoder(name: str, _default_random_state: int | None = None, **kwargs) -> BaseEncoder:
    """Instantiate a registered encoder by name (e.g. encoder="plr").

    ``_default_random_state`` seeds encoders that sample parameters (e.g.
    "periodic") when the caller did not pass ``random_state`` explicitly;
    encoders without that argument ignore it.
    """
    if name not in _ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown encoder {name!r}. Available encoders: {sorted(_ENCODER_REGISTRY)}"
        )
    cls = _ENCODER_REGISTRY[name]
    if (
        _default_random_state is not None
        and "random_state" not in kwargs
        and "random_state" in inspect.signature(cls.__init__).parameters
    ):
        kwargs["random_state"] = _default_random_state
    return cls(**kwargs)


def encoder_from_config(name: str, config: dict) -> BaseEncoder:
    """Rebuild an encoder (possibly projection-wrapped) from serialized config."""
    if name == RandomProjectionEncoder.name:
        base = make_encoder(config["base_name"], **config["base_config"])
        return RandomProjectionEncoder(
            base, out_dim=config["out_dim"], random_state=config["random_state"]
        )
    return make_encoder(name, **config)


__all__ = [
    "BaseEncoder",
    "IdentityEncoder",
    "SimplePLREncoder",
    "PeriodicEncoder",
    "CrossInteractionEncoder",
    "TorchPeriodicEncoder",
    "TorchPLREncoder",
    "TorchPeriodicPLREncoder",
    "TorchMLPEncoder",
    "RandomProjectionEncoder",
    "make_encoder",
    "encoder_from_config",
]
