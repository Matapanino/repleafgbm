"""Encoders: numerical feature -> representation z_theta(x).

v0 ships NumPy-based frozen encoders only. Future encoders (periodic
embeddings, RealMLP-style blocks, PyTorch modules) must implement
:class:`~repleafgbm.encoders.base.BaseEncoder` and register here.
"""

from __future__ import annotations

import inspect

from repleafgbm.encoders.base import BaseEncoder
from repleafgbm.encoders.identity import IdentityEncoder
from repleafgbm.encoders.periodic import PeriodicEncoder
from repleafgbm.encoders.plr import SimplePLREncoder
from repleafgbm.encoders.projection import RandomProjectionEncoder
from repleafgbm.encoders.torch_encoders import TorchPeriodicEncoder, TorchPLREncoder

_ENCODER_REGISTRY: dict[str, type[BaseEncoder]] = {
    "identity": IdentityEncoder,
    "plr": SimplePLREncoder,
    "periodic": PeriodicEncoder,
    # Learned encoders: torch needed only at fit time (see torch_encoders).
    "torch_periodic": TorchPeriodicEncoder,
    "torch_plr": TorchPLREncoder,
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
    "TorchPeriodicEncoder",
    "TorchPLREncoder",
    "RandomProjectionEncoder",
    "make_encoder",
    "encoder_from_config",
]
