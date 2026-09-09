"""Public STQE inference API.

This file is safe to publish together with the compiled stqe_core shared object.
The network architecture itself is implemented inside stqe_core.so.
"""
from pathlib import Path
import torch
from stqe_core import STQEInference as _STQEInference


class STQEModel:
    """Minimal public inference wrapper for STQE."""

    __slots__ = ("__backend",)

    def __init__(self, checkpoint_path=None, device=None, strict=True):
        if checkpoint_path is not None:
            checkpoint_path = str(Path(checkpoint_path).expanduser())
        self.__backend = _STQEInference(
            checkpoint_path=checkpoint_path,
            device=device,
            strict=strict,
        )

    @property
    def device(self):
        return self.__backend.device

    def load_checkpoint(self, checkpoint_path, strict=True):
        self.__backend.load_checkpoint(
            str(Path(checkpoint_path).expanduser()),
            strict=strict,
        )
        return self

    def to(self, device):
        self.__backend.set_device(device)
        return self

    def eval(self):
        # The compiled backend is permanently kept in eval mode.
        return self

    def __call__(self, x):
        return self.__backend.forward(x)

    def forward(self, x):
        return self.__backend.forward(x)

    def __repr__(self):
        return f"<STQEModel compiled inference backend, device={self.device}>"


def load_stqe(checkpoint_path, device=None, strict=True):
    """Convenience loader used by release/evaluation code."""
    return STQEModel(checkpoint_path, device=device, strict=strict)
