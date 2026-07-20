"""Activation-Hessian collection for GPTQ-style PTQ calibration.

For each target Linear with weight ``[out, in]``, GPTQ needs the input second-moment
``H = Σ x xᵀ`` over calibration tokens (shape ``[in, in]``). This collector attaches
forward-pre-hooks to the target modules and accumulates ``H`` in-place, streaming so no
activations are retained. Pure PyTorch, unit-tested on CPU.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from bite.quant.quantlinear import QuantLinear


def quant_linear_modules(model: nn.Module) -> dict[str, nn.Module]:
    """All :class:`QuantLinear` modules by name — the PTQ/Hessian targets."""
    return {n: m for n, m in model.named_modules() if isinstance(m, QuantLinear)}


class HessianCollector:
    """Accumulates ``H = Σ xᵀx`` per attached module over calibration forward passes."""

    def __init__(self) -> None:
        self.H: dict[str, Tensor] = {}
        self.tokens: dict[str, int] = {}
        self._handles: list = []

    def _pre_hook(self, name: str):
        def hook(_module, args):
            x = args[0].detach()
            x = x.reshape(-1, x.shape[-1]).float()
            gram = x.t() @ x
            if name in self.H:
                self.H[name] += gram
                self.tokens[name] += x.shape[0]
            else:
                self.H[name] = gram
                self.tokens[name] = x.shape[0]

        return hook

    def attach(self, modules: dict[str, nn.Module]) -> "HessianCollector":
        for name, module in modules.items():
            self._handles.append(module.register_forward_pre_hook(self._pre_hook(name)))
        return self

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self) -> "HessianCollector":
        return self

    def __exit__(self, *exc) -> None:
        self.detach()
