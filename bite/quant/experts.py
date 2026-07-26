"""Fake-quantization for fused MoE expert weights.

Qwen3.5-MoE stores its 256 experts not as ``nn.Linear`` modules but as **fused 3D
``nn.Parameter`` tensors** (``gate_up_proj`` ``(E, hidden, 2*inter)``, ``down_proj``
``(E, inter, hidden)``) for grouped-GEMM. The :class:`~bite.quant.quantlinear.QuantLinear`
swap therefore misses them entirely — yet they are the bulk of the model and the whole point
of MoE quantization.

This module quantizes those tensors via :func:`torch.nn.utils.parametrize`: a fake-quant
parametrization is registered on each fused expert Parameter, so every forward access returns
the STE-quantized weight (group-wise along the last dim) while the latent full-precision
Parameter stays trainable for QAD. Works with any MoE forward that reads ``self.gate_up_proj``.
The math reuses :func:`bite.quant.fakequant.fake_quantize` (shape-agnostic, groups last dim).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn.utils import parametrize

from bite.quant.fakequant import fake_quantize, quantize_binary, quantize_ternary
from bite.quant.quantlinear import _is_excluded

# fused expert projection parameter names in the Qwen MoE block
EXPERT_PARAM_NAMES = ("gate_up_proj", "down_proj", "gate_proj", "up_proj")


class FakeQuantParam(nn.Module):
    """Parametrization: maps a latent weight to its STE fake-quantized view on each access."""

    def __init__(
        self,
        mode: str = "ternary",
        group_size: int = 128,
        threshold_ratio: float | str | None = None,
        clip_ste: bool = False,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.group_size = group_size
        self.threshold_ratio = threshold_ratio
        self.clip_ste = clip_ste

    def forward(self, w: Tensor) -> Tensor:
        return fake_quantize(w, self.mode, self.group_size, self.threshold_ratio, self.clip_ste)


def install_expert_fakequant(
    model: nn.Module,
    *,
    mode: str = "ternary",
    group_size: int = 128,
    threshold_ratio: float | str | None = None,
    clip_ste: bool = False,
    exclude_prefixes: tuple[str, ...] = (),
) -> dict[str, str]:
    """Register fake-quant on every fused 3D expert Parameter. Returns ``{name.param: mode}``.

    Only 3D parameters with a known expert projection name are targeted — the 2D router gate
    and shared expert (kept higher precision) and 3D DeltaNet conv weights (different names)
    are left untouched, as is anything under ``exclude_prefixes`` (the vision tower).
    """
    installed: dict[str, str] = {}
    for name, module in model.named_modules():
        if _is_excluded(name, exclude_prefixes):
            continue
        for pname in EXPERT_PARAM_NAMES:
            p = getattr(module, pname, None)
            if isinstance(p, nn.Parameter) and p.dim() == 3:
                parametrize.register_parametrization(
                    module,
                    pname,
                    FakeQuantParam(mode, group_size, threshold_ratio, clip_ste=clip_ste),
                )
                installed[f"{name}.{pname}"] = mode
    return installed


def _quantize_like(w: Tensor, fq: FakeQuantParam) -> Tensor:
    if fq.mode == "binary":
        return quantize_binary(w, fq.group_size)[0]
    return quantize_ternary(w, fq.group_size, fq.threshold_ratio)[0]


def ptq_init_experts(model: nn.Module) -> int:
    """Set each parametrized expert latent Parameter to its on-grid (quantized) value."""
    count = 0
    for module in model.modules():
        if not parametrize.is_parametrized(module):
            continue
        for pname in list(module.parametrizations.keys()):
            fq = module.parametrizations[pname][0]
            if not isinstance(fq, FakeQuantParam):
                continue
            orig = module.parametrizations[pname].original
            with torch.no_grad():
                orig.copy_(_quantize_like(orig, fq))
            count += 1
    return count


def expert_latent_params(model: nn.Module) -> list[nn.Parameter]:
    """The trainable latent expert Parameters (for QAD optimizer construction)."""
    params: list[nn.Parameter] = []
    for module in model.modules():
        if parametrize.is_parametrized(module):
            for pname in module.parametrizations:
                if isinstance(module.parametrizations[pname][0], FakeQuantParam):
                    params.append(module.parametrizations[pname].original)
    return params
