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
from bite.quant.policy import KEEP
from bite.quant.quantlinear import _is_excluded

# fused expert projection parameter names in the Qwen MoE block
EXPERT_PARAM_NAMES = ("gate_up_proj", "down_proj", "gate_proj", "up_proj")


class FakeQuantParam(nn.Module):
    """Parametrization: maps a latent weight to its STE fake-quantized view on each access.

    ``keep_indices`` names experts (indices along dim 0 of the fused ``(E, ...)`` tensor) that
    stay at full precision — the mechanism for **expert-level mixed precision**: quantize most
    experts, spend a few bits of budget on the sensitive minority. Kept rows pass through
    untouched (real gradients, not STE).
    """

    def __init__(
        self,
        mode: str = "ternary",
        group_size: int = 128,
        threshold_ratio: float | str | None = None,
        clip_ste: bool = False,
        keep_indices: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.group_size = group_size
        self.threshold_ratio = threshold_ratio
        self.clip_ste = clip_ste
        self.keep_indices = tuple(keep_indices) if keep_indices else None

    def forward(self, w: Tensor) -> Tensor:
        q = fake_quantize(w, self.mode, self.group_size, self.threshold_ratio, self.clip_ste)
        if not self.keep_indices:
            return q
        # broadcast an expert-index mask over the trailing dims of the fused tensor
        idx = torch.as_tensor(self.keep_indices, device=w.device, dtype=torch.long)
        keep = torch.zeros(w.shape[0], dtype=torch.bool, device=w.device)
        keep[idx] = True
        return torch.where(keep.view(-1, *([1] * (w.dim() - 1))), w, q)

    def effective_bpw(self, num_experts: int, scale_bits: int = 16) -> float:
        """Bits/weight of this tensor under the mixed-precision split (kept experts at 16-bit)."""
        from bite.quant.fakequant import effective_bits

        kept = len(self.keep_indices or ())
        frac = kept / max(num_experts, 1)
        return frac * 16.0 + (1 - frac) * effective_bits(self.mode, self.group_size, scale_bits)


def install_expert_fakequant(
    model: nn.Module,
    *,
    mode: str = "ternary",
    group_size: int = 128,
    threshold_ratio: float | str | None = None,
    clip_ste: bool = False,
    exclude_prefixes: tuple[str, ...] = (),
    keep_patterns: tuple[str, ...] = (),
    keep_expert_indices: tuple[int, ...] | None = None,
) -> dict[str, str]:
    """Register fake-quant on every fused 3D expert Parameter. Returns ``{name.param: mode}``.

    Only 3D parameters with a known expert projection name are targeted — the 2D router gate
    and shared expert (kept higher precision) and 3D DeltaNet conv weights (different names)
    are left untouched, as is anything under ``exclude_prefixes`` (the vision tower).

    ``keep_patterns`` (regexes matched against ``name.param``) leave whole expert tensors at
    full precision — the policy only governs ``nn.Linear`` swaps, so without this there is no
    way to ablate the fused experts, which are the bulk of the model.
    ``keep_expert_indices`` leaves those expert slots high-precision inside every quantized
    tensor (expert-level mixed precision).
    """
    import re

    keep_res = [re.compile(p) for p in keep_patterns]
    installed: dict[str, str] = {}
    for name, module in model.named_modules():
        if _is_excluded(name, exclude_prefixes):
            continue
        for pname in EXPERT_PARAM_NAMES:
            p = getattr(module, pname, None)
            if isinstance(p, nn.Parameter) and p.dim() == 3:
                full = f"{name}.{pname}"
                if any(r.search(full) for r in keep_res):
                    installed[full] = KEEP
                    continue
                parametrize.register_parametrization(
                    module,
                    pname,
                    FakeQuantParam(
                        mode, group_size, threshold_ratio, clip_ste=clip_ste,
                        keep_indices=keep_expert_indices,
                    ),
                )
                installed[full] = mode
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


def apply_expert_mixed_precision(
    model: nn.Module, frac: float, by: str = "error"
) -> dict[str, int]:
    """Hold ``frac`` of each fused tensor's expert slots at high precision. Returns per-tensor counts.

    Selection runs AFTER the build because it needs the weights: ``by="error"`` keeps the experts
    whose ternary round-trip error is largest (the ones the low-bit grid represents worst), while
    ``by="first"`` keeps slots 0..k as a control for "does *which* experts we keep matter, or just
    how many". Per-tensor selection (not one global slot set) lets every layer keep its own worst
    experts. Mutates each :class:`FakeQuantParam`'s ``keep_indices`` in place.
    """
    if not 0.0 < frac <= 1.0:
        raise ValueError(f"frac must be in (0, 1], got {frac}")
    chosen: dict[str, int] = {}
    for name, module in model.named_modules():
        if not parametrize.is_parametrized(module):
            continue
        for pname in module.parametrizations:
            fq = module.parametrizations[pname][0]
            if not isinstance(fq, FakeQuantParam):
                continue
            w = module.parametrizations[pname].original.detach()
            n_experts = w.shape[0]
            k = max(1, int(round(frac * n_experts)))
            if by == "first":
                idx = tuple(range(k))
            else:
                per_expert = w.float() - _quantize_like(w.float(), fq)
                err = per_expert.flatten(1).pow(2).mean(dim=1)  # MSE per expert slot
                idx = tuple(int(i) for i in err.topk(k).indices.tolist())
            fq.keep_indices = idx
            chosen[f"{name}.{pname}"] = len(idx)
    return chosen


def expert_latent_params(model: nn.Module) -> list[nn.Parameter]:
    """The trainable latent expert Parameters (for QAD optimizer construction)."""
    params: list[nn.Parameter] = []
    for module in model.modules():
        if parametrize.is_parametrized(module):
            for pname in module.parametrizations:
                if isinstance(module.parametrizations[pname][0], FakeQuantParam):
                    params.append(module.parametrizations[pname].original)
    return params
