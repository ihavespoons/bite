"""``QuantLinear`` and the module-swap that installs it across a model.

``QuantLinear`` keeps a full-precision *latent* weight (trainable) and applies group-wise
fake-quantization in its forward pass, so a model swapped this way trains end-to-end under
QAD while its forward numerics match the low-bit representation.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from bite.quant.fakequant import fake_quantize
from bite.quant.policy import KEEP, PrecisionPolicy


class QuantLinear(nn.Module):
    """Drop-in replacement for :class:`nn.Linear` with STE fake-quantized weights."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        mode: str = "ternary",
        group_size: int = 128,
        threshold_ratio: float | None = None,
        clip_ste: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.mode = mode
        self.group_size = group_size
        self.threshold_ratio = threshold_ratio
        self.clip_ste = clip_ste
        # latent full-precision weight, trained through the STE
        self.weight = nn.Parameter(torch.empty(out_features, in_features, **factory))
        self.bias = nn.Parameter(torch.empty(out_features, **factory)) if bias else None

    @classmethod
    def from_linear(cls, linear: nn.Linear, **kw) -> "QuantLinear":
        ql = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
            **kw,
        )
        with torch.no_grad():
            ql.weight.copy_(linear.weight)
            if linear.bias is not None:
                ql.bias.copy_(linear.bias)
        return ql

    def quantized_weight(self) -> Tensor:
        return fake_quantize(
            self.weight,
            mode=self.mode,
            group_size=self.group_size,
            threshold_ratio=self.threshold_ratio,
            clip_ste=self.clip_ste,
        )

    def forward(self, x: Tensor) -> Tensor:
        return nn.functional.linear(x, self.quantized_weight(), self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, mode={self.mode}, group_size={self.group_size}"
        )


def _set_submodule(root: nn.Module, dotted: str, new: nn.Module) -> None:
    parent = root
    *path, last = dotted.split(".")
    for part in path:
        parent = getattr(parent, part)
    setattr(parent, last, new)


def swap_linears(
    model: nn.Module,
    policy: PrecisionPolicy,
    *,
    group_size: int = 128,
    threshold_ratio: float | None = None,
    clip_ste: bool = False,
) -> dict[str, str]:
    """Replace policy-selected :class:`nn.Linear` modules with :class:`QuantLinear`.

    Returns a mapping ``{module_name: mode}`` for the swapped modules (for logging/tests).
    Modules the policy resolves to ``keep`` — and every non-Linear module — are left
    untouched.
    """
    targets: list[tuple[str, nn.Linear, str]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            mode = policy.resolve(name)
            if mode != KEEP:
                targets.append((name, module, mode))

    swapped: dict[str, str] = {}
    for name, linear, mode in targets:
        ql = QuantLinear.from_linear(
            linear,
            mode=mode,
            group_size=group_size,
            threshold_ratio=threshold_ratio,
            clip_ste=clip_ste,
        )
        _set_submodule(model, name, ql)
        swapped[name] = mode
    return swapped
