"""Export healed low-bit weights to GGUF for the PrismML llama.cpp fork.

Runtime target (Stage 0 finding #6): ``PrismML-Eng/llama.cpp @ prism``, whose **g128** blocks
match our scheme exactly — **ternary -> ``Q2_0``** (2.125 bpw), **binary -> ``Q1_0``** (1.125
bpw) — and which runs on CUDA (Hopper/H200), Metal, x86 and Vulkan with a DSpark drafter. The
fork's quantize already handles 3D MoE expert tensors and keeps expert gating high-precision;
the open risk the Stage 0 spike validates is this path on a **256-expert** MoE (never exercised
at 1-bit) plus overriding the stock ``n_expert>=4 -> Q4_K`` auto-bump for a true end-to-end
low-bit export.

This module packs a :class:`QuantLinear` into codes + FP16 group scales (the on-disk layout);
wiring those tensors into a fork GGUF writer for this architecture is the runner-side step.
"""

from __future__ import annotations

import torch
from torch import Tensor

from bite.quant.fakequant import quantize_binary, quantize_ternary
from bite.quant.quantlinear import QuantLinear


def pack_quantlinear(layer: QuantLinear) -> dict[str, Tensor]:
    """Return the on-disk tensors for a healed layer: integer codes + FP16 group scales.

    Codes are stored as int8 in ``{-1,0,1}`` (ternary) or ``{-1,1}`` (binary); scales are FP16
    per group of ``layer.group_size``. Bit-packing into GGUF blocks happens in the writer.
    """
    w = layer.weight.detach()
    if layer.mode == "ternary":
        _, codes, scales = quantize_ternary(w, layer.group_size, layer.threshold_ratio)
    elif layer.mode == "binary":
        _, codes, scales = quantize_binary(w, layer.group_size)
    else:
        raise ValueError(f"unexpected mode {layer.mode!r}")
    out = {
        "codes": codes.to(torch.int8),
        "scales": scales.squeeze(-1).to(torch.float16),
        "mode": layer.mode,
        "group_size": layer.group_size,
    }
    if layer.bias is not None:
        out["bias"] = layer.bias.detach().to(torch.float16)
    return out


def export_model(model, out_path: str) -> None:  # pragma: no cover - runner-side
    """Write the full GGUF (LM low-bit + high-precision tail). See module docstring."""
    raise NotImplementedError(
        "runner-side: pack all QuantLinear layers via pack_quantlinear and emit GGUF with "
        "convert_hf_to_gguf.py (TQ1_0/TQ2_0). Validate the Stage 0 round-trip first."
    )
