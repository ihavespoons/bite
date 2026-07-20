"""Export healed low-bit weights to GGUF for llama.cpp / bitnet.cpp.

Ternary -> ``TQ1_0``/``TQ2_0``; binary -> 1-bit ``g128``. The **Stage 0 feasibility spike**
must confirm that ``convert_hf_to_gguf.py`` understands the Qwen3.6 hybrid (Gated DeltaNet +
MoE) architecture and that these low-bit quant types round-trip for it; if not, we either
contribute the converter or fall back to the in-framework fake-quant eval harness.

This module packs a :class:`QuantLinear` into codes + FP16 group scales (the on-disk layout);
wiring those tensors into a GGUF writer for this specific architecture is the runner-side step.
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
