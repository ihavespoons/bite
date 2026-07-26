"""Model loading and teacher/student wiring for Qwen3.6-35B-A3B.

The model is a multimodal ``Qwen3_5MoeForConditionalGeneration``. Two consequences handled here:

1. **Loading** needs a multimodal-aware auto class + ``trust_remote_code`` (plain
   ``AutoModelForCausalLM`` may not resolve it); the full model still produces text logits from
   ``forward(input_ids).logits``.
2. **Scoping** — the vision tower is held at 4-bit (HQQ), so the low-bit swap must exclude it.
   :func:`find_vision_prefixes` + :func:`swap_language_model` restrict quantization to the
   language weights. These operate on plain ``nn.Module``s and are unit-tested on CPU.

``transformers`` is imported lazily so the package imports without it.
"""

from __future__ import annotations

from torch import nn

from bite.quant.policy import PrecisionPolicy, default_policy
from bite.quant.quantlinear import swap_linears

DEFAULT_MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
# submodule attribute names that hold the vision tower across Qwen VL variants
VISION_ATTRS = ("visual", "vision_model", "vision_tower")


def find_vision_prefixes(model: nn.Module) -> tuple[str, ...]:
    """Dotted names of vision-tower submodules to exclude from low-bit quantization."""
    names = set()
    for name, _module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf in VISION_ATTRS:
            names.add(name)
    # keep only top-most prefixes (drop nested children of an already-listed prefix)
    return tuple(sorted(n for n in names if not any(n != m and n.startswith(m + ".") for m in names)))


def swap_language_model(
    model: nn.Module,
    policy: PrecisionPolicy,
    *,
    group_size: int = 128,
    clip_ste: bool = False,
) -> dict[str, str]:
    """Install :class:`QuantLinear` across the language weights only (vision tower excluded)."""
    return swap_linears(
        model,
        policy,
        group_size=group_size,
        clip_ste=clip_ste,
        exclude_prefixes=find_vision_prefixes(model),
    )


def _load_multimodal(model_id: str, device_map: str):  # pragma: no cover - runner-side
    import torch
    from transformers import AutoModelForImageTextToText

    return AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map=device_map, trust_remote_code=True
    )


def load_tokenizer(model_id: str = DEFAULT_MODEL_ID):  # pragma: no cover - runner-side
    from transformers import AutoProcessor, AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except (ValueError, OSError):
        return AutoProcessor.from_pretrained(model_id, trust_remote_code=True).tokenizer


def load_teacher(model_id: str = DEFAULT_MODEL_ID, device_map: str = "auto"):  # pragma: no cover
    """Frozen BF16 teacher used to generate distillation targets."""
    model = _load_multimodal(model_id, device_map)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def build_student(  # pragma: no cover - runner-side
    model_id: str = DEFAULT_MODEL_ID,
    *,
    mode: str = "ternary",
    group_size: int = 128,
    threshold_ratio: float | str | None = None,
    policy: PrecisionPolicy | None = None,
    device_map: str = "auto",
    clip_ste: bool = False,
):
    """Load the model, quantize the language weights: ``QuantLinear`` for the ``nn.Linear``
    attention/lm-head, and fused-expert fake-quant for the MoE experts (3D Parameters).

    Returns ``(model, swapped_linears, quantized_experts)``. Vision tower untouched (4-bit HQQ,
    Stage 5); router gate and shared expert stay higher-precision.
    """
    from bite.quant.experts import install_expert_fakequant

    model = _load_multimodal(model_id, device_map)
    policy = policy or default_policy(mode)
    vision = find_vision_prefixes(model)
    swapped = swap_linears(
        model,
        policy,
        group_size=group_size,
        threshold_ratio=threshold_ratio,
        clip_ste=clip_ste,
        exclude_prefixes=vision,
    )
    experts = install_expert_fakequant(
        model,
        mode=mode,
        group_size=group_size,
        threshold_ratio=threshold_ratio,
        clip_ste=clip_ste,
        exclude_prefixes=vision,
    )
    return model, swapped, experts
