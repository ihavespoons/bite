"""Model loading and teacher/student wiring for Qwen3.6-35B-A3B.

Heavy dependencies (``transformers``) are imported lazily so the pipeline package imports
without them; the actual loading runs on the cloud runner (single H200 holds the 35B in
bf16 for inference-shaped stages).
"""

from __future__ import annotations

from bite.quant.policy import PrecisionPolicy, default_policy
from bite.quant.quantlinear import swap_linears

DEFAULT_MODEL_ID = "Qwen/Qwen3.6-35B-A3B"


def load_tokenizer(model_id: str = DEFAULT_MODEL_ID):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


def load_teacher(model_id: str = DEFAULT_MODEL_ID, device_map: str = "auto"):
    """Frozen FP16/BF16 teacher used to generate distillation targets."""
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=device_map, trust_remote_code=True
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def build_student(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    mode: str = "ternary",
    group_size: int = 128,
    policy: PrecisionPolicy | None = None,
    device_map: str = "auto",
    clip_ste: bool = False,
):
    """Load the model and install :class:`QuantLinear` per the precision policy.

    Returns ``(student_model, swapped_map)``. ``swapped_map`` is ``{name: mode}`` for logging
    the low-bit coverage (attention proj / routed-expert MLP / LM head), with the router gate
    and shared expert left in higher precision.
    """
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=device_map, trust_remote_code=True
    )
    policy = policy or default_policy(mode)
    swapped = swap_linears(
        model, policy, group_size=group_size, clip_ste=clip_ste
    )
    return model, swapped
