"""Vision tower quantization — held at 4-bit HQQ, like Bonsai.

The language model goes to 1-bit/ternary; the small vision tower stays at 4-bit (HQQ) and
ships as a separate ``mmproj`` component. Runner-side only (needs ``hqq`` + the model).
"""

from __future__ import annotations

VISION_TOWER_ATTR = "visual"  # Qwen VL vision submodule


def quantize_vision_tower(model, nbits: int = 4, group_size: int = 64):  # pragma: no cover
    """Quantize the vision tower in-place to ``nbits`` with HQQ and return it.

    Runner-side: locate ``model.<VISION_TOWER_ATTR>``, apply ``hqq`` 4-bit to its Linear
    layers, and export as a standalone ``mmproj`` for multimodal input.
    """
    raise NotImplementedError(
        "runner-side: apply hqq 4-bit to the vision tower and export mmproj"
    )
