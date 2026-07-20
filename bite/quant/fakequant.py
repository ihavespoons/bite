"""Group-wise fake-quantization for extreme (ternary / binary) weight quantization.

Implements the two weight representations from the Bonsai whitepaper:

* **ternary** ``g128``: codes in ``{-1, 0, +1}`` with one FP16 scale per group of
  ``group_size`` weights (~1.71 bits/weight),
* **binary**  ``g128``: codes in ``{-1, +1}`` with one FP16 scale per group
  (~1.125 bits/weight).

Quantization is applied along the last (input/contraction) dimension of a weight,
grouped into blocks of ``group_size``. Training uses a **straight-through estimator
(STE)**: the forward pass sees the quantized weight while gradients flow to the latent
full-precision weight, so the model stays trainable under quantization-aware
distillation (QAD).

All functions are pure PyTorch and run on CPU — the mathematical core is unit-tested
without needing the 35B model or a GPU.
"""

from __future__ import annotations

import torch
from torch import Tensor

EPS = 1e-8

Mode = str  # "ternary" | "binary"


def _to_groups(w: Tensor, group_size: int) -> tuple[Tensor, int]:
    """Reshape ``w[..., N]`` into ``w[..., N // g, g]``.

    ``group_size <= 0`` (or ``None``) means "one group per row" (whole last dim).
    Requires ``N`` divisible by ``group_size``; MoE expert/attention dims in
    Qwen3.6-35B-A3B are multiples of 128, so this holds for the shipped config.
    """
    n = w.shape[-1]
    g = n if (group_size is None or group_size <= 0) else group_size
    if n % g != 0:
        raise ValueError(
            f"last dim {n} not divisible by group_size {g}; "
            "pad the tensor or choose a compatible group_size"
        )
    return w.reshape(*w.shape[:-1], n // g, g), g


def quantize_ternary(
    w: Tensor, group_size: int = 128, threshold_ratio: float | None = None
) -> tuple[Tensor, Tensor, Tensor]:
    """Quantize ``w`` to ternary ``{-1, 0, +1}`` with per-group scales.

    Two scale rules:

    * ``threshold_ratio is None`` — BitNet-1.58 *absmean* rounding: ``scale = mean(|w|)``
      per group, ``code = clamp(round(w / scale), -1, 1)`` (zeroes ``|w| < 0.5*scale``).
    * ``threshold_ratio`` set — TWN-style: keep ``|w| >= ratio * mean(|w|)``, scale is the
      mean magnitude of the kept entries (more MSE-optimal for peaky groups).

    Returns ``(w_hat, codes, scales)`` where ``w_hat`` and ``codes`` have ``w``'s shape and
    ``scales`` is group-shaped ``[..., n_groups, 1]``.
    """
    wg, _ = _to_groups(w, group_size)
    abs_mean = wg.abs().mean(dim=-1, keepdim=True)
    if threshold_ratio is None:
        scale = abs_mean.clamp_min(EPS)
        codes = torch.clamp(torch.round(wg / scale), -1.0, 1.0)
    else:
        thresh = threshold_ratio * abs_mean
        keep = (wg.abs() >= thresh).to(w.dtype)
        denom = keep.sum(dim=-1, keepdim=True).clamp_min(1.0)
        scale = ((wg.abs() * keep).sum(dim=-1, keepdim=True) / denom).clamp_min(EPS)
        codes = torch.sign(wg) * keep
    w_hat = (codes * scale).reshape(w.shape)
    return w_hat, codes.reshape(w.shape), scale


def quantize_binary(w: Tensor, group_size: int = 128) -> tuple[Tensor, Tensor, Tensor]:
    """Quantize ``w`` to binary ``{-1, +1}`` with per-group scales.

    Uses the XNOR-Net optimal scale ``scale = mean(|w|)`` (minimizes group MSE for a sign
    quantizer). ``sign(0)`` is mapped to ``+1``. Returns ``(w_hat, codes, scales)``.
    """
    wg, _ = _to_groups(w, group_size)
    scale = wg.abs().mean(dim=-1, keepdim=True).clamp_min(EPS)
    codes = torch.sign(wg)
    codes = torch.where(codes == 0, torch.ones_like(codes), codes)
    w_hat = (codes * scale).reshape(w.shape)
    return w_hat, codes.reshape(w.shape), scale


class _STE(torch.autograd.Function):
    """Forward returns ``w_hat``; backward multiplies the upstream grad by ``grad_mask``."""

    @staticmethod
    def forward(ctx, w: Tensor, w_hat: Tensor, grad_mask: Tensor) -> Tensor:  # noqa: ANN001
        ctx.save_for_backward(grad_mask)
        return w_hat

    @staticmethod
    def backward(ctx, grad_out: Tensor):  # noqa: ANN001
        (grad_mask,) = ctx.saved_tensors
        return grad_out * grad_mask, None, None


def fake_quantize(
    w: Tensor,
    mode: Mode = "ternary",
    group_size: int = 128,
    threshold_ratio: float | None = None,
    clip_ste: bool = False,
) -> Tensor:
    """Return the STE-quantized view of ``w`` for use inside a forward pass.

    Forward value equals the quantized weight; the gradient w.r.t. ``w`` is the identity
    (plain STE) unless ``clip_ste`` is set, in which case gradients are zeroed where the
    latent weight lies outside the representable range ``|w| > scale`` (LLM-QAT clipping).
    """
    if mode == "ternary":
        w_hat, _, scale = quantize_ternary(w, group_size, threshold_ratio)
    elif mode == "binary":
        w_hat, _, scale = quantize_binary(w, group_size)
    else:
        raise ValueError(f"unknown mode {mode!r}; expected 'ternary' or 'binary'")

    if not clip_ste:
        # detach trick: forward = w_hat, backward = identity to w
        return w + (w_hat - w).detach()

    wg, _ = _to_groups(w, group_size)
    grad_mask = (wg.abs() <= scale).to(w.dtype).reshape(w.shape)
    return _STE.apply(w, w_hat.detach(), grad_mask)


def effective_bits(mode: Mode, group_size: int = 128, scale_bits: int = 16) -> float:
    """Idealized bits/weight for a representation, matching the whitepaper's accounting."""
    code_bits = {"ternary": torch.log2(torch.tensor(3.0)).item(), "binary": 1.0}[mode]
    return code_bits + scale_bits / group_size
