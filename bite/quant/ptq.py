"""Post-training quantization (PTQ) initialization for the low-bit weights.

Provides a strong starting point for QAD by rounding each weight to its ternary/binary
grid with **GPTQ-style error compensation**: weights are rounded column-by-column and the
rounding error is propagated to the not-yet-quantized columns through the inverse Hessian
``H = X Xᵀ`` estimated from calibration activations. This minimizes the *output* error
``‖(W − Ŵ)X‖`` rather than the naive weight error ‖W − Ŵ‖.

The Hessian comes from calibration forward passes on the cloud runner (see
:mod:`bite.moe.calibration`); the rounding algorithm here is pure PyTorch and unit-tested
on CPU, including the invariant that with ``H = I`` it reduces to plain group rounding.
"""

from __future__ import annotations

import torch
from torch import Tensor

from bite.quant.fakequant import EPS, _to_groups


def group_absmean_scales(w: Tensor, group_size: int) -> Tensor:
    """Per-group absmean scale, shape ``[out, n_groups]`` for a ``[out, in]`` weight."""
    wg, _ = _to_groups(w, group_size)
    return wg.abs().mean(dim=-1).clamp_min(EPS)


def _round_to_grid(col: Tensor, scale: Tensor, mode: str) -> Tensor:
    """Round one weight column to its group's grid (``scale`` is per-row for this column)."""
    if mode == "binary":
        q = torch.sign(col)
        q = torch.where(q == 0, torch.ones_like(q), q)
        return q * scale
    level = torch.clamp(torch.round(col / scale), -1.0, 1.0)
    return level * scale


def gptq_quantize(
    w: Tensor,
    hessian: Tensor | None = None,
    *,
    mode: str = "ternary",
    group_size: int = 128,
    percdamp: float = 0.01,
) -> tuple[Tensor, Tensor]:
    """GPTQ-style error-compensated PTQ of ``w`` to a ternary/binary grid.

    Args:
        w: weight ``[out_features, in_features]``.
        hessian: ``[in_features, in_features]`` calibration Hessian ``X Xᵀ``. ``None`` uses
            the identity (equivalent to naive per-group rounding).
        mode: ``"ternary"`` or ``"binary"``.
        group_size: scale group size along ``in_features``.
        percdamp: Hessian diagonal damping as a fraction of its mean diagonal.

    Returns ``(w_hat, scales)`` with ``scales`` group-shaped ``[out, n_groups]``.
    """
    dev, dtype = w.device, w.dtype
    W = w.detach().float().clone()
    rows, cols = W.shape
    scales = group_absmean_scales(W, group_size)  # [out, n_groups]

    H = torch.eye(cols, device=dev) if hessian is None else hessian.detach().float().clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0
    H[range(cols), range(cols)] += percdamp * torch.diag(H).mean()

    # Hinv = (upper Cholesky of H^{-1}); standard GPTQ preconditioning
    hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)

    Q = torch.zeros_like(W)
    for i in range(cols):
        g = i // group_size
        d = hinv[i, i]
        col = W[:, i]
        q = _round_to_grid(col, scales[:, g], mode)
        Q[:, i] = q
        err = (col - q) / d
        W[:, i + 1 :] -= err.unsqueeze(1) * hinv[i, i + 1 :].unsqueeze(0)

    return Q.to(dtype), scales.to(dtype)
