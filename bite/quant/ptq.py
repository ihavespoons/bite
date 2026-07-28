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

from bite.quant.fakequant import (
    EPS,
    _to_groups,
    parse_mode,
    quantize_binary,
    quantize_ternary,
    quantize_uniform,
)


def group_absmean_scales(w: Tensor, group_size: int) -> Tensor:
    """Per-group absmean scale, shape ``[out, n_groups]`` for a ``[out, in]`` weight."""
    wg, _ = _to_groups(w, group_size)
    return wg.abs().mean(dim=-1).clamp_min(EPS)


def _round_to_grid(col: Tensor, scale: Tensor, mode: str) -> Tensor:
    """Round one weight column to its group's grid (``scale`` is per-row for this column)."""
    kind, bits = parse_mode(mode)
    if kind == "binary":
        q = torch.sign(col)
        q = torch.where(q == 0, torch.ones_like(q), q)
        return q * scale
    qmax = 1.0 if kind == "ternary" else float(2 ** (bits - 1) - 1)
    level = torch.clamp(torch.round(col / scale), -qmax, qmax)
    return level * scale


def gptq_quantize(
    w: Tensor,
    hessian: Tensor | None = None,
    *,
    mode: str = "ternary",
    group_size: int = 128,
    threshold_ratio: float | str | None = None,
    percdamp: float = 0.01,
) -> tuple[Tensor, Tensor]:
    """GPTQ-style error-compensated PTQ of ``w`` to a ternary/binary grid.

    Args:
        w: weight ``[out_features, in_features]``.
        hessian: ``[in_features, in_features]`` calibration Hessian ``X Xᵀ``. ``None`` uses
            the identity (equivalent to plain per-group rounding).
        mode: ``"ternary"`` or ``"binary"``.
        group_size: scale group size along ``in_features``.
        threshold_ratio: ternary scale rule (see :func:`quantize_ternary`); must match the
            forward's rule so the init lands on-grid. Hessian path is absmean-only for now.
        percdamp: Hessian diagonal damping as a fraction of its mean diagonal.

    Returns ``(w_hat, scales)`` with ``scales`` group-shaped ``[out, n_groups]``.
    """
    dtype = w.dtype
    if parse_mode(mode)[0] == "int":
        _, bits_ = parse_mode(mode)
        wg, _ = _to_groups(w.detach().float(), group_size)
        scales = (wg.abs().amax(dim=-1) / (2 ** (bits_ - 1) - 1)).clamp_min(EPS)
    else:
        scales = group_absmean_scales(w.detach().float(), group_size)  # [out, n_groups]

    # No Hessian -> identity preconditioner -> error feedback does nothing, so the GPTQ column
    # loop is exactly plain per-group rounding. Take the fast vectorized path (critical: the
    # loop over a 35B model's thousands of expert Linears would otherwise take hours).
    kind, bits = parse_mode(mode)
    if hessian is None:
        if kind == "binary":
            w_hat, _, _ = quantize_binary(w, group_size)
        elif kind == "int":
            # uniform N-bit uses an ABSMAX scale, not absmean — keep scales rule-consistent
            w_hat, _, gs = quantize_uniform(w, bits, group_size)
            scales = gs.squeeze(-1).float()
        else:
            w_hat, _, gs = quantize_ternary(w, group_size, threshold_ratio)
            scales = gs.squeeze(-1).float()  # rule-consistent scales, [out, n_groups]
        return w_hat.to(dtype), scales.to(dtype)

    if threshold_ratio is not None:
        raise NotImplementedError(
            "GPTQ Hessian path only supports absmean rounding; got "
            f"threshold_ratio={threshold_ratio!r}"
        )

    dev = w.device
    W = w.detach().float().clone()
    rows, cols = W.shape

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


def ptq_init_model(
    model,
    hessians: dict | None = None,
    *,
    percdamp: float = 0.01,
) -> dict[str, int]:
    """PTQ-initialize every :class:`QuantLinear`'s latent weight to its GPTQ grid point.

    Sets ``module.weight`` to the error-compensated quantized weights so QAD starts near a good
    low-bit solution. Uses ``hessians[name]`` when present (a :class:`HessianCollector`'s ``H``),
    else the identity (naive rounding). The first fake-quant forward re-quantizes these on-grid
    weights near-idempotently; QAD heals the small residual. Returns ``{name: cols_quantized}``.
    """
    from bite.quant.quantlinear import QuantLinear  # local to avoid an import cycle

    hessians = hessians or {}
    done: dict[str, int] = {}
    for name, module in model.named_modules():
        if not isinstance(module, QuantLinear):
            continue
        w_hat, _ = gptq_quantize(
            module.weight.data,
            hessians.get(name),
            mode=module.mode,
            group_size=module.group_size,
            threshold_ratio=module.threshold_ratio,
            percdamp=percdamp,
        )
        module.weight.data.copy_(w_hat)
        done[name] = module.weight.shape[1]
    return done
