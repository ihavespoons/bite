"""Offline (one-shot) weight quantization for PTQ *evaluation* — no parametrizations, no STE.

The QAD path in :mod:`bite.quant.experts` / :mod:`bite.quant.quantlinear` re-quantizes on every
forward so gradients can reach the latent weights. For forward-only PTQ measurement that is pure
overhead, and it makes two things awkward that we now need:

**Grouping axis.** Fake-quant groups the LAST dim. For an ``nn.Linear`` weight ``[out, in]`` that
is the contraction dim — correct. But the fused MoE experts are ``(E, hidden, 2*inter)`` and
``(E, inter, hidden)``, contracting over dim **1**, so grouping the last dim shares one scale
across weights that never meet in a dot product. llama.cpp's Q*_0 blocks (and standard practice)
group along the contraction axis. ``axis`` here fixes that.

**Rotation (incoherence processing).** Quantizing in a Hadamard-rotated basis spreads outliers
and is what lets QuaRot/QuIP#-style methods work near 2 bits. Simulating it offline is exact:
a real rotated implementation computes ``(xR) · q(WR)ᵀ``, and storing ``q(WR)Rᵀ`` then doing a
plain ``x · Wᵀ`` gives ``x R q(WR)ᵀ`` — the same value. So the accuracy measured here is the
accuracy such an implementation would deliver; only the runtime Hadamard is elided.
"""

from __future__ import annotations

import math
import re

import torch
from torch import Tensor, nn

from bite.quant.fakequant import parse_mode, quantize_binary, quantize_ternary, quantize_uniform

_HADAMARD_CACHE: dict[tuple[int, str, torch.dtype], Tensor] = {}


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def hadamard_matrix(n: int, device="cpu", dtype=torch.float32) -> Tensor:
    """Normalized Sylvester Hadamard ``H`` (``H @ H.T == I``). Requires a power-of-two ``n``."""
    if not is_power_of_two(n):
        raise ValueError(f"Hadamard needs a power-of-two size, got {n}")
    key = (n, str(device), dtype)
    if key not in _HADAMARD_CACHE:
        h = torch.ones(1, 1, device=device, dtype=torch.float32)
        while h.shape[0] < n:
            h = torch.cat(
                [torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0
            )
        _HADAMARD_CACHE[key] = (h / math.sqrt(n)).to(dtype)
    return _HADAMARD_CACHE[key]


def _quantize_last_dim(
    w: Tensor, mode: str, group_size: int, threshold_ratio: float | str | None
) -> Tensor:
    kind, bits = parse_mode(mode)
    if kind == "binary":
        return quantize_binary(w, group_size)[0]
    if kind == "int":
        return quantize_uniform(w, bits, group_size)[0]
    return quantize_ternary(w, group_size, threshold_ratio)[0]


def quantize_tensor_offline(
    w: Tensor,
    *,
    mode: str = "ternary",
    group_size: int = 128,
    threshold_ratio: float | str | None = None,
    axis: int = -1,
    rotate: bool = False,
) -> tuple[Tensor, bool]:
    """Quantize ``w`` along ``axis``, optionally in a Hadamard basis. Returns ``(w_hat, rotated)``.

    ``rotated`` reports whether rotation was actually applied — it is skipped (not an error) when
    the axis length is not a power of two, so a sweep over a real model degrades gracefully
    instead of dying on one odd tensor.
    """
    orig_dtype = w.dtype
    x = w.detach().float()
    if axis != -1 and axis != x.dim() - 1:
        x = x.transpose(axis, -1)

    did_rotate = False
    n = x.shape[-1]
    if rotate and is_power_of_two(n):
        h = hadamard_matrix(n, x.device, x.dtype)
        xr = x @ h                                        # into the rotated basis
        q = _quantize_last_dim(xr, mode, group_size, threshold_ratio)
        x_hat = q @ h.transpose(0, 1)                     # back out (see module docstring)
        did_rotate = True
    else:
        x_hat = _quantize_last_dim(x, mode, group_size, threshold_ratio)

    if axis != -1 and axis != w.dim() - 1:
        x_hat = x_hat.transpose(axis, -1)
    return x_hat.to(orig_dtype), did_rotate


# fused 3D expert params: (E, hidden, 2*inter) and (E, inter, hidden) both contract over dim 1
EXPERT_CONTRACT_AXIS = 1


def quantize_model_offline(  # pragma: no cover - runner-side (needs a real model)
    model: nn.Module,
    *,
    mode: str = "ternary",
    group_size: int = 128,
    threshold_ratio: float | str | None = None,
    rotate: bool = False,
    expert_axis: int = EXPERT_CONTRACT_AXIS,
    keep_patterns: tuple[str, ...] = (),
    expert_param_names: tuple[str, ...] = ("gate_up_proj", "down_proj", "gate_proj", "up_proj"),
    verbose: bool = True,
) -> dict:
    """Quantize every eligible weight IN PLACE. Returns counts and how much got rotated."""
    keeps = [re.compile(p) for p in keep_patterns]
    kept = lambda name: any(r.search(name) for r in keeps)  # noqa: E731

    stats = {
        "linears": 0, "linear_params": 0, "linears_rotated": 0,
        "experts": 0, "expert_params": 0, "experts_rotated": 0,
        "skipped": 0,
    }
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if kept(name):
                    stats["skipped"] += 1
                    continue
                w_hat, rot = quantize_tensor_offline(
                    module.weight.data, mode=mode, group_size=group_size,
                    threshold_ratio=threshold_ratio, axis=-1, rotate=rotate,
                )
                module.weight.data.copy_(w_hat)
                stats["linears"] += 1
                stats["linear_params"] += module.weight.numel()
                stats["linears_rotated"] += int(rot)
                continue
            for pname in expert_param_names:
                p = getattr(module, pname, None)
                if isinstance(p, nn.Parameter) and p.dim() == 3:
                    full = f"{name}.{pname}"
                    if kept(full):
                        stats["skipped"] += 1
                        continue
                    w_hat, rot = quantize_tensor_offline(
                        p.data, mode=mode, group_size=group_size,
                        threshold_ratio=threshold_ratio, axis=expert_axis, rotate=rotate,
                    )
                    p.data.copy_(w_hat)
                    stats["experts"] += 1
                    stats["expert_params"] += p.numel()
                    stats["experts_rotated"] += int(rot)
    if verbose:
        print(
            f"offline quant: mode={mode} g={group_size} rotate={rotate} expert_axis={expert_axis}"
            f" | {stats['linears']} linears ({stats['linear_params'] / 1e9:.2f}B params,"
            f" {stats['linears_rotated']} rotated) | {stats['experts']} expert tensors"
            f" ({stats['expert_params'] / 1e9:.2f}B params, {stats['experts_rotated']} rotated)"
            f" | {stats['skipped']} kept",
            flush=True,
        )
    return stats
