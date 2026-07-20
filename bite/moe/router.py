"""Router stability under quantization.

The MoE router/gate stays in higher precision (see :mod:`bite.quant.policy`), but routing can
still drift as expert weights are quantized. This module (1) freezes router parameters and
(2) adds a router-distillation term that keeps the student's routing distribution close to the
teacher's, preserving which experts fire for which tokens.

Tensor helpers are unit-tested on CPU; hook attachment runs on the cloud runner.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

ROUTER_NAME_PATTERNS = (".gate",)


def freeze_router(model: nn.Module) -> list[str]:
    """Set ``requires_grad=False`` on router/gate parameters; return their names."""
    frozen: list[str] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.endswith("gate"):
            for p in module.parameters():
                p.requires_grad_(False)
            frozen.append(name)
    return frozen


def router_distill_loss(
    student_router_logits: Tensor, teacher_router_logits: Tensor
) -> Tensor:
    """KL(teacher ‖ student) over the full expert distribution, keeping routing aligned."""
    teacher_logp = F.log_softmax(teacher_router_logits, dim=-1)
    student_logp = F.log_softmax(student_router_logits, dim=-1)
    return (teacher_logp.exp() * (teacher_logp - student_logp)).sum(-1).mean()


def top_k_agreement(
    student_router_logits: Tensor, teacher_router_logits: Tensor, k: int = 8
) -> float:
    """Fraction of the teacher's top-k experts also in the student's top-k (routing drift)."""
    s = student_router_logits.topk(k, dim=-1).indices
    t = teacher_router_logits.topk(k, dim=-1).indices
    inter = torch.tensor(
        [len(set(si.tolist()) & set(ti.tolist())) for si, ti in zip(s.reshape(-1, k), t.reshape(-1, k))]
    )
    return float(inter.float().mean() / k)
