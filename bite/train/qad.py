"""Quantization-Aware Distillation (QAD) — Stage 3/4.

Heals a fake-quantized *student* against a frozen FP16 *teacher*. The default schedule is
**block-wise**: one transformer block is distilled at a time against precomputed teacher
hidden states / top-k logits, so only a single block's optimizer state is resident and the
run fits a single H200. An optional end-to-end polish pass (8-bit optimizer) follows.

The loss functions are pure PyTorch and unit-tested on CPU; the training loop is executed on
the cloud runner where the model and precomputed teacher targets are available.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


def topk_kl_loss(
    student_logits: Tensor,
    teacher_topk_values: Tensor,
    teacher_topk_indices: Tensor,
    temperature: float = 1.0,
) -> Tensor:
    """KL(teacher ‖ student) over the teacher's top-k logits (memory-cheap distillation).

    ``student_logits``: ``[..., vocab]``. Teacher targets are the top-k logits and their
    indices (as precomputed and stored to disk), shaped ``[..., k]``.
    """
    t = temperature
    teacher_logp = F.log_softmax(teacher_topk_values / t, dim=-1)
    student_at_k = torch.gather(student_logits / t, -1, teacher_topk_indices)
    student_logp = F.log_softmax(student_at_k, dim=-1)
    kl = (teacher_logp.exp() * (teacher_logp - student_logp)).sum(-1)
    return (kl * (t * t)).mean()


def hidden_state_mse(student_h: Tensor, teacher_h: Tensor) -> Tensor:
    """MSE between student and teacher hidden states (block-wise feature matching)."""
    return F.mse_loss(student_h.float(), teacher_h.float())


@dataclass
class QADLossWeights:
    kl: float = 1.0
    hidden: float = 1.0
    ce: float = 0.5


def qad_loss(
    student_logits: Tensor,
    teacher_topk_values: Tensor,
    teacher_topk_indices: Tensor,
    labels: Tensor | None = None,
    student_h: Tensor | None = None,
    teacher_h: Tensor | None = None,
    weights: QADLossWeights | None = None,
    temperature: float = 1.0,
) -> tuple[Tensor, dict[str, float]]:
    """Combined QAD objective: top-k KL + optional hidden-MSE + optional LM cross-entropy."""
    w = weights or QADLossWeights()
    parts: dict[str, Tensor] = {
        "kl": w.kl * topk_kl_loss(student_logits, teacher_topk_values, teacher_topk_indices, temperature)
    }
    if student_h is not None and teacher_h is not None:
        parts["hidden"] = w.hidden * hidden_state_mse(student_h, teacher_h)
    if labels is not None:
        parts["ce"] = w.ce * F.cross_entropy(
            student_logits.reshape(-1, student_logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
        )
    total = sum(parts.values())
    return total, {k: float(v.detach()) for k, v in parts.items()}


# --- cloud-runner training loop (lazy heavy imports; not exercised by CPU tests) ---


def run_blockwise_qad(config: dict) -> None:  # pragma: no cover - requires model + GPU
    """Block-wise QAD entrypoint. See ``scripts/run_qad.py`` and ``configs/*.yaml``.

    Steps (executed on the H200 runner):
      1. load student (swapped via :func:`bite.quant.quantlinear.swap_linears`) + PTQ init,
      2. stream calibration/QAD batches; per block, minimize :func:`qad_loss` against the
         precomputed teacher targets while an 8-bit optimizer updates that block's latent
         weights (and, if enabled, distills router logits via :mod:`bite.moe.router`),
      3. checkpoint each healed block; evaluate with :mod:`bite.eval.harness`.
    """
    raise NotImplementedError(
        "run on the cloud runner via scripts/run_qad.py; see module docstring for the schedule"
    )
