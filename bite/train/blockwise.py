"""Block-wise QAD engine — heal one transformer block at a time.

The schedule that fits a single H200: walk the decoder blocks in order; for each, optimize
only that block's low-bit latent weights to match the FP16 teacher block's output on the
running (already-healed) activations, so just one block's optimizer state is resident. A final
optional end-to-end logit-KL polish (8-bit optimizer) uses the precomputed teacher top-k logits.

The core pieces — block discovery, trainable-param selection (which auto-freezes router,
shared expert and norms since only :class:`QuantLinear` weights are trained), optimizer
construction, and the per-block distill step — are pure PyTorch and unit-tested on CPU. The
full-model orchestration is runner-side.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from bite.quant.quantlinear import QuantLinear


def iter_blocks(model: nn.Module, layers_path: str | None = None) -> list[nn.Module]:
    """Return the decoder blocks. Uses ``layers_path`` (e.g. ``model.layers``) if given,
    else the longest :class:`nn.ModuleList` in the model."""
    if layers_path:
        obj = model
        for part in layers_path.split("."):
            obj = getattr(obj, part)
        return list(obj)
    best: list[nn.Module] = []
    for _, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > len(best):
            best = list(module)
    return best


def quant_parameters(module: nn.Module) -> list[nn.Parameter]:
    """The trainable low-bit latent weights in a block — i.e. only ``QuantLinear.weight``.

    Everything else (router gate, shared expert, RMSNorm, DeltaNet params, biases) is left out,
    which is exactly the precision policy: those stay in higher precision and are not healed.
    """
    return [m.weight for m in module.modules() if isinstance(m, QuantLinear)]


def build_optimizer(params: list[nn.Parameter], *, lr: float = 1e-4, kind: str = "adam"):
    """Adam optimizer; ``kind='adam8bit'`` uses bitsandbytes for the end-to-end polish pass."""
    if kind == "adam8bit":
        import bitsandbytes as bnb

        return bnb.optim.Adam8bit(params, lr=lr)
    return torch.optim.AdamW(params, lr=lr)


def distill_block(
    student_block: nn.Module,
    teacher_block: nn.Module,
    inputs: Iterable[Tensor],
    *,
    steps: int = 100,
    lr: float = 1e-4,
    optimizer: str = "adam",
    forward_fn: Callable[[nn.Module, Tensor], Tensor] = lambda b, x: b(x),
) -> list[float]:
    """Heal ``student_block`` to match ``teacher_block`` outputs on ``inputs`` (hidden-MSE).

    Only the block's :class:`QuantLinear` latent weights train (via STE). Returns the loss
    history so the caller can gate/early-stop. ``forward_fn`` adapts to real decoder-block
    call signatures (hidden states + position/mask); the default calls ``block(x)``.
    """
    params = quant_parameters(student_block)
    if not params:
        return []
    opt = build_optimizer(params, lr=lr, kind=optimizer)
    teacher_block.eval()
    for p in teacher_block.parameters():
        p.requires_grad_(False)

    batches = list(inputs)
    history: list[float] = []
    for step in range(steps):
        x = batches[step % len(batches)]
        opt.zero_grad()
        with torch.no_grad():
            target = forward_fn(teacher_block, x)
        out = forward_fn(student_block, x)
        loss = F.mse_loss(out.float(), target.float())
        loss.backward()
        opt.step()
        history.append(float(loss.detach()))
    return history


def run_blockwise_qad(config: dict) -> None:  # pragma: no cover - requires model + GPU
    """Full-model block-wise QAD orchestration (runner-side).

    Steps on the H200 runner:
      1. load the PTQ-initialized student (Stage 2) and a source of FP16 teacher block weights;
      2. ``h = embed(batch)``; for each ``(student_block, teacher_block)`` in
         :func:`iter_blocks` order, run :func:`distill_block` (loading the teacher block's FP16
         weights just for that step), checkpoint the healed block, then advance ``h`` through the
         healed student block so later blocks see realistic (quant-accumulated) inputs;
      3. optional end-to-end polish: unfreeze all blocks, minimize
         :func:`bite.train.qad.qad_loss` against the precomputed teacher top-k logits with an
         ``adam8bit`` optimizer;
      4. evaluate with :mod:`bite.eval.harness`; gate on ternary ≥ ~93% of FP16.
    """
    raise NotImplementedError(
        "run on the cloud runner via scripts/run_qad.py; see docstring for the block schedule"
    )
