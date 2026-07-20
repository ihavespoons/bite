"""Teacher-logit precompute — the memory lever that removes the resident teacher from QAD.

Running the FP16 teacher once over the QAD data and storing only the **top-k logits per token**
(values + indices) lets Stage 3 distill against those tensors instead of keeping a second 35B
model resident — which is what makes block-wise QAD fit a single H200. The top-k extractor is a
pure function (unit-tested); the storage driver is runner-side (needs the teacher + data).
"""

from __future__ import annotations

import torch
from torch import Tensor


def topk_teacher_logits(logits: Tensor, k: int = 64) -> tuple[Tensor, Tensor]:
    """Return ``(values, indices)`` of the top-``k`` logits along the vocab dim.

    Consumed by :func:`bite.train.qad.topk_kl_loss`. ``k`` is clamped to the vocab size.
    """
    k = min(k, logits.shape[-1])
    values, indices = logits.topk(k, dim=-1)
    return values, indices


def precompute_teacher_logits(  # pragma: no cover - runner-side (teacher + data)
    teacher,
    batches,
    out_dir: str,
    *,
    k: int = 64,
) -> None:
    """Stream the QAD data through the frozen teacher; store top-k logits as shards.

    Runner-side: for each batch, forward the teacher under ``torch.no_grad``, take
    :func:`topk_teacher_logits`, and write ``values``/``indices`` (fp16/int32) to
    ``out_dir`` as safetensors shards keyed by batch id — later loaded as an HF dataset by the
    QAD loop. Store on the HF Pro 1TB quota.
    """
    import safetensors.torch as st

    for i, batch in enumerate(batches):
        with torch.no_grad():
            logits = teacher(**batch).logits
        values, indices = topk_teacher_logits(logits, k)
        st.save_file(
            {"values": values.to(torch.float16).cpu(), "indices": indices.to(torch.int32).cpu()},
            f"{out_dir}/teacher_topk_{i:06d}.safetensors",
        )
