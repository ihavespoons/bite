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


def load_teacher_shard(path: str) -> tuple[Tensor, Tensor, Tensor]:  # pragma: no cover - runner I/O
    """Load a shard as ``(input_ids, values, indices)`` — self-contained QAD training example."""
    import safetensors.torch as st

    d = st.load_file(path)
    return d["input_ids"], d["values"], d["indices"]


def precompute_teacher_logits(  # pragma: no cover - runner-side (teacher + data)
    teacher,
    batches,
    out_dir: str,
    *,
    k: int = 64,
    start_index: int = 0,
    log_every: int = 50,
) -> None:
    """Stream the QAD data through the frozen teacher; store (input_ids, top-k logits) shards.

    Each shard bundles the **input_ids** with the teacher's top-k ``values``/``indices`` so QAD is
    driven entirely by the shards — no re-streaming of c4, which isn't order-reproducible, so the
    targets always align with the exact tokens they were computed on. Shards go to ``out_dir`` and
    are uploaded to the HF dataset by ``run_ptq --push-repo``.
    """
    import os

    import safetensors.torch as st

    os.makedirs(out_dir, exist_ok=True)
    tokens = 0
    for i, batch in enumerate(batches):
        with torch.no_grad():
            logits = teacher(**batch).logits
        values, indices = topk_teacher_logits(logits, k)
        st.save_file(
            {
                "input_ids": batch["input_ids"].to(torch.int32).cpu(),
                "values": values.to(torch.float16).cpu(),
                "indices": indices.to(torch.int32).cpu(),
            },
            f"{out_dir}/teacher_topk_{start_index + i:06d}.safetensors",
        )
        tokens += batch["input_ids"].numel()
        if log_every and (i + 1) % log_every == 0:
            print(f"teacher shards: {i + 1} written ({tokens / 1e6:.1f}M tokens)", flush=True)
