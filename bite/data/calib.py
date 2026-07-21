"""Calibration data for Stage 2 — stream text, tokenize, pack into fixed-length batches.

The packing core (:func:`batch_token_ids`) is pure and unit-tested on CPU; the dataset
streaming (:func:`stream_calibration`) is runner-side (needs ``datasets`` + the tokenizer).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import torch


def batch_token_ids(
    ids: Iterable[int], seq_len: int, batch_size: int, *, device: str = "cpu"
) -> Iterator[dict]:
    """Pack a flat stream of token ids into ``{"input_ids": [B, seq_len]}`` batches.

    Drops a trailing partial *sequence*; emits a final short *batch* if sequences remain.
    """
    seqs: list[list[int]] = []
    buf: list[int] = []
    for tid in ids:
        buf.append(int(tid))
        if len(buf) == seq_len:
            seqs.append(buf)
            buf = []
            if len(seqs) == batch_size:
                yield {"input_ids": torch.tensor(seqs, device=device)}
                seqs = []
    if seqs:
        yield {"input_ids": torch.tensor(seqs, device=device)}


def stream_calibration(  # pragma: no cover - runner-side (datasets + tokenizer)
    cfg: dict,
    tokenizer,
    *,
    device: str = "cuda",
    max_seqs: int | None = None,
) -> Iterator[dict]:
    """Yield tokenized calibration batches from the configured HF dataset.

    Defaults to a diverse streamed corpus so the 256 experts get broad coverage. ``max_seqs``
    caps the number of packed sequences for cheap validation runs.
    """
    from datasets import load_dataset

    ds_id = cfg["calibration"].get("dataset") or "allenai/c4"
    seq_len = cfg["calibration"].get("seq_len", 2048)
    batch_size = cfg["eval"].get("batch_size", 8)
    text_field = cfg["calibration"].get("text_field", "text")

    if ds_id == "allenai/c4":
        ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    else:
        ds = load_dataset(ds_id, split="train", streaming=True)

    def token_stream() -> Iterator[int]:
        emitted = 0
        for row in ds:
            text = row.get(text_field) or ""
            if not text:
                continue
            for tid in tokenizer(text).input_ids:
                yield tid
                emitted += 1
            if max_seqs and emitted >= max_seqs * seq_len:
                return

    n = 0
    for batch in batch_token_ids(token_stream(), seq_len, batch_size, device=device):
        yield batch
        n += batch["input_ids"].shape[0]
        if max_seqs and n >= max_seqs:
            return
