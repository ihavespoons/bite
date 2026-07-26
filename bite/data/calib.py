"""Calibration/training data — stream text, tokenize, pack into fixed-length batches.

The packing core (:func:`batch_token_ids`) and the weighted document interleave
(:func:`interleave_weighted`) are pure and unit-tested on CPU; the dataset streaming
(:func:`stream_calibration`) is runner-side (needs ``datasets`` + the tokenizer).

For the slope run (fresh, diverse tokens — the e2e go/no-go showed token diversity is the
binding constraint), ``calibration.mixture`` in the config selects a weighted mixture of
streamed corpora instead of single-source c4::

    calibration:
      mixture:
        - { dataset: HuggingFaceFW/fineweb-edu, subset: sample-10BT, weight: 0.7 }
        - { dataset: allenai/c4, subset: en, weight: 0.3 }
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import torch


def interleave_weighted(streams: list[Iterator], weights: list[float]) -> Iterator:
    """Deterministically interleave item streams in long-run proportion to ``weights``.

    Smooth weighted round-robin (nginx-style): each step, every live stream accrues its
    weight as credit and the stream with the most credit emits one item and pays back the
    total. No RNG — the schedule is reproducible, which keeps teacher shards aligned with
    the exact documents they were computed on. A stream that ends is dropped; the rest
    continue at their relative weights.
    """
    if len(streams) != len(weights):
        raise ValueError(f"{len(streams)} streams but {len(weights)} weights")
    if any(w <= 0 for w in weights):
        raise ValueError(f"weights must be positive, got {weights}")
    live = list(zip(streams, [float(w) for w in weights], strict=True))
    credit = [0.0] * len(live)
    while live:
        total = sum(w for _, w in live)
        for i, (_, w) in enumerate(live):
            credit[i] += w
        i = max(range(len(live)), key=credit.__getitem__)
        try:
            yield next(live[i][0])
            credit[i] -= total
        except StopIteration:
            del live[i]
            del credit[i]


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


def _doc_texts(entry: dict) -> Iterator[str]:  # pragma: no cover - runner-side (datasets)
    """Stream non-empty document texts for one corpus entry (dataset/subset/text_field/skip_docs)."""
    from datasets import load_dataset

    ds_id = entry.get("dataset") or "allenai/c4"
    subset = entry.get("subset") or ("en" if ds_id == "allenai/c4" else None)
    split = entry.get("split", "train")
    if subset:
        ds = load_dataset(ds_id, subset, split=split, streaming=True)
    else:
        ds = load_dataset(ds_id, split=split, streaming=True)
    skip = int(entry.get("skip_docs") or 0)
    if skip:  # doc-level offset so later shard-generation jobs see fresh tokens
        ds = ds.skip(skip)
    field = entry.get("text_field", "text")
    for row in ds:
        text = row.get(field) or ""
        if text:
            yield text


def stream_calibration(  # pragma: no cover - runner-side (datasets + tokenizer)
    cfg: dict,
    tokenizer,
    *,
    device: str = "cuda",
    max_seqs: int | None = None,
) -> Iterator[dict]:
    """Yield tokenized calibration batches from the configured HF dataset(s).

    ``calibration.mixture`` (a list of ``{dataset, subset, weight, text_field, skip_docs}``
    entries) streams a weighted document mixture via :func:`interleave_weighted`; otherwise
    the single ``calibration.dataset`` corpus (default c4-en) is streamed. ``max_seqs`` caps
    the number of packed sequences for cheap validation runs.
    """
    calib = cfg["calibration"]
    seq_len = calib.get("seq_len", 2048)
    batch_size = cfg["eval"].get("batch_size", 8)

    mixture = calib.get("mixture")
    if mixture:
        docs = interleave_weighted(
            [_doc_texts(e) for e in mixture], [e.get("weight", 1.0) for e in mixture]
        )
    else:
        docs = _doc_texts({"dataset": calib.get("dataset"), "text_field": calib.get("text_field", "text")})

    def token_stream() -> Iterator[int]:
        emitted = 0
        for text in docs:
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
