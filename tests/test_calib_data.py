import torch

from bite.data.calib import batch_token_ids


def test_packs_into_fixed_shape_batches():
    batches = list(batch_token_ids(range(4 * 8 * 2), seq_len=8, batch_size=4))
    assert len(batches) == 2
    assert batches[0]["input_ids"].shape == (4, 8)
    assert batches[0]["input_ids"].dtype == torch.int64


def test_drops_trailing_partial_sequence_keeps_short_batch():
    # 3 full seqs of len 4 (+3 leftover tokens dropped) -> one short batch of 3
    batches = list(batch_token_ids(range(3 * 4 + 3), seq_len=4, batch_size=8))
    assert len(batches) == 1
    assert batches[0]["input_ids"].shape == (3, 4)


def test_contiguous_ids_preserved():
    b = next(batch_token_ids(range(16), seq_len=4, batch_size=2))
    assert b["input_ids"][0].tolist() == [0, 1, 2, 3]
    assert b["input_ids"][1].tolist() == [4, 5, 6, 7]


# --- weighted document interleave (slope-run mixture) ---

from bite.data.calib import interleave_weighted


def test_interleave_weighted_proportions():
    a = iter(["a"] * 1000)
    b = iter(["b"] * 1000)
    c = iter(["c"] * 1000)
    out = []
    it = interleave_weighted([a, b, c], [0.6, 0.25, 0.15])
    for _ in range(400):
        out.append(next(it))
    # long-run proportions track the weights (within a couple of scheduling quanta)
    assert abs(out.count("a") / 400 - 0.60) < 0.02
    assert abs(out.count("b") / 400 - 0.25) < 0.02
    assert abs(out.count("c") / 400 - 0.15) < 0.02


def test_interleave_weighted_is_deterministic():
    def make():
        return interleave_weighted([iter(range(0, 50)), iter(range(100, 150))], [2.0, 1.0])

    assert list(make()) == list(make())


def test_interleave_weighted_exhausted_stream_drops_out():
    out = list(interleave_weighted([iter(["a"] * 3), iter(["b"] * 9)], [1.0, 1.0]))
    assert len(out) == 12
    assert out.count("a") == 3 and out.count("b") == 9
    # after 'a' runs dry the tail is all 'b'
    assert out[-6:] == ["b"] * 6


def test_interleave_weighted_validates_inputs():
    import pytest

    with pytest.raises(ValueError):
        list(interleave_weighted([iter([1])], [1.0, 2.0]))
    with pytest.raises(ValueError):
        list(interleave_weighted([iter([1])], [0.0]))
