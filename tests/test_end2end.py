import torch

from bite.train.end2end import end2end_loss, iter_shard_examples


def test_end2end_loss_positive_and_parts():
    torch.manual_seed(0)
    B, T, V, k = 2, 8, 50, 5
    logits = torch.randn(B, T, V, requires_grad=True)
    # teacher top-k drawn from a (different) random logit tensor
    teacher = torch.randn(B, T, V)
    values, indices = teacher.topk(k, dim=-1)
    input_ids = torch.randint(0, V, (B, T))

    loss, parts = end2end_loss(logits, values, indices, input_ids, w_kl=1.0, w_ce=0.5)
    assert loss.item() > 0
    assert set(parts) == {"kl", "ce"}
    loss.backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0


def test_end2end_loss_kl_zero_when_student_matches_teacher():
    B, T, V, k = 1, 4, 20, 4
    teacher_logits = torch.randn(B, T, V)
    values, indices = teacher_logits.topk(k, dim=-1)
    input_ids = torch.randint(0, V, (B, T))
    # student == teacher -> KL term ~ 0 (loss is then pure CE)
    loss_match, parts_match = end2end_loss(teacher_logits, values, indices, input_ids, w_ce=0.0)
    assert parts_match["kl"] < 1e-5
    assert abs(loss_match.item()) < 1e-4


def test_iter_shard_examples_splits_batches(tmp_path):
    import pytest

    st = pytest.importorskip("safetensors.torch")  # runner-side dep (model extra)

    B, T, k = 3, 6, 4
    st.save_file(
        {
            "input_ids": torch.randint(0, 100, (B, T)).to(torch.int32),
            "values": torch.randn(B, T, k).to(torch.float16),
            "indices": torch.randint(0, 100, (B, T, k)).to(torch.int32),
        },
        f"{tmp_path}/teacher_topk_000000.safetensors",
    )
    examples = list(iter_shard_examples(str(tmp_path)))
    assert len(examples) == B
    assert examples[0]["input_ids"].shape == (T,)
    assert examples[0]["values"].shape == (T, k)
