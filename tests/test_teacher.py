import torch

from bite.train.qad import topk_kl_loss
from bite.train.teacher import topk_teacher_logits


def test_topk_shapes_and_values():
    logits = torch.randn(3, 5, 100)
    v, idx = topk_teacher_logits(logits, k=64)
    assert v.shape == (3, 5, 64)
    assert idx.shape == (3, 5, 64)
    # gathered values match the topk values
    assert torch.allclose(torch.gather(logits, -1, idx), v)


def test_k_clamped_to_vocab():
    logits = torch.randn(2, 8)
    v, idx = topk_teacher_logits(logits, k=64)
    assert v.shape[-1] == 8


def test_precomputed_targets_feed_qad_loss_to_zero():
    # teacher top-k of the teacher's own logits -> KL(student=teacher) == 0
    logits = torch.randn(4, 50)
    v, idx = topk_teacher_logits(logits, k=16)
    assert topk_kl_loss(logits.clone(), v, idx).item() < 1e-5
