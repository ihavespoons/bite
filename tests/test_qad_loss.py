import torch

from bite.train.qad import QADLossWeights, hidden_state_mse, qad_loss, topk_kl_loss


def _teacher_topk(logits, k):
    v, idx = logits.topk(k, dim=-1)
    return v, idx


def test_kl_zero_when_student_matches_teacher():
    torch.manual_seed(0)
    logits = torch.randn(4, 32)
    v, idx = _teacher_topk(logits, k=8)
    loss = topk_kl_loss(logits.clone(), v, idx)
    assert loss.item() < 1e-5


def test_kl_positive_when_student_differs():
    torch.manual_seed(0)
    teacher = torch.randn(4, 32)
    v, idx = _teacher_topk(teacher, k=8)
    loss = topk_kl_loss(torch.randn(4, 32), v, idx)
    assert loss.item() > 0


def test_hidden_mse_zero_on_identical():
    h = torch.randn(2, 8, 16)
    assert hidden_state_mse(h, h.clone()).item() == 0.0


def test_combined_loss_components_present_and_differentiable():
    torch.manual_seed(0)
    student = torch.randn(3, 5, 32, requires_grad=True)
    teacher = torch.randn(3, 5, 32)
    v, idx = _teacher_topk(teacher, k=8)
    labels = torch.randint(0, 32, (3, 5))
    sh = torch.randn(3, 5, 16, requires_grad=True)
    th = torch.randn(3, 5, 16)

    total, parts = qad_loss(
        student, v, idx, labels=labels, student_h=sh, teacher_h=th,
        weights=QADLossWeights(kl=1.0, hidden=0.5, ce=0.25),
    )
    assert {"kl", "hidden", "ce"} == set(parts)
    total.backward()
    assert student.grad is not None and sh.grad is not None
