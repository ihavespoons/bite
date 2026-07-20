import torch
from torch import nn

from bite.moe.router import freeze_router, router_distill_loss, top_k_agreement


def test_freeze_router_targets_only_gate():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = nn.Linear(8, 4)
            self.q_proj = nn.Linear(8, 8)

    m = M()
    frozen = freeze_router(m)
    assert frozen == ["gate"]
    assert not m.gate.weight.requires_grad
    assert m.q_proj.weight.requires_grad


def test_router_distill_zero_when_identical():
    logits = torch.randn(5, 16)
    assert router_distill_loss(logits.clone(), logits.clone()).item() < 1e-6


def test_top_k_agreement_perfect_and_partial():
    logits = torch.randn(10, 16)
    assert top_k_agreement(logits.clone(), logits.clone(), k=8) == 1.0
    # fully shuffled logits -> agreement below 1
    assert top_k_agreement(torch.randn(10, 16), logits, k=4) < 1.0
