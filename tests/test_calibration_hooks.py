import torch
from torch import nn

from bite.moe.calibration import (
    ExpertCoverage,
    attach_coverage_hooks,
    inverse_frequency_weights,
    sample_weights,
)


class _ToyMoE(nn.Module):
    def __init__(self, d=8, n_experts=4):
        super().__init__()
        self.gate = nn.Linear(d, n_experts, bias=False)          # router -> hooked
        self.experts_gate_proj = nn.Linear(d, 16, bias=False)     # ends in proj, ignored
        self.attn_gate = nn.Linear(d, d, bias=False)              # width != n_experts, ignored

    def forward(self, x):
        _ = self.gate(x)
        _ = self.experts_gate_proj(x)
        return self.attn_gate(x)


def test_coverage_hook_only_targets_router_gate():
    m = _ToyMoE(d=8, n_experts=4)
    cov = ExpertCoverage(num_experts=4, top_k=2)
    handles = attach_coverage_hooks(m, cov)
    assert len(handles) == 1  # only .gate with out_features == num_experts
    m(torch.randn(5, 8))
    assert cov.tokens == 5
    assert cov.counts.sum() == 5 * 2  # top_k assignments per token
    for h in handles:
        h.remove()


def test_inverse_frequency_weights_favor_rare_experts():
    counts = torch.tensor([100, 100, 100, 1])  # expert 3 is rare
    w = inverse_frequency_weights(counts)
    assert w[3] > w[0]


def test_sample_weights_prefer_samples_hitting_rare_experts():
    expert_w = inverse_frequency_weights(torch.tensor([100, 100, 100, 1]))
    weights = sample_weights([[0, 1], [2, 3]], expert_w)  # 2nd sample hits rare expert 3
    assert weights[1] > weights[0]
