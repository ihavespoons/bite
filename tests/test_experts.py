import torch
from torch import nn
from torch.nn.utils import parametrize

from bite.quant.experts import (
    expert_latent_params,
    install_expert_fakequant,
    ptq_init_experts,
)


class _ToyExperts(nn.Module):
    """Mimics a fused MoE block: 3D expert projections + a 3D conv + a 2D router gate."""

    def __init__(self, E=4, hidden=128, inter=64):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(E, hidden, 2 * inter))
        self.down_proj = nn.Parameter(torch.randn(E, inter, hidden))
        self.conv1d = nn.Parameter(torch.randn(E, 4, 4))  # 3D but not an expert proj -> skip
        self.gate = nn.Parameter(torch.randn(hidden, E))  # 2D router -> skip

    def forward(self, x):  # x: (E, tokens, hidden)
        return torch.bmm(x, self.gate_up_proj)


def _on_ternary_grid(w, group_size):
    wg = w.reshape(*w.shape[:-1], w.shape[-1] // group_size, group_size)
    scale = wg.abs().amax(-1, keepdim=True).clamp_min(1e-8)
    ratio = torch.round(wg / scale)
    return set(ratio.unique().tolist()).issubset({-1.0, 0.0, 1.0})


def test_install_targets_only_3d_expert_projections():
    m = _ToyExperts()
    installed = install_expert_fakequant(m, mode="ternary", group_size=128)
    # keys are "<module>.<param>"; toy is the root so names are ".<param>"
    assert {k.rsplit(".", 1)[-1] for k in installed} == {"gate_up_proj", "down_proj"}
    assert parametrize.is_parametrized(m, "gate_up_proj")
    assert not parametrize.is_parametrized(m, "conv1d")


def test_forward_access_returns_quantized_weights():
    m = _ToyExperts()
    install_expert_fakequant(m, mode="ternary", group_size=128)
    assert _on_ternary_grid(m.gate_up_proj, 128)
    assert _on_ternary_grid(m.down_proj, 128)


def test_gradient_flows_to_latent_expert_param():
    m = _ToyExperts()
    install_expert_fakequant(m, mode="binary", group_size=128)
    m(torch.randn(4, 8, 128)).sum().backward()
    latent = m.parametrizations.gate_up_proj.original
    assert latent.grad is not None and latent.grad.abs().sum() > 0


def test_ptq_init_experts_sets_latent_on_grid():
    m = _ToyExperts()
    install_expert_fakequant(m, mode="ternary", group_size=128)
    assert ptq_init_experts(m) == 2
    assert _on_ternary_grid(m.parametrizations.gate_up_proj.original, 128)


def test_expert_latent_params_collected():
    m = _ToyExperts()
    install_expert_fakequant(m, mode="ternary", group_size=128)
    assert len(expert_latent_params(m)) == 2
