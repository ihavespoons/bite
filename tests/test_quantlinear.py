import torch
from torch import nn

from bite.quant.policy import default_policy
from bite.quant.quantlinear import QuantLinear, swap_linears


class _Expert(nn.Module):
    def __init__(self, d=128, h=256):
        super().__init__()
        self.gate_proj = nn.Linear(d, h, bias=False)
        self.up_proj = nn.Linear(d, h, bias=False)
        self.down_proj = nn.Linear(h, d, bias=False)


class _MoE(nn.Module):
    def __init__(self, d=128, n_experts=4):
        super().__init__()
        self.gate = nn.Linear(d, n_experts, bias=False)  # router -> keep
        self.experts = nn.ModuleList(_Expert(d) for _ in range(n_experts))
        self.shared_expert = _Expert(d)  # -> keep


class _Block(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.o_proj = nn.Linear(d, d)
        self.mlp = _MoE(d)


def test_quantlinear_forward_shape_and_from_linear():
    lin = nn.Linear(128, 64)
    ql = QuantLinear.from_linear(lin, mode="ternary", group_size=128)
    assert torch.allclose(ql.weight, lin.weight)
    x = torch.randn(3, 128)
    assert ql(x).shape == (3, 64)


def test_quantlinear_gradient_flows_to_latent_weight():
    ql = QuantLinear(128, 64, mode="binary", group_size=128)
    nn.init.normal_(ql.weight)
    ql(torch.randn(3, 128)).sum().backward()
    assert ql.weight.grad is not None
    assert ql.weight.grad.abs().sum() > 0


def test_swap_keeps_router_and_shared_expert():
    model = _Block()
    swapped = swap_linears(model, default_policy("ternary"), group_size=128)

    # attention + routed-expert projections are swapped
    assert isinstance(model.q_proj, QuantLinear)
    assert isinstance(model.mlp.experts[0].gate_proj, QuantLinear)
    assert "mlp.experts.0.gate_proj" in swapped

    # router gate and shared expert stay full precision
    assert isinstance(model.mlp.gate, nn.Linear) and not isinstance(model.mlp.gate, QuantLinear)
    assert isinstance(model.mlp.shared_expert.gate_proj, nn.Linear)
    assert not isinstance(model.mlp.shared_expert.gate_proj, QuantLinear)
    assert "mlp.gate" not in swapped
    assert not any(k.startswith("mlp.shared_expert") for k in swapped)


def test_swapped_model_still_runs_forward():
    model = _Block()
    swap_linears(model, default_policy("ternary"), group_size=128)
    out = model.q_proj(torch.randn(2, 128))
    assert out.shape == (2, 128)
