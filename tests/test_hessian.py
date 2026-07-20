import torch

from bite.quant.hessian import HessianCollector, quant_linear_modules
from bite.quant.policy import default_policy
from bite.quant.quantlinear import QuantLinear, swap_linears
from torch import nn


def test_hessian_equals_gram_of_inputs():
    lin = QuantLinear(4, 3, bias=False, group_size=4)
    nn.init.normal_(lin.weight)
    x = torch.randn(10, 4)

    hc = HessianCollector().attach({"lin": lin})
    with torch.no_grad():
        lin(x)
    hc.detach()

    assert torch.allclose(hc.H["lin"], x.t() @ x, atol=1e-4)
    assert hc.tokens["lin"] == 10


def test_hessian_accumulates_across_batches():
    lin = QuantLinear(4, 3, bias=False, group_size=4)
    x1, x2 = torch.randn(6, 4), torch.randn(4, 4)
    with HessianCollector().attach({"lin": lin}) as hc:
        with torch.no_grad():
            lin(x1)
            lin(x2)
    expected = x1.t() @ x1 + x2.t() @ x2
    assert torch.allclose(hc.H["lin"], expected, atol=1e-4)
    assert hc.tokens["lin"] == 10


def test_quant_linear_modules_finds_swapped_layers():
    model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))
    swap_linears(model, default_policy("ternary"), group_size=8)
    mods = quant_linear_modules(model)
    assert len(mods) == 2
    assert all(isinstance(m, QuantLinear) for m in mods.values())
