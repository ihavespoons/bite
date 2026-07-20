import torch
from torch import nn

from bite.quant.hessian import HessianCollector, quant_linear_modules
from bite.quant.policy import default_policy
from bite.quant.ptq import ptq_init_model
from bite.quant.quantlinear import QuantLinear, swap_linears


def _toy():
    # in_features must be divisible by group_size (128); grouping is along in_features
    model = nn.Sequential(nn.Linear(128, 256, bias=False), nn.Linear(256, 128, bias=False))
    swap_linears(model, default_policy("ternary"), group_size=128)
    return model


def test_ptq_init_puts_latent_weights_on_ternary_grid():
    model = _toy()
    for m in model:
        nn.init.normal_(m.weight)
    done = ptq_init_model(model, hessians=None)
    assert set(done) == {"0", "1"}

    # each latent weight is now scale * {-1,0,1} per group
    for m in quant_linear_modules(model).values():
        w = m.weight.data
        scale = w.abs().reshape(w.shape[0], -1, 128).amax(-1, keepdim=True).clamp_min(1e-8)
        ratio = torch.round(w.reshape(w.shape[0], -1, 128) / scale)
        assert set(ratio.unique().tolist()).issubset({-1.0, 0.0, 1.0})


def test_ptq_with_hessian_runs_and_changes_weights():
    model = _toy()
    for m in model:
        nn.init.normal_(m.weight)
    before = model[0].weight.data.clone()

    hc = HessianCollector().attach(quant_linear_modules(model))
    with torch.no_grad():
        model(torch.randn(32, 128))
    hc.detach()

    ptq_init_model(model, hc.H)
    assert not torch.allclose(before, model[0].weight.data)  # quantized in place
