import torch
from torch import nn
from torch.nn.utils import parametrize

from bite.quant.experts import (
    FakeQuantParam,
    apply_expert_mixed_precision,
    expert_latent_params,
    install_expert_fakequant,
    ptq_init_experts,
)
from bite.quant.fakequant import quantize_ternary


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


# --- expert-level mixed precision + whole-tensor keeps (sensitivity ablations) ---


def test_keep_indices_leaves_those_experts_untouched():
    """Kept expert slots must be bit-identical to the latent; others must be on the grid."""
    torch.manual_seed(0)
    E, d, inter = 8, 64, 32
    w = torch.randn(E, d, inter)
    fq = FakeQuantParam("ternary", group_size=32, keep_indices=(1, 5))
    out = fq(w)
    for e in range(E):
        if e in (1, 5):
            assert torch.equal(out[e], w[e]), f"expert {e} should be untouched"
        else:
            q, _, _ = quantize_ternary(w[e], 32)
            assert torch.allclose(out[e], q), f"expert {e} should be quantized"


def test_keep_indices_gradients_flow_to_both_paths():
    torch.manual_seed(1)
    w = torch.randn(4, 32, 32, requires_grad=True)
    out = FakeQuantParam("ternary", group_size=32, keep_indices=(0,))(w)
    out.sum().backward()
    # kept row: true gradient of identity (ones); quantized rows: STE (also ones) -> both nonzero
    assert w.grad is not None
    assert torch.allclose(w.grad[0], torch.ones_like(w.grad[0]))
    assert w.grad[1].abs().sum() > 0


def test_effective_bpw_mixed_precision_accounting():
    fq = FakeQuantParam("ternary", group_size=128, keep_indices=tuple(range(26)))  # ~10% of 256
    bpw = fq.effective_bpw(num_experts=256)
    # 10% at 16 bits + 90% at ~1.71 -> ~3.1 bpw; all-ternary is ~1.71
    assert 3.0 < bpw < 3.3
    assert FakeQuantParam("ternary", 128).effective_bpw(256) < 1.8


def test_install_keep_patterns_skips_whole_expert_tensors():
    class _Blk(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = nn.Parameter(torch.randn(4, 32, 64))
            self.down_proj = nn.Parameter(torch.randn(4, 32, 32))

    m = nn.Module()
    m.mlp = _Blk()
    installed = install_expert_fakequant(m, mode="ternary", group_size=32,
                                         keep_patterns=(r"down_proj",))
    assert installed["mlp.gate_up_proj"] == "ternary"
    assert installed["mlp.down_proj"] == "keep"
    assert parametrize.is_parametrized(m.mlp, "gate_up_proj")
    assert not parametrize.is_parametrized(m.mlp, "down_proj")


def test_install_keep_expert_indices_threaded_through():
    class _Blk(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = nn.Parameter(torch.randn(6, 32, 64))

    m = nn.Module()
    m.mlp = _Blk()
    install_expert_fakequant(m, mode="ternary", group_size=32, keep_expert_indices=(2, 3))
    fq = m.mlp.parametrizations.gate_up_proj[0]
    assert fq.keep_indices == (2, 3)
    latent = m.mlp.parametrizations.gate_up_proj.original
    assert torch.equal(m.mlp.gate_up_proj[2], latent[2])  # kept slot survives the round-trip


def test_apply_expert_mixed_precision_selects_worst_experts():
    """by='error' must keep the slots the ternary grid represents worst."""
    torch.manual_seed(0)

    class _Blk(nn.Module):
        def __init__(self):
            super().__init__()
            w = torch.randn(4, 32, 32) * 0.01
            w[2] = torch.randn(32, 32) * 5.0  # expert 2: high magnitude -> largest abs error
            self.gate_up_proj = nn.Parameter(w)

    m = nn.Module()
    m.mlp = _Blk()
    install_expert_fakequant(m, mode="ternary", group_size=32)
    counts = apply_expert_mixed_precision(m, frac=0.25, by="error")

    assert counts["mlp.gate_up_proj"] == 1
    fq = m.mlp.parametrizations.gate_up_proj[0]
    assert fq.keep_indices == (2,), f"expected the high-error expert, got {fq.keep_indices}"
    # and that slot now round-trips exactly
    latent = m.mlp.parametrizations.gate_up_proj.original
    assert torch.equal(m.mlp.gate_up_proj[2], latent[2])
    assert not torch.equal(m.mlp.gate_up_proj[0], latent[0])


def test_apply_expert_mixed_precision_first_is_a_control():
    class _Blk(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = nn.Parameter(torch.randn(8, 32, 32))

    m = nn.Module()
    m.mlp = _Blk()
    install_expert_fakequant(m, mode="ternary", group_size=32)
    apply_expert_mixed_precision(m, frac=0.25, by="first")
    assert m.mlp.parametrizations.gate_up_proj[0].keep_indices == (0, 1)


def test_apply_expert_mixed_precision_validates_frac():
    import pytest

    m = nn.Module()
    with pytest.raises(ValueError):
        apply_expert_mixed_precision(m, frac=0.0)
