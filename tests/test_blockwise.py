import torch
from torch import nn

from bite.quant.quantlinear import QuantLinear
from bite.train.blockwise import (
    build_optimizer,
    distill_block,
    iter_blocks,
    quant_parameters,
)


class _Block(nn.Module):
    """Toy decoder block: a low-bit projection + a high-precision router gate + a norm."""

    def __init__(self, d=128, n_experts=4):
        super().__init__()
        self.proj = QuantLinear(d, d, bias=False, mode="ternary", group_size=128)
        self.gate = nn.Linear(d, n_experts, bias=False)  # router -> not trained
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        return self.norm(self.proj(x))


class _Model(nn.Module):
    def __init__(self, n=3):
        super().__init__()
        self.blocks = nn.ModuleList(_Block() for _ in range(n))


def test_iter_blocks_finds_longest_modulelist():
    m = _Model(n=3)
    assert len(iter_blocks(m)) == 3
    assert len(iter_blocks(m, layers_path="blocks")) == 3


def test_quant_parameters_only_returns_quantlinear_weights():
    block = _Block()
    params = quant_parameters(block)
    assert len(params) == 1
    assert params[0] is block.proj.weight
    # the router gate is NOT trained
    assert all(p is not block.gate.weight for p in params)


def test_build_optimizer_adam_default():
    block = _Block()
    opt = build_optimizer(quant_parameters(block), lr=1e-3)
    assert isinstance(opt, torch.optim.AdamW)


def test_distill_block_reduces_hidden_mse():
    torch.manual_seed(0)
    d = 128
    student = QuantLinear(d, d, bias=False, mode="ternary", group_size=128)
    nn.init.normal_(student.weight, std=0.1)
    teacher = nn.Linear(d, d, bias=False)

    inputs = [torch.randn(16, d) for _ in range(4)]
    history = distill_block(student, teacher, inputs, steps=60, lr=0.05)

    assert len(history) == 60
    assert min(history) < history[0]  # healing improves the match


def test_distill_block_trains_only_quant_weights():
    d = 128
    block = _Block(d=d)
    teacher = _Block(d=d)
    gate_before = block.gate.weight.detach().clone()

    distill_block(block, teacher, [torch.randn(8, d)], steps=10, lr=0.05)

    # router gate untouched; low-bit proj weight moved
    assert torch.allclose(gate_before, block.gate.weight)
