import torch
from torch import nn

from bite.quant.experts import expert_latent_params, install_expert_fakequant, ptq_init_experts
from bite.quant.quantlinear import QuantLinear
from bite.train.blockwise import (
    BlockInput,
    build_optimizer,
    capture_block_inputs,
    distill_block,
    forward_block,
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


class _MoEBlock(nn.Module):
    """Toy MoE block: a QuantLinear attention proj + fused 3D expert projections + a router gate."""

    def __init__(self, d=128, E=4, inter=64):
        super().__init__()
        self.attn = QuantLinear(d, d, bias=False, mode="ternary", group_size=128)
        self.gate_up_proj = nn.Parameter(torch.randn(E, d, 2 * inter))
        self.down_proj = nn.Parameter(torch.randn(E, inter, d))
        self.gate = nn.Parameter(torch.randn(d, E))  # 2D router -> not quantized/healed


def test_quant_parameters_includes_fused_expert_latents():
    block = _MoEBlock()
    install_expert_fakequant(block, mode="ternary", group_size=128)
    params = quant_parameters(block)
    # QuantLinear.weight + the two fused expert latents (gate_up_proj, down_proj)
    assert block.attn.weight in params
    latents = expert_latent_params(block)
    assert len(latents) == 2
    assert all(any(lp is p for p in params) for lp in latents)
    # the 2D router gate is NOT trained
    assert all(p is not block.gate for p in params)
    assert len(params) == 3


# --- per-block forward adapter (real blocks take kwargs and may return a tuple) ---


class _KwBlock(nn.Module):
    """Mimics a real decoder block: takes hidden + a rest arg + kwargs, returns a tuple."""

    def __init__(self, d=32):
        super().__init__()
        self.proj = QuantLinear(d, d, bias=False, mode="ternary", group_size=32)

    def forward(self, hidden, scale, *, mask, pos):  # note: kwargs like a real block
        return (self.proj(hidden) * scale + mask + pos, None)


def test_forward_block_threads_hidden_and_normalizes_tuple():
    block = _KwBlock()
    item = BlockInput(
        hidden=torch.randn(4, 32),
        rest=(1.0,),
        kwargs={"mask": torch.zeros(4, 32), "pos": torch.zeros(4, 32)},
    )
    out = forward_block(block, item)
    assert isinstance(out, torch.Tensor)  # tuple was unwrapped
    assert out.shape == (4, 32)


def test_distill_block_with_forward_block_adapter():
    torch.manual_seed(0)
    student = _KwBlock()
    nn.init.normal_(student.proj.weight, std=0.1)
    teacher = _KwBlock()
    kw = {"mask": torch.zeros(4, 32), "pos": torch.zeros(4, 32)}
    items = [BlockInput(torch.randn(4, 32), (1.0,), kw) for _ in range(3)]
    history = distill_block(student, teacher, items, steps=40, lr=0.01, forward_fn=forward_block)
    assert len(history) == 40
    assert min(history) < history[0]


def test_capture_block_inputs_records_signature():
    d = 32

    class _Emb(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList(_KwBlock(d) for _ in range(3))

        def forward(self, x, mask, pos):
            h = x
            for b in self.blocks:
                h = b(h, 1.0, mask=mask, pos=pos)[0]
            return h

    model = _Emb()
    blocks = list(model.blocks)
    batch = {"x": torch.randn(4, d), "mask": torch.zeros(4, d), "pos": torch.zeros(4, d)}
    h0, caps = capture_block_inputs(model, blocks, batch)
    assert h0.shape == (4, d)
    assert len(caps) == 3
    # rest arg (scale=1.0) captured positionally; mask/pos captured as kwargs
    rest, kwargs = caps[0]
    assert rest == (1.0,)
    assert set(kwargs) == {"mask", "pos"}
