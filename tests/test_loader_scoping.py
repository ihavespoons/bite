import torch
from torch import nn

from bite.models.loader import find_vision_prefixes, swap_language_model
from bite.quant.policy import default_policy
from bite.quant.quantlinear import QuantLinear


class _Vision(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.blocks = nn.ModuleList(nn.Linear(d, d) for _ in range(3))
        self.proj = nn.Linear(d, 128)


class _Attn(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.q_proj = nn.Linear(d, d)
        self.o_proj = nn.Linear(d, d)


class _MoE(nn.Module):
    def __init__(self, d=128, n=4):
        super().__init__()
        self.gate = nn.Linear(d, n)               # router -> keep
        self.experts = nn.ModuleList(nn.Linear(d, d) for _ in range(n))
        self.shared_expert = nn.Linear(d, d)      # -> keep


class _Block(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.self_attn = _Attn(d)
        self.mlp = _MoE(d)


class _LM(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.layers = nn.ModuleList(_Block(d) for _ in range(2))


class _MM(nn.Module):
    """Multimodal: vision tower + language model + head, like the real Qwen VL model."""

    def __init__(self, d=128):
        super().__init__()
        self.visual = _Vision()
        self.model = _LM(d)
        self.lm_head = nn.Linear(d, 1000)


def test_find_vision_prefixes():
    assert find_vision_prefixes(_MM()) == ("visual",)


def test_swap_excludes_vision_tower():
    mm = _MM()
    swap_language_model(mm, default_policy("ternary"), group_size=128)

    # language weights swapped
    assert isinstance(mm.model.layers[0].self_attn.q_proj, QuantLinear)
    assert isinstance(mm.model.layers[0].mlp.experts[0], QuantLinear)
    assert isinstance(mm.lm_head, QuantLinear)

    # vision tower untouched (held at 4-bit HQQ separately)
    assert not isinstance(mm.visual.proj, QuantLinear)
    assert all(not isinstance(b, QuantLinear) for b in mm.visual.blocks)

    # router gate + shared expert kept high-precision
    assert not isinstance(mm.model.layers[0].mlp.gate, QuantLinear)
    assert not isinstance(mm.model.layers[0].mlp.shared_expert, QuantLinear)


def test_swapped_multimodal_language_path_runs():
    mm = _MM()
    swap_language_model(mm, default_policy("binary"), group_size=128)
    out = mm.lm_head(mm.model.layers[0].self_attn.q_proj(torch.randn(2, 128)))
    assert out.shape == (2, 1000)
