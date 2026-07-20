import torch

from bite.export.gguf import pack_quantlinear
from bite.quant.quantlinear import QuantLinear


def test_pack_ternary_shapes_and_alphabet():
    ql = QuantLinear(256, 64, bias=True, mode="ternary", group_size=128)
    torch.nn.init.normal_(ql.weight)
    packed = pack_quantlinear(ql)
    assert packed["codes"].shape == (64, 256)
    assert packed["codes"].dtype == torch.int8
    assert set(packed["codes"].unique().tolist()).issubset({-1, 0, 1})
    assert packed["scales"].shape == (64, 256 // 128)  # one scale per group
    assert packed["scales"].dtype == torch.float16
    assert packed["bias"].shape == (64,)


def test_pack_binary_alphabet():
    ql = QuantLinear(128, 32, bias=False, mode="binary", group_size=128)
    torch.nn.init.normal_(ql.weight)
    packed = pack_quantlinear(ql)
    assert set(packed["codes"].unique().tolist()).issubset({-1, 1})
    assert "bias" not in packed
