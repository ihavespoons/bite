import torch

from bite.quant.fakequant import quantize_ternary, quantize_uniform
from bite.quant.offline import (
    hadamard_matrix,
    is_power_of_two,
    quantize_tensor_offline,
)


def test_hadamard_is_orthonormal():
    for n in (2, 8, 128, 2048):
        h = hadamard_matrix(n)
        assert torch.allclose(h @ h.T, torch.eye(n), atol=1e-5), n


def test_hadamard_rejects_non_power_of_two():
    import pytest

    with pytest.raises(ValueError):
        hadamard_matrix(96)
    assert is_power_of_two(2048) and not is_power_of_two(768)


def test_rotation_is_exactly_invertible_without_quantization():
    """Sanity on the identity the offline simulation relies on: q(WH)Hᵀ -> W when q == identity."""
    torch.manual_seed(0)
    w = torch.randn(4, 128)
    h = hadamard_matrix(128)
    assert torch.allclose((w @ h) @ h.T, w, atol=1e-5)


def test_offline_matches_fakequant_when_no_rotation_and_last_axis():
    torch.manual_seed(0)
    w = torch.randn(8, 256)
    got, rot = quantize_tensor_offline(w, mode="int4", group_size=128, rotate=False)
    assert not rot
    assert torch.allclose(got, quantize_uniform(w, 4, 128)[0], atol=1e-6)
    got_t, _ = quantize_tensor_offline(w, mode="ternary", group_size=128, rotate=False)
    assert torch.allclose(got_t, quantize_ternary(w, 128)[0], atol=1e-6)


def test_axis_selection_equals_manual_transpose():
    """Quantizing a 3D tensor along dim 1 must equal transposing, quantizing last, transposing back."""
    torch.manual_seed(1)
    w = torch.randn(4, 256, 64)  # (E, contract, out) like a fused expert tensor
    got, _ = quantize_tensor_offline(w, mode="int4", group_size=128, axis=1, rotate=False)
    manual = quantize_uniform(w.transpose(1, -1), 4, 128)[0].transpose(1, -1)
    assert torch.equal(got, manual)
    # and it must DIFFER from grouping the last dim (that was the bug being fixed)
    last_axis, _ = quantize_tensor_offline(w, mode="int4", group_size=64, axis=-1, rotate=False)
    assert not torch.allclose(got, last_axis)


def _mse(w, wh):
    return (w - wh).pow(2).mean().item()


def test_rotation_helps_ternary_on_heavy_tailed_weights():
    """Rotation's real win: heavy-tailed sources under a ternary grid (measured 0.5-0.7x error).

    Real LLM weights are heavy-tailed rather than Gaussian, which is why this is the case that
    matters. Ternary + absmean is badly hurt by a few large entries; spreading them helps.
    """
    torch.manual_seed(0)
    for w in (
        torch.distributions.Laplace(0.0, 1.0).sample((16, 512)),
        torch.distributions.StudentT(3.0).sample((16, 512)),
    ):
        plain, _ = quantize_tensor_offline(w, mode="ternary", group_size=128, rotate=False)
        rot, did = quantize_tensor_offline(w, mode="ternary", group_size=128, rotate=True)
        assert did
        assert _mse(w, rot) < _mse(w, plain), (_mse(w, rot), _mse(w, plain))


def test_rotation_is_not_universally_better_at_2bit():
    """Guards a real trap: a SPARSE grid matches a sparse-spiky source, and rotation destroys
    that. Measured ~1.9x WORSE for int2 on spiky weights — so rotation must be validated per
    (mode, distribution), not assumed. This is why QuIP#/QTIP pair rotation with lattice
    codebooks matched to the rotated Gaussian, not with scalar grids."""
    torch.manual_seed(0)
    w = torch.randn(16, 512) * 0.02
    w[:, ::64] += torch.randn(16, 8) * 2.0
    plain, _ = quantize_tensor_offline(w, mode="int2", group_size=128, rotate=False)
    rot, _ = quantize_tensor_offline(w, mode="int2", group_size=128, rotate=True)
    assert _mse(w, rot) > _mse(w, plain)


def test_rotation_skipped_gracefully_on_non_power_of_two():
    w = torch.randn(2, 96)
    got, did = quantize_tensor_offline(w, mode="int4", group_size=32, rotate=True)
    assert not did and got.shape == w.shape and torch.isfinite(got).all()


def test_dtype_and_shape_preserved():
    w = torch.randn(4, 128, dtype=torch.bfloat16)
    got, _ = quantize_tensor_offline(w, mode="int3", group_size=128, rotate=True)
    assert got.dtype == torch.bfloat16 and got.shape == w.shape


# --- model-level walker: the real model has heterogeneous shapes and a vision tower ---


class _Odd(torch.nn.Module):
    """Mimics the awkward parts of the real model that killed the first offline sweep."""

    def __init__(self):
        super().__init__()
        # names mirror the real model: the router is `...mlp.gate`, so the keep pattern
        # `\.gate$` needs the dot — a flat `gate` attribute would not match, and did not.
        self.self_attn = torch.nn.Module()
        self.self_attn.q_proj = torch.nn.Linear(256, 128, bias=False)   # divisible -> quantized
        self.self_attn.odd_proj = torch.nn.Linear(4304, 64, bias=False)  # 4304 % 128 -> skipped
        self.mlp = torch.nn.Module()
        self.mlp.gate = torch.nn.Linear(256, 8, bias=False)             # router -> kept
        self.mlp.gate_up_proj = torch.nn.Parameter(torch.randn(4, 256, 128))  # (E, contract, out)
        self.mlp.down_proj = torch.nn.Parameter(torch.randn(4, 100, 256))     # 100 % 128 -> skip
        self.visual = torch.nn.Module()
        self.visual.patch = torch.nn.Linear(256, 256, bias=False)       # vision -> excluded


def test_model_walker_skips_vision_odd_shapes_and_keeps():
    from bite.quant.offline import quantize_model_offline

    torch.manual_seed(0)
    m = _Odd()
    before = {
        "odd": m.self_attn.odd_proj.weight.detach().clone(),
        "gate": m.mlp.gate.weight.detach().clone(),
        "vis": m.visual.patch.weight.detach().clone(),
        "q": m.self_attn.q_proj.weight.detach().clone(),
        "down": m.mlp.down_proj.detach().clone(),
    }
    stats = quantize_model_offline(
        m, mode="int4", group_size=128, keep_patterns=(r"\.gate$",), verbose=False
    )

    # quantized: q_proj + gate_up_proj
    assert stats["linears"] == 1 and stats["experts"] == 1
    assert not torch.equal(m.self_attn.q_proj.weight, before["q"])
    # skipped for the right reasons, and left BIT-IDENTICAL
    assert stats["skipped_indivisible"] == 2, stats      # odd_proj + down_proj
    assert stats["skipped"] == 1, stats                  # router gate
    assert stats["skipped_vision"] >= 1, stats           # visual.*
    assert torch.equal(m.self_attn.odd_proj.weight, before["odd"])
    assert torch.equal(m.mlp.gate.weight, before["gate"])
    assert torch.equal(m.visual.patch.weight, before["vis"])
    assert torch.equal(m.mlp.down_proj, before["down"])


def test_model_walker_rotation_counts_only_power_of_two_axes():
    from bite.quant.offline import quantize_model_offline

    m = _Odd()
    stats = quantize_model_offline(
        m, mode="ternary", group_size=128, rotate=True, keep_patterns=(r"\.gate$",), verbose=False
    )
    assert stats["linears_rotated"] == 1 and stats["experts_rotated"] == 1
