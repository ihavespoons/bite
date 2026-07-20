import torch

from bite.quant.fakequant import (
    effective_bits,
    fake_quantize,
    quantize_binary,
    quantize_ternary,
)


def test_ternary_codes_are_in_alphabet():
    w = torch.randn(4, 256)
    _, codes, _ = quantize_ternary(w, group_size=128)
    assert codes.shape == w.shape
    assert set(codes.unique().tolist()).issubset({-1.0, 0.0, 1.0})


def test_binary_codes_are_in_alphabet():
    w = torch.randn(4, 256)
    _, codes, _ = quantize_binary(w, group_size=128)
    assert codes.shape == w.shape
    assert set(codes.unique().tolist()).issubset({-1.0, 1.0})


def test_binary_sign_zero_maps_to_plus_one():
    w = torch.zeros(1, 128)
    _, codes, _ = quantize_binary(w, group_size=128)
    assert torch.all(codes == 1.0)


def test_group_scale_is_absmean_and_exact_on_constant_groups():
    # two groups of 128 with constant magnitudes -> sign*absmean reconstructs exactly
    w = torch.cat([torch.full((1, 128), 2.0), torch.full((1, 128), -4.0)], dim=1)
    w_hat, codes, scales = quantize_binary(w, group_size=128)
    assert torch.allclose(scales.flatten(), torch.tensor([2.0, 4.0]))
    assert torch.allclose(w_hat, w)
    assert torch.allclose(codes[0, :128], torch.ones(128))
    assert torch.allclose(codes[0, 128:], -torch.ones(128))


def test_ternary_reconstruction_beats_zero():
    w = torch.randn(8, 512)
    w_hat, _, _ = quantize_ternary(w, group_size=128)
    assert (w - w_hat).norm() < w.norm()


def test_ternary_threshold_variant_produces_zeros():
    w = torch.randn(4, 128)
    _, codes, _ = quantize_ternary(w, group_size=128, threshold_ratio=0.75)
    assert (codes == 0.0).any()
    assert set(codes.unique().tolist()).issubset({-1.0, 0.0, 1.0})


def test_ste_forward_equals_quantized():
    w = torch.randn(4, 256)
    q = fake_quantize(w, mode="ternary", group_size=128)
    w_hat, _, _ = quantize_ternary(w, group_size=128)
    assert torch.allclose(q, w_hat)


def test_ste_gradient_is_identity_without_clip():
    w = torch.randn(4, 256, requires_grad=True)
    fake_quantize(w, mode="binary", group_size=128).sum().backward()
    assert torch.allclose(w.grad, torch.ones_like(w))


def test_ste_clip_zeroes_out_of_range_gradients():
    # one huge entry per group is far outside |w| <= scale -> its grad is zeroed
    w = torch.randn(2, 128)
    w[:, 0] = 100.0
    w = w.detach().requires_grad_(True)
    fake_quantize(w, mode="binary", group_size=128, clip_ste=True).sum().backward()
    assert torch.all(w.grad[:, 0] == 0.0)
    assert w.grad.sum() > 0  # in-range entries still pass gradient


def test_non_divisible_group_size_raises():
    w = torch.randn(4, 100)
    try:
        quantize_ternary(w, group_size=128)
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-divisible group_size")


def test_effective_bits_matches_whitepaper():
    assert abs(effective_bits("ternary", 128) - 1.71) < 0.01
    assert abs(effective_bits("binary", 128) - 1.125) < 0.01
