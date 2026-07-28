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


# --- exact MSE-optimal ternary ("optimal" rule) ---


def _mse(w, w_hat):
    return (w - w_hat).pow(2).mean().item()


def test_optimal_ternary_codes_and_shapes():
    w = torch.randn(4, 256)
    w_hat, codes, scales = quantize_ternary(w, group_size=128, threshold_ratio="optimal")
    assert w_hat.shape == codes.shape == w.shape
    assert scales.shape == (4, 2, 1)
    assert set(codes.unique().tolist()).issubset({-1.0, 0.0, 1.0})
    assert (scales > 0).all()


def test_optimal_ternary_matches_bruteforce_on_small_groups():
    # exhaustively check the closed form against brute force over all support sizes
    torch.manual_seed(3)
    w = torch.randn(1, 8)
    w_hat, _, _ = quantize_ternary(w, group_size=8, threshold_ratio="optimal")
    a, _ = w.abs().sort(descending=True)
    best = float("inf")
    for k in range(1, 9):
        scale = a[0, :k].mean()
        cand = torch.sign(w) * (w.abs() >= a[0, k - 1]).float() * scale
        best = min(best, _mse(w, cand))
    assert abs(_mse(w, w_hat) - best) < 1e-7


def test_optimal_ternary_beats_absmean_and_fixed_ratios():
    torch.manual_seed(0)
    for shape in [(8, 256), (4, 64, 128)]:  # 2D linear + 3D fused-expert layouts
        w = torch.randn(*shape)
        opt = _mse(w, quantize_ternary(w, 128, "optimal")[0])
        assert opt <= _mse(w, quantize_ternary(w, 128, None)[0]) + 1e-8
        for ratio in (0.5, 0.7, 0.9):
            assert opt <= _mse(w, quantize_ternary(w, 128, ratio)[0]) + 1e-8


def test_optimal_ternary_is_idempotent():
    # PTQ init writes w_hat into the latent; the next forward must reproduce it exactly
    torch.manual_seed(1)
    w = torch.randn(4, 256)
    w_hat, _, _ = quantize_ternary(w, 128, "optimal")
    w_hat2, _, _ = quantize_ternary(w_hat, 128, "optimal")
    assert torch.allclose(w_hat, w_hat2, atol=1e-6)


def test_optimal_ternary_ste_gradient_flows():
    w = torch.randn(4, 256, requires_grad=True)
    out = fake_quantize(w, "ternary", 128, threshold_ratio="optimal")
    out.sum().backward()
    assert torch.allclose(w.grad, torch.ones_like(w))


def test_optimal_ternary_all_zero_group_is_safe():
    w = torch.zeros(2, 128)
    w_hat, codes, scales = quantize_ternary(w, 128, "optimal")
    assert torch.isfinite(w_hat).all() and (codes == 0).all()


def test_optimal_ternary_chunked_matches_unchunked(monkeypatch):
    import bite.quant.fakequant as fq

    torch.manual_seed(2)
    w = torch.randn(6, 4, 256)  # 3D fused-expert layout
    ref = fq.quantize_ternary_optimal(w, 128)
    monkeypatch.setattr(fq, "_OPTIMAL_CHUNK_ELEMS", 1024)  # force ~1-row slices
    chunked = fq.quantize_ternary_optimal(w, 128)
    for r, c in zip(ref, chunked):
        assert torch.equal(r, c)


# --- uniform N-bit (the missing bitwidth curve: where is this model's cliff?) ---

from bite.quant.fakequant import parse_mode, quantize_uniform


def test_parse_mode():
    assert parse_mode("ternary") == ("ternary", 0)
    assert parse_mode("binary") == ("binary", 0)
    assert parse_mode("int4") == ("int", 4)
    for bad in ("int1", "int9", "fp8", "int", "nonsense"):
        try:
            parse_mode(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad!r}")


def test_uniform_code_alphabet_and_symmetry():
    torch.manual_seed(0)
    w = torch.randn(4, 256)
    for bits, qmax in ((2, 1), (3, 3), (4, 7), (8, 127)):
        w_hat, codes, scales = quantize_uniform(w, bits, 128)
        assert codes.abs().max() <= qmax, bits
        assert codes.eq(codes.round()).all()
        assert (scales > 0).all()
        assert w_hat.shape == w.shape


def test_uniform_error_falls_monotonically_with_bits():
    torch.manual_seed(0)
    w = torch.randn(8, 512)
    errs = [(w - quantize_uniform(w, b, 128)[0]).pow(2).mean().item() for b in (2, 3, 4, 6, 8)]
    assert all(a > b for a, b in zip(errs, errs[1:])), errs
    # 2-bit uniform (3 levels used symmetrically) should be in the ballpark of ternary
    tern = (w - quantize_ternary(w, 128)[0]).pow(2).mean().item()
    assert 0.2 < errs[0] / tern < 5.0


def test_uniform_absmax_scale_means_no_clipping():
    w = torch.randn(2, 128)
    w_hat, _, _ = quantize_uniform(w, 4, 128)
    assert w_hat.abs().max() <= w.abs().max() * 1.001  # absmax scaling never exceeds the input


def test_effective_bits_for_intn():
    assert abs(effective_bits("int4", 128) - (4 + 16 / 128)) < 1e-9
    assert abs(effective_bits("int2", 128) - (2 + 16 / 128)) < 1e-9
    assert effective_bits("ternary", 128) < effective_bits("int2", 128)


def test_fake_quantize_intn_ste_gradient():
    w = torch.randn(4, 128, requires_grad=True)
    fake_quantize(w, "int4", 128).sum().backward()
    assert torch.allclose(w.grad, torch.ones_like(w.grad))
