import torch

from bite.quant.fakequant import quantize_binary, quantize_ternary
from bite.quant.ptq import gptq_quantize, group_absmean_scales


def test_identity_hessian_matches_plain_ternary_rounding():
    torch.manual_seed(0)
    w = torch.randn(8, 128)
    w_hat_gptq, _ = gptq_quantize(w, hessian=None, mode="ternary", group_size=128, percdamp=0.0)
    w_hat_plain, _, _ = quantize_ternary(w, group_size=128)
    assert torch.allclose(w_hat_gptq, w_hat_plain, atol=1e-5)


def test_identity_hessian_matches_plain_binary_rounding():
    torch.manual_seed(0)
    w = torch.randn(8, 128)
    w_hat_gptq, _ = gptq_quantize(w, hessian=None, mode="binary", group_size=128, percdamp=0.0)
    w_hat_plain, _, _ = quantize_binary(w, group_size=128)
    assert torch.allclose(w_hat_gptq, w_hat_plain, atol=1e-5)


def test_gptq_reduces_output_error_vs_naive_under_real_hessian():
    torch.manual_seed(0)
    out_f, in_f, n_cal = 16, 128, 512
    w = torch.randn(out_f, in_f)
    x = torch.randn(in_f, n_cal)
    hessian = x @ x.t()

    w_gptq, _ = gptq_quantize(w, hessian=hessian, mode="ternary", group_size=128)
    w_naive, _, _ = quantize_ternary(w, group_size=128)

    err_gptq = ((w - w_gptq) @ x).norm()
    err_naive = ((w - w_naive) @ x).norm()
    assert err_gptq < err_naive


def test_codes_stay_in_alphabet_after_gptq():
    w = torch.randn(8, 128)
    scales = group_absmean_scales(w, 128)
    w_hat, out_scales = gptq_quantize(w, mode="ternary", group_size=128)
    assert out_scales.shape == scales.shape
    # each quantized weight is scale * {-1,0,1}
    ratio = torch.round(w_hat / out_scales.repeat_interleave(128, dim=1))
    assert set(ratio.unique().tolist()).issubset({-1.0, 0.0, 1.0})
