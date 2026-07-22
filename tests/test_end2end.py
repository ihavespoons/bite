import torch
from torch import nn

from bite.quant.experts import install_expert_fakequant
from bite.quant.quantlinear import QuantLinear
from bite.train.blockwise import quant_parameters
from bite.train.end2end import end2end_loss, iter_shard_examples


def test_end2end_loss_positive_and_parts():
    torch.manual_seed(0)
    B, T, V, k = 2, 8, 50, 5
    logits = torch.randn(B, T, V, requires_grad=True)
    # teacher top-k drawn from a (different) random logit tensor
    teacher = torch.randn(B, T, V)
    values, indices = teacher.topk(k, dim=-1)
    input_ids = torch.randint(0, V, (B, T))

    loss, parts = end2end_loss(logits, values, indices, input_ids, w_kl=1.0, w_ce=0.5)
    assert loss.item() > 0
    assert set(parts) == {"kl", "ce"}
    loss.backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0


def test_end2end_loss_kl_zero_when_student_matches_teacher():
    B, T, V, k = 1, 4, 20, 4
    teacher_logits = torch.randn(B, T, V)
    values, indices = teacher_logits.topk(k, dim=-1)
    input_ids = torch.randint(0, V, (B, T))
    # student == teacher -> KL term ~ 0 (loss is then pure CE)
    loss_match, parts_match = end2end_loss(teacher_logits, values, indices, input_ids, w_ce=0.0)
    assert parts_match["kl"] < 1e-5
    assert abs(loss_match.item()) < 1e-4


class _TinyQuantModel(nn.Module):
    """QuantLinear + fused expert param — the two structures the e2e init load must survive."""

    def __init__(self, d=64, E=2, inter=32):
        super().__init__()
        self.attn = QuantLinear(d, d, bias=False, mode="ternary", group_size=32)
        self.gate_up_proj = nn.Parameter(torch.randn(E, d, 2 * inter))
        self.gate = nn.Parameter(torch.randn(d, E))  # kept-precision router


def test_assign_load_keeps_quant_parameters_live_and_trainable():
    """Mirrors the e2e init path: build -> assign-load -> re-collect -> mark trainable -> grads.

    assign=True replaces the Parameter OBJECTS, so anything collected before the load is stale;
    quant_parameters re-walks live modules and must return the loaded (new) params, which must
    train after requires_grad marking.
    """
    torch.manual_seed(0)
    src = _TinyQuantModel()
    install_expert_fakequant(src, mode="ternary", group_size=32)
    sd = {k: v.clone() for k, v in src.state_dict().items()}
    assert any("parametrizations.gate_up_proj.original" in k for k in sd)  # healed-ckpt key shape

    dst = _TinyQuantModel()
    install_expert_fakequant(dst, mode="ternary", group_size=32)
    stale = quant_parameters(dst)  # collected BEFORE the load -> must become stale
    missing, unexpected = dst.load_state_dict(sd, strict=False, assign=True)
    assert not missing and not unexpected

    for p in dst.parameters():
        p.requires_grad_(False)
    live = quant_parameters(dst)
    for p in live:
        p.requires_grad_(True)

    # the re-collected params are the loaded objects, not the stale pre-load ones
    assert len(live) == len(stale) == 2
    assert all(all(lp is not sp for sp in stale) for lp in live)
    # loaded values match the source and gradients flow through the parametrization
    latent = dst.parametrizations.gate_up_proj.original
    assert torch.equal(latent, src.parametrizations.gate_up_proj.original)
    out = torch.bmm(torch.randn(2, 4, 64), dst.gate_up_proj).sum() + dst.attn(torch.randn(4, 64)).sum()
    out.backward()
    assert latent.grad is not None and latent.grad.abs().sum() > 0
    assert dst.attn.weight.grad is not None
    assert dst.gate.grad is None  # kept-precision tail stays frozen


def test_iter_shard_examples_splits_batches(tmp_path):
    import pytest

    st = pytest.importorskip("safetensors.torch")  # runner-side dep (model extra)

    B, T, k = 3, 6, 4
    st.save_file(
        {
            "input_ids": torch.randint(0, 100, (B, T)).to(torch.int32),
            "values": torch.randn(B, T, k).to(torch.float16),
            "indices": torch.randint(0, 100, (B, T, k)).to(torch.int32),
        },
        f"{tmp_path}/teacher_topk_000000.safetensors",
    )
    examples = list(iter_shard_examples(str(tmp_path)))
    assert len(examples) == B
    assert examples[0]["input_ids"].shape == (T,)
    assert examples[0]["values"].shape == (T, k)
