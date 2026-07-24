#!/usr/bin/env python
"""Toy repro: does ZeRO-3 + parametrize + reentrant checkpointing leak memory across backward?

The 35B e2e smokes OOM in backward with ~1.6GB/layer accumulating regardless of resident
headroom (param-offload freed 9GB; the peak stayed ~74GB — OOM just moved a few layers later).
This reproduces the exact structural combo at 1/1000 the cost on a T4: a stack of blocks, each
with a QuantLinear and a parametrized fused 3D "expert" param, under ZeRO-3 with a client bnb
Adam8bit and per-block torch checkpointing. Prints CUDA allocated per block during forward and
backward — a monotone climb through backward reproduces the leak; flat is healthy.

    deepspeed --num_gpus=1 scripts/repro_zero3_parametrize.py --mode reentrant
    ... --mode nonreentrant | --mode nocheckpoint

Cost: pennies on t4-small.
"""

from __future__ import annotations

import argparse

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from bite.quant.experts import install_expert_fakequant
from bite.quant.quantlinear import QuantLinear
from bite.train.blockwise import quant_parameters


class _RealMoeBlock(nn.Module):
    """QuantLinear attn + the REAL transformers Qwen3_5MoeSparseMoeBlock (router + grouped-GEMM
    experts + shared expert) — reproduces the exact MoE compute path of the 35B runs."""

    def __init__(self, d: int, E: int, inter: int, group_size: int, impl: str, deltanet: bool = False):
        super().__init__()
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeGatedDeltaNet,
            Qwen3_5MoeSparseMoeBlock,
            Qwen3_5MoeTextConfig,
        )

        cfg = Qwen3_5MoeTextConfig(
            hidden_size=d,
            moe_intermediate_size=inter,
            num_experts=E,
            num_experts_per_tok=4,
            shared_expert_intermediate_size=inter // 2,
            hidden_act="silu",
            linear_num_key_heads=4,
            linear_num_value_heads=8,
            linear_key_head_dim=64,
            linear_value_head_dim=64,
            linear_conv_kernel_dim=4,
        )
        cfg._experts_implementation = impl  # 'grouped_mm' (real path) | 'batched_mm' (candidate fix)
        self.attn = QuantLinear(d, d, bias=False, mode="ternary", group_size=group_size)
        # real GatedDeltaNet (uses the fla chunk kernel when installed — the real 35B path)
        self.deltanet = Qwen3_5MoeGatedDeltaNet(cfg, layer_idx=0) if deltanet else None
        self.moe = Qwen3_5MoeSparseMoeBlock(cfg)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        h = self.attn(x)
        if self.deltanet is not None:
            dn = self.deltanet(h)
            h = h + (dn[0] if isinstance(dn, tuple) else dn)
        out = self.moe(h)
        out = out[0] if isinstance(out, tuple) else out
        return self.norm(x + out)


class _Block(nn.Module):
    """Toy decoder block: QuantLinear attn + parametrized fused 3D expert param (the combo)."""

    def __init__(self, d: int, E: int, inter: int, group_size: int = 128):
        super().__init__()
        self.attn = QuantLinear(d, d, bias=False, mode="ternary", group_size=group_size)
        self.gate_up_proj = nn.Parameter(torch.randn(E, d, inter) * 0.02)
        self.down_proj = nn.Parameter(torch.randn(E, inter, d) * 0.02)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):  # x: [B, T, d]
        h = self.attn(x)
        e = torch.einsum("btd,edi->ebti", h, self.gate_up_proj)  # all experts, like grouped GEMM
        e = torch.einsum("ebti,eid->btd", torch.nn.functional.gelu(e), self.down_proj)
        return self.norm(x + e / self.gate_up_proj.shape[0])


class _Model(nn.Module):
    def __init__(
        self,
        n_layers: int,
        d: int,
        E: int,
        inter: int,
        mode: str,
        group_size: int = 128,
        moe_impl: str | None = None,
        deltanet: bool = False,
    ):
        super().__init__()
        self.embed = nn.Linear(d, d, bias=False)
        mk = (lambda: _RealMoeBlock(d, E, inter, group_size, moe_impl, deltanet=deltanet)) if moe_impl else (
            lambda: _Block(d, E, inter, group_size)
        )
        self.blocks = nn.ModuleList(mk() for _ in range(n_layers))
        self.mode = mode

    def forward(self, x):
        h = self.embed(x)
        for i, blk in enumerate(self.blocks):
            if self.mode == "reentrant":
                h = checkpoint(blk, h, use_reentrant=True)
            elif self.mode == "nonreentrant":
                h = checkpoint(blk, h, use_reentrant=False)
            else:
                h = blk(h)
            if i % 4 == 0:
                print(f"  fwd block {i}: alloc {torch.cuda.memory_allocated() / 1e9:.2f}GB")
        return h


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="reentrant", choices=("reentrant", "nonreentrant", "nocheckpoint"))
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation steps (accum=2 tests the non-boundary reduction path)")
    ap.add_argument("--assign-load", action="store_true", help="init via safetensors mmap + load_state_dict(assign=True) — mirrors the real e2e init path")
    ap.add_argument("--reduce-bucket", type=float, default=5e8, help="reduce_bucket_size; set BELOW param numel to test the oversized-grad path")
    ap.add_argument("--max-live", type=float, default=1e8, help="stage3_max_live_parameters; set BELOW param numel to test oversized-param release")
    ap.add_argument("--real-moe", default=None, choices=(None, "grouped_mm", "batched_mm"), help="use the REAL transformers Qwen3_5MoeSparseMoeBlock with this experts implementation")
    ap.add_argument("--deltanet", action="store_true", help="add the REAL Qwen3_5MoeGatedDeltaNet to each block (fla chunk kernel when installed)")
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--experts", type=int, default=32)
    ap.add_argument("--inter", type=int, default=512)
    ap.add_argument("--local_rank", type=int, default=0)
    args = ap.parse_args()

    import bitsandbytes as bnb
    import deepspeed

    # each block's expert params: 32*1024*512*2 * 2B(bf16 via DS) ≈ 67MB -> visible per-layer signal
    model = _Model(args.layers, args.dim, args.experts, args.inter, args.mode, moe_impl=args.real_moe, deltanet=args.deltanet)
    install_expert_fakequant(model, mode="ternary", group_size=128)
    if args.assign_load:
        # mirror the real e2e init: save -> fresh model -> mmap safetensors -> assign-load
        import safetensors.torch as st

        st.save_file({k: v.contiguous() for k, v in model.state_dict().items()}, "/tmp/toy_ckpt.safetensors")
        model = _Model(args.layers, args.dim, args.experts, args.inter, args.mode)
        install_expert_fakequant(model, mode="ternary", group_size=128)
        missing, unexpected = model.load_state_dict(st.load_file("/tmp/toy_ckpt.safetensors"), strict=False, assign=True)
        assert not missing and not unexpected
        print("assign-loaded toy from mmap safetensors")
    for p in model.parameters():
        p.requires_grad_(False)
    trainable = quant_parameters(model)
    for p in trainable:
        p.requires_grad_(True)
    model.train()
    print(f"toy: {args.layers} blocks, {len(trainable)} trainable latents, mode={args.mode}")

    opt = bnb.optim.Adam8bit(trainable, lr=1e-4)
    engine, _, _, _ = deepspeed.initialize(
        model=model,
        optimizer=opt,
        config={
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": args.accum,
            "bf16": {"enabled": True},
            "zero_allow_untested_optimizer": True,
            "zero_optimization": {
                "stage": 3,
                "reduce_bucket_size": args.reduce_bucket,
                "stage3_prefetch_bucket_size": 5e6,
                "stage3_param_persistence_threshold": 1e4,
                "stage3_max_live_parameters": args.max_live,
            },
        },
    )
    print(f"post-shard alloc {torch.cuda.memory_allocated() / 1e9:.2f}GB")

    # per-block BACKWARD memory: hook on each block's first param's grad
    def bwd_probe(i):
        def hook(_g):
            print(f"  bwd block {i}: alloc {torch.cuda.memory_allocated() / 1e9:.2f}GB")
            return None

        return hook

    for i, blk in enumerate(engine.module.blocks):
        if i % 4 == 0:
            quant_parameters(blk)[-1].register_hook(bwd_probe(i))

    for step in range(2 * args.accum + 1):  # cross at least two optimizer-step boundaries
        # requires_grad mirrors enable_input_require_grads() in the real run: with reentrant
        # checkpointing + frozen embed, the checkpoint chain needs a grad-requiring input
        x = torch.randn(1, 512, args.dim, device=engine.device, dtype=torch.bfloat16, requires_grad=True)
        torch.cuda.reset_peak_memory_stats()
        out = engine(x)
        loss = out.float().pow(2).mean()
        engine.backward(loss)
        engine.step()
        print(
            f"step {step}: loss {loss.item():.4f} peak {torch.cuda.max_memory_allocated() / 1e9:.2f}GB "
            f"end-alloc {torch.cuda.memory_allocated() / 1e9:.2f}GB"
        )
    # verdict: healthy = bwd-block prints roughly FLAT and step peaks stable across steps;
    # leak = monotone climb through bwd blocks (late blocks are processed FIRST in backward,
    # so a climb from high block idx down to 0 mirrors the 35B failure)
    st = {v.dtype for s in opt.state.values() for v in s.values() if torch.is_tensor(v)}
    print(f"bnb state dtypes: {st}")
    print("REPRO DONE")


if __name__ == "__main__":
    main()
