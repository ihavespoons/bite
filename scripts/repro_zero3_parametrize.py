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
    def __init__(self, n_layers: int, d: int, E: int, inter: int, mode: str, group_size: int = 128):
        super().__init__()
        self.embed = nn.Linear(d, d, bias=False)
        self.blocks = nn.ModuleList(_Block(d, E, inter, group_size) for _ in range(n_layers))
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
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--experts", type=int, default=32)
    ap.add_argument("--inter", type=int, default=512)
    ap.add_argument("--local_rank", type=int, default=0)
    args = ap.parse_args()

    import bitsandbytes as bnb
    import deepspeed

    # each block's expert params: 32*1024*512*2 * 2B(bf16 via DS) ≈ 67MB -> visible per-layer signal
    model = _Model(args.layers, args.dim, args.experts, args.inter, args.mode)
    install_expert_fakequant(model, mode="ternary", group_size=128)
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
            "gradient_accumulation_steps": 1,
            "bf16": {"enabled": True},
            "zero_allow_untested_optimizer": True,
            "zero_optimization": {
                "stage": 3,
                "stage3_prefetch_bucket_size": 5e6,
                "stage3_param_persistence_threshold": 1e4,
                "stage3_max_live_parameters": 1e8,
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
            blk.parametrizations.gate_up_proj.original.register_hook(bwd_probe(i))

    for step in range(3):
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
