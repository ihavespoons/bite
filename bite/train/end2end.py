"""End-to-end Quantization-Aware Distillation — the primary ternary recovery mechanism.

The Stage-3 diagnostic showed naive ternary PTQ collapses this MoE to chance MMLU and that
block-wise hidden-MSE healing does NOT recover it: local matching can't undo the error that
compounds across 40 blocks. Recovery needs a **global** signal — distilling the student's full
next-token distribution against the frozen FP16 teacher's precomputed top-k logits (plus a CE
term), with STE gradients flowing to every quantized latent weight (attention + fused experts).

Training all ~33B trainable params exceeds one H200, so this runs under **DeepSpeed ZeRO-3 with
CPU offload** (params/grads/optimizer offloaded; one layer's params gathered to GPU at a time)
plus gradient checkpointing. The loss and shard-iteration are pure/CPU-testable; the ZeRO-3
driver is runner-side.
"""

from __future__ import annotations

import glob
from collections.abc import Iterator

import torch
import torch.nn.functional as F
from torch import Tensor

from bite.train.qad import topk_kl_loss


def end2end_loss(
    logits: Tensor,
    teacher_values: Tensor,
    teacher_indices: Tensor,
    input_ids: Tensor,
    *,
    w_kl: float = 1.0,
    w_ce: float = 0.5,
    temperature: float = 1.0,
) -> tuple[Tensor, dict[str, float]]:
    """Distillation loss: per-position top-k KL(teacher‖student) + next-token cross-entropy.

    ``logits``: student ``[B, T, V]``. ``teacher_values``/``teacher_indices``: ``[B, T, k]`` —
    the teacher's top-k next-token logits at each position (KL aligns position-wise, no shift).
    CE is the standard shifted next-token objective on the ground-truth ``input_ids``.
    """
    kl = topk_kl_loss(logits, teacher_values, teacher_indices, temperature)
    ce = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        input_ids[:, 1:].reshape(-1),
    )
    total = w_kl * kl + w_ce * ce
    return total, {"kl": float(kl.detach()), "ce": float(ce.detach())}


def iter_shard_examples(shard_dir: str) -> Iterator[dict]:  # pragma: no cover - runner I/O
    """Yield per-sequence training examples from the precomputed teacher-logit shards.

    Each shard bundles ``input_ids``/``values``/``indices`` for a batch of sequences; we split
    into single sequences so the training driver controls the micro-batch size.
    """
    from bite.train.teacher import load_teacher_shard

    for path in sorted(glob.glob(f"{shard_dir}/*.safetensors")):
        input_ids, values, indices = load_teacher_shard(path)
        for i in range(input_ids.shape[0]):
            yield {
                "input_ids": input_ids[i],
                "values": values[i],
                "indices": indices[i],
            }


def download_teacher_shards(repo: str, local_dir: str) -> str:  # pragma: no cover - runner I/O
    """Download the ``teacher_topk/`` folder of an HF dataset for training."""
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        allow_patterns="teacher_topk/*",
        local_dir=local_dir,
    )
    return f"{local_dir}/teacher_topk"


def _zero3_offload_config(lr: float, *, micro_batch: int, accum: int) -> dict:
    """DeepSpeed ZeRO-3 config: offload params + optimizer to CPU so a 35B fits one H200."""
    return {
        "train_micro_batch_size_per_gpu": micro_batch,
        "gradient_accumulation_steps": accum,
        "bf16": {"enabled": True},
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
            "offload_param": {"device": "cpu", "pin_memory": True},
            "stage3_prefetch_bucket_size": 5e7,
            "stage3_param_persistence_threshold": 1e5,
            "stage3_max_live_parameters": 1e9,
        },
        "optimizer": {"type": "AdamW", "params": {"lr": lr, "betas": [0.9, 0.999]}},
    }


def run_end2end_qad(  # pragma: no cover - requires model + GPU + deepspeed
    config: dict,
    *,
    model_id: str | None = None,
    teacher_repo: str | None = None,
    steps: int | None = None,
    micro_batch: int = 1,
    accum: int = 8,
    eval_tasks: list[str] | None = None,
    eval_limit: int | None = None,
    out_dir: str = "outputs/e2e",
    push_repo: str | None = None,
    skip_save: bool = False,
    smoke: bool = False,
) -> dict:
    """End-to-end ternary QAD under DeepSpeed ZeRO-3 CPU offload (runner-side).

    1. Build the PTQ-init student, freeze the kept-precision tail, mark the quantized latents
       (QuantLinear weights + fused expert originals) trainable, enable gradient checkpointing.
    2. ``deepspeed.initialize`` with ZeRO-3 offload; iterate the precomputed teacher shards,
       minimizing :func:`end2end_loss` (top-k logit KL + next-token CE) via STE.
    3. Eval MMLU vs the FP16 baseline. ``smoke=True`` runs a couple of steps only (integration
       check) — no eval/save.
    """
    import json
    import os

    import deepspeed

    from bite.eval.harness import run_lm_eval_model
    from bite.models.loader import build_student, load_tokenizer
    from bite.quant.experts import ptq_init_experts
    from bite.quant.ptq import ptq_init_model
    from bite.train.blockwise import quant_parameters

    mid = model_id or config["model"]["id"]
    lr = float(config["qad"].get("lr", 1e-4))
    steps = int(steps if steps is not None else 200)
    temperature = float(config["qad"].get("temperature", 1.0))
    lw = config["qad"].get("loss_weights", {}) or {}
    os.makedirs(out_dir, exist_ok=True)

    tokenizer = load_tokenizer(mid)
    student, swapped, experts = build_student(
        mid, mode=config["quant"]["mode"], group_size=config["quant"]["group_size"]
    )
    ptq_init_model(student, hessians=None, percdamp=config["ptq"]["percdamp"])
    ptq_init_experts(student)
    if hasattr(student, "config"):
        student.config.use_cache = False
    if hasattr(student, "gradient_checkpointing_enable"):
        student.gradient_checkpointing_enable()

    # only the quantized latents train; the kept-precision tail (router, shared expert, norms) is frozen
    for p in student.parameters():
        p.requires_grad_(False)
    trainable = quant_parameters(student)
    for p in trainable:
        p.requires_grad_(True)
    print(f"student: {len(swapped)} linears + {len(experts)} expert tensors; {len(trainable)} trainable latents")

    engine, _, _, _ = deepspeed.initialize(
        model=student,
        model_parameters=[p for p in student.parameters() if p.requires_grad],
        config=_zero3_offload_config(lr, micro_batch=micro_batch, accum=accum),
    )
    device = engine.device

    shard_dir = download_teacher_shards(teacher_repo, out_dir) if teacher_repo else f"{out_dir}/teacher_topk"
    w_kl = float(lw.get("kl", 1.0))
    w_ce = float(lw.get("ce", 0.5))

    history: list[dict] = []
    step = 0
    done = False
    while not done:
        for ex in iter_shard_examples(shard_dir):
            input_ids = ex["input_ids"].long().unsqueeze(0).to(device)
            values = ex["values"].to(device).float().unsqueeze(0)
            indices = ex["indices"].long().unsqueeze(0).to(device)
            logits = engine(input_ids=input_ids).logits
            loss, parts = end2end_loss(
                logits, values, indices, input_ids, w_kl=w_kl, w_ce=w_ce, temperature=temperature
            )
            engine.backward(loss)
            engine.step()
            history.append({"step": step, "loss": float(loss.detach()), **parts})
            if step % 10 == 0:
                print(f"step {step}: loss {float(loss.detach()):.4f} (kl {parts['kl']:.4f} ce {parts['ce']:.4f})")
            step += 1
            if step >= (2 if smoke else steps):
                done = True
                break

    result: dict = {"steps": step, "loss_first": history[0]["loss"], "loss_last": history[-1]["loss"]}
    print(f"end-to-end QAD done: loss {result['loss_first']:.4f} -> {result['loss_last']:.4f}")
    if smoke:
        print("SMOKE OK: deepspeed ZeRO-3 + parametrized student forward/backward/step work")
        return result

    with open(f"{out_dir}/e2e_metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    if push_repo:
        from huggingface_hub import HfApi

        HfApi().upload_file(
            path_or_fileobj=f"{out_dir}/e2e_metrics.json",
            path_in_repo="e2e_metrics.json",
            repo_id=push_repo,
            repo_type="dataset",
        )

    if eval_tasks:
        try:
            res = run_lm_eval_model(
                engine.module, tokenizer, eval_tasks, batch_size=config["eval"]["batch_size"],
                **({"limit": eval_limit} if eval_limit else {}),
            )
            mmlu = res["results"].get("mmlu", {}).get("acc,none")
            base = (config.get("eval", {}) or {}).get("teacher_baseline") or {}
            if mmlu is not None:
                result["mmlu"] = mmlu
                if base.get("mmlu"):
                    result["mmlu_retained"] = mmlu / base["mmlu"]
                    print(f"MMLU {mmlu:.4f} vs FP16 {base['mmlu']:.4f} -> {result['mmlu_retained']:.1%} retained")
            with open(f"{out_dir}/e2e_metrics.json", "w") as f:
                json.dump(result, f, indent=2)
            if push_repo:
                from huggingface_hub import HfApi

                HfApi().upload_file(
                    path_or_fileobj=f"{out_dir}/e2e_metrics.json",
                    path_in_repo="e2e_metrics.json",
                    repo_id=push_repo,
                    repo_type="dataset",
                )
        except Exception as e:  # noqa: BLE001
            print(f"WARN: eval failed ({e})")

    return result
