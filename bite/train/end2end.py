"""End-to-end Quantization-Aware Distillation — the primary ternary recovery mechanism.

The Stage-3 diagnostic showed naive ternary PTQ collapses this MoE to chance MMLU and that
block-wise hidden-MSE healing does NOT recover it: local matching can't undo the error that
compounds across 40 blocks. Recovery needs a **global** signal — distilling the student's full
next-token distribution against the frozen FP16 teacher's precomputed top-k logits (plus a CE
term), with STE gradients flowing to every quantized latent weight (attention + fused experts).

Memory design (the hard-won part — four OOMs taught us this):
- **DeepSpeed ZeRO-3, everything on-GPU, no offload.** fp32 Adam state for ~33B params is
  ~400GB — it fits nowhere as fp32+offload. Instead a **bitsandbytes 8-bit Adam** is passed as
  the *client* optimizer (production precedent: HF Trainer ``adamw_bnb_8bit`` + ZeRO-3): DS
  replaces its param groups with fp32 flat partitions and calls its CUDA step on them. Budget
  per GPU (of 141GB): params 17.5 + grads 16.5 + fp32 masters 33 + 8-bit states 16.6 ≈ 84GB
  resident, ~105GB peak with fake-quant temporaries/logits/comm buffers.
- The fp32 flat master ZeRO-3 keeps is not waste — STE recovery *needs* it (bf16's ~8 mantissa
  bits swallow the tiny updates that accumulate toward a ternary boundary flip).
- **Init is loaded, not computed**: each rank assign-loads the block-wise-healed checkpoint via
  safetensors mmap (zero-copy, read-only) so 4 ranks share ~70GB of page cache instead of
  materializing 4×70GB (in-run PTQ init *writes*, which COW-materializes every page).

The loss and shard-iteration are pure/CPU-testable; the ZeRO-3 driver is runner-side.
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


def _zero3_config(*, micro_batch: int, accum: int, offload_param: bool = False) -> dict:
    """DeepSpeed ZeRO-3 config: training state on-GPU, sharded across ranks.

    The optimizer is a *client* bnb ``Adam8bit`` (CUDA-only step; optimizer offload would break
    it — NEVER offload it), hence ``zero_allow_untested_optimizer``. No ``optimizer`` block —
    DS must use the client instance. ``sub_group_size`` stays at its 1e9 default (bnb kernels
    use 32-bit indexing internally). Gradient clipping is DS's job.

    ``offload_param=True`` moves only the bf16 param *shards* to (pinned) CPU — gathered to GPU
    per-module on demand. Costs ~PCIe transfer per step but frees ~9GB/rank resident: the knob
    that closes the ~5GB backward-peak overshoot on 80GB A100s (H200s don't need it).
    """
    # KNOB FLOOR: the fused gate_up_proj is 5.37e8 elements — BIGGER than the default
    # reduce_bucket_size (5e8) and any tighter max_live cap. Params/grads that exceed these
    # knobs hit DS's oversized special-case paths, which retained every layer's full expert
    # grads through backward (+1.6GB/layer -> OOM). Keep BOTH knobs above the largest param.
    zero: dict = {
        "stage": 3,
        "reduce_bucket_size": 6e8,
        "stage3_prefetch_bucket_size": 5e7,
        "stage3_param_persistence_threshold": 1e5,
        "stage3_max_live_parameters": 1.5e9,
        "stage3_gather_16bit_weights_on_model_save": True,
        "overlap_comm": False,  # sync reduction: bounded backward memory (async gave no benefit)
    }
    if offload_param:
        zero["offload_param"] = {"device": "cpu", "pin_memory": True}
    return {
        "train_micro_batch_size_per_gpu": micro_batch,
        "gradient_accumulation_steps": accum,
        "bf16": {"enabled": True},
        "gradient_clipping": 1.0,
        "zero_allow_untested_optimizer": True,
        "zero_optimization": zero,
    }


def _host_mem(tag: str) -> None:  # pragma: no cover - runner-side telemetry
    """Log host memory at load milestones (validates the shared-page-cache design)."""
    try:
        import psutil

        vm = psutil.virtual_memory()
        print(f"[mem:{tag}] host available {vm.available / 1e9:.0f}GB of {vm.total / 1e9:.0f}GB")
    except Exception:  # noqa: BLE001 - telemetry only
        pass


def load_init_checkpoint(student, repo: str, out_dir: str) -> None:  # pragma: no cover - runner I/O
    """Assign-load the block-wise-healed student checkpoint via safetensors mmap (read-only).

    ``load_state_dict(..., assign=True)`` swaps the Parameter *objects* for mmap-backed tensors,
    so all ranks share the file page cache (~70GB total, not 70GB × ranks) and nothing is
    written on CPU. Anything collected from the model **before** this call (e.g. a
    ``quant_parameters`` list) is stale afterwards — always re-collect.
    """
    import safetensors.torch as st
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo, repo_type="dataset", allow_patterns="qad_student/*", local_dir=out_dir
    )
    sd: dict = {}
    for shard in sorted(glob.glob(f"{out_dir}/qad_student/*.safetensors")):
        sd.update(st.load_file(shard))  # zero-copy mmap on CPU
    missing, unexpected = student.load_state_dict(sd, strict=False, assign=True)
    real_missing = [k for k in missing if "inv_freq" not in k]  # non-persistent buffers are fine
    assert not unexpected and not real_missing, (real_missing[:5], list(unexpected)[:5])
    print(f"assign-loaded {len(sd)} tensors from {repo}/qad_student (mmap, shared page cache)")


def run_end2end_qad(  # pragma: no cover - requires model + GPU + deepspeed
    config: dict,
    *,
    model_id: str | None = None,
    teacher_repo: str | None = None,
    init_repo: str | None = None,
    ptq_init: bool = False,
    steps: int | None = None,
    micro_batch: int = 1,
    accum: int = 8,
    offload_param: bool = False,
    mem_probe: bool = False,
    out_dir: str = "outputs/e2e",
    push_repo: str | None = None,
    skip_save: bool = False,
    smoke: bool = False,
) -> dict:
    """End-to-end ternary QAD under DeepSpeed ZeRO-3, all on-GPU (runner-side).

    1. Build the fake-quant student on CPU (no device_map hooks — they block ZeRO-3 sharding),
       assign-load the healed checkpoint from ``init_repo`` (mmap, read-only; ``ptq_init=True``
       is the escape hatch that recomputes naive PTQ instead — costs 70GB CPU per rank).
    2. Freeze everything, re-collect + mark the quantized latents trainable, build a client
       bnb ``Adam8bit``, ``deepspeed.initialize`` (ZeRO-3, no offload).
    3. Iterate the precomputed teacher shards rank-sharded, minimizing :func:`end2end_loss`.
    4. Save a consolidated 16-bit checkpoint; MMLU runs in a separate 1-GPU reload-eval job
       (``eval_quant.py --load-weights``). ``smoke=True`` runs ``accum+1`` micro-steps so ONE
       real optimizer step happens, then asserts state/params actually moved — no eval/save.
    """
    import json
    import os

    import deepspeed

    from bite.models.loader import build_student, load_tokenizer
    from bite.quant.fakequant import quantize_ternary
    from bite.train.blockwise import quant_parameters

    mid = model_id or config["model"]["id"]
    lr = float(config["qad"].get("lr", 1e-4))
    steps = int(steps if steps is not None else 200)
    temperature = float(config["qad"].get("temperature", 1.0))
    lw = config["qad"].get("loss_weights", {}) or {}
    group_size = config["quant"]["group_size"]
    os.makedirs(out_dir, exist_ok=True)

    _host_mem("start")
    tokenizer = load_tokenizer(mid)
    # CPU build: device_map to GPU installs accelerate hooks that block ZeRO-3 partitioning
    # (that was OOM #3); deepspeed.initialize owns placement and shards from here.
    student, swapped, experts = build_student(
        mid,
        mode=config["quant"]["mode"],
        group_size=group_size,
        device_map="cpu",
    )
    _host_mem("built")
    if ptq_init:
        from bite.quant.experts import ptq_init_experts
        from bite.quant.ptq import ptq_init_model

        ptq_init_model(student, hessians=None, percdamp=config["ptq"]["percdamp"])
        ptq_init_experts(student)
    else:
        load_init_checkpoint(student, init_repo or "ihavespoons/bite-baseline", out_dir)
        # Identity is proven by the strict key match in load_init_checkpoint (only our saved
        # student has parametrizations.*.original keys). The grid distance is telemetry only:
        # healed latents start on-grid at PTQ init and drift off-grid as block-wise QAD trains
        # them, so a small-but-nonzero distance is the EXPECTED signature of the healed ckpt.
        if config["quant"]["mode"] == "ternary":
            probe = quant_parameters(student)[-1].detach().float()
            err = (probe - quantize_ternary(probe, group_size)[0]).abs().max().item()
            print(f"init checkpoint grid distance (telemetry): max {err:.3e}")
    _host_mem("loaded")
    if hasattr(student, "config"):
        student.config.use_cache = False
    if hasattr(student, "gradient_checkpointing_enable"):
        # use_reentrant=True: the non-reentrant variant's saved-tensor metadata check breaks
        # under ZeRO-3 (params are re-partitioned to numel-0 after each module forward, so the
        # backward recompute sees different metadata -> CheckpointError). Reentrant + ZeRO-3 is
        # the battle-tested combo (HF Trainer default).
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    if hasattr(student, "enable_input_require_grads"):
        # reentrant checkpointing needs a grad-requiring input to anchor each block's backward;
        # our embeddings are frozen, so mark the embedding OUTPUT (standard PEFT-style fix)
        student.enable_input_require_grads()
    # from_pretrained returns the model in EVAL mode and transformers gates checkpointing on
    # self.training — without train() every layer's fake-quant weight output (~1.6GB × 40
    # layers ≈ 64GB) is retained for backward, which OOM'd the a100x8 forward at 75.6/79GB.
    student.train()

    # freeze the kept-precision tail; re-collect latents NOW (assign-load replaced the objects)
    for p in student.parameters():
        p.requires_grad_(False)
    trainable = quant_parameters(student)
    for p in trainable:
        p.requires_grad_(True)
    print(f"student: {len(swapped)} linears + {len(experts)} expert tensors; {len(trainable)} trainable latents")

    import bitsandbytes as bnb

    opt = bnb.optim.Adam8bit(trainable, lr=lr, betas=(0.9, 0.999))  # NOT Paged* under ZeRO-3
    engine, _, _, _ = deepspeed.initialize(
        model=student,
        optimizer=opt,
        config=_zero3_config(micro_batch=micro_batch, accum=accum, offload_param=offload_param),
    )
    device = engine.device
    _host_mem("sharded")
    # verify the two memory levers are actually engaged before burning forward passes:
    # ZeRO-3 partitioned the params (resident GPU ≈ shard size, not 70GB) and checkpointing
    # will actually run (train mode survived deepspeed.initialize)
    assert engine.module.training, "model must be in train() mode or checkpointing is skipped"
    resident = torch.cuda.memory_allocated() / 1e9
    print(f"post-shard resident {resident:.1f}GB/GPU "
          f"(checkpointing={getattr(engine.module, 'is_gradient_checkpointing', '?')})")
    assert resident < 40, f"params not sharded: {resident:.1f}GB resident (expect ~shard size)"

    import torch.distributed as dist

    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    shard_dir = download_teacher_shards(teacher_repo, out_dir) if teacher_repo else f"{out_dir}/teacher_topk"
    w_kl = float(lw.get("kl", 1.0))
    w_ce = float(lw.get("ce", 0.5))

    # probe: a slice of one expert latent, cloned pre-training — the smoke asserts the optimizer
    # step actually moved it (a silent bnb no-op on DS flat partitions would otherwise pass)
    probe_param = trainable[-1]
    with deepspeed.zero.GatheredParameters([probe_param]):
        probe_before = probe_param.flatten()[:4096].detach().clone().cpu()

    if mem_probe and rank == 0:
        # per-decoder-layer memory probes (toy showed the quant machinery is leak-free; this
        # identifies WHICH real layers grow the allocation — the 3xDeltaNet+1xattn pattern
        # fingerprints the culprit). fwd: module pre-hook; bwd: grad hook on a block latent.
        from bite.train.blockwise import iter_blocks, quant_parameters as _qp

        blocks = iter_blocks(engine.module, config["qad"].get("layers_path"))

        def _fwd_probe(i):
            def hook(_m, _args, _kwargs):
                print(f"  fwd L{i:02d}: alloc {torch.cuda.memory_allocated() / 1e9:.1f}GB")

            return hook

        def _bwd_probe(i):
            def hook(_g):
                print(f"  bwd L{i:02d}: alloc {torch.cuda.memory_allocated() / 1e9:.1f}GB")
                return None

            return hook

        for i, blk in enumerate(blocks):
            if i % 4 == 0:
                blk.register_forward_pre_hook(_fwd_probe(i), with_kwargs=True)
            bp = _qp(blk)
            if bp:
                bp[0].register_hook(_bwd_probe(i))
        print(f"mem probes on {len(blocks)} blocks (fwd every 4th, bwd all)")

    # smoke: accum+1 micro-steps guarantees crossing ONE real optimizer-step boundary — the
    # previous smoke (2 micro-steps, accum=8) never actually called optimizer.step()
    target_micro_steps = (accum + 1) if smoke else steps
    history: list[dict] = []
    step = 0
    done = False
    while not done:
        # rank-aware data-parallel sharding: each GPU trains on a disjoint slice of sequences
        for ex_idx, ex in enumerate(iter_shard_examples(shard_dir)):
            if ex_idx % world != rank:
                continue
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
            if step % 10 == 0 or smoke:
                alloc = torch.cuda.max_memory_allocated() / 1e9
                print(
                    f"step {step}: loss {float(loss.detach()):.4f} (kl {parts['kl']:.4f} "
                    f"ce {parts['ce']:.4f}) peak {alloc:.0f}GB"
                )
            step += 1
            if step >= target_micro_steps:
                done = True
                break

    result: dict = {"steps": step, "loss_first": history[0]["loss"], "loss_last": history[-1]["loss"]}
    print(f"end-to-end QAD done: loss {result['loss_first']:.4f} -> {result['loss_last']:.4f}")

    if smoke:
        # 1. loss finite on every micro-step
        assert all(h["loss"] == h["loss"] and abs(h["loss"]) < 1e6 for h in history), "non-finite loss"
        # 2. bnb 8-bit state materialized on CUDA (DS hands it fp32 flat partitions)
        state_dtypes = {
            v.dtype for st_ in opt.state.values() for v in st_.values() if torch.is_tensor(v)
        }
        assert torch.uint8 in state_dtypes, f"bnb 8-bit state missing (dtypes: {state_dtypes})"
        # 3. the probed latent slice actually moved
        with deepspeed.zero.GatheredParameters([probe_param]):
            probe_after = probe_param.flatten()[:4096].detach().cpu()
        moved = (probe_after - probe_before).abs().max().item()
        assert moved > 0, "optimizer step did not change the probed latent"
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"SMOKE OK: optimizer step verified (probe moved {moved:.2e}, "
              f"8-bit state on CUDA, peak {peak:.0f}GB/rank)")
        return result

    # Save a CONSOLIDATED 16-bit checkpoint (ZeRO-3 gathers the sharded params to rank 0). MMLU is
    # evaluated in a separate 1-GPU job that reloads this into a fresh student — evaluating under
    # ZeRO-3 sharding in-process is fragile, and the fake-quant structure is rebuilt on reload.
    if not skip_save:
        engine.save_16bit_model(f"{out_dir}/student16", "model.safetensors")
        print(f"saved consolidated 16-bit student -> {out_dir}/student16")

    if rank == 0:
        with open(f"{out_dir}/e2e_metrics.json", "w") as f:
            json.dump(result, f, indent=2)
        if push_repo:
            from huggingface_hub import HfApi

            api = HfApi()
            api.upload_file(
                path_or_fileobj=f"{out_dir}/e2e_metrics.json",
                path_in_repo="e2e_metrics.json",
                repo_id=push_repo,
                repo_type="dataset",
            )
            if not skip_save:
                api.upload_file(
                    path_or_fileobj=f"{out_dir}/student16/model.safetensors",
                    path_in_repo="e2e_student/model.safetensors",
                    repo_id=push_repo,
                    repo_type="dataset",
                )
            print(f"uploaded metrics + checkpoint -> {push_repo}")

    return result
