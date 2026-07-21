"""Block-wise QAD engine — heal one transformer block at a time.

The schedule that fits a single H200: walk the decoder blocks in order; for each, optimize
only that block's low-bit latent weights to match the FP16 teacher block's output on the
running (already-healed) activations, so just one block's optimizer state is resident. A final
optional end-to-end logit-KL polish (8-bit optimizer) uses the precomputed teacher top-k logits.

The core pieces — block discovery, trainable-param selection (which auto-freezes router,
shared expert and norms since only :class:`QuantLinear` weights are trained), optimizer
construction, and the per-block distill step — are pure PyTorch and unit-tested on CPU. The
full-model orchestration is runner-side.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from bite.quant.experts import expert_latent_params
from bite.quant.quantlinear import QuantLinear


def iter_blocks(model: nn.Module, layers_path: str | None = None) -> list[nn.Module]:
    """Return the decoder blocks. Uses ``layers_path`` (e.g. ``model.layers``) if given,
    else the longest :class:`nn.ModuleList` in the model."""
    if layers_path:
        obj = model
        for part in layers_path.split("."):
            obj = getattr(obj, part)
        return list(obj)
    best: list[nn.Module] = []
    for _, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > len(best):
            best = list(module)
    return best


def quant_parameters(module: nn.Module) -> list[nn.Parameter]:
    """The trainable low-bit latent weights in a block: ``QuantLinear.weight`` **and** the fused
    MoE expert latent Parameters (``gate_up_proj``/``down_proj``, via their parametrization).

    In this MoE the experts are the bulk of the low-bit weights and live as fused 3D
    ``nn.Parameter``s, not ``nn.Linear`` — so healing them requires collecting
    :func:`bite.quant.experts.expert_latent_params` alongside the ``QuantLinear`` weights.
    Everything else (router gate, shared expert, RMSNorm, DeltaNet params, biases) is left out,
    which is exactly the precision policy: those stay in higher precision and are not healed.
    """
    params = [m.weight for m in module.modules() if isinstance(m, QuantLinear)]
    params.extend(expert_latent_params(module))
    return params


def build_optimizer(params: list[nn.Parameter], *, lr: float = 1e-4, kind: str = "adam"):
    """Adam optimizer; ``kind='adam8bit'`` uses bitsandbytes for the end-to-end polish pass."""
    if kind == "adam8bit":
        import bitsandbytes as bnb

        return bnb.optim.Adam8bit(params, lr=lr)
    return torch.optim.AdamW(params, lr=lr)


def distill_block(
    student_block: nn.Module,
    teacher_block: nn.Module,
    inputs: Iterable[Tensor],
    *,
    steps: int = 100,
    lr: float = 1e-4,
    optimizer: str = "adam",
    forward_fn: Callable[[nn.Module, Tensor], Tensor] = lambda b, x: b(x),
) -> list[float]:
    """Heal ``student_block`` to match ``teacher_block`` outputs on ``inputs`` (hidden-MSE).

    Only the block's :class:`QuantLinear` latent weights train (via STE). Returns the loss
    history so the caller can gate/early-stop. ``forward_fn`` adapts to real decoder-block
    call signatures (hidden states + position/mask); the default calls ``block(x)``.
    """
    params = quant_parameters(student_block)
    if not params:
        return []
    opt = build_optimizer(params, lr=lr, kind=optimizer)
    teacher_block.eval()
    for p in teacher_block.parameters():
        p.requires_grad_(False)

    batches = list(inputs)
    # The teacher block is frozen and each input is fixed, so its targets are constant across
    # steps — compute them once (fewer forwards) and, crucially, so the teacher's forward never
    # shares/mutates tensors that the student's autograd graph depends on.
    with torch.no_grad():
        targets = [forward_fn(teacher_block, x).detach() for x in batches]

    history: list[float] = []
    for step in range(steps):
        idx = step % len(batches)
        opt.zero_grad()
        out = forward_fn(student_block, batches[idx])
        loss = F.mse_loss(out.float(), targets[idx].float())
        loss.backward()
        opt.step()
        history.append(float(loss.detach()))
    # Free this block's gradients before the optimizer is discarded. Otherwise each healed
    # block's .grad tensors (the fused expert latents alone are ~1.6GB/block) stay resident and
    # accumulate across all 40 blocks -> OOM partway through (was hitting it at block ~34).
    opt.zero_grad(set_to_none=True)
    return history


# --- per-block forward adapter -------------------------------------------------------------
#
# Real Qwen3.5 decoder blocks are NOT called ``block(hidden)``: a full-attention layer takes
# ``(hidden_states, position_embeddings=..., attention_mask=..., position_ids=...)`` while a
# GatedDeltaNet layer takes a different mix, and both may return a tuple. Rather than hard-code
# either signature, we **capture** the exact ``(args, kwargs)`` each block is called with during
# one real model forward (:func:`capture_block_inputs`) and replay them, threading only the
# hidden state between blocks. This makes the DeltaNet-vs-attention difference a non-issue.


@dataclass
class BlockInput:
    """A captured block call with the hidden state pulled out so it can be re-threaded."""

    hidden: Tensor
    rest: tuple = ()
    kwargs: dict = field(default_factory=dict)


def _block_output(out):
    """Normalize a decoder-block return (tensor or ``(hidden, ...)`` tuple) to the hidden tensor."""
    return out[0] if isinstance(out, (tuple, list)) else out


def forward_block(block: nn.Module, item: BlockInput) -> Tensor:
    """Run ``block`` on a :class:`BlockInput`, returning its output hidden state.

    Suitable as ``distill_block``'s ``forward_fn`` for real decoder blocks: it reuses the
    captured non-hidden kwargs (rotary embeddings, masks) and substitutes the threaded hidden.
    """
    return _block_output(block(item.hidden, *item.rest, **item.kwargs))


def capture_block_inputs(  # pragma: no cover - requires model + GPU
    model: nn.Module, blocks: list[nn.Module], batch: dict
) -> tuple[Tensor, list[tuple]]:
    """One real forward; return ``(block0_hidden, per_block_(rest_args, kwargs))``.

    Uses forward-pre-hooks with ``with_kwargs=True`` so we record exactly what each block
    receives — no assumptions about the (differing) DeltaNet vs full-attention signatures.
    """
    caps: list[tuple | None] = [None] * len(blocks)
    grabbed: dict[str, Tensor] = {}

    def mk(i: int):
        def pre(_mod, args, kwargs):
            hidden = args[0] if args else kwargs["hidden_states"]
            if i == 0:
                grabbed["h0"] = hidden.detach()
            rest = tuple(args[1:]) if args else ()
            # Drop cache/state kwargs: a threaded KV/recurrent cache is mutated in-place by the
            # block, which corrupts the autograd graph when we replay the (frozen) inputs for
            # healing. use_cache=False below also prevents the model from creating one.
            kw = {
                k: v
                for k, v in kwargs.items()
                if k not in ("hidden_states", "past_key_value", "past_key_values", "use_cache")
            }
            caps[i] = (rest, kw)

        return pre

    handles = [b.register_forward_pre_hook(mk(i), with_kwargs=True) for i, b in enumerate(blocks)]
    try:
        with torch.no_grad():
            try:
                model(**batch, use_cache=False)
            except TypeError:
                model(**batch)  # model forward doesn't take use_cache (e.g. a toy stack)
    finally:
        for h in handles:
            h.remove()
    return grabbed["h0"], [c for c in caps if c is not None]


def run_blockwise_qad(  # pragma: no cover - requires model + GPU
    config: dict,
    *,
    model_id: str | None = None,
    max_seqs: int | None = None,
    max_blocks: int | None = None,
    steps: int | None = None,
    eval_tasks: list[str] | None = None,
    eval_limit: int | None = None,
    out_dir: str = "outputs/qad",
    push_repo: str | None = None,
    skip_save: bool = False,
) -> dict:
    """Full-model block-wise QAD orchestration (runner-side).

    Schedule (fits one H200 — only the student is resident on GPU; the FP16 teacher stays on CPU
    and one block at a time is streamed to GPU to produce targets):

      1. Build the PTQ-initialized student (``build_student`` + ``ptq_init_model`` +
         ``ptq_init_experts``) and load the frozen FP16 teacher onto CPU.
      2. For a modest calibration set, capture each block's real call signature
         (:func:`capture_block_inputs`) and the block-0 hidden input.
      3. Walk blocks in order; for each, move the teacher block to GPU, run :func:`distill_block`
         (hidden-MSE, STE through the student block's quant latents — QuantLinear **and** fused
         experts), move the teacher block back, then advance the running hidden through the
         healed student block so later blocks see realistic (quant-accumulated) inputs.
      4. Save the healed student; caller evaluates MMLU vs the FP16 baseline (gate ≥ ~93%).

    ``max_seqs``/``max_blocks`` cap the run for cheap GPU validation of the forward adapter.
    Returns a metrics dict (per-block first/last loss) for logging/persistence.
    """
    import json
    import os

    from bite.data.calib import stream_calibration
    from bite.models.loader import build_student, load_teacher, load_tokenizer
    from bite.quant.experts import ptq_init_experts
    from bite.quant.ptq import ptq_init_model

    mid = model_id or config["model"]["id"]
    layers_path = config["qad"].get("layers_path")
    steps = int(steps if steps is not None else config["qad"].get("block_steps", 200))
    lr = float(config["qad"].get("lr", 1e-4))
    os.makedirs(out_dir, exist_ok=True)

    tokenizer = load_tokenizer(mid)

    # 1. PTQ-init student on GPU
    student, swapped, experts = build_student(
        mid, mode=config["quant"]["mode"], group_size=config["quant"]["group_size"]
    )
    device = str(next(student.parameters()).device)
    ptq_init_model(student, hessians=None, percdamp=config["ptq"]["percdamp"])
    ptq_init_experts(student)
    student.eval()
    if hasattr(student, "config"):
        student.config.use_cache = False  # stateless forward -> autograd-safe DeltaNet backward
    log = f"student: {len(swapped)} linears + {len(experts)} expert tensors ({config['quant']['mode']})"
    print(log)

    student_blocks = iter_blocks(student, layers_path)
    if max_blocks:
        student_blocks = student_blocks[:max_blocks]

    # 2. capture block IO on a modest calibration set (activations stay on GPU — keep it small)
    hiddens: list[Tensor] = []
    caps_per_batch: list[list[tuple]] = []
    for batch in stream_calibration(config, tokenizer, device=device, max_seqs=max_seqs):
        h0, caps = capture_block_inputs(student, iter_blocks(student, layers_path), batch)
        hiddens.append(h0)
        caps_per_batch.append(caps)
    print(f"captured block IO for {len(hiddens)} calibration batches")

    # 3. teacher on CPU; stream one block at a time to GPU for targets
    teacher = load_teacher(mid, device_map="cpu")
    teacher_blocks = iter_blocks(teacher, layers_path)

    metrics: dict[str, list[float]] = {}
    for i, sblk in enumerate(student_blocks):
        tblk = teacher_blocks[i].to(device)
        items = [
            BlockInput(hiddens[b], caps_per_batch[b][i][0], caps_per_batch[b][i][1])
            for b in range(len(hiddens))
        ]
        history = distill_block(
            sblk, tblk, items, steps=steps, lr=lr, forward_fn=forward_block
        )
        teacher_blocks[i].to("cpu")
        torch.cuda.empty_cache()
        # advance the running hidden through the healed student block (no grad)
        with torch.no_grad():
            for b in range(len(hiddens)):
                hiddens[b] = forward_block(sblk, items[b]).detach()
        metrics[f"block_{i:02d}"] = [history[0], history[-1]] if history else []
        print(f"block {i:02d}: loss {history[0]:.4e} -> {history[-1]:.4e}" if history else f"block {i:02d}: no quant params")

    del teacher
    torch.cuda.empty_cache()

    result: dict = {"blocks": metrics}

    # in-job eval on the healed in-memory student (the fake-quant structure doesn't round-trip
    # through from_pretrained, so evaluate here rather than reloading a checkpoint)
    if eval_tasks:
        from bite.eval.harness import run_lm_eval_model

        print(f"evaluating healed student on {eval_tasks} ...")
        eval_kw = {"limit": eval_limit} if eval_limit else {}
        res = run_lm_eval_model(
            student, tokenizer, eval_tasks, batch_size=config["eval"]["batch_size"], **eval_kw
        )
        result["eval"] = {t: dict(res["results"].get(t, {})) for t in res.get("results", {})}
        baseline = (config.get("eval", {}) or {}).get("teacher_baseline") or {}
        mmlu = res["results"].get("mmlu", {}).get("acc,none")
        if mmlu is not None:
            result["mmlu"] = mmlu
            if baseline.get("mmlu"):
                result["mmlu_retained"] = mmlu / baseline["mmlu"]
                print(f"MMLU {mmlu:.4f} vs FP16 {baseline['mmlu']:.4f} -> {result['mmlu_retained']:.1%} retained")

    # secure the (small, critical) metrics before the 70GB checkpoint upload
    with open(f"{out_dir}/qad_metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    if push_repo:
        from huggingface_hub import HfApi

        HfApi().upload_file(
            path_or_fileobj=f"{out_dir}/qad_metrics.json",
            path_in_repo="qad_metrics.json",
            repo_id=push_repo,
            repo_type="dataset",
        )
        print(f"uploaded qad_metrics.json -> {push_repo}")

    if not skip_save:
        try:
            student.save_pretrained(f"{out_dir}/student")
            print(f"saved healed student -> {out_dir}/student")
            if push_repo:
                from huggingface_hub import HfApi

                HfApi().upload_folder(
                    folder_path=f"{out_dir}/student",
                    path_in_repo="qad_student",
                    repo_id=push_repo,
                    repo_type="dataset",
                )
                print(f"uploaded healed student -> {push_repo}/qad_student")
        except Exception as e:  # noqa: BLE001 - metrics already secured; don't lose the run on a save error
            print(f"WARN: checkpoint save/upload failed ({e}); metrics were persisted")

    return result
