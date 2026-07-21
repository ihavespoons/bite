#!/usr/bin/env python
"""Stage 2: expert-coverage calibration -> PTQ init -> (optional) teacher-logit precompute.

Runner-side. Uses **naive per-group PTQ init** (scalable): GPTQ-with-Hessian needs a
layer-sequential rewrite to fit a 256-expert MoE in memory (holding one Hessian per expert
Linear at once is hundreds of GB) — deferred; QAD heals the naive init.

    python scripts/run_ptq.py --config configs/ternary.yaml --model /model --max-seqs 64
"""

from __future__ import annotations

import argparse

from bite.config import load_config


def main() -> None:  # pragma: no cover - runner-side
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ternary.yaml")
    ap.add_argument("--out", default="outputs/ptq")
    ap.add_argument("--model", default=None, help="model path/id override (e.g. mounted /model)")
    ap.add_argument("--max-seqs", type=int, default=None, help="cap calibration sequences (cheap runs)")
    ap.add_argument("--skip-teacher-logits", action="store_true")
    ap.add_argument("--skip-save", action="store_true", help="don't write the 70GB student (validation)")
    ap.add_argument("--teacher-only", action="store_true", help="only precompute teacher logits (skip student/coverage/PTQ)")
    ap.add_argument("--push-repo", default=None, help="HF dataset repo to persist coverage + teacher logits")
    args = ap.parse_args()
    cfg = load_config(args.config)
    model_id = args.model or cfg["model"]["id"]

    import json
    import os

    import torch

    from bite.data.calib import stream_calibration
    from bite.models.loader import build_student, load_teacher, load_tokenizer
    from bite.moe.calibration import ExpertCoverage, attach_coverage_hooks
    from bite.quant.experts import ptq_init_experts
    from bite.quant.ptq import ptq_init_model
    from bite.train.teacher import precompute_teacher_logits

    tokenizer = load_tokenizer(model_id)
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"
    have_coverage = False

    if not args.teacher_only:
        # 1. student: QuantLinear (attention/lm-head) + fused-expert fake-quant (the MoE bulk)
        student, swapped, experts = build_student(
            model_id, mode=cfg["quant"]["mode"], group_size=cfg["quant"]["group_size"]
        )
        device = str(next(student.parameters()).device)
        print(f"quantized {len(swapped)} linears + {len(experts)} fused-expert tensors ({cfg['quant']['mode']})")

        # 2. calibration pass: per-expert coverage (forward only)
        coverage = ExpertCoverage(cfg["moe"]["num_experts"], cfg["moe"]["routed_experts"])
        handles = attach_coverage_hooks(student, coverage)
        print(f"coverage hooks attached: {len(handles)}")
        student.eval()
        n_batches = 0
        with torch.no_grad():
            for batch in stream_calibration(cfg, tokenizer, device=device, max_seqs=args.max_seqs):
                student(**batch)
                n_batches += 1
        for h in handles:
            h.remove()
        print(f"calibration batches processed: {n_batches}")
        summary = coverage.summary()
        print("expert coverage:", summary)
        if coverage.dead_experts():
            print(f"WARN: {len(coverage.dead_experts())} dead experts — widen calibration coverage")
        with open(f"{args.out}/coverage.json", "w") as f:
            json.dump(summary, f, indent=2)
        have_coverage = True

        # 3. naive per-group PTQ init: QuantLinear weights + fused-expert latents
        done = ptq_init_model(student, hessians=None, percdamp=cfg["ptq"]["percdamp"])
        n_experts = ptq_init_experts(student)
        print(f"PTQ-initialized {len(done)} quant linears + {n_experts} expert tensors")
        if not args.skip_save:
            student.save_pretrained(f"{args.out}/student")
            print(f"saved student -> {args.out}/student")
        del student
        torch.cuda.empty_cache()

    # 4. teacher-logit precompute (student already freed above if it was built)
    if not args.skip_teacher_logits:
        teacher = load_teacher(model_id)
        precompute_teacher_logits(
            teacher,
            stream_calibration(cfg, tokenizer, device=device, max_seqs=args.max_seqs),
            f"{args.out}/teacher_topk",
            k=cfg["qad"]["teacher_topk"],
        )
        print(f"saved teacher top-{cfg['qad']['teacher_topk']} logits -> {args.out}/teacher_topk")

    # 5. persist artifacts to HF so a detached job's output survives (coverage always; logits if made)
    if args.push_repo:
        from huggingface_hub import HfApi

        api = HfApi()
        if have_coverage:
            api.upload_file(
                path_or_fileobj=f"{args.out}/coverage.json",
                path_in_repo="coverage.json",
                repo_id=args.push_repo,
                repo_type="dataset",
            )
        if not args.skip_teacher_logits:
            api.upload_folder(
                folder_path=f"{args.out}/teacher_topk",
                path_in_repo="teacher_topk",
                repo_id=args.push_repo,
                repo_type="dataset",
            )
        print(f"uploaded artifacts -> {args.push_repo}")


if __name__ == "__main__":
    main()
