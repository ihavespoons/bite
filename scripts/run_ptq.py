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
    args = ap.parse_args()
    cfg = load_config(args.config)
    model_id = args.model or cfg["model"]["id"]

    import torch

    from bite.data.calib import stream_calibration
    from bite.models.loader import build_student, load_teacher, load_tokenizer
    from bite.moe.calibration import ExpertCoverage, attach_coverage_hooks
    from bite.quant.ptq import ptq_init_model
    from bite.train.teacher import precompute_teacher_logits

    tokenizer = load_tokenizer(model_id)

    # 1. student with QuantLinear across the language weights (vision excluded)
    student, swapped = build_student(
        model_id, mode=cfg["quant"]["mode"], group_size=cfg["quant"]["group_size"]
    )
    device = str(next(student.parameters()).device)
    print(f"swapped {len(swapped)} linears to {cfg['quant']['mode']}")

    # 2. calibration pass: per-expert coverage (forward only)
    coverage = ExpertCoverage(cfg["moe"]["num_experts"], cfg["moe"]["routed_experts"])
    handles = attach_coverage_hooks(student, coverage)
    student.eval()
    with torch.no_grad():
        for batch in stream_calibration(cfg, tokenizer, device=device, max_seqs=args.max_seqs):
            student(**batch)
    for h in handles:
        h.remove()
    print("expert coverage:", coverage.summary())
    if coverage.dead_experts():
        print(f"WARN: {len(coverage.dead_experts())} dead experts — widen calibration coverage")

    # 3. naive per-group PTQ init of every QuantLinear latent weight
    done = ptq_init_model(student, hessians=None, percdamp=cfg["ptq"]["percdamp"])
    print(f"PTQ-initialized {len(done)} quant linears")
    if not args.skip_save:
        student.save_pretrained(f"{args.out}/student")
        print(f"saved student -> {args.out}/student")

    # 4. teacher-logit precompute (free the student first so both 35B models don't co-reside)
    if not args.skip_teacher_logits:
        del student
        torch.cuda.empty_cache()
        teacher = load_teacher(model_id)
        precompute_teacher_logits(
            teacher,
            stream_calibration(cfg, tokenizer, device=device, max_seqs=args.max_seqs),
            f"{args.out}/teacher_topk",
            k=cfg["qad"]["teacher_topk"],
        )
        print(f"saved teacher top-{cfg['qad']['teacher_topk']} logits -> {args.out}/teacher_topk")


if __name__ == "__main__":
    main()
