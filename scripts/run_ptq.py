#!/usr/bin/env python
"""Stage 2 orchestration: coverage -> Hessian calibration -> GPTQ PTQ init -> teacher logits.

Runner-side (needs the model + a GPU). The mechanisms it wires together — coverage hooks,
Hessian collection, GPTQ init, top-k teacher logits — are all unit-tested on CPU.
"""

from __future__ import annotations

import argparse

from bite.config import load_config


def iter_calibration_batches(cfg: dict):  # pragma: no cover - runner-side (datasets + tokenizer)
    """Yield tokenized, expert-balanced calibration batches (HF datasets + tokenizer)."""
    raise NotImplementedError(
        "runner-side: load cfg['calibration']['dataset'], tokenize, and (2nd pass) oversample "
        "the rare-expert tail via bite.moe.calibration.sample_weights"
    )


def main() -> None:  # pragma: no cover - runner-side
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ternary.yaml")
    ap.add_argument("--out", default="outputs/ptq")
    args = ap.parse_args()
    cfg = load_config(args.config)

    import torch

    from bite.models.loader import build_student, load_teacher
    from bite.moe.calibration import ExpertCoverage, attach_coverage_hooks
    from bite.quant.hessian import HessianCollector, quant_linear_modules
    from bite.quant.ptq import ptq_init_model
    from bite.train.teacher import precompute_teacher_logits

    # 1. student with QuantLinear installed per the precision policy
    student, swapped = build_student(
        cfg["model"]["id"], mode=cfg["quant"]["mode"], group_size=cfg["quant"]["group_size"]
    )
    print(f"swapped {len(swapped)} linears to low-bit")

    # 2. one calibration pass: per-expert coverage + input Hessians
    coverage = ExpertCoverage(cfg["moe"]["num_experts"], cfg["moe"]["routed_experts"])
    cov_handles = attach_coverage_hooks(student, coverage)
    hess = HessianCollector().attach(quant_linear_modules(student))
    student.eval()
    with torch.no_grad():
        for batch in iter_calibration_batches(cfg):
            student(**batch)
    for h in cov_handles:
        h.remove()
    hess.detach()
    print("expert coverage:", coverage.summary())
    if coverage.dead_experts():
        print(f"WARN: {len(coverage.dead_experts())} dead experts — increase calibration coverage")

    # 3. GPTQ PTQ init of every QuantLinear latent weight
    ptq_init_model(student, hess.H, percdamp=cfg["ptq"]["percdamp"])
    student.save_pretrained(f"{args.out}/student")

    # 4. precompute top-k teacher logits (removes the resident teacher from QAD)
    teacher = load_teacher(cfg["model"]["id"])
    precompute_teacher_logits(
        teacher, iter_calibration_batches(cfg), f"{args.out}/teacher_topk", k=cfg["qad"]["teacher_topk"]
    )


if __name__ == "__main__":
    main()
