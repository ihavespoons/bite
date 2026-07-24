#!/usr/bin/env python
"""Stage 3 diagnostic: eval PTQ-init-only quantized students (no QAD) on a capped MMLU.

The first full block-wise QAD run collapsed to chance MMLU (0.245 vs 0.839). Before spending on
the end-to-end polish, isolate the cause cheaply (forward-only): compare the PTQ-init floor with
lm_head ternarized (current policy) vs lm_head kept FP16. If keeping lm_head high-precision lifts
MMLU off the ~25% floor, the 2048->248320 output projection was a primary culprit.

    python scripts/eval_quant.py --config configs/ternary.yaml --eval-limit 100 --push-repo ...
"""

from __future__ import annotations

import argparse

from bite.config import load_config


def _eval_variant(model_id, cfg, tokenizer, *, keep_lmhead: bool, tasks, limit, load_weights=None):  # pragma: no cover - runner
    import torch

    from bite.models.loader import build_student
    from bite.quant.policy import PrecisionPolicy, _default_keep
    from bite.quant.experts import ptq_init_experts
    from bite.quant.ptq import ptq_init_model
    from bite.eval.harness import run_lm_eval_model

    mode = cfg["quant"]["mode"]
    keep = _default_keep() + ((r"lm_head",) if keep_lmhead else ())
    policy = PrecisionPolicy(default=mode, keep_patterns=keep)
    student, swapped, experts = build_student(
        model_id, mode=mode, group_size=cfg["quant"]["group_size"], policy=policy
    )
    if load_weights:
        # eval a QAD-trained checkpoint: rebuild the fake-quant structure, then load the trained
        # latents (the parametrizations re-apply fake-quant on forward). NOTE: DeepSpeed's
        # save_16bit_model writes a torch pickle regardless of the filename's extension.
        try:
            import safetensors.torch as st

            sd = st.load_file(load_weights)
        except Exception:
            sd = torch.load(load_weights, map_location="cpu", weights_only=True)
            print("loaded as torch pickle (DeepSpeed save_16bit_model format)")
        missing, unexpected = student.load_state_dict(sd, strict=False)
        tag = f"e2e-trained (missing={len(missing)}, unexpected={len(unexpected)})"
        print(f"[{tag}] loaded {load_weights}")
    else:
        tag = "lm_head=FP16" if keep_lmhead else "lm_head=ternary"
        print(f"[{tag}] quantized {len(swapped)} linears + {len(experts)} expert tensors")
        ptq_init_model(student, hessians=None, percdamp=cfg["ptq"]["percdamp"])
        ptq_init_experts(student)
    student.eval()
    if hasattr(student, "config"):
        student.config.use_cache = False

    res = run_lm_eval_model(
        student, tokenizer, tasks, batch_size=cfg["eval"]["batch_size"], limit=limit
    )
    mmlu = res["results"].get("mmlu", {}).get("acc,none")
    base = (cfg.get("eval", {}) or {}).get("teacher_baseline") or {}
    out = {"variant": tag, "swapped": len(swapped), "mmlu": mmlu}
    if mmlu is not None and base.get("mmlu"):
        out["mmlu_retained"] = mmlu / base["mmlu"]
        print(f"[{tag}] MMLU {mmlu:.4f} vs FP16 {base['mmlu']:.4f} -> {out['mmlu_retained']:.1%} retained")
    # fully release the 70GB model before the next variant: lm-eval's HFLM leaves a reference
    # cycle, so del + empty_cache alone don't reclaim it (was OOMing the 2nd build)
    import gc

    del student
    gc.collect()
    torch.cuda.empty_cache()
    return out


def main() -> None:  # pragma: no cover - runner
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ternary.yaml")
    ap.add_argument("--model", default=None)
    ap.add_argument("--eval-tasks", default="mmlu")
    ap.add_argument("--variant", default="both", choices=("both", "ternary", "fp16"), help="which lm_head variant(s) to eval")
    ap.add_argument("--load-weights", default=None, help="eval a QAD-trained consolidated checkpoint (safetensors) instead of PTQ-init; local path or repo-relative (downloaded from --load-repo)")
    ap.add_argument("--load-repo", default="ihavespoons/bite-baseline", help="HF dataset to download --load-weights from when not a local path")
    ap.add_argument("--eval-limit", type=int, default=100, help="examples per MMLU subtask (fast directional read)")
    ap.add_argument("--out", default="outputs/diag/eval_quant.json")
    ap.add_argument("--push-repo", default=None)
    args = ap.parse_args()

    import json
    import os

    from bite.models.loader import load_tokenizer

    cfg = load_config(args.config)
    model_id = args.model or cfg["model"]["id"]
    tasks = args.eval_tasks.split(",")
    tokenizer = load_tokenizer(model_id)

    results = []
    if args.load_weights:
        weights = args.load_weights
        if not os.path.exists(weights):
            from huggingface_hub import hf_hub_download

            weights = hf_hub_download(args.load_repo, weights, repo_type="dataset")
            print(f"downloaded checkpoint: {weights}")
        results.append(
            _eval_variant(model_id, cfg, tokenizer, keep_lmhead=False, tasks=tasks, limit=args.eval_limit, load_weights=weights)
        )
    else:
        variants = {"both": (False, True), "ternary": (False,), "fp16": (True,)}[args.variant]
        for keep_lmhead in variants:
            results.append(
                _eval_variant(model_id, cfg, tokenizer, keep_lmhead=keep_lmhead, tasks=tasks, limit=args.eval_limit)
            )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("diagnostic results:", results)
    if args.push_repo:
        from huggingface_hub import HfApi

        HfApi().upload_file(
            path_or_fileobj=args.out,
            path_in_repo=os.path.basename(args.out),
            repo_id=args.push_repo,
            repo_type="dataset",
        )
        print("uploaded ->", args.push_repo)


if __name__ == "__main__":
    main()
