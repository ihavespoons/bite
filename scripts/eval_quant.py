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


def _eval_variant(
    model_id,
    cfg,
    tokenizer,
    *,
    keep_lmhead: bool,
    tasks,
    limit,
    load_weights=None,
    threshold_ratio=None,
    keep_patterns=(),
    expert_keep_patterns=(),
    mode_override: str | None = None,
    offline: bool = False,
    rotate: bool = False,
    expert_axis: int = 1,
    keep_expert_frac: float = 0.0,
    keep_expert_by: str = "error",
    label=None,
):  # pragma: no cover - runner
    import torch

    from bite.models.loader import build_student
    from bite.quant.policy import PrecisionPolicy, _default_keep
    from bite.quant.experts import apply_expert_mixed_precision, ptq_init_experts
    from bite.quant.ptq import ptq_init_model
    from bite.eval.harness import run_lm_eval_model
    from bite.quant.fakequant import effective_bits as _effective_bits

    mode = mode_override or cfg["quant"]["mode"]
    if offline:
        # forward-only measurement path: no parametrizations/STE. Lets us (a) group the fused
        # expert tensors along their CONTRACTION axis (the fake-quant path groups the last dim,
        # which for (E, hidden, 2*inter) is the output axis — scales shared across weights that
        # never meet in a dot product) and (b) simulate Hadamard rotation exactly.
        from bite.models.loader import load_teacher
        from bite.quant.offline import quantize_model_offline
        from bite.quant.policy import _default_keep as _dk

        student = load_teacher(model_id)
        stats = quantize_model_offline(
            student, mode=mode, group_size=cfg["quant"]["group_size"],
            threshold_ratio=threshold_ratio, rotate=rotate, expert_axis=expert_axis,
            keep_patterns=_dk() + tuple(keep_patterns) + tuple(expert_keep_patterns),
        )
        tag = label or f"offline {mode} rotate={rotate} expert_axis={expert_axis}"
        swapped, experts, mixed = {}, {}, {}
        print(f"[{tag}] {stats}", flush=True)
    else:
        keep = _default_keep() + ((r"lm_head",) if keep_lmhead else ()) + tuple(keep_patterns)
        policy = PrecisionPolicy(default=mode, keep_patterns=keep)
        student, swapped, experts = build_student(
            model_id,
            mode=mode,
            group_size=cfg["quant"]["group_size"],
            threshold_ratio=threshold_ratio,
            policy=policy,
            expert_keep_patterns=tuple(expert_keep_patterns),
        )
        # expert-level mixed precision is applied AFTER the build: selecting the worst-represented
        # experts needs the weights themselves
        mixed = (
            apply_expert_mixed_precision(student, keep_expert_frac, keep_expert_by)
            if keep_expert_frac
            else {}
        )
    if not offline and load_weights:
        # eval a QAD-trained checkpoint: rebuild the fake-quant structure, then load the trained
        # latents (the parametrizations re-apply fake-quant on forward). NOTE: DeepSpeed's
        # save_16bit_model writes a torch pickle regardless of the filename's extension.
        try:
            import safetensors.torch as st

            sd = st.load_file(load_weights)
        except Exception:
            # modern torch.load dispatches on the .safetensors EXTENSION and would bounce back
            # into safetensors, so read through a file object to bypass that
            with open(load_weights, "rb") as fh:
                sd = torch.load(fh, map_location="cpu", weights_only=True)
            print("loaded as torch pickle (DeepSpeed save_16bit_model format)")
        missing, unexpected = student.load_state_dict(sd, strict=False)
        tag = f"e2e-trained (missing={len(missing)}, unexpected={len(unexpected)})"
        print(f"[{tag}] loaded {load_weights}")
    elif not offline:
        rule = "absmean" if threshold_ratio is None else str(threshold_ratio)
        rule = rule if mode.startswith(("ternary", "binary")) else "n/a"  # intN has no scale rule
        tag = label or (("lm_head=FP16" if keep_lmhead else "lm_head=ternary") + f", rule={rule}")
        print(f"[{tag}] quantized {len(swapped)} linears + {len(experts)} expert tensors")
        ptq_init_model(student, hessians=None, percdamp=cfg["ptq"]["percdamp"])
        ptq_init_experts(student)
    student.eval()
    if hasattr(student, "config"):
        student.config.use_cache = False

    # perplexity tasks use ROLLING loglikelihood: window size defaults to the model's huge
    # context and each window materializes full-vocab (248K) logits -> batch 8 OOM'd an H200
    # next to the 70GB model. Run them separately at batch 1 with a 2048-token window.
    ppl_tasks = [t for t in tasks if t in {"wikitext"}]
    ll_tasks = [t for t in tasks if t not in ppl_tasks]
    res: dict = {"results": {}}
    if ll_tasks:
        r = run_lm_eval_model(
            student, tokenizer, ll_tasks, batch_size=cfg["eval"]["batch_size"], limit=limit
        )
        res["results"].update(r["results"])
    if ppl_tasks:
        r = run_lm_eval_model(
            student, tokenizer, ppl_tasks, batch_size=1, max_length=2048, limit=limit
        )
        res["results"].update(r["results"])
    mmlu = res["results"].get("mmlu", {}).get("acc,none")
    base = (cfg.get("eval", {}) or {}).get("teacher_baseline") or {}
    out = {
        "variant": tag,
        "swapped": len(swapped),
        "experts_quantized": sum(1 for v in experts.values() if v != "keep"),
        "expert_tensors_kept": sum(1 for v in experts.values() if v == "keep"),
        "kept_expert_slots_per_tensor": (max(mixed.values()) if mixed else 0),
        "keep_expert_frac": keep_expert_frac,
        "mode": mode,
        "bpw": _effective_bits(mode, cfg["quant"]["group_size"]),
        "mmlu": mmlu,
    }
    for t in tasks:  # headline numerics per requested task (e.g. wikitext perplexities)
        r = res["results"].get(t) or {}
        out[t] = {k: v for k, v in r.items() if isinstance(v, (int, float))}
    print(f"[{tag}] metrics: {out}")  # survives a later-variant crash (the JSON dump may not)
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


def _parse_rule(s):
    """CLI/config -> quantize_ternary threshold_ratio: None (absmean), 'optimal', or a float."""
    if s in (None, "", "absmean", "none", "null"):
        return None
    if isinstance(s, (int, float)):
        return float(s)
    return "optimal" if s == "optimal" else float(s)


def main() -> None:  # pragma: no cover - runner
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ternary.yaml")
    ap.add_argument("--model", default=None)
    ap.add_argument("--eval-tasks", default="mmlu")
    ap.add_argument("--variant", default="both", choices=("both", "ternary", "fp16"), help="which lm_head variant(s) to eval")
    ap.add_argument(
        "--rules",
        default=None,
        help="comma-separated ternary scale rules to compare (absmean|optimal|<float ratio>); "
        "default: the config's quant.ternary_threshold",
    )
    ap.add_argument("--load-weights", default=None, help="eval a QAD-trained consolidated checkpoint (safetensors) instead of PTQ-init; local path or repo-relative (downloaded from --load-repo)")
    ap.add_argument("--load-repo", default="ihavespoons/bite-baseline", help="HF dataset to download --load-weights from when not a local path")
    ap.add_argument("--eval-limit", type=int, default=100, help="examples per MMLU subtask (fast directional read)")
    ap.add_argument("--keep-patterns", default="", help="comma-separated regexes: nn.Linear modules to hold at high precision (sensitivity ablations)")
    ap.add_argument("--expert-keep-patterns", default="", help="comma-separated regexes: FUSED expert tensors to hold at high precision (the policy only covers Linears)")
    ap.add_argument("--keep-expert-frac", type=float, default=0.0, help="expert-level mixed precision: fraction of expert SLOTS kept high-precision in every quantized tensor")
    ap.add_argument("--keep-expert-by", default="error", choices=("error", "first"), help="which slots to keep: highest ternary quantization error, or the first N (control)")
    ap.add_argument("--offline", action="store_true", help="one-shot offline quantization instead of fake-quant parametrizations (enables --rotate and correct expert grouping axis)")
    ap.add_argument("--rotate", action="store_true", help="quantize in a Hadamard-rotated basis (incoherence processing); offline only")
    ap.add_argument("--expert-axis", type=int, default=1, help="axis of the fused expert tensors to group along; 1 = contraction (correct), -1 = last (the fake-quant default)")
    ap.add_argument("--mode", default=None, help="quantization mode override: ternary | binary | intN (2-8). Default: config quant.mode")
    ap.add_argument("--label", default=None, help="label recorded in the results JSON")
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

    keep_patterns = tuple(p for p in args.keep_patterns.split(",") if p)
    expert_keep_patterns = tuple(p for p in args.expert_keep_patterns.split(",") if p)

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
        if args.rules:
            rules = [_parse_rule(r) for r in args.rules.split(",")]
        else:
            rules = [_parse_rule((cfg.get("quant") or {}).get("ternary_threshold"))]
        variants = {"both": (False, True), "ternary": (False,), "fp16": (True,)}[args.variant]
        for rule in rules:
            for keep_lmhead in variants:
                results.append(
                    _eval_variant(
                        model_id,
                        cfg,
                        tokenizer,
                        keep_lmhead=keep_lmhead,
                        tasks=tasks,
                        limit=args.eval_limit,
                        threshold_ratio=rule,
                        keep_patterns=keep_patterns,
                        expert_keep_patterns=expert_keep_patterns,
                        mode_override=args.mode,
                        offline=args.offline,
                        rotate=args.rotate,
                        expert_axis=args.expert_axis,
                        keep_expert_frac=args.keep_expert_frac,
                        keep_expert_by=args.keep_expert_by,
                        label=args.label,
                    )
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
