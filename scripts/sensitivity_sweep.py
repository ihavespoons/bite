#!/usr/bin/env python
"""Forward-only sensitivity sweep: WHICH tensors destroy MMLU under ternary quantization?

Motivation (2026-07-27): end-to-end QAD does not recover MMLU. 1M c4 tokens x14 epochs gave
0.2395; 5M *fresh, diverse, expert-balanced* tokens gave 0.2405 — statistically identical, and
both at chance (~0.25) versus FP16's 0.8393. Five times more and better data changed nothing, so
the constraint is unlikely to be the data. That points at the representation, which is testable
WITHOUT training: hold one group of tensors at high precision, PTQ the rest, measure MMLU.

Each variant runs as a SEPARATE SUBPROCESS. The 70GB model downloads once into the HF cache and
is reused, but a fresh process per variant is what makes the sweep survivable: lm-eval's HFLM
leaves reference cycles that previously OOM'd the second in-process build. Results append to the
output JSON and upload after EVERY variant, so a crash or a killed pod still yields the points
already measured.

    python scripts/sensitivity_sweep.py --config configs/slope.yaml --eval-limit 100 \
        --push-repo ihavespoons/bite-baseline --only experts_fp16,attn_fp16
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

# Qwen3.6-35B-A3B naming: attention projections are q/k/v/o_proj; the fused MoE experts are
# 3D params named gate_up_proj / down_proj; the router is `...mlp.gate` (already kept).
VARIANTS: list[dict] = [
    # --- control: the current all-ternary policy (expected ~0.2695 with the optimal rule) ---
    {"name": "baseline_ternary", "why": "control"},
    # --- experiment 1: which half carries the damage, the experts or the attention path? ---
    {"name": "experts_fp16", "why": "all fused MoE experts high-precision; attn+lm_head ternary",
     "expert_keep_patterns": r"gate_up_proj,down_proj"},
    {"name": "attn_fp16", "why": "all attention projections high-precision; experts ternary",
     "keep_patterns": r"q_proj,k_proj,v_proj,o_proj"},
    {"name": "lmhead_fp16", "why": "output projection only (re-test under the optimal rule)",
     "keep_lmhead": True},
    # --- which expert projection matters more: the up/gate fan-out or the down fan-in? ---
    {"name": "down_proj_fp16", "why": "expert output projection only",
     "expert_keep_patterns": r"down_proj"},
    {"name": "gate_up_proj_fp16", "why": "expert input projection only",
     "expert_keep_patterns": r"gate_up_proj"},
    # --- experiment 3: structural ablations — do edge layers carry outsized damage? ---
    {"name": "first4_layers_fp16", "why": "layers 0-3 high-precision (all tensor types)",
     "keep_patterns": r"layers\.[0-3]\.", "expert_keep_patterns": r"layers\.[0-3]\."},
    {"name": "last4_layers_fp16", "why": "layers 36-39 high-precision (all tensor types)",
     "keep_patterns": r"layers\.3[6-9]\.", "expert_keep_patterns": r"layers\.3[6-9]\."},
    # --- experiment 2: expert-level mixed precision (worst-represented experts kept) ---
    {"name": "mixed_10pct_experts", "why": "10% worst-error expert slots high-precision",
     "keep_expert_frac": 0.10},
    {"name": "mixed_25pct_experts", "why": "25% worst-error expert slots high-precision",
     "keep_expert_frac": 0.25},
    {"name": "mixed_10pct_first", "why": "control for the above: first 10% of slots instead",
     "keep_expert_frac": 0.10, "keep_expert_by": "first"},
]


def run_variant(v: dict, args, out_path: str) -> dict:
    """Run one variant in a fresh process; return its parsed result (or an error record)."""
    tmp = f"{out_path}.{v['name']}.json"
    cmd = [
        sys.executable, os.path.join(os.path.dirname(__file__), "eval_quant.py"),
        "--config", args.config,
        "--variant", "fp16" if v.get("keep_lmhead") else "ternary",
        "--eval-tasks", args.eval_tasks,
        "--eval-limit", str(args.eval_limit),
        "--label", f"{v['name']} ({v['why']})",
        "--out", tmp,
    ]
    if args.model:
        cmd += ["--model", args.model]
    if v.get("keep_patterns"):
        cmd += ["--keep-patterns", v["keep_patterns"]]
    if v.get("expert_keep_patterns"):
        cmd += ["--expert-keep-patterns", v["expert_keep_patterns"]]
    if v.get("keep_expert_frac"):
        cmd += ["--keep-expert-frac", str(v["keep_expert_frac"]),
                "--keep-expert-by", v.get("keep_expert_by", "error")]

    print(f"\n=== variant {v['name']}: {v['why']}\n    {' '.join(cmd[2:])}", flush=True)
    proc = subprocess.run(cmd, capture_output=False)
    if proc.returncode != 0:
        print(f"!!! variant {v['name']} failed (rc={proc.returncode}) — continuing", flush=True)
        return {"variant": v["name"], "why": v["why"], "error": f"rc={proc.returncode}"}
    with open(tmp) as f:
        got = json.load(f)
    rec = (got[0] if isinstance(got, list) else got) | {"name": v["name"], "why": v["why"]}
    return rec


def main() -> None:  # pragma: no cover - runner
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/slope.yaml")
    ap.add_argument("--model", default=None)
    ap.add_argument("--eval-tasks", default="mmlu")
    ap.add_argument("--eval-limit", type=int, default=100)
    ap.add_argument("--only", default="", help="comma-separated variant names to run (default: all)")
    ap.add_argument("--out", default="outputs/diag/sensitivity_sweep.json")
    ap.add_argument("--push-repo", default=None)
    args = ap.parse_args()

    only = {s for s in args.only.split(",") if s}
    todo = [v for v in VARIANTS if not only or v["name"] in only]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    print(f"sweep: {len(todo)} variants -> {args.out}", flush=True)

    results: list[dict] = []
    for v in todo:
        results.append(run_variant(v, args, args.out))
        with open(args.out, "w") as f:  # persist + upload after EVERY variant
            json.dump(results, f, indent=2)
        if args.push_repo:
            try:
                from huggingface_hub import HfApi

                HfApi().upload_file(
                    path_or_fileobj=args.out,
                    path_in_repo=os.path.basename(args.out),
                    repo_id=args.push_repo,
                    repo_type="dataset",
                )
            except Exception as e:  # never lose the run over an upload hiccup
                print(f"WARN: upload failed: {e}", flush=True)
        done = [r for r in results if r.get("mmlu")]
        print(f"--- {len(results)}/{len(todo)} done; ranking so far:", flush=True)
        for r in sorted(done, key=lambda r: -(r["mmlu"].get("acc,none") or 0)):
            print(f"      {r['mmlu']['acc,none']:.4f}  {r['name']}", flush=True)

    print("\nsweep complete:", json.dumps([r.get("name") for r in results]), flush=True)
    sys.stdout.flush()
    os._exit(0)  # datasets/HF threads can hang interpreter shutdown; uploads are synchronous


if __name__ == "__main__":
    main()
