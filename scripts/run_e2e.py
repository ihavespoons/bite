#!/usr/bin/env python
"""Stage 3: end-to-end ternary QAD under DeepSpeed ZeRO-3 CPU offload. Runner-side.

Launched via the deepspeed launcher so torch.distributed is set up (single GPU is fine):

    deepspeed --num_gpus=1 scripts/run_e2e.py --config configs/ternary.yaml \
        --teacher-repo ihavespoons/bite-baseline --smoke

Smoke first (2 steps, no eval/save) to validate the ZeRO-3 + parametrized-model integration,
then a short go/no-go (--steps N --eval-tasks mmlu --eval-limit 100) to see if MMLU moves off
the ~0.25 floor.
"""

import argparse

from bite.config import load_config
from bite.train.end2end import run_end2end_qad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ternary.yaml")
    ap.add_argument("--model", default=None)
    ap.add_argument("--teacher-repo", default=None, help="HF dataset with teacher_topk/ shards")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--eval-tasks", default=None)
    ap.add_argument("--eval-limit", type=int, default=None)
    ap.add_argument("--out", default="outputs/e2e")
    ap.add_argument("--push-repo", default=None)
    ap.add_argument("--skip-save", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="2 steps, no eval/save — integration check")
    # deepspeed launcher injects --local_rank
    ap.add_argument("--local_rank", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    run_end2end_qad(
        cfg,
        model_id=args.model,
        teacher_repo=args.teacher_repo,
        steps=args.steps,
        micro_batch=args.micro_batch,
        accum=args.accum,
        eval_tasks=args.eval_tasks.split(",") if args.eval_tasks else None,
        eval_limit=args.eval_limit,
        out_dir=args.out,
        push_repo=args.push_repo,
        skip_save=args.skip_save,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
