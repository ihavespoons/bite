#!/usr/bin/env python
"""Stage 3/4: block-wise Quantization-Aware Distillation against the frozen teacher. Runner-side.

Validate cheap first (the per-block forward adapter is the main unknown):

    python scripts/run_qad.py --config configs/ternary.yaml --model /model \
        --max-seqs 4 --max-blocks 2 --skip-save

then scale up (all blocks, save + persist the healed student's metrics):

    python scripts/run_qad.py --config configs/ternary.yaml --push-repo ihavespoons/bite-baseline
"""

import argparse

from bite.config import load_config
from bite.train.qad import run_blockwise_qad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ternary.yaml")
    ap.add_argument("--model", default=None, help="model path/id override (e.g. mounted /model)")
    ap.add_argument("--out", default="outputs/qad")
    ap.add_argument("--max-seqs", type=int, default=None, help="cap calibration sequences (cheap runs)")
    ap.add_argument("--max-blocks", type=int, default=None, help="heal only the first N blocks (validation)")
    ap.add_argument("--steps", type=int, default=None, help="optimizer steps per block (overrides config block_steps)")
    ap.add_argument("--eval-tasks", default=None, help="comma-separated lm-eval tasks to run on the healed student (e.g. mmlu)")
    ap.add_argument("--eval-limit", type=int, default=None, help="cap eval examples per task (cheap runs)")
    ap.add_argument("--skip-save", action="store_true", help="don't write the healed student (validation)")
    ap.add_argument("--push-repo", default=None, help="HF dataset repo to persist metrics + healed student")
    args = ap.parse_args()
    cfg = load_config(args.config)
    metrics = run_blockwise_qad(
        cfg,
        model_id=args.model,
        max_seqs=args.max_seqs,
        max_blocks=args.max_blocks,
        steps=args.steps,
        eval_tasks=args.eval_tasks.split(",") if args.eval_tasks else None,
        eval_limit=args.eval_limit,
        out_dir=args.out,
        push_repo=args.push_repo,
        skip_save=args.skip_save,
    )
    print("QAD metrics:", metrics)


if __name__ == "__main__":
    main()
