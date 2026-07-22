#!/usr/bin/env python
"""Stage 3: end-to-end ternary QAD — DeepSpeed ZeRO-3, all on-GPU, client bnb Adam8bit.

Launched via the deepspeed launcher (4×H200) so torch.distributed is set up:

    deepspeed --num_gpus=4 scripts/run_e2e.py --config configs/ternary.yaml \
        --teacher-repo ihavespoons/bite-baseline --smoke

Smoke = accum+1 micro-steps (crosses ONE real optimizer step) + assertions that the bnb 8-bit
state materialized and a probed latent moved — no eval/save. Then the go/no-go run:

    ... scripts/run_e2e.py --config configs/ternary.yaml --steps 500 \
        --teacher-repo ihavespoons/bite-baseline --push-repo ihavespoons/bite-baseline

MMLU is evaluated afterwards in a separate 1-GPU job: eval_quant.py --load-weights.
"""

import argparse

from bite.config import load_config
from bite.train.end2end import run_end2end_qad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ternary.yaml")
    ap.add_argument("--model", default=None)
    ap.add_argument("--teacher-repo", default=None, help="HF dataset with teacher_topk/ shards")
    ap.add_argument("--init-repo", default="ihavespoons/bite-baseline", help="HF dataset with the qad_student/ init checkpoint")
    ap.add_argument("--ptq-init", action="store_true", help="escape hatch: recompute naive PTQ init instead of loading the checkpoint (70GB CPU per rank)")
    ap.add_argument("--steps", type=int, default=200, help="total micro-steps (ignored for --smoke)")
    ap.add_argument("--micro-batch", type=int, default=1, help="keep at 1 — the memory budget assumes it")
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--offload-param", action="store_true", help="offload bf16 param shards to CPU (needed on 80GB A100s; not on H200s)")
    ap.add_argument("--out", default="outputs/e2e")
    ap.add_argument("--push-repo", default=None)
    ap.add_argument("--skip-save", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="accum+1 micro-steps + optimizer-step assertions; no eval/save")
    # deepspeed launcher injects --local_rank
    ap.add_argument("--local_rank", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    run_end2end_qad(
        cfg,
        model_id=args.model,
        teacher_repo=args.teacher_repo,
        init_repo=args.init_repo,
        ptq_init=args.ptq_init,
        steps=args.steps,
        micro_batch=args.micro_batch,
        accum=args.accum,
        offload_param=args.offload_param,
        out_dir=args.out,
        push_repo=args.push_repo,
        skip_save=args.skip_save,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
