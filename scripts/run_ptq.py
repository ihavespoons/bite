#!/usr/bin/env python
"""Stage 2: expert-balanced calibration -> per-expert coverage report + GPTQ ternary PTQ init.

Also precomputes top-k teacher logits over the QAD data (stored as an HF dataset). Runner-side.
"""

import argparse

from bite.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ternary.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    # Runner-side wiring:
    #   1. build_student(...) and attach ExpertCoverage hooks to each router gate
    #   2. stream the calibration set; log ExpertCoverage.summary() (dead experts, gini)
    #   3. per Linear: accumulate H = X Xᵀ, call gptq_quantize(W, H, mode, group_size)
    #   4. precompute + store top-{teacher_topk} teacher logits for the QAD data
    raise SystemExit(
        f"run on the cloud runner; config loaded ok: mode={cfg['quant']['mode']}, "
        f"group_size={cfg['quant']['group_size']}, teacher_topk={cfg['qad']['teacher_topk']}"
    )


if __name__ == "__main__":
    main()
