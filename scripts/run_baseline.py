#!/usr/bin/env python
"""Stage 0: load the FP16 teacher, run the eval suite -> the 100% reference. Runner-side."""

import argparse

from bite.config import load_config
from bite.eval.harness import run_lm_eval


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ternary.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    results = run_lm_eval(cfg["model"]["id"], cfg["eval"]["tasks"])
    print(results["results"])  # persist as eval.teacher_baseline for later comparison


if __name__ == "__main__":
    main()
