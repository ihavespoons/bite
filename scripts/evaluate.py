#!/usr/bin/env python
"""Evaluate a quantized checkpoint: benchmark suite + brittle-collapse probes vs FP16. Runner-side."""

import argparse

from bite.config import load_config
from bite.eval.harness import retained_fraction, run_lm_eval


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ternary.yaml")
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    results = run_lm_eval(args.checkpoint, cfg["eval"]["tasks"])
    baseline = cfg["eval"].get("teacher_baseline")
    if baseline:
        for task, score in results["results"].items():
            print(task, "retained:", retained_fraction(score, baseline[task]))


if __name__ == "__main__":
    main()
