#!/usr/bin/env python
"""Stage 3/4: block-wise Quantization-Aware Distillation against the frozen teacher. Runner-side."""

import argparse

from bite.config import load_config
from bite.train.qad import run_blockwise_qad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ternary.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_blockwise_qad(cfg)


if __name__ == "__main__":
    main()
