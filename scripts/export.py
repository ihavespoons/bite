#!/usr/bin/env python
"""Stage 5/6: quantize the vision tower (HQQ 4-bit) and export the LM to GGUF. Runner-side."""

import argparse

from bite.config import load_config
from bite.export.gguf import export_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ternary.yaml")
    ap.add_argument("--checkpoint", required=True, help="healed student checkpoint dir")
    args = ap.parse_args()
    cfg = load_config(args.config)
    export_model(args.checkpoint, cfg["export"]["out_dir"])


if __name__ == "__main__":
    main()
