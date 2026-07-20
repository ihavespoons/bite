#!/usr/bin/env python
"""Submit a bite pipeline stage as an HF Job.

Runs locally (needs ``huggingface_hub`` + ``hf auth login``); it submits the job and prints its
URL. The job clones this repo, installs the right extras, and runs the stage on the chosen GPU
flavor. See ``docs/running_on_hf.md``.

    python scripts/launch_hf.py --stage tests    --flavor t4-small
    python scripts/launch_hf.py --stage baseline --flavor h200 --timeout 6h
    python scripts/launch_hf.py --stage ptq      --flavor h200 --timeout 4h --mount-model
"""

from __future__ import annotations

import argparse
import os

IMAGE = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel"
REPO = "https://github.com/ihavespoons/bite"

# stage -> (pip extras, shell command run inside the cloned repo)
STAGES = {
    "tests": (
        "dev",
        "python -m pytest -q && python -c \"import torch;from bite.quant.fakequant import "
        "fake_quantize;w=torch.randn(4,256,device='cuda',requires_grad=True);"
        "fake_quantize(w,'ternary',128).sum().backward();print('CUDA quant OK',torch.cuda.get_device_name())\"",
    ),
    "baseline": ("model,eval", "python scripts/run_baseline.py --config {config}"),
    "ptq": ("model", "python scripts/run_ptq.py --config {config} --out outputs/ptq"),
    "qad": ("model,train", "python scripts/run_qad.py --config {config}"),
    "spike": ("model", "echo 'build the PrismML fork first; see scripts/stage0_spike.py'"),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, choices=sorted(STAGES))
    ap.add_argument("--flavor", default="h200")
    ap.add_argument("--timeout", default="2h")
    ap.add_argument("--config", default="configs/ternary.yaml")
    ap.add_argument("--ref", default="main", help="git ref of the repo to run")
    ap.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    ap.add_argument("--mount-model", action="store_true", help="mount the model repo at /model")
    args = ap.parse_args()

    from huggingface_hub import run_job

    extras, stage_cmd = STAGES[args.stage]
    stage_cmd = stage_cmd.format(config=args.config)
    inner = (
        f"set -e; git clone -q --branch {args.ref} {REPO} && cd bite "
        f"&& pip install -q -e '.[{extras}]' && {stage_cmd}"
    )

    kwargs: dict = {
        "image": IMAGE,
        "command": ["bash", "-lc", inner],
        "flavor": args.flavor,
        "timeout": args.timeout,
    }
    token = os.environ.get("HF_TOKEN")
    if token:
        kwargs["secrets"] = {"HF_TOKEN": token}
    if args.mount_model:
        from huggingface_hub import Volume

        kwargs["volumes"] = [Volume(type="model", source=args.model, mount_path="/model")]

    job = run_job(**kwargs)
    print("submitted:", job.url)
    print("logs:      hf jobs logs", job.id)


if __name__ == "__main__":
    main()
