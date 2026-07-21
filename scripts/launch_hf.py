#!/usr/bin/env python
"""Submit a bite pipeline stage as an HF Job.

Runs locally (needs ``huggingface_hub`` + ``hf auth login``); submits the job and prints its
URL. The job installs git, clones this (private) repo via a GitHub token, installs the right
extras, and runs the stage on the chosen GPU flavor. See ``docs/running_on_hf.md``.

Set tokens in the env so they're passed as encrypted job secrets (never printed):
    GITHUB_TOKEN=$(gh auth token) HF_TOKEN=$(hf auth token) \
        python scripts/launch_hf.py --stage ptq --flavor h200 --mount-model \
        --extra-args "--max-seqs 64 --skip-teacher-logits"
"""

from __future__ import annotations

import argparse
import os

IMAGE = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel"
REPO = "github.com/ihavespoons/bite"

# stage -> (pip extras, command template; {config}/{model_arg}/{extra} filled below)
STAGES = {
    "tests": ("dev", "python -m pytest -q && python scripts/cuda_smoke.py {extra}"),
    "checktasks": ("eval", "python scripts/check_tasks.py {extra}"),
    "baseline": ("model,eval", "python scripts/run_baseline.py --config {config} {extra}"),
    "ptq": ("model", "python scripts/run_ptq.py --config {config} --out outputs/ptq {model_arg} {extra}"),
    "qad": ("model,train", "python scripts/run_qad.py --config {config} {extra}"),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, choices=sorted(STAGES))
    ap.add_argument("--flavor", default="h200")
    ap.add_argument("--timeout", default="2h")
    ap.add_argument("--config", default="configs/ternary.yaml")
    ap.add_argument("--ref", default="main", help="git ref to run")
    ap.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    ap.add_argument("--mount-model", action="store_true", help="mount the model repo at /model")
    ap.add_argument("--extra-args", default="", help="appended to the stage command")
    ap.add_argument("--detach", action="store_true", help="return immediately (survives disconnect)")
    args = ap.parse_args()

    from huggingface_hub import run_job

    extras, template = STAGES[args.stage]
    model_arg = "--model /model" if args.mount_model else ""
    stage_cmd = template.format(config=args.config, model_arg=model_arg, extra=args.extra_args).strip()

    gh = os.environ.get("GITHUB_TOKEN", "")
    clone_url = f"https://x-access-token:${{GITHUB_TOKEN}}@{REPO}" if gh else f"https://{REPO}"
    inner = (
        "set -e; apt-get -qq update && apt-get -qq install -y git >/dev/null 2>&1; "
        f"git clone -q --branch {args.ref} {clone_url}; cd bite; "
        f"pip install -q -e '.[{extras}]'; {stage_cmd}"
    )

    secrets = {k: os.environ[k] for k in ("GITHUB_TOKEN", "HF_TOKEN") if os.environ.get(k)}
    kwargs: dict = {
        "image": IMAGE,
        "command": ["bash", "-c", inner],
        "flavor": args.flavor,
        "timeout": args.timeout,
        "detach": args.detach,
    }
    if secrets:
        kwargs["secrets"] = secrets
    if args.mount_model:
        from huggingface_hub import Volume

        kwargs["volumes"] = [Volume(type="model", source=args.model, mount_path="/model")]

    job = run_job(**kwargs)
    print("submitted:", job.url)
    print("logs:     hf jobs logs", job.id)


if __name__ == "__main__":
    main()
