#!/usr/bin/env python
"""Submit a bite pipeline stage as a RunPod GPU pod (~1.7x cheaper than HF Jobs).

Reuses launch_hf.py's STAGES. Differences from HF Jobs, all handled here:
  * pods bill until removed -> the pod SELF-TERMINATES after the stage command (and a
    ``timeout`` guard bounds a hang; a dead-on-arrival pod still needs manual --terminate);
  * pod logs vanish at termination -> stdout/stderr are tee'd and uploaded to the HF
    dataset (``logs/<name>.log``) before the pod removes itself;
  * no volume mounts -> the model/init downloads from the HF hub inside the pod.

Tokens come from the env and are passed into the pod's env (never printed/committed):
    RUNPOD_API_KEY=... HF_TOKEN=$(hf auth token) \
        python scripts/launch_runpod.py --stage e2e --gpu-count 8 --extra-args "..."

Ops:  --status <pod_id> | --terminate <pod_id> | --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import urllib.request

from launch_hf import IMAGE, REPO, STAGES  # same stages, same docker image

REST = "https://rest.runpod.io/v1"


def _rest(method: str, path: str, api_key: str, body: dict | None = None) -> dict | list:
    req = urllib.request.Request(
        f"{REST}{path}",
        data=json.dumps(body).encode() if body is not None else None,
        method=method,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:  # surface RunPod's error body, not just the code
        sys.exit(f"RunPod API {e.code} on {method} {path}: {e.read().decode(errors='replace')}")
    return json.loads(raw) if raw else {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=sorted(STAGES))
    ap.add_argument("--status", metavar="POD_ID", help="print a pod's current status and exit")
    ap.add_argument("--terminate", metavar="POD_ID", help="remove a pod and exit")
    ap.add_argument("--gpu-type", default="NVIDIA A100-SXM4-80GB", help="RunPod gpuTypeId (H100: 'NVIDIA H100 80GB HBM3')")
    ap.add_argument("--gpu-count", type=int, default=1)
    ap.add_argument("--cloud", default="SECURE", choices=("SECURE", "COMMUNITY"))
    ap.add_argument("--disk", type=int, default=400, help="container disk GB (model 70 + init 70 + ckpt 70 + slack)")
    ap.add_argument("--max-seconds", type=int, default=36_000, help="hard timeout around the stage command (default 10h)")
    ap.add_argument("--config", default="configs/ternary.yaml")
    ap.add_argument("--ref", default="main", help="git ref to run")
    ap.add_argument("--extra-pip", default="", help="extra pip spec(s) installed after the project extras")
    ap.add_argument("--extra-args", default="", help="appended to the stage command")
    ap.add_argument("--log-repo", default="ihavespoons/bite-baseline", help="HF dataset for the job log ('' disables)")
    ap.add_argument("--name", default=None, help="pod name (default bite-<stage>)")
    ap.add_argument("--dry-run", action="store_true", help="print the request body (secrets redacted) and exit")
    args = ap.parse_args()

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        sys.exit("set RUNPOD_API_KEY in the environment (never commit it)")

    if args.status:
        print(json.dumps(_rest("GET", f"/pods/{args.status}", api_key), indent=2, default=str))
        return
    if args.terminate:
        _rest("DELETE", f"/pods/{args.terminate}", api_key)
        print(f"terminated {args.terminate}")
        return
    if not args.stage:
        sys.exit("--stage is required (or --status/--terminate)")

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        sys.exit("set HF_TOKEN in the environment (artifacts/logs push to the HF dataset)")

    name = args.name or f"bite-{args.stage}"
    extras, template = STAGES[args.stage]
    stage_cmd = template.format(
        config=args.config, model_arg="", extra=args.extra_args, num_gpus=args.gpu_count
    ).strip()
    extra_pip = f"pip install -q {args.extra_pip}; " if args.extra_pip else ""

    log_upload = ""
    if args.log_repo:
        py = (
            "from huggingface_hub import HfApi; "
            f"HfApi().upload_file(path_or_fileobj='/workspace/job.log', path_in_repo='logs/{name}-'"
            "+__import__('os').environ.get('RUNPOD_POD_ID','pod')+'.log', "
            f"repo_id='{args.log_repo}', repo_type='dataset')"
        )
        log_upload = f"python -c {shlex.quote(py)} || true; "

    # self-termination is unconditional (success or failure) — logs are preserved above
    inner = (
        "set -e; cd /workspace; apt-get -qq update && apt-get -qq install -y git curl >/dev/null 2>&1; "
        f"git clone -q --branch {args.ref} https://{REPO}; cd bite; "
        f"pip install -q -e '.[{extras}]'; {extra_pip}"
        f"set +e; set -o pipefail; timeout {args.max_seconds} bash -c {shlex.quote(stage_cmd)} 2>&1 | tee /workspace/job.log; "
        f"code=$?; cd /workspace; {log_upload}"
        'curl -s -X DELETE "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" '
        '-H "Authorization: Bearer $RUNPOD_API_KEY"; exit $code'
    )

    body = {
        "name": name,
        "imageName": IMAGE,
        "cloudType": args.cloud,
        "computeType": "GPU",
        "gpuTypeIds": [args.gpu_type],
        "gpuCount": args.gpu_count,
        "containerDiskInGb": args.disk,
        "volumeInGb": 0,
        "env": {"HF_TOKEN": hf_token, "RUNPOD_API_KEY": api_key},
        "dockerEntrypoint": ["bash", "-c", inner],
    }

    if args.dry_run:
        red = dict(body, env={k: "<redacted>" for k in body["env"]})
        print(json.dumps(red, indent=2))
        return

    pod = _rest("POST", "/pods", api_key, body)
    pid = pod.get("id") if isinstance(pod, dict) else None
    print(f"pod created: {pid}")
    print(f"status:      python scripts/launch_runpod.py --status {pid}")
    print(f"terminate:   python scripts/launch_runpod.py --terminate {pid}")
    print(f"console:     https://console.runpod.io/pods (log also uploads to {args.log_repo or 'nowhere'})")


if __name__ == "__main__":
    main()
