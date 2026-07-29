#!/usr/bin/env python
"""Ship a running job's log to the HF dataset periodically — the pod's only log surface.

RunPod exposes no log API (only the browser console), and overriding a pod's entrypoint stops
its runtime telemetry from reporting, so ``runtime: null`` says nothing about job health. That
blind spot cost a real run: a healthy job looked dead and was killed at 34 min.

This uploads the tail of ``--log`` every ``--interval`` seconds to ``logs/<name>.log`` in an HF
dataset, plus a ``logs/<name>.status`` file holding ``<unix_ts> <state>``. Two consumers:

* humans/agents get progress without console access;
* ``launch_runpod.py --watch`` treats a stale status file as "container died" and terminates the
  pod — a liveness signal that comes from INSIDE the container, unlike RunPod's telemetry.

Runs as a background process next to the job; exits when ``--stop-file`` appears.
"""

from __future__ import annotations

import argparse
import os
import time


def _tail(path: str, max_bytes: int) -> str:
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
            return f.read().decode("utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def _upload(api, repo: str, name: str, text: str, state: str, log_too: bool = True) -> None:
    """One commit for the status file, and the log only when asked.

    HF caps repository commits (256/hour) and a busy day of pods blew through it, which silently
    broke result uploads. So: status every beat (tiny, and it is the liveness signal), log only
    every Nth beat or at exit.
    """
    if log_too:
        api.upload_file(
            path_or_fileobj=text.encode(),
            path_in_repo=f"logs/{name}.log",
            repo_id=repo,
            repo_type="dataset",
        )
    api.upload_file(
        path_or_fileobj=f"{int(time.time())} {state}\n".encode(),
        path_in_repo=f"logs/{name}.status",
        repo_id=repo,
        repo_type="dataset",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="log file to tail")
    ap.add_argument("--repo", required=True, help="HF dataset to upload into")
    ap.add_argument("--name", required=True, help="basename for logs/<name>.log|.status")
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--max-bytes", type=int, default=256_000, help="tail size per upload")
    ap.add_argument("--log-every", type=int, default=4, help="upload the log every Nth beat (HF caps commits at 256/hour); the status file goes every beat")
    ap.add_argument("--stop-file", default="/workspace/hb.stop")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    beat = 0
    while True:
        stopped = os.path.exists(args.stop_file)
        state = open(args.stop_file).read().strip() if stopped else "running"
        try:
            _upload(
                api, args.repo, args.name, _tail(args.log, args.max_bytes), state or "done",
                log_too=stopped or beat % args.log_every == 0,
            )
        except Exception as e:  # never let telemetry kill the job
            print(f"[heartbeat] upload failed: {e}", flush=True)
        if stopped:
            return
        beat += 1
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
