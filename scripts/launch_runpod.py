#!/usr/bin/env python
"""Submit a bite pipeline stage as a RunPod GPU pod (~1.7x cheaper GPU-hours than HF Jobs).

Reuses launch_hf.py's STAGES. Pods differ from HF Jobs in two dangerous ways, both handled here:

**1. Pods bill until deleted.** A container that exits — including a failure during setup —
leaves the pod rented and charging. Termination therefore runs from a bash ``trap ... EXIT``
covering EVERY exit path, plus a host-side watchdog (``--watch``) as an independent backstop.
(An earlier version terminated only on the success path and only from inside the container;
a setup failure would have billed indefinitely.)

**2. Startup is slow and silent.** A cold machine can take 25+ MINUTES to pull the image; the
pod is rented ("RUNNING") but ``runtime`` stays null and nothing executes. Treating that silence
as failure killed two pods here — the second one 13 seconds after it finally came alive. So
``--watch`` separates "still pulling" (long ``--start-timeout``) from "container up but not
reporting" (``--startup-grace``, measured from container start) from "died mid-run" (``--stale-seconds``).

Live logs come from RunPod's own SSE endpoint, ``GET api.runpod.io/v2/pods/{id}/logs``.
``pod_heartbeat.py`` additionally archives the log to the HF dataset because that endpoint
404s the moment a pod is terminated — which is exactly when you most want the post-mortem.

Tokens come from the env, never printed or committed:
    RUNPOD_API_KEY=... HF_TOKEN=$(hf auth token) \
        python scripts/launch_runpod.py --stage e2e --gpu-count 8 --extra-args "..."

Ops:  --watch <pod_id> (attach to a running pod) | --status <id> | --terminate <id> | --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
import urllib.error
import urllib.request

from launch_hf import IMAGE, REPO, STAGES  # same stages, same docker image

REST = "https://rest.runpod.io/v1"
HF_RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"


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
    except urllib.error.HTTPError as e:
        sys.exit(f"RunPod API {e.code} on {method} {path}: {e.read().decode(errors='replace')}")
    return json.loads(raw) if raw else {}


def _hf_get(repo: str, path: str, token: str) -> str | None:
    """Fetch a file from an HF dataset, bypassing caches. None if it doesn't exist yet."""
    req = urllib.request.Request(
        HF_RESOLVE.format(repo=repo, path=path),
        headers={
            "Authorization": f"Bearer {token}",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        if e.code in (401, 403, 404):
            return None
        raise


def fetch_logs(pod_id: str, api_key: str, since: str | None, read_seconds: int = 8,
               tail: int = 200) -> tuple[list[str], str | None]:
    """Pull new container log lines from RunPod's SSE log API.

    ``GET api.runpod.io/v2/pods/{id}/logs`` streams Server-Sent Events; we read for a few
    seconds, then close and return what arrived. Note the stream 404s once a pod is terminated,
    which is why pod_heartbeat.py still archives the log to HF for post-mortems.
    """
    q = f"source=container&{'since=' + since if since else f'tail={tail}'}"
    req = urllib.request.Request(
        f"https://api.runpod.io/v2/pods/{pod_id}/logs?{q}",
        headers={"Authorization": f"Bearer {api_key}", "Accept": "text/event-stream"},
    )
    lines: list[str] = []
    last = since
    try:
        with urllib.request.urlopen(req, timeout=read_seconds) as r:
            deadline = time.time() + read_seconds
            for raw in r:
                if time.time() > deadline:
                    break
                s = raw.decode("utf-8", errors="replace").strip()
                if not s.startswith("data:"):
                    continue
                try:
                    ev = json.loads(s[5:].strip())
                except json.JSONDecodeError:
                    continue
                if ev.get("line"):
                    lines.append(ev["line"].rstrip())
                if ev.get("ts"):
                    last = ev["ts"]
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError):
        pass  # transient stream errors / pod gone — the caller's checks decide what it means
    return lines, last


def terminate(pod_id: str, api_key: str, why: str = "") -> None:
    _rest("DELETE", f"/pods/{pod_id}", api_key)
    print(f"terminated {pod_id}{f' ({why})' if why else ''}", flush=True)


def watch(pod_id: str, log_name: str, api_key: str, hf_token: str, repo: str,
          poll: int, stale: int, grace: int, start_timeout: int) -> int:
    """Stream the pod's shipped log; terminate the pod if it goes silent or finishes.

    Three distinct phases, because conflating them killed a pod that was merely slow to start:

    1. **Image pull** — RunPod rents the pod immediately but the container can take 25+ MINUTES
       to start on a cold machine (9GB image). ``runtime`` is null throughout and the pod is
       billed little or nothing, so this phase gets its own generous ``start_timeout``.
    2. **Container setup** — ``runtime`` is populated; the first heartbeat should follow within
       minutes (apt, pip, clone). ``grace`` is measured FROM CONTAINER START, not from launch.
    3. **Running** — a heartbeat older than ``stale`` means the container died; terminate.

    Ctrl-C detaches WITHOUT terminating.
    """
    started, shown, container_start = time.time(), 0, None
    since: str | None = None
    print(f"watching {pod_id} — live container logs + heartbeat (poll {poll}s | pull timeout "
          f"{start_timeout}s | setup grace {grace}s | stale {stale}s) — Ctrl-C detaches", flush=True)
    while True:
        # primary visibility: RunPod's own log stream (works the moment the container starts)
        new_lines, since = fetch_logs(pod_id, api_key, since)
        for ln in new_lines:
            print(f"  | {ln}", flush=True)
        status = _hf_get(repo, f"logs/{log_name}.status", hf_token)
        now = time.time()
        if status is None:
            waited = now - started
            rt = _rest("GET", f"/pods/{pod_id}", api_key).get("runtime") or {}
            uptime = rt.get("uptime")
            if uptime is None:  # phase 1: container not started yet
                if waited > start_timeout:
                    terminate(pod_id, api_key, f"container never started within {start_timeout}s")
                    return 1
                print(f"  [{int(waited)}s] pulling image / scheduling (no container yet; "
                      f"timeout {start_timeout}s)", flush=True)
            else:  # phase 2: container up, waiting on its first heartbeat
                if container_start is None:
                    container_start = now - uptime
                    print(f"  container started (uptime {uptime}s) — setup grace {grace}s begins", flush=True)
                since_start = now - container_start
                if since_start > grace:
                    terminate(pod_id, api_key, f"no heartbeat within {grace}s of container start")
                    return 1
                print(f"  [{int(waited)}s] container up {int(since_start)}s, awaiting first heartbeat", flush=True)
        else:
            parts = status.split()
            ts, state = int(parts[0]), (parts[1] if len(parts) > 1 else "running")
            age = int(now - ts)
            log = _hf_get(repo, f"logs/{log_name}.log", hf_token) or ""
            if len(log) > shown:  # print only what's new since last poll
                print(log[shown:], end="", flush=True)
                shown = len(log)
            if state != "running":
                print(f"\n=== job finished: {state} ===", flush=True)
                terminate(pod_id, api_key, "job complete")
                return 0 if state.endswith(" 0") else 1
            if age > stale:
                terminate(pod_id, api_key, f"heartbeat stale by {age}s")
                return 1
        time.sleep(poll)


def build_entrypoint(args, log_name_expr: str) -> str:
    """The in-container script. Order matters — see the numbered comments."""
    extras, template = STAGES[args.stage]
    stage_cmd = template.format(
        config=args.config, model_arg="", extra=args.extra_args, num_gpus=args.gpu_count
    ).strip()
    extra_pip = f"pip install -q {args.extra_pip}; " if args.extra_pip else ""
    hb = (
        f"python /workspace/bite/scripts/pod_heartbeat.py --log /workspace/job.log "
        f"--repo {shlex.quote(args.log_repo)} --name \"$NAME\""
    )
    return (
        # NOT `set -e`: a failing setup step must still reach the trap, not abort the shell in a
        # way that skips termination. Every step is followed by the trap on exit regardless.
        "set -uo pipefail; "
        f"NAME={log_name_expr}; "
        'kill_pod() { curl -s -X DELETE "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" '
        '-H "Authorization: Bearer $RUNPOD_API_KEY" >/dev/null 2>&1 || true; }; '
        # 1. trap = the ONLY termination path, so success, failure, and timeout all converge
        #    here. It flushes a final log synchronously before deleting the pod.
        'finish() { code=$?; echo "exit $code" > /workspace/hb.stop; '
        f'[ -f /workspace/bite/scripts/pod_heartbeat.py ] && {hb} --interval 0 >/dev/null 2>&1; '  # final flush
        'kill_pod; }; '
        "trap finish EXIT; "
        # 2. capture everything from here on into the log the heartbeat ships
        # /workspace only exists when a volume is mounted; we run volume-less, so create it
        # BEFORE the redirect below (a failed exec redirect kills the shell instantly)
        "mkdir -p /workspace; cd /workspace; "
        "exec > >(tee -a /workspace/job.log) 2>&1; "
        "apt-get -qq update && apt-get -qq install -y git curl >/dev/null 2>&1; "
        # 3. huggingface_hub FIRST and the heartbeat started BEFORE the heavy install, so a hang
        #    or failure during setup is visible instead of looking like a dead pod
        "pip install -q huggingface_hub; "
        f"git clone -q --branch {args.ref} https://{REPO}; cd bite; "
        f"({hb} --interval {args.heartbeat} &) ; "
        f"pip install -q -e '.[{extras}]'; {extra_pip}"
        f"timeout {args.max_seconds} bash -c {shlex.quote(stage_cmd)}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=sorted(STAGES))
    ap.add_argument("--status", metavar="POD_ID")
    ap.add_argument("--terminate", metavar="POD_ID")
    ap.add_argument("--watch", metavar="POD_ID", help="attach to an already-running pod")
    ap.add_argument("--watch-name", default=None, help="log basename for --watch (default bite-<stage>-<pod_id>)")
    ap.add_argument("--gpu-type", default="NVIDIA A100-SXM4-80GB")
    ap.add_argument("--gpu-count", type=int, default=1)
    ap.add_argument("--cloud", default="SECURE", choices=("SECURE", "COMMUNITY"))
    ap.add_argument("--disk", type=int, default=400, help="container disk GB (model 70 + init 70 + ckpts)")
    ap.add_argument("--max-seconds", type=int, default=36_000, help="hard timeout around the stage command")
    ap.add_argument("--config", default="configs/ternary.yaml")
    ap.add_argument("--ref", default="main")
    ap.add_argument("--extra-pip", default="")
    ap.add_argument("--extra-args", default="")
    ap.add_argument("--log-repo", default="ihavespoons/bite-baseline", help="HF dataset the job ships logs to")
    ap.add_argument("--name", default=None)
    ap.add_argument("--heartbeat", type=int, default=300, help="seconds between log uploads")
    ap.add_argument("--poll", type=int, default=120, help="--watch poll interval")
    ap.add_argument("--stale-seconds", type=int, default=1200, help="terminate if the heartbeat is older than this")
    ap.add_argument("--startup-grace", type=int, default=900, help="seconds from CONTAINER START to first heartbeat")
    ap.add_argument("--start-timeout", type=int, default=3600, help="seconds to allow for image pull before the container starts (cold pulls run 25min+)")
    ap.add_argument("--no-watch", action="store_true", help="create the pod and exit (watch later with --watch)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        sys.exit("set RUNPOD_API_KEY in the environment (never commit it)")
    hf_token = os.environ.get("HF_TOKEN", "")

    if args.status:
        print(json.dumps(_rest("GET", f"/pods/{args.status}", api_key), indent=2, default=str))
        return
    if args.terminate:
        terminate(args.terminate, api_key)
        return
    if args.watch:
        if not hf_token:
            sys.exit("--watch needs HF_TOKEN to read the shipped log")
        name = args.watch_name or f"bite-{args.stage or 'e2e'}-{args.watch}"
        sys.exit(watch(args.watch, name, api_key, hf_token, args.log_repo,
                       args.poll, args.stale_seconds, args.startup_grace, args.start_timeout))
    if not args.stage:
        sys.exit("--stage is required (or --status/--terminate/--watch)")
    if not hf_token:
        sys.exit("set HF_TOKEN in the environment (artifacts and logs push to the HF dataset)")

    pod_name = args.name or f"bite-{args.stage}"
    body = {
        "name": pod_name,
        "imageName": IMAGE,
        "cloudType": args.cloud,
        "computeType": "GPU",
        "gpuTypeIds": [args.gpu_type],
        "gpuCount": args.gpu_count,
        "containerDiskInGb": args.disk,
        "volumeInGb": 0,
        "env": {"HF_TOKEN": hf_token, "RUNPOD_API_KEY": api_key},
        # $RUNPOD_POD_ID is injected by RunPod, so the log name is derivable on both sides
        "dockerEntrypoint": ["bash", "-c", build_entrypoint(args, f"{pod_name}-$RUNPOD_POD_ID")],
    }

    if args.dry_run:
        print(json.dumps(dict(body, env={k: "<redacted>" for k in body["env"]}), indent=2))
        return

    pod = _rest("POST", "/pods", api_key, body)
    pid = pod.get("id") if isinstance(pod, dict) else None
    log_name = f"{pod_name}-{pid}"
    print(f"pod created: {pid}  (${args.gpu_count} x {args.gpu_type})")
    print(f"live log:    https://huggingface.co/datasets/{args.log_repo}/blob/main/logs/{log_name}.log")
    print(f"attach:      python scripts/launch_runpod.py --watch {pid} --watch-name {log_name}")
    print(f"terminate:   python scripts/launch_runpod.py --terminate {pid}")
    if args.no_watch:
        print("NOTE: --no-watch means no host-side backstop; the in-container trap still terminates.")
        return
    sys.exit(watch(pid, log_name, api_key, hf_token, args.log_repo,
                   args.poll, args.stale_seconds, args.startup_grace, args.start_timeout))


if __name__ == "__main__":
    main()
