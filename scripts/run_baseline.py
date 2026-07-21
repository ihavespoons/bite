#!/usr/bin/env python
"""Stage 0: load the FP16 teacher, run the eval suite -> the 100% reference. Runner-side.

Persists results to JSON and (optionally) uploads them to an HF dataset so a multi-hour run
survives a dropped stream. ``--allow-code`` enables humaneval's code execution.
"""

import argparse
import json


def _jsonable(o):
    return o.item() if hasattr(o, "item") else str(o)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ternary.yaml")
    ap.add_argument("--tasks", nargs="*", help="override eval.tasks from the config")
    ap.add_argument("--limit", type=float, default=None, help="cap samples/task (smoke runs)")
    ap.add_argument("--out", default="outputs/baseline.json")
    ap.add_argument("--push-repo", default=None, help="HF dataset repo to upload results to")
    ap.add_argument("--allow-code", action="store_true", help="enable humaneval code execution")
    args = ap.parse_args()

    from bite.config import load_config
    from bite.eval.harness import run_lm_eval

    cfg = load_config(args.config)
    tasks = args.tasks or cfg["eval"]["tasks"]
    kw = {}
    if args.limit:
        kw["limit"] = args.limit
    if args.allow_code:
        kw["confirm_run_unsafe_code"] = True

    results = run_lm_eval(
        cfg["model"]["id"], tasks, batch_size=cfg["eval"].get("batch_size", 8), **kw
    )
    res = results["results"]
    print(res)

    import os

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, default=_jsonable)
    print("saved", args.out)

    if args.push_repo:
        from huggingface_hub import HfApi

        HfApi().upload_file(
            path_or_fileobj=args.out,
            path_in_repo="baseline.json",
            repo_id=args.push_repo,
            repo_type="dataset",
        )
        print("uploaded to", args.push_repo)


if __name__ == "__main__":
    main()
