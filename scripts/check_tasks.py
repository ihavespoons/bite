#!/usr/bin/env python
"""Cheap CPU check: every eval task's config + deps load (no model download).

Catches missing task dependencies (langdetect, nltk, pass_at_k, ...) for pennies on CPU
instead of ~$1.5 of H200 per failure (the model loads before task configs on the real run).
"""

import sys

from lm_eval.tasks import TaskManager, get_task_dict

from bite.config import load_config
from bite.eval.harness import _ensure_nltk


def main() -> None:
    tasks = sys.argv[1:] or load_config("configs/ternary.yaml")["eval"]["tasks"]
    _ensure_nltk()
    td = get_task_dict(tasks, TaskManager())
    names = [str(k) for k in td]  # keys mix ConfigurableGroup (mmlu) and str -> stringify
    print(f"tasks load OK ({len(td)} built): {names[:6]}{' ...' if len(td) > 6 else ''}")


if __name__ == "__main__":
    main()
