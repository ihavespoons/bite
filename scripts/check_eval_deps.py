#!/usr/bin/env python
"""Cheap import check for the eval stack (no model download) — validates the lm-eval /
typing_extensions compatibility on the runner image before spending on a GPU baseline."""

from lm_eval import simple_evaluate  # noqa: F401
from lm_eval.models.huggingface import HFLM  # noqa: F401

from bite.eval.harness import run_lm_eval  # noqa: F401

print("eval deps import OK")
