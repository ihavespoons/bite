"""Evaluation — % of FP16 retained, plus the whitepaper's brittle-collapse probes.

Perplexity alone misses the qualitative sub-4-bit failure modes the whitepaper flags, so we
track them explicitly: chain-of-thought self-consistency, tool-call JSON parse rate, and
multi-turn/agentic coherence. Benchmark scoring wraps ``lm-eval``; the probe scorers here are
pure functions and unit-tested on CPU.
"""

from __future__ import annotations

import json
from collections.abc import Sequence


def tool_call_parse_rate(outputs: Sequence[str]) -> float:
    """Fraction of model outputs that parse as valid JSON tool calls with a ``name`` field."""
    if not outputs:
        return 0.0
    ok = 0
    for o in outputs:
        try:
            obj = json.loads(o)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(obj, dict) and "name" in obj:
            ok += 1
    return ok / len(outputs)


def cot_self_consistency(answers: Sequence[str]) -> float:
    """Share of the modal answer across sampled CoT traces (higher = more consistent)."""
    if not answers:
        return 0.0
    counts: dict[str, int] = {}
    for a in answers:
        key = a.strip()
        counts[key] = counts.get(key, 0) + 1
    return max(counts.values()) / len(answers)


def retained_fraction(quantized_score: float, teacher_score: float) -> float:
    """Percent of FP16 retained, the whitepaper's headline metric."""
    if teacher_score == 0:
        return 0.0
    return quantized_score / teacher_score


def run_lm_eval(model_id_or_path: str, tasks: list[str], **kw):  # pragma: no cover - runner
    """Wrapper around ``lm-eval`` for the fixed benchmark suite. Runs on the cloud runner."""
    from lm_eval import simple_evaluate

    return simple_evaluate(model="hf", model_args=f"pretrained={model_id_or_path}", tasks=tasks, **kw)
