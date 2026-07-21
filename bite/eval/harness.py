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


def _ensure_nltk() -> None:  # pragma: no cover - runner-side
    """Fetch the nltk tokenizer data ifeval needs (no-op if already present)."""
    import nltk

    for pkg in ("punkt", "punkt_tab"):
        try:
            nltk.download(pkg, quiet=True)
        except Exception:  # noqa: BLE001 - best-effort; ifeval will error clearly if truly missing
            pass


def run_lm_eval(
    model_id_or_path: str,
    tasks: list[str],
    batch_size: int = 8,
    apply_chat_template: bool = False,
    max_gen_toks: int | None = None,
    **kw,
):  # pragma: no cover - runner
    """Text-benchmark suite via ``lm-eval``, robust to the multimodal model class.

    Loads the model with our multimodal-aware loader and wraps it in an ``HFLM`` instance, so
    ``lm-eval`` evaluates the already-loaded model instead of re-loading it with the default
    ``AutoModelForCausalLM`` (which may not resolve ``...ForConditionalGeneration``).

    ``batch_size`` is a fixed integer, not ``"auto"``: with this model's ~248K vocab, an
    auto-selected large batch overflows 32-bit CUDA index math on the logits tensor.

    For generative tasks on this instruction/thinking model, pass ``apply_chat_template=True``
    (formats prompts with the model's chat template) and a larger ``max_gen_toks`` so long CoT
    isn't truncated before the answer — without these the generative scores are artificially low.
    """
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    from bite.models.loader import load_teacher, load_tokenizer

    _ensure_nltk()  # ifeval tokenizes with nltk punkt at eval time
    model = load_teacher(model_id_or_path)
    hflm_kw = {"max_gen_toks": max_gen_toks} if max_gen_toks else {}
    lm = HFLM(
        pretrained=model, tokenizer=load_tokenizer(model_id_or_path), batch_size=batch_size, **hflm_kw
    )
    if apply_chat_template:
        kw["apply_chat_template"] = True
        kw.setdefault("fewshot_as_multiturn", True)
    return simple_evaluate(model=lm, tasks=tasks, **kw)
