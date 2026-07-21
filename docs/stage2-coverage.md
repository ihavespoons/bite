# Stage 2 — expert coverage (FP16, job stage2-coverage)

First MoE-specific result. Quantization path confirmed on the real `qwen3_5_moe` model, and the
router-coverage instrumentation validated.

## Run
- 256 calibration sequences (2048 tokens) streamed from c4, 32 batches, on H200.
- `quantized 191 linears + 80 fused-expert tensors (ternary)` — attention/lm-head via
  `QuantLinear`, all 40 MoE layers' fused `gate_up_proj`/`down_proj` via expert fake-quant.

## Coverage (aggregate over all 40 layer-routers)
| metric | value | reading |
|---|---|---|
| routers hooked | 40 | one per MoE layer (`Qwen3_5MoeTopKRouter`) |
| dead experts | **0 / 256** | every expert fires at least once |
| Gini | **0.51** | markedly imbalanced routing |
| max expert share | 0.174 | a few dominant experts (uniform top-8/256 ≈ 0.031) |
| min expert share | 4.6e-5 | rare experts get almost no signal |

## Implication for quantization
Routing is heavily skewed: the rarest experts receive ~0.005% of assignments, so a naive
calibration set barely exercises them — their PTQ scales / QAD gradients would be unreliable.
This is the concrete motivation for **expert-balanced calibration** (oversampling the rare-expert
tail; see `bite.moe.calibration.sample_weights`, currently a TODO 2nd pass).

## Caveats / next
- `coverage.tokens` aggregates across layers (counted per-router-call), so it's ~40× the distinct
  token count; fractions/Gini are ratios and remain meaningful, but as a **global** mix, not
  per-layer. Per-layer coverage is a useful refinement.
- Naive-PTQ student not saved (deterministic, re-derived at QAD start).

## Teacher logits (job teacher-logits) — QAD input ready ✅
- 512 seqs (~1M tokens), **top-64 logits + input_ids** per token, 64 self-contained shards
  (~407 MB) at `datasets/ihavespoons/bite-baseline/teacher_topk/`.
- Each shard bundles input_ids so QAD trains against the exact tokens the targets came from
  (no c4 re-stream). Load with `bite.train.teacher.load_teacher_shard`.
- Scalable: re-run `run_ptq --teacher-only --max-seqs N --push-repo ...` for a larger set once
  the QAD loop is validated.
