# bite

Extreme (1-bit / ternary) quantization pipeline for a **Mixture-of-Experts** LLM —
`Qwen/Qwen3.6-35B-A3B` (35B total / 3B active, 256 experts, hybrid Gated-DeltaNet + attention).

Goal: move the whole language model into **ternary `{-1,0,+1}`** and **binary `{-1,+1}`**
weights with minimal quality loss, starting from the pretrained model (no from-scratch
pretraining), and export to an existing low-bit runtime (GGUF / llama.cpp / bitnet.cpp).
Inspired by PrismML's *Bonsai 27B*, but pushing into the open question Bonsai left untouched:
doing it on an **MoE** model rather than a dense one.

## Method

**PTQ init + Quantization-Aware Distillation (QAD)** with a straight-through estimator:
round to the ternary/binary grid with GPTQ-style error compensation, then heal a
fake-quantized *student* against a frozen FP16 *teacher* (top-k KL + hidden-state MSE + LM CE).

MoE-specific core (`bite/moe/`): expert-balanced calibration with per-expert coverage
tracking, router freezing + distillation to prevent routing drift, and keeping the router
gate + always-on shared expert in higher precision.

## Layout

| Module | Purpose | Status |
|---|---|---|
| `bite/quant/fakequant.py` | ternary/binary g128 quantizers + STE | ✅ built + tested |
| `bite/quant/quantlinear.py` | `QuantLinear` + model-wide module swap | ✅ built + tested |
| `bite/quant/policy.py` | per-tensor precision policy | ✅ built + tested |
| `bite/quant/ptq.py` | GPTQ PTQ init + `ptq_init_model` driver | ✅ built + tested |
| `bite/quant/hessian.py` | activation-Hessian collector for GPTQ | ✅ built + tested |
| `bite/moe/calibration.py` | expert coverage + router hooks + tail balancing | ✅ built + tested |
| `bite/moe/router.py` | router freeze + distillation | ✅ built + tested |
| `bite/train/teacher.py` | top-k teacher-logit precompute | ◐ extractor tested |
| `bite/train/qad.py` | QAD losses (tested) + block-wise loop (runner) | ◐ losses tested |
| `bite/eval/harness.py` | brittle-collapse probes (tested) + lm-eval (runner) | ◐ probes tested |
| `bite/models/loader.py`, `bite/export/*` | model load / GGUF / vision HQQ | ○ runner-side scaffold |

`✅` unit-tested on CPU · `◐` pure-function parts tested, GPU parts scaffolded · `○` runner-side.

## Dev setup (CPU core)

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch numpy pyyaml pytest
.venv/bin/python -m pytest -q
```

## Stage 0 — feasibility: **GO** ✅

See [`docs/stage0_findings.md`](docs/stage0_findings.md). Architecture conversion is supported
(the llama.cpp org itself publishes a `qwen3_5_moe` GGUF); export splits into LM + vision
`mmproj` + MTP drafter; and no sub-2-bit build exists for this model yet (novel). **Runtime
target: the PrismML fork** (`PrismML-Eng/llama.cpp @ prism`), whose g128 **`Q2_0` (ternary) and
`Q1_0` (binary)** both run on CUDA/Metal/CPU/Vulkan + DSpark — so `quant.group_size: 128` and
**1-bit is a shippable, runnable target** (mainline has no binary type). Caveat: no Bonsai MoE
exists, so the 256-expert path at 1-bit is unexercised — `scripts/stage0_spike.py` runs the
empirical `Q2_0`/`Q1_0` round-trip on a fork build to validate it.

## Cloud stages (HF Pro hub + pay-as-you-go H200)

Inference-shaped stages run on **ZeroGPU** (single H200 141GB holds the 35B in bf16, free in
Pro quota); only the sustained QAD loop uses a pay-as-you-go H200. See the plan for the
compute table and cost gates.

```bash
python scripts/run_baseline.py --config configs/ternary.yaml   # Stage 0: FP16 reference
python scripts/run_ptq.py      --config configs/ternary.yaml   # Stage 2: coverage + PTQ + teacher logits
python scripts/run_qad.py      --config configs/ternary.yaml   # Stage 3: block-wise QAD
python scripts/export.py       --config configs/ternary.yaml --checkpoint <healed>
python scripts/evaluate.py     --config configs/ternary.yaml --checkpoint <healed>
```

Ternary is the milestone (target ≥93% of FP16); binary (`configs/binary.yaml`) re-inits from
the healed ternary checkpoint.
