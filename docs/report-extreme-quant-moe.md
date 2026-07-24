# Extreme Quantization of a Mixture-of-Experts LLM: Collapse, Healing, and the Token-Budget Question

*bite project technical report — draft v0.1, 2026-07-24*
*Ben Gittins, with engineering by Claude (Anthropic)*

## Summary

We built and open-sourced a complete pipeline for compressing **Qwen3.6-35B-A3B** — a 35B-parameter
Mixture-of-Experts model with ~3B active parameters — toward **ternary {-1,0,+1} weights (1.71
bits/weight)**, using only public techniques: absmean post-training quantization (PTQ), straight-through-estimator
(STE) fake-quantization, and quantization-aware distillation (QAD) against the frozen FP16 teacher.
PrismML's Bonsai demonstrated this transformation for *dense* models with a proprietary method; to our
knowledge no public pipeline exists for **MoE** models, whose fused expert tensors, sparse routing, and
per-expert data starvation make them a distinct and unexplored target.

Findings so far, on a total compute budget of **≈ $150** of rented GPU time:

1. **Naive ternary PTQ collapses the model to chance** (MMLU 25.3% vs the FP16 baseline's 83.9%), and
   the collapse is *systemic* — keeping the 248K-vocab `lm_head` at FP16 changes nothing (25.4%).
2. **Block-wise (local) healing is insufficient.** Healing each transformer block to match its FP16
   teacher block reduces per-block hidden-state error by 1–2 orders of magnitude, yet end-task accuracy
   stays at chance (24.5%): quantization error *compounds across depth* faster than local matching can
   correct.
3. **End-to-end logit distillation demonstrably heals the collapsed model.** With top-64 teacher-logit
   KL + next-token CE and STE gradients into all ~33B quantized latent weights, training loss fell from
   CE ≈ 8.3 (perplexity ≈ 4000) to CE ≈ 1.5 (perplexity ≈ 4.6) in 500 micro-steps over a deliberately
   tiny ~1M-token corpus. **[PLACEHOLDER: held-out MMLU after healing = X.XX / XX% of FP16 —
   generalization verdict]**
4. **A systems contribution:** DeepSpeed ZeRO-3's partitioned-parameter coordinator does not release
   gathered parameters during its first (trace-recording) backward pass. Invisible at world size 1 and
   modest scale, this hoards the *entire model's* gathered parameters (~64 GB here) during the first
   backward at 35B scale and OOMs any 80 GB GPU. We provide a minimal reproducer, a per-class memory
   attribution methodology, and a working mitigation (per-block release hooks using DeepSpeed's own
   force-clear pattern), enabling full-model 35B QAD training on commodity 8×A100 nodes.

The open question this work sharpens — and the subject of proposed follow-up — is the **token budget for
QAD recovery of an MoE at ternary precision**: with top-8-of-256 routing, each expert observes only ~3%
of training tokens, suggesting MoE recovery may require substantially more data than the billions of
tokens reported for dense models. Nobody has measured this. Our pipeline makes the measurement a matter
of GPU-hours.

---

## 1. Background and motivation

Sub-2-bit quantization of pretrained LLMs was long considered destructive; from-scratch 1-bit training
(BitNet, TriLM) sidestepped rather than solved the conversion problem. PrismML's Bonsai (2026) showed a
*dense* pretrained model (Qwen3.6-27B) can be converted to ternary at ~95% of FP16 quality and binary at
~90% — but the transformation method is proprietary. The public recipe believed to underlie such results
is **PTQ initialization + QAD**: round to the low-bit grid, then continue-train the fake-quantized
student against its own FP16 teacher.

MoE models are the natural next target — inference-efficient (A3B: 35B capability at ~3B-active cost)
and increasingly the frontier architecture — but present unique obstacles:

- Experts are stored as **fused 3D tensors** (e.g. `gate_up_proj`: 256 experts × 5.4×10⁸ elements per
  layer), not `nn.Linear` modules — standard quantization tooling misses them entirely.
- **Routing is heavily skewed** (we measure Gini 0.51; rarest experts receive 0.005% of assignments), so
  calibration and healing signal per expert is scarce and uneven.
- The **router/expert interaction** under weight quantization is unstudied.

## 2. Target model and quantization scheme

**Model:** `Qwen/Qwen3.6-35B-A3B` (`qwen3_5_moe` architecture): 35B total / ~3B active parameters;
40 layers (30 GatedDeltaNet linear-attention + 10 GQA full-attention); 256 experts per layer, top-8
routed + 1 shared; hidden 2048; vocabulary 248,320; multimodal (vision tower held out of scope, planned
4-bit HQQ).

**Scheme (all validated on the live model):**

| Component | Treatment |
|---|---|
| Fused expert tensors (80 tensors, ~32B params — the bulk) | ternary fake-quant via `torch.nn.utils.parametrize` on the 3D parameters |
| Attention/lm_head `nn.Linear` (191 modules) | `QuantLinear` swap, ternary |
| Router gates, shared experts, norms, DeltaNet conv/gates, embeddings | kept high precision (the "small, high-leverage tail") |
| Grouping | 128 elements/group, FP16 scale (matches the PrismML llama.cpp fork's `Q2_0`/`Q1_0`, so healed = shippable) |

Effective weight width: **1.71 bpw** ternary (binary 1.125 bpw is the staged follow-on). Deliverable
target: GGUF on the PrismML fork — the final artifact is **~8–9 GB** and runs on a 16 GB laptop.

**FP16 reference:** MMLU **83.9%** (loglikelihood, lm-eval 0.4.12, batch 8). Generative benchmarks
(gsm8k, ifeval) are depressed by thinking-model evaluation confounds and are tracked only as secondary
references; MMLU is the degradation metric throughout.

## 3. Result 1 — Naive PTQ collapses the MoE, and it isn't the lm_head

Absmean ternary rounding (per-group MSE-optimal scales, no data) yields MMLU **25.32%** — the 4-choice
floor is 25%. An ablation keeping the 2048×248,320 `lm_head` at FP16 yields **25.39%**: the collapse is
not attributable to the output projection but is distributed across the quantized weights. Router
coverage measured on 256 calibration sequences confirms the machinery works as intended (40/40 routers
hooked, 0/256 dead experts).

## 4. Result 2 — Local (block-wise) healing does not transfer to end-task accuracy

BRECQ-style block-wise healing — walking the 40 decoder blocks, training each block's quantized latents
to match the FP16 teacher block's outputs on running activations — achieves large local improvements
(hidden-MSE down 1–2 orders of magnitude per block) but **MMLU 24.5%**, indistinguishable from the
unhealed PTQ floor. The per-block loss profile is diagnostic: pre-heal error grows monotonically with
depth (block 3: 1.2×10⁻⁴ → block 39: 1.1×10⁻¹, a 1000× amplification), and deep blocks heal to floors
10–100× worse than shallow ones. Local matching cannot undo *compounding* drift: by the final layers the
healed model's hidden states — and hence its logits — remain unusable. Extreme quantization of this MoE
requires a global training signal.

## 5. Result 3 — End-to-end QAD heals the collapse (training-side)

**Setup:** loss = KL(teacher-top-64 ‖ student) + 0.5·CE(next token), teacher logits precomputed once
(64 self-contained shards, ~407 MB — the teacher is never resident during training); STE gradients into
all 271 quantized latent tensors (~33B parameters); bitsandbytes 8-bit Adam as DeepSpeed ZeRO-3 client
optimizer; bf16; non-reentrant gradient checkpointing; 8×A100-80GB; 500 micro-steps ≈ 8M trained tokens
(~8 epochs over a deliberately tiny ~1M-token c4 sample).

**Healing curve (training set):**

| micro-step | KL | CE | ≈ perplexity |
|---|---|---|---|
| 0 | 1.90 | 7.81 | ~2500 |
| 100 | 1.84 | 6.28 | ~530 |
| 200 | 1.39 | 3.67 | ~39 |
| 300 | 0.64 | 2.81 | ~17 |
| 400 | 0.50 | 1.91 | ~7 |
| 450 | 0.48 | 1.53 | ~4.6 |

The optimizer demonstrably converts gradient signal into recovered language modeling through the ternary
constraint — the failure mode "STE cannot move a fully collapsed ternary MoE" is ruled out. At ~8 epochs
over 1M tokens, late-run figures partially reflect memorization; the held-out verdict is:

**[PLACEHOLDER — held-out MMLU after e2e healing: X.XX (XX% of FP16). Interpretation.]**

## 6. Result 4 — Systems: making 35B QAD fit commodity GPUs

Full-model QAD of a 35B MoE was blocked by a sequence of memory walls, each diagnosed with cheap,
controlled experiments (a toy reproducer at 1/1000 scale, ~$0.05–0.25 per run; per-class memory
attribution separating gathered-parameter bytes, live-gradient bytes, and reduction events). Findings of
general interest:

1. **fp32 Adam is structurally infeasible** for 33B trainable at this scale (~400 GB of state);
   bitsandbytes 8-bit Adam functions correctly as a ZeRO-3 *client* optimizer (uint8 state on fp32 flat
   partitions), cutting optimizer memory 6×.
2. **`device_map`-based loading silently defeats ZeRO-3 sharding** (accelerate hooks pin full replicas);
   models must be built on CPU and handed to `deepspeed.initialize`.
3. **Checkpoint init must be read-only:** `safetensors` mmap + `load_state_dict(assign=True)` lets N
   ranks share one page-cache copy of a 70 GB checkpoint; any in-run weight mutation (e.g. PTQ init)
   copy-on-write-materializes N private copies.
4. **Eval-mode silently disables gradient checkpointing** in transformers (`from_pretrained` returns
   eval mode); the resulting retention of per-layer dequantized weights (~64 GB) masquerades as a leak.
5. **Non-reentrant checkpointing requires `determinism_check="none"` under ZeRO-3** (parameters are
   re-partitioned between save and recompute, so the metadata check always trips).
6. **Main finding — ZeRO-3 first-iteration gathered-parameter hoard:** during the parameter
   coordinator's trace-recording first iteration, gathered parameters are never released in the backward
   pass. At world size 1 gathering allocates nothing (the "gather" is the partition itself), making the
   behavior invisible in small-scale tests; at world > 1 on a 40-layer 35B model it accumulates ~1.6
   GB/layer (~64 GB) and OOMs any 80 GB device — insensitive to optimizer choice, communication mode,
   bucket sizes, reuse distance, and DeepSpeed version (reproduced on 0.15.4 and current). *Mitigation:*
   a `register_full_backward_hook` per decoder block that force-clears `ds_active_sub_modules` and
   re-partitions the block's parameters once its backward completes (the pattern DeepSpeed itself uses in
   `release_and_reset_all`). Result: backward memory flat at ~47 GB across all 40 layers, peak 70
   GB/rank, stable across steps. We intend to file this upstream with the reproducer.

**Cost discipline:** the entire investigation — 15+ controlled discriminator experiments plus the fix
validation — cost ≈ $13. The full project to date, including the FP16 baseline, coverage analysis,
teacher-logit precompute, block-wise runs, all diagnostics, and the 500-step healing run: **≈ $150.**

## 7. The open question: token budget for MoE recovery

Dense-model QAD recovery reportedly requires 10⁹–10¹⁰ tokens. For an MoE with top-8-of-256 routing, each
routed expert participates in ~3% of token forward passes — and our measured routing skew (Gini 0.51)
concentrates even that. A priori, MoE recovery could require **10–30× the dense token budget** for
equivalent per-expert signal — or far less, if shared/attention pathways carry most of the recovery.
**This number is unknown, cheap to measure with this pipeline, and decision-relevant for anyone hoping
to ship extreme-quantized MoEs.**

Proposed measurement (the "slope run"): 20–50M fresh teacher tokens (teacher-logit generation is ~$0.28
per 512 sequences), 1–2 epochs with periodic held-out MMLU checkpoints → a retention-vs-tokens curve
whose extrapolation prices full recovery. Estimated cost at current rentals: **$150–400.** Full recovery
attempts at 10⁹ tokens: ~$3.5–5k. Expert-balanced data (oversampling rare-expert-activating text) is a
second, testable lever.

## 8. Artifacts

- **Pipeline** (open source): `github.com/ihavespoons/bite` — quantization core (STE fake-quant,
  QuantLinear, fused-expert parametrization), PTQ, block-wise and end-to-end QAD under ZeRO-3, eval
  harness, HF Jobs launchers, and the DeepSpeed toy reproducer with per-class memory probes.
- **Dataset** (`ihavespoons/bite-baseline`): FP16 baselines, router-coverage report, teacher top-64
  logit shards, block-wise-healed checkpoint (`qad_student/`), e2e-healed checkpoint (`e2e_student/`),
  metrics JSON for every run.
- **Runtime target:** PrismML llama.cpp fork (`Q2_0` ternary / `Q1_0` binary, group 128) — the healed
  weights are export-compatible by construction.

## Appendix A — Reproducibility

Every stage is a config-driven script launched via `scripts/launch_hf.py` (HF Jobs, pay-per-second).
Key jobs (IDs in repo history): baseline-v4 (FP16 reference), stage2-coverage, teacher-logits,
block-wise QAD, PTQ/lm_head ablations, the discriminator series, and the 500-step healing run.
CPU-testable core: 78 unit tests.

**[PLACEHOLDER — final numbers table after eval: e2e MMLU, % retained, tokens trained, wall-clock,
total cost.]**
