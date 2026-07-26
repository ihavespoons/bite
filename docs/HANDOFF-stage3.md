# bite — handoff (start of Stage 3: block-wise QAD)

Snapshot for resuming after context compaction. Everything below is committed/pushed to
`github.com/ihavespoons/bite` (private) and the HF dataset `datasets/ihavespoons/bite-baseline`.

## What bite is
Pipeline to compress **`Qwen/Qwen3.6-35B-A3B`** to **ternary {-1,0,+1}** and **binary {-1,+1}**
weights with minimal quality loss, via **PTQ init + Quantization-Aware Distillation (QAD)**.
Inspired by PrismML's Bonsai (dense); the novelty is doing it on an **MoE**. Deliverable:
low-bit weights that run on the **PrismML llama.cpp fork**.

## Target model facts (verified)
- "Qwen3.6" is the **`qwen3_5_moe`** architecture (3.6 = release, 3.5 = arch). Built into
  transformers (no remote code); multimodal `Qwen3_5MoeForConditionalGeneration`.
- 35B total / **~3B active** (A3B) → inference is fast/cheap (~3B-param speed), NOT 35B.
- **256 experts** (8 routed + 1 shared), 40 layers = 30 GatedDeltaNet (linear-attn) + 10 full
  attn (GQA 16Q/2KV), hidden 2048, moe_intermediate 512, vocab 248320, MTP head, vision tower.
- **CRITICAL: experts are fused 3D `nn.Parameter`s** (`gate_up_proj (E,hidden,2*inter)`,
  `down_proj (E,inter,hidden)`), NOT `nn.Linear`. Router is a custom `Qwen3_5MoeTopKRouter`
  (weight `(num_experts, hidden)`, not nn.Linear).

## Runtime target (locked)
**PrismML-Eng/llama.cpp @ `prism`** — has **g128** `Q1_0` (binary, 1.125 bpw) + `Q2_0` (ternary).
Mainline llama.cpp only has g64 ternary and NO binary type. → **group_size = 128**. Fork's
quantize already handles 3D expert tensors + keeps router high-precision. No Bonsai MoE exists,
so the MoE-at-1-bit path is unexercised (validate at export).

## Quantization approach (built + validated on the live model)
- `nn.Linear` (attention q/k/v/o + lm_head) → `QuantLinear` swap: **191 linears**
  (`bite/quant/quantlinear.py`, `swap_linears` with `exclude_prefixes` for vision).
- **Fused experts** → `torch.nn.utils.parametrize` fake-quant on the 3D params: **80 tensors**
  (`bite/quant/experts.py`: `install_expert_fakequant`, `ptq_init_experts`, `expert_latent_params`).
- Kept high precision: router gate, shared expert, norms, DeltaNet conv, vision tower (→4-bit HQQ).
- **Naive PTQ init** (deterministic absmean rounding, no data, re-derivable in seconds).
  GPTQ-with-Hessian is DEFERRED — the all-experts Hessian OOMs on 256 experts; needs a
  layer-sequential rewrite (`bite/quant/hessian.py` + `ptq.gptq_quantize` exist, unused for now).
- STE fake-quant core: `bite/quant/fakequant.py` (group-wise last dim). effective_bits ternary
  1.71 / binary 1.125 (g128).

## Baseline (FP16 reference) — job baseline-v4
- **MMLU 83.9% is the PRIMARY degradation metric** (loglikelihood, robust). Use for % retained.
- gsm8k 0.28/0.30, ifeval 0.34/0.47 are **depressed by eval config** (thinking model; chat
  template made gsm8k WORSE → 0.0). Don't chase them; MMLU is the signal.
- Files: `docs/baseline-v4-fp16.json`, `docs/baseline-chat-fp16.json`; dataset `baseline.json`.
- Stored in config as `eval.teacher_baseline` (configs/base.yaml).

## Stage 2 — DONE
- Coverage confirmed: **40 routers hooked, 0 dead experts, Gini 0.51** (routing imbalanced;
  rare experts get ~0.005% → motivates expert-balanced calibration, a TODO). `docs/stage2-coverage.md`.
- **Teacher top-64 logits: 64 shards (~407 MB)** at `datasets/ihavespoons/bite-baseline/teacher_topk/`.
  Each shard bundles **`input_ids` + `values` + `indices`** (self-contained; QAD does NOT
  re-stream c4). Load with `bite.train.teacher.load_teacher_shard(path) -> (input_ids, values, indices)`.
  512 seqs (~1M tokens); scalable via `run_ptq --teacher-only --max-seqs N --push-repo ...`.

## Stage 3 — TO BUILD (block-wise QAD)
Scaffold exists and is unit-tested: `bite/train/blockwise.py` (`iter_blocks`, `quant_parameters`,
`build_optimizer` [adam/adam8bit], `distill_block` — heals a block to match the FP16 teacher block
via hidden-MSE/STE, tested), `bite/train/qad.py` (`qad_loss` = top-k KL + hidden MSE + CE, tested),
`scripts/run_qad.py` → `run_blockwise_qad`.

**The genuinely new pieces to build:**
1. **Per-block forward adapter** — real Qwen3.5 decoder blocks take `(hidden_states, position_ids,
   attention_mask, ...)`, not just hidden states; `distill_block`'s `forward_fn` needs a correct
   adapter (and the DeltaNet vs full-attn blocks differ). This is the main unknown; validate with
   a cheap capped GPU run.
2. **QAD optimizer must include expert latents** — `quant_parameters()` currently returns only
   `QuantLinear.weight`; extend it to also collect `expert_latent_params(model)` (the fused expert
   originals), or the experts won't heal.
3. Wire `run_blockwise_qad` to: load PTQ-init student (build_student + ptq_init_model +
   ptq_init_experts), stream teacher shards (`load_teacher_shard`), heal each block, checkpoint,
   eval MMLU vs 83.9%.
4. Ternary first (target ≥93% of FP16 MMLU); then binary re-init from healed ternary.

## HOW TO RUN (HF Jobs) — critical operational facts
- Launch: `scripts/launch_hf.py` (Python `run_job`, server-side/detached, survives disconnect).
  `GITHUB_TOKEN=$(gh auth token) HF_TOKEN=$(.venv/bin/hf auth token) .venv/bin/python scripts/launch_hf.py --stage <s> --flavor h200 --extra-args "..."`.
- Private repo cloned in-job via `GITHUB_TOKEN` secret; **git installed via apt** in the job
  (pytorch devel image lacks it). Both tokens passed as encrypted `--secrets`.
- **h200 flavor, $5/hr.** Model **download is fast** (2.6 GB/s, 70 GB in ~27 s) — **do NOT use
  `--mount-model`** (lazy mount stalls; download is better).
- **A3B forward is fast/cheap**: teacher logits over 512 seqs took **3.4 min (~$0.28)**. Recalibrate
  cost intuition DOWN — do not over-quote. Only QAD (fwd+bwd+opt) is meaningfully costly.
- **Always validate cheap first**: CPU dep-check (`scripts/check_tasks.py`), tiny `--max-seqs`,
  internal limit-2 gates — before any full run. Persist outputs to HF (`--push-repo`) so detached
  jobs' results survive.
- Monitor with `hf jobs wait <id>` backgrounded (streaming `logs -f` drops locally); fetch results
  from the dataset, not the local stream.
- Env: local `.venv` is Python 3.12 (torch etc.); `hf` CLI at `.venv/bin/hf`, logged in as `ihavespoons`.

## lm-eval gotchas (pinned lm-eval==0.4.12)
- humaneval broken in 0.4.12 (missing `pass_at_k`) → dropped. Tasks: `[gsm8k, mmlu, ifeval]`.
- ifeval needs `langdetect`/`immutabledict`/`nltk` (in `[eval]` extra) + nltk punkt (auto-fetched).
- **batch_size fixed = 8**, NOT "auto" (248K vocab overflows 32-bit CUDA index at large batch).
- `typing_extensions>=4.13` needed on py<3.13.
- Eval loads the model via `AutoModelForImageTextToText` (multimodal) wrapped in `HFLM`.

## Spend so far: ~$35-40 total (mostly one-time baseline + cheap validations).
```

---

## ADDENDUM (2026-07-22): e2e QAD state — PAUSED (credits), resume point

Block-wise QAD ran end-to-end but ternary collapsed to chance MMLU (0.245 vs 0.839; PTQ-only
0.253; lm_head precision irrelevant). Recovery lever = END-TO-END logit-KL QAD (built:
bite/train/end2end.py, scripts/run_e2e.py, launch stage `e2e`; init = mmap assign-load of the
healed qad_student ckpt; client bnb Adam8bit under ZeRO-3 — validated clean in toy).

BLOCKER: real-35B backward retains every layer's full expert grads (+1.6GB/layer -> OOM ~74GB)
regardless of headroom. Toy (scripts/repro_zero3_parametrize.py, a10g ~$0.05) is ALWAYS flat —
eliminated: parametrize/ckpt machinery, bnb-under-ZeRO3, accum>=2, assign-load, overlap_comm,
oversized-vs-bucket knobs. REMAINING SUSPECTS: transformers grouped-GEMM experts forward
(extend toy with the real Qwen3_5Moe experts module — NEXT), then fla GatedDeltaNet.
Requirements found on the way (all committed): model.train() (eval silently disables ckpt),
enable_input_require_grads, use_reentrant=True (non-reentrant breaks under ZeRO-3),
device_map="cpu" load (GPU device_map blocks sharding), --mem-probe for per-layer prints.
Branch stage3-qad @ 732c595. Reverie: bite-stage3-e2e-paused-resume-point.

---

## ADDENDUM 2 (2026-07-26): e2e verdict + next phase

**Stage 3 e2e complete.** Training heals on-distribution (CE 7.81->1.26 over 500 micro-steps,
8xA100, peak 70GB/rank with the gathered-param release fix) but held-out MMLU = 0.2395 (chance):
1M tokens x14 epochs = memorization without generalization. Token diversity, not optimization,
is the binding constraint. Full write-up: docs/report-extreme-quant-moe.md (complete, no
placeholders). Repo PUBLIC (Apache-2.0) + dataset public with card (e2e_student checkpoint is a
TORCH PICKLE named .safetensors — DeepSpeed save_16bit_model quirk; eval_quant handles it).

**NEXT PHASE (agreed order):**
0. RunPod migration FIRST (user-funded account; ~half HF's $/GPU-hr): scripts/launch_runpod.py
   mirroring launch_hf.py stages; repo is public now so no GITHUB_TOKEN needed to clone;
   artifacts still push to the HF dataset. Keep launch_hf.py as fallback.
1. Better PTQ init (~$30): layer-sequential GPTQ (hessian.py scaffold), AWQ-style scaling,
   ternary threshold search. Biggest single lever — start closer, heal less.
2. Throughput pass (~$10): micro_batch=2, compiled causal-conv1d, checkpoint granularity
   (currently only ~1.7K tok/s on 8xA100 → expect 2-3x).
3. Slope run (~$50-150 at RunPod prices): 20-50M fresh tokens, FineWeb-Edu-heavy mixture,
   expert-balanced sampling + hard-example mining, periodic MMLU checkpoints.
Fallbacks: expert-level mixed precision, scale-only warmup, progressive precision.
