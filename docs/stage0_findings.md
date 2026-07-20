# Stage 0 — Feasibility findings

The gating question before any training spend: **can we ship a sub-2-bit `qwen3_5_moe` to an
existing runtime (GGUF / llama.cpp)?** The desk-research portion is settled below; the
empirical Q2_0 round-trip is confirmed by `scripts/stage0_spike.py` on the runner.

## Model facts (from the Hub `config.json`)

- `model_type: qwen3_5_moe`, class `Qwen3_5MoeForConditionalGeneration`, 35.95B params.
- **40 layers**, `layer_types` = 3× `linear_attention` (Gated DeltaNet) + 1× `full_attention`
  (interval 4) → 30 linear / 10 full.
- Full attention: 16 Q / 2 KV heads (GQA), `head_dim` 256, partial RoPE 0.25, output gate.
- Linear attention: 16 key / 32 value heads, head dim 128, conv kernel 4.
- **MoE:** 256 experts, top-8, `moe_intermediate_size` 512, **shared expert** (512).
- `hidden_size` 2048, separate `lm_head` (`tie_word_embeddings: false`), vocab 248320.
- **MTP** head (`mtp_num_hidden_layers: 1`) — the speculative-decode drafter.
- Vision tower: depth 27, hidden 1152, out 2048.
- **All quantizable dims (2048, 512, 4096, 256) are divisible by 64 and 128** → group-wise
  scaling works everywhere; no ragged groups.

## Finding 1 — architecture conversion is supported ✅

`convert_hf_to_gguf.py` handles `qwen3_5_moe` including the linear-attention backbone and MTP.
Proven by published GGUFs, incl. the llama.cpp org's own
[`ggml-org/Qwen3.6-35B-A3B-GGUF`](https://hf.co/ggml-org/Qwen3.6-35B-A3B-GGUF) (plus unsloth,
bartowski, lmstudio-community). The scary unknown (a novel hybrid arch llama.cpp can't read)
is **closed**.

## Finding 2 — component structure matches our plan ✅

The ggml-org repo ships the model as **separate GGUFs**:

| Component | File | BF16 size | Maps to |
|---|---|---|---|
| Language model | `Qwen3.6-35B-A3B-BF16.gguf` | 69.4 GB | our low-bit target |
| Vision tower | `mmproj-*.gguf` | 0.90 GB | our 4-bit HQQ vision (Stage 5) |
| MTP drafter | `mtp-*.gguf` | 3.7 GB | Bonsai "DSpark" speculative layer |

So the export splits cleanly: LM low-bit, vision as `mmproj`, MTP as an optional drafter.

## Finding 3 — the sub-2-bit target is **Q2_0 @ group-64**, not g128 ⚠️

llama.cpp discussion [#22019](https://github.com/ggml-org/llama.cpp/discussions/22019)
(*"Supporting Ternary Bonsai… group-128 ternary format"*) is directly on point. Outcome:

- Maintainer **rejected a g128 ternary type** as "suboptimal"; mainline adopted **`Q2_0` at
  group size 64** ("<6% overhead, more practical, better quality"). Several backends merged.
- The Bonsai team keeps their **g128 format on a fork**, off mainline.
- `TQ1_0`/`TQ2_0` are BitNet-style **group-256** ternary — a different layout again.

**Implication for us:** to hit the "existing runtime" deliverable with no custom kernels,
**heal and export at group-64 → `Q2_0`**. Healing at our original g128 would require a
re-quant to g64 at export (a small quality gap). We therefore set `quant.group_size: 64` as
the default (see `configs/base.yaml`). g128 remains available for a research-optimal run that
accepts an export re-quant.

## Finding 4 — no sub-2-bit built for this arch yet ✅ (novel)

The lowest published quant for `Qwen3.6-35B-A3B` is **Q4_K_M** (~20 GB); no Q2_0/TQ/1-bit
build exists publicly. The sub-2-bit MoE result is genuinely unclaimed — this is new work,
and `scripts/stage0_spike.py` is what proves the Q2_0 round-trip empirically.

## Finding 5 — binary (1-bit) has no mainline runtime home ⚠️

There is **no mainline binary `{-1,+1}` GGUF type** (`TQ1_0` is 1.69 bpw *ternary*). Under the
locked "existing runtime, no custom kernels" constraint:

- **Ternary is the shippable deliverable** (`Q2_0` g64).
- **Binary stays a research result** — evaluated via the in-framework fake-quant harness (or
  the Bonsai fork / a custom kernel), not stock llama.cpp. This sharpens the staged plan:
  ternary → ship; binary → prove the quality, defer the runtime.

## Gate decision

**GO.** Architecture, component split, and a concrete ternary runtime target (`Q2_0` g64) are
all confirmed without spend. Remaining runner-side confirmations before Stage 3 training:

1. `scripts/stage0_spike.py` — empirical `Q2_0` quantize + generate round-trip (esp. that the
   256 routed-expert tensors quantize cleanly).
2. `scripts/run_baseline.py` — FP16 eval suite → the 100% reference.
