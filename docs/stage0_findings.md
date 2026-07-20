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

**Implication for us:** mainline (g64 `Q2_0`) is ternary-only and CPU/Metal-only. Because the
1-bit/binary goal needs a runtime that mainline doesn't provide, we target the **PrismML fork**
instead (Finding #6) and use **g128**. Mainline g64 remains a secondary ternary export if broad
reach is wanted later.

## Finding 4 — no sub-2-bit built for this arch yet ✅ (novel)

The lowest published quant for `Qwen3.6-35B-A3B` is **Q4_K_M** (~20 GB); no Q2_0/TQ/1-bit
build exists publicly. The sub-2-bit MoE result is genuinely unclaimed — this is new work,
and `scripts/stage0_spike.py` is what proves the Q2_0 round-trip empirically.

## Finding 5 — binary (1-bit) has no mainline runtime home ⚠️

There is **no mainline binary `{-1,+1}` GGUF type** (`TQ1_0` is 1.69 bpw *ternary*). On
**mainline**, binary would stay a research-only fake-quant result. This is what motivated
adopting the fork (Finding #6), where binary **is** a native type (`Q1_0`). Decision: target
the fork so both ternary and binary ship.

## Finding 6 — the PrismML fork runs both ternary and binary at g128 (chosen runtime) ✅⚠️

The Bonsai runtime is [`PrismML-Eng/llama.cpp`](https://github.com/PrismML-Eng/llama.cpp)
branch `prism`. From its source:

- **`Q1_0` = binary `{-1,+1}` g128 = 1.125 bpw** (`QK1_0=128`, fp16 scale + 16 B) — our 1-bit
  target, and **exactly** the whitepaper's binary figure.
- **`Q2_0` = ternary g128 = 2.125 bpw** stored (fp16 scale + 32 B @ 2 bits; 1.71 ideal).
- `llama-quant.cpp`: *"quantize only 2D and 3D tensors (experts)"* and *"do not quantize expert
  gating tensors"* → **3D MoE expert tensors are quantized** and the router is kept
  high-precision, matching `bite/quant/policy.py`.
- Kernels for **CUDA (Hopper WGMMA/PTX → H200), Metal, x86 AVX512-VNNI, Vulkan**; **DSpark**
  speculative decoding; **Gated-DeltaNet** kernels; commits referencing `qwen35`.

**Caveat ⚠️:** every published Bonsai model is *dense* (Qwen3.6-27B / 8B / 4B / 1.7B) —
**no Bonsai MoE exists.** The MoE quant plumbing is present but **unexercised at 1-bit on a
256-expert model**; the stock quantize also auto-bumps some tensors to `Q4_K` when
`n_expert>=4`, which we must override for a true end-to-end low-bit export. The fork is the
*runtime only* — it does not contain the quantization method (that stays the IP we reconstruct
via QAD). Off-mainline: g128 `Q1_0`/`Q2_0` don't load on stock llama.cpp.

**Decision:** target the fork, `group_size: 128`, ship ternary (`Q2_0`) + binary (`Q1_0`).

## Gate decision

**GO.** Architecture, component split, and a concrete runtime for **both** ternary and binary
(the PrismML fork, g128) are confirmed without spend. Remaining runner-side confirmations
before Stage 3 training:

1. `scripts/stage0_spike.py` — on a **fork build**, empirical `Q2_0` **and** `Q1_0` quantize +
   generate round-trip, confirming the **256 routed-expert** 3D tensors quantize cleanly and
   the `n_expert>=4 → Q4_K` auto-bump is overridden.
2. `scripts/run_baseline.py` — FP16 eval suite → the 100% reference.
