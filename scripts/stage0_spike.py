#!/usr/bin/env python
"""Stage 0 feasibility spike: prove the sub-2-bit GGUF round-trip for qwen3_5_moe.

The desk-research portion is already settled (see ``docs/stage0_findings.md``):
architecture conversion is supported (ggml-org and others publish GGUFs), the mainline
sub-2-bit ternary target is **Q2_0 at group-64** (not g128), and there is no mainline
binary type. This script empirically confirms the remaining unknown on the cloud runner:
that ``llama-quantize`` can produce a **Q2_0** build for this MoE arch (including the routed
expert tensors) and that it loads and generates coherently.

Runner-side; needs a built llama.cpp and the model. Pure stdlib so ``--help`` and arg
parsing work anywhere.

Example:
    python scripts/stage0_spike.py \
        --llama-cpp ~/llama.cpp/build/bin \
        --bf16-gguf ggml-org/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-BF16.gguf \
        --work-dir outputs/stage0
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

GATE = "Q2_0"  # mainline group-64 ternary-capable type (see docs/stage0_findings.md)
EXPERT_TENSOR_HINT = "ffn_"  # MoE expert tensors we most need to confirm quantize cleanly


def _tool(bin_dir: Path, name: str) -> Path:
    p = bin_dir / name
    if not p.exists():
        sys.exit(f"missing llama.cpp tool: {p} (build llama.cpp and pass --llama-cpp)")
    return p


def run(cmd: list[str]) -> str:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    out = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out.stdout + out.stderr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--llama-cpp", type=Path, required=True, help="dir with llama-quantize / llama-cli")
    ap.add_argument("--bf16-gguf", type=Path, required=True, help="BF16 LM GGUF to quantize")
    ap.add_argument("--work-dir", type=Path, default=Path("outputs/stage0"))
    ap.add_argument("--qtype", default=GATE, help=f"quant type to test (default {GATE})")
    ap.add_argument("--prompt", default="Explain, step by step, why the sky is blue.")
    args = ap.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    quantize = _tool(args.llama_cpp, "llama-quantize")
    cli = _tool(args.llama_cpp, "llama-cli")
    out_gguf = args.work_dir / f"qwen3_5_moe-{args.qtype}.gguf"

    # 1. quantize BF16 -> target sub-2-bit type; this is the round-trip that can fail on
    #    an unsupported arch / tensor shape (esp. the 256 routed experts).
    qlog = run([str(quantize), str(args.bf16_gguf), str(out_gguf), args.qtype])
    if EXPERT_TENSOR_HINT not in qlog:
        print(f"WARN: no '{EXPERT_TENSOR_HINT}*' tensors seen in quantize log — verify experts quantized")

    # 2. load + generate; assert non-empty coherent output (routing didn't collapse to garbage)
    gen = run([str(cli), "-m", str(out_gguf), "-p", args.prompt, "-n", "128", "-no-cnv"])
    produced = gen.split(args.prompt, 1)[-1].strip()

    size_gb = out_gguf.stat().st_size / 1e9
    print("\n=== Stage 0 gate ===")
    print(f"type={args.qtype}  size={size_gb:.2f} GB  bytes/param≈{size_gb*1e9/35.95e9*8:.3f} bpw")
    ok = len(produced) > 40
    print("PASS: Q2_0 round-trip generated coherent text" if ok else "FAIL: empty/short generation")
    print("gate ->", "GO (train, then export to this type)" if ok
          else "NO-GO (contribute converter or use fake-quant eval harness)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
