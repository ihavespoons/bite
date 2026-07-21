# Running bite on Hugging Face (HF Jobs)

Our stages are CLI scripts, so the right HF mechanism is **HF Jobs** (`hf jobs`) — pay-per-second
GPU, Docker-style — not ZeroGPU Spaces (which are for interactive/inference bursts). Jobs need a
**positive credit balance** (HF Pro includes credits).

## Prereqs (once)

```bash
pip install -U "huggingface_hub[cli]"
hf auth login          # token needs write + Jobs; check billing has credit
hf jobs hardware       # list flavors + prices
```

## Hardware we care about (from `hf jobs hardware`)

| flavor | GPU | $/hr | use |
|---|---|---|---|
| `t4-small` | T4 16 GB | 0.40 | smoke / unit tests on CUDA |
| `a100-large` | A100 80 GB | 2.50 | baseline (tight), PTQ |
| **`h200`** | **H200 141 GB** | **5.00** | **baseline, PTQ, block-wise QAD** — holds 35B bf16 (~70 GB) with room |
| `h200x4` | 4×H200 | 20.00 | end-to-end QAD polish |

**Default job timeout is 30 min — always pass `--timeout` for real runs.** Jobs have large scratch
disk (h200 = 3 TB), so a 70 GB model download per run is fine; mount the repo to avoid re-downloading.

## Run 1 — Jobs + GPU smoke (~$0.01)

```bash
hf jobs run --flavor t4-small \
  pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
  python -c "import torch; print(torch.cuda.get_device_name())"
```

## Run 2 — bite on CUDA: unit tests + quant-core GPU sanity (~$0.10)

```bash
hf jobs run --flavor t4-small --timeout 20m \
  pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
  bash -lc "set -e; git clone -q https://github.com/ihavespoons/bite && cd bite \
    && pip install -q -e '.[dev]' && python -m pytest -q \
    && python -c \"import torch; from bite.quant.fakequant import fake_quantize; \
w=torch.randn(4,256,device='cuda',requires_grad=True); fake_quantize(w,'ternary',128).sum().backward(); \
print('CUDA quant OK grad=', w.grad is not None)\""
```

## Run 3 — FP16 baseline, the 100% reference (~hours on `h200`)

```bash
hf jobs run --flavor h200 --timeout 6h --secrets HF_TOKEN=$HF_TOKEN \
  pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
  bash -lc "set -e; git clone -q https://github.com/ihavespoons/bite && cd bite \
    && pip install -q -e '.[model,eval]' \
    && python scripts/run_baseline.py --config configs/ternary.yaml"
```

Start with **one small task** first (edit `eval.tasks` to e.g. `[gsm8k]`) to confirm the model loads
and scores, then scale to the full suite.

**Multimodal loading is handled:** `Qwen3.6-35B-A3B` is a multimodal `...ForConditionalGeneration`, so
`bite.models.loader` loads it with `AutoModelForImageTextToText` + `trust_remote_code`, the low-bit
swap **excludes the vision tower** (`swap_language_model`), and `run_lm_eval` wraps the loaded model in
an `HFLM` so `lm-eval` doesn't re-load it with the wrong auto-class. If the exact processor/tokenizer
path differs on the runner, that's the one spot to adjust.

## Managing jobs

```bash
hf jobs ps                 # running jobs
hf jobs logs <job_id>      # stream logs
hf jobs inspect <job_id>   # status + URL
hf jobs cancel <job_id>
```

## Model mount: not recommended (measured)

`--mount-model` exists but **don't use it.** Measured on an actual run, HF's intra-datacenter
download hits **~2.6 GB/s → the 70 GB model in ~27 s** (~$0.04). A read-only model *mount* does slow
lazy reads of a 70 GB checkpoint and **stalled** at load in testing — strictly worse than the fast
download. The real per-run cost is model-load-into-GPU + pip install, which a mount doesn't help.
So just let each job download; it's negligible.

`scripts/launch_hf.py` wraps the stages (tests/checktasks/baseline/ptq/qad) with the right flavor,
timeout, and secrets via the Python `run_job` API.
```bash
python scripts/launch_hf.py --stage tests   --flavor t4-small
python scripts/launch_hf.py --stage baseline --flavor h200 --timeout 6h
```
