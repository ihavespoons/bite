#!/usr/bin/env python
"""Minimal GPU sanity: the ternary STE quant runs and backprops on CUDA."""

import torch

from bite.quant.fakequant import fake_quantize


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    w = torch.randn(4, 256, device=dev, requires_grad=True)
    fake_quantize(w, "ternary", 128).sum().backward()
    name = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
    print(f"CUDA quant OK grad={w.grad is not None} | device={name}")


if __name__ == "__main__":
    main()
