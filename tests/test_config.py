from pathlib import Path

from bite.config import load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_ternary_inherits_base_and_overrides_mode():
    cfg = load_config(CONFIGS / "ternary.yaml")
    assert cfg["quant"]["mode"] == "ternary"
    assert cfg["quant"]["group_size"] == 128          # from base (fork g128, Stage 0 finding #6)
    assert cfg["moe"]["num_experts"] == 256           # from base
    assert cfg["export"]["gguf_type"]["ternary"] == "Q2_0"   # fork ternary block
    assert cfg["export"]["gguf_type"]["binary"] == "Q1_0"    # fork binary block


def test_binary_override_mode_and_init_from():
    cfg = load_config(CONFIGS / "binary.yaml")
    assert cfg["quant"]["mode"] == "binary"
    assert cfg["quant"]["group_size"] == 128           # fork Q1_0 block
    assert cfg["qad"]["init_from"].endswith("ternary/healed")
    assert cfg["qad"]["teacher_topk"] == 64            # deep-merged from base
