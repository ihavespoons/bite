from bite.quant.policy import KEEP, PrecisionPolicy, default_policy


def test_router_gate_is_kept_but_expert_gate_proj_is_not():
    p = default_policy("ternary")
    assert p.resolve("model.layers.0.mlp.gate") == KEEP
    assert p.resolve("model.layers.0.mlp.experts.3.gate_proj") == "ternary"


def test_shared_expert_is_kept():
    p = default_policy("ternary")
    assert p.resolve("model.layers.0.mlp.shared_expert.up_proj") == KEEP


def test_default_applies_to_attention_projections():
    assert default_policy("ternary").resolve("model.layers.0.self_attn.q_proj") == "ternary"
    assert default_policy("binary").resolve("model.layers.0.self_attn.q_proj") == "binary"


def test_precedence_keep_over_bit_patterns():
    p = PrecisionPolicy(default="ternary", binary_patterns=(r"gate",))
    # keep patterns win over explicit binary override
    assert p.resolve("model.layers.0.mlp.gate") == KEEP


def test_explicit_binary_override():
    p = PrecisionPolicy(default="ternary", binary_patterns=(r"lm_head",))
    assert p.resolve("lm_head") == "binary"
    assert p.resolve("model.layers.0.self_attn.o_proj") == "ternary"


def test_intn_default_is_accepted():
    """intN became a first-class target (bitwidth curve); it must resolve like any other mode."""
    assert PrecisionPolicy(default="int4").resolve("model.layers.0.self_attn.q_proj") == "int4"
    assert PrecisionPolicy(default="int2").resolve("x.gate") == "keep"  # keeps still win


def test_invalid_default_raises():
    for bad in ("int1", "int9", "fp8", "nonsense", ""):
        try:
            PrecisionPolicy(default=bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for default={bad!r}")
