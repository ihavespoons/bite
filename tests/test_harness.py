from bite.eval.harness import (
    cot_self_consistency,
    retained_fraction,
    tool_call_parse_rate,
)


def test_tool_call_parse_rate():
    outs = ['{"name": "search", "args": {}}', "not json", '{"noname": 1}', '{"name": "x"}']
    assert tool_call_parse_rate(outs) == 0.5
    assert tool_call_parse_rate([]) == 0.0


def test_cot_self_consistency():
    assert cot_self_consistency(["42", "42", "7", "42"]) == 0.75
    assert cot_self_consistency([]) == 0.0


def test_retained_fraction():
    assert retained_fraction(80.49, 85.07) > 0.94
    assert retained_fraction(1.0, 0.0) == 0.0
