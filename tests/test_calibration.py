import torch

from bite.moe.calibration import ExpertCoverage


def test_counts_accumulate_topk():
    cov = ExpertCoverage(num_experts=4, top_k=2)
    # token routes strongly to experts 0 and 1
    logits = torch.tensor([[10.0, 9.0, 0.0, -1.0]])
    cov.update(logits)
    assert cov.counts.tolist() == [1, 1, 0, 0]
    assert cov.tokens == 1


def test_dead_experts_detected():
    cov = ExpertCoverage(num_experts=4, top_k=1)
    cov.update(torch.tensor([[5.0, 0.0, 0.0, 0.0]]).repeat(10, 1))
    assert cov.dead_experts() == [1, 2, 3]


def test_gini_zero_when_balanced():
    cov = ExpertCoverage(num_experts=4, top_k=4)
    cov.update(torch.randn(100, 4))  # top_k == num_experts -> every expert every token
    assert cov.coverage_gini() == 0.0


def test_wrong_width_raises():
    cov = ExpertCoverage(num_experts=8, top_k=2)
    try:
        cov.update(torch.randn(3, 4))
    except ValueError:
        return
    raise AssertionError("expected ValueError for mismatched expert width")


def test_summary_keys():
    cov = ExpertCoverage(num_experts=4, top_k=2)
    cov.update(torch.randn(20, 4))
    s = cov.summary()
    assert {"num_experts", "tokens", "dead", "gini", "min_frac", "max_frac"} <= set(s)
