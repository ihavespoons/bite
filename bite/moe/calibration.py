"""Expert-coverage tracking for MoE calibration — the MoE-specific core of the pipeline.

With 256 experts (8 routed + 1 shared) and sparse routing, a naive calibration set leaves
rarely-fired experts under-covered, so their PTQ scales and QAD updates are unreliable. This
module counts per-expert token assignments from router logits so the calibration/QAD data
can be balanced and the long tail oversampled.

``ExpertCoverage`` is pure tensor bookkeeping and unit-tested on CPU; the router-logit hook
that feeds it is attached on the cloud runner where the real model routes tokens.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ExpertCoverage:
    """Accumulates per-expert top-k routing counts across many batches/layers."""

    def __init__(self, num_experts: int, top_k: int = 8) -> None:
        self.num_experts = num_experts
        self.top_k = top_k
        self.counts = torch.zeros(num_experts, dtype=torch.long)
        self.tokens = 0

    def update(self, router_logits: Tensor) -> None:
        """Accumulate counts from ``router_logits`` shaped ``[n_tokens, num_experts]``."""
        if router_logits.shape[-1] != self.num_experts:
            raise ValueError(
                f"router_logits last dim {router_logits.shape[-1]} != num_experts {self.num_experts}"
            )
        flat = router_logits.reshape(-1, self.num_experts)
        topk = flat.topk(self.top_k, dim=-1).indices  # [n_tokens, top_k]
        self.counts += torch.bincount(topk.reshape(-1), minlength=self.num_experts).cpu()
        self.tokens += flat.shape[0]

    def fraction(self) -> Tensor:
        """Per-expert share of routed slots (sums to ``top_k`` if fully used)."""
        total = max(self.tokens, 1)
        return self.counts.float() / total

    def dead_experts(self, min_tokens: int = 1) -> list[int]:
        """Experts that received fewer than ``min_tokens`` assignments."""
        return torch.nonzero(self.counts < min_tokens).flatten().tolist()

    def coverage_gini(self) -> float:
        """Gini coefficient of the count distribution (0 = perfectly balanced)."""
        c = self.counts.float().sort().values
        n = c.numel()
        if c.sum() == 0:
            return 0.0
        idx = torch.arange(1, n + 1, dtype=torch.float)
        return float(((2 * idx - n - 1) * c).sum() / (n * c.sum()))

    def summary(self) -> dict[str, float | int]:
        return {
            "num_experts": self.num_experts,
            "tokens": self.tokens,
            "dead": len(self.dead_experts()),
            "gini": round(self.coverage_gini(), 4),
            "min_frac": round(float(self.fraction().min()), 6),
            "max_frac": round(float(self.fraction().max()), 6),
        }


def attach_coverage_hooks(model: nn.Module, coverage: ExpertCoverage) -> list:
    """Register hooks on each MoE router so ``coverage`` sees the routing logits.

    Matches any module named ``...gate`` with a 2-D ``weight`` Parameter of shape
    ``(num_experts, hidden)`` — covering both a plain ``nn.Linear`` router and Qwen3.5's custom
    ``Qwen3_5MoeTopKRouter`` (whose ``weight`` uses the same convention). A forward-*pre* hook
    recomputes the logits ``F.linear(hidden, weight)`` from the input, so it's robust to whatever
    the router's forward returns. Returns the handles; ``.remove()`` them when calibration ends.
    """
    import torch.nn.functional as F

    def make_hook(cov: ExpertCoverage):
        def pre_hook(module, args):
            hidden = args[0].detach()
            cov.update(F.linear(hidden, module.weight))

        return pre_hook

    handles = []
    for name, module in model.named_modules():
        w = getattr(module, "weight", None)
        if (
            name.endswith("gate")
            and isinstance(w, nn.Parameter)
            and w.dim() == 2
            and w.shape[0] == coverage.num_experts
        ):
            handles.append(module.register_forward_pre_hook(make_hook(coverage)))
    return handles


def inverse_frequency_weights(counts: Tensor, eps: float = 1.0) -> Tensor:
    """Per-expert weights ∝ 1/frequency (rarely-fired experts get more weight), mean ≈ 1."""
    inv = 1.0 / (counts.float() + eps)
    return inv / inv.mean()


def sample_weights(expert_id_lists: list[list[int]], expert_weights: Tensor) -> Tensor:
    """Weight each calibration sample by the mean inverse-frequency of the experts it fires.

    Feeds an oversampling sampler so the QAD/calibration data covers the long tail of experts.
    """
    return torch.stack(
        [expert_weights[torch.tensor(ids, dtype=torch.long)].mean() for ids in expert_id_lists]
    )
