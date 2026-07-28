"""Per-tensor precision policy: which Linear layers go low-bit, and at what precision.

Follows the whitepaper's "no large high-precision escape hatch" principle: attention
projections, routed-expert MLP projections, and the LM head go to the low-bit target,
while a small, high-leverage tail stays in higher precision — the MoE **router/gate** and
the always-on **shared expert** (Qwen3.6-35B-A3B specific). Non-Linear tensors (RMSNorm,
DeltaNet ``A_log``/``dt_bias``/short-conv, biases) are never swapped and so are kept
implicitly by the module-swap in :mod:`bite.quant.quantlinear`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

KEEP = "keep"


def _default_keep() -> tuple[str, ...]:
    # Router gate is named ``...mlp.gate`` (a Linear); expert projections are
    # ``...experts.<i>.gate_proj`` — anchor on the segment end so we don't match those.
    return (r"\.gate$", r"shared_expert")


@dataclass
class PrecisionPolicy:
    """Resolve a module's target precision from its dotted name.

    Precedence: ``keep`` patterns win, then explicit ``binary``/``ternary`` overrides,
    then ``default``.
    """

    default: str = "ternary"  # "ternary" | "binary"
    keep_patterns: tuple[str, ...] = field(default_factory=_default_keep)
    binary_patterns: tuple[str, ...] = ()
    ternary_patterns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        from bite.quant.fakequant import parse_mode

        parse_mode(self.default)  # raises for anything but ternary/binary/intN
        self._keep = [re.compile(p) for p in self.keep_patterns]
        self._binary = [re.compile(p) for p in self.binary_patterns]
        self._ternary = [re.compile(p) for p in self.ternary_patterns]

    @staticmethod
    def _any(patterns: list[re.Pattern[str]], name: str) -> bool:
        return any(p.search(name) for p in patterns)

    def resolve(self, name: str) -> str:
        """Return ``"keep"``, ``"binary"``, or ``"ternary"`` for a module name."""
        if self._any(self._keep, name):
            return KEEP
        if self._any(self._binary, name):
            return "binary"
        if self._any(self._ternary, name):
            return "ternary"
        return self.default


def default_policy(mode: str = "ternary") -> PrecisionPolicy:
    """Policy for Qwen3.6-35B-A3B: everything low-bit except router gate + shared expert."""
    return PrecisionPolicy(default=mode)
