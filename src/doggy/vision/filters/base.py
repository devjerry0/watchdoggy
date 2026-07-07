from __future__ import annotations

from typing import Protocol, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from doggy.core.config import TunableSettings
    from doggy.vision.analysis import FrameAnalysis


class DetectionFilter(Protocol):
    """One link in the detection filter chain (Chain of Responsibility).

    Each link mutates the shared `FrameAnalysis` in place -- narrowing `dogs`
    and/or `candidates` -- and returns nothing. Links decide for themselves
    whether they apply based on `cfg`.
    """

    def apply(self, analysis: "FrameAnalysis", cfg: "TunableSettings") -> None: ...


class FilterChain:
    """Runs a fixed sequence of detection filters, in order, over one analysis."""

    def __init__(self, filters: Sequence[DetectionFilter]) -> None:
        self._filters = list(filters)

    def run(self, analysis: "FrameAnalysis", cfg: "TunableSettings") -> None:
        for f in self._filters:
            f.apply(analysis, cfg)
