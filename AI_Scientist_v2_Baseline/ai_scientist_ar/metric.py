"""Metric values that compare by *better*, not by larger.

Ported from AI-Scientist-v2 (`ai_scientist/treesearch/utils/metric.py`) and
trimmed to the single-scalar case, which is all the autoresearch task needs:
the ground-truth metric is `val_bpb` and lower is better.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import total_ordering
from typing import Any


@dataclass
@total_ordering
class MetricValue:
    """A metric value to be optimized. Comparisons are by quality, not magnitude."""

    value: float | int | None
    maximize: bool | None = field(default=None, kw_only=True)
    name: str | None = field(default=None, kw_only=True)
    description: str | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if self.value is not None:
            self.value = float(self.value)

    def __gt__(self, other: "MetricValue") -> bool:
        """True if self is a *better* (not necessarily larger) metric value than other."""
        if self.value is None:
            return False
        if other.value is None:
            return True
        if self.value == other.value:
            return False
        comp = self.value > other.value
        return comp if self.maximize else not comp

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, MetricValue):
            return NotImplemented
        return self.value == other.value

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        opt_dir = "?" if self.maximize is None else ("↑" if self.maximize else "↓")
        metric_name = f"({self.name})" if self.name else ""
        return f"Metric{opt_dir}{metric_name}({self.value_npsafe:.6f})"

    @property
    def is_worst(self) -> bool:
        return self.value is None

    @property
    def value_npsafe(self) -> float:
        return self.value if self.value is not None else float("nan")


@dataclass
class WorstMetricValue(MetricValue):
    """An invalid metric value (buggy / crashed run). Always compares worst."""

    value: None = None
