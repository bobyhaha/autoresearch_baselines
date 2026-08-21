from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

from .config import SearchConfig


@dataclass(frozen=True)
class SearchDecision:
    stage: str
    parent_id: str
    secondary_parent_id: str | None
    reason: str


def scored_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if record.get("metric") is not None and record.get("status") in {"keep", "discard", "baseline"}
    ]


def champion(records: list[dict[str, Any]]) -> dict[str, Any]:
    scored = scored_records(records)
    if not scored:
        raise ValueError("No valid baseline exists")
    return min(scored, key=lambda record: (float(record["metric"]), str(record["trial_id"])))


def _deterministic_fraction(seed: int, index: int) -> float:
    digest = hashlib.sha256(f"{seed}:{index}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def _children_count(records: list[dict[str, Any]], trial_id: str) -> int:
    return sum(1 for record in records if record.get("parent_id") == trial_id)


def _ucb_parent(records: list[dict[str, Any]], config: SearchConfig) -> dict[str, Any]:
    candidates = sorted(scored_records(records), key=lambda record: float(record["metric"]))[
        : config.elite_size
    ]
    if len(candidates) == 1:
        return candidates[0]
    metrics = [float(record["metric"]) for record in candidates]
    span = max(metrics) - min(metrics)
    span = span if span > 1e-12 else max(abs(min(metrics)) * 0.001, 1e-6)
    total = max(1, len(records))

    def priority(record: dict[str, Any]) -> tuple[float, str]:
        quality = (max(metrics) - float(record["metric"])) / span
        visits = _children_count(records, str(record["trial_id"]))
        bonus = config.ucb_weight * math.sqrt(math.log(total + 1) / (visits + 1))
        return quality + bonus, str(record["trial_id"])

    return max(candidates, key=priority)


def _failure_streak(records: list[dict[str, Any]]) -> int:
    count = 0
    for record in reversed([r for r in records if r.get("trial_id") != "b0000"]):
        if record.get("promoted"):
            break
        count += 1
    return count


def choose_search_decision(
    records: list[dict[str, Any]], config: SearchConfig
) -> SearchDecision:
    best = champion(records)
    nonbaseline = [record for record in records if record.get("trial_id") != "b0000"]
    index = len(nonbaseline) + 1
    last = nonbaseline[-1] if nonbaseline else None

    if (
        last
        and last.get("status") in {"failed", "timeout"}
        and int(last.get("debug_depth", 0)) < config.max_debug_retries
        and last.get("workspace")
    ):
        return SearchDecision(
            stage="debug",
            parent_id=str(last["trial_id"]),
            secondary_parent_id=None,
            reason="Repair the most recent executable failure before abandoning its hypothesis.",
        )

    elites = sorted(scored_records(records), key=lambda record: float(record["metric"]))[
        : config.elite_size
    ]
    if config.recombine_every > 0 and index % config.recombine_every == 0 and len(elites) >= 2:
        return SearchDecision(
            stage="recombine",
            parent_id=str(elites[0]["trial_id"]),
            secondary_parent_id=str(elites[1]["trial_id"]),
            reason="Recombine two independently strong branches at the scheduled synthesis step.",
        )

    escape = _failure_streak(records) >= config.failure_escape_after
    explore = escape or _deterministic_fraction(config.seed, index) < config.exploration_fraction
    if explore:
        return SearchDecision(
            stage="explore",
            parent_id=str(best["trial_id"]),
            secondary_parent_id=None,
            reason=(
                "The current line is stalled; make a structurally different bet."
                if escape
                else "Spend the configured search share on a distinct architectural or recipe direction."
            ),
        )

    parent = _ucb_parent(records, config)
    return SearchDecision(
        stage="refine",
        parent_id=str(parent["trial_id"]),
        secondary_parent_id=None,
        reason="Exploit a top-K branch selected by quality plus an under-exploration bonus.",
    )

