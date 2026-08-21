from __future__ import annotations

from pathlib import Path
from typing import Any

from .policy import champion
from .util import atomic_write_json, atomic_write_text


def _fmt_metric(value: Any) -> str:
    return "-" if value is None else f"{float(value):.6f}"


def render_reports(campaign_root: Path, records: list[dict[str, Any]]) -> None:
    reports = campaign_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    if not records:
        atomic_write_text(reports / "STATUS.md", "# Campaign status\n\nNo baseline yet.\n")
        return

    best = champion(records)
    valid = sum(record.get("metric") is not None for record in records)
    promoted = sum(bool(record.get("promoted")) for record in records)
    status_lines = [
        "# Fengheguai campaign status",
        "",
        f"- Champion: `{best['trial_id']}`",
        f"- Best `val_bpb`: `{_fmt_metric(best.get('metric'))}`",
        "- Training contract: exactly 300 seconds, lower is better",
        f"- Completed nodes: {len(records)} ({valid} valid, {promoted} promoted)",
        "",
        "| node | parent | stage | val_bpb | delta vs parent | status | promoted |",
        "|---|---|---|---:|---:|---|---|",
    ]
    for record in records:
        status_lines.append(
            "| {trial} | {parent} | {stage} | {metric} | {delta} | {status} | {promoted} |".format(
                trial=record.get("trial_id", "-"),
                parent=record.get("parent_id") or "-",
                stage=record.get("stage") or "-",
                metric=_fmt_metric(record.get("metric")),
                delta=_fmt_metric(record.get("delta_vs_parent")),
                status=record.get("status", "-"),
                promoted="yes" if record.get("promoted") else "no",
            )
        )
    atomic_write_text(reports / "STATUS.md", "\n".join(status_lines) + "\n")

    tsv = ["trial\tparent\tstage\tval_bpb\ttraining_seconds\tstatus\tdescription"]
    for record in records:
        proposal = record.get("proposal") or {}
        description = str(proposal.get("change_summary") or record.get("finding") or "").replace(
            "\t", " "
        ).replace("\n", " ")
        tsv.append(
            "\t".join(
                [
                    str(record.get("trial_id", "")),
                    str(record.get("parent_id") or ""),
                    str(record.get("stage") or ""),
                    _fmt_metric(record.get("metric")),
                    _fmt_metric(record.get("training_seconds")),
                    str(record.get("status", "")),
                    description,
                ]
            )
        )
    atomic_write_text(reports / "results.tsv", "\n".join(tsv) + "\n")

    nodes = []
    edges = []
    for record in records:
        nodes.append(
            {
                "id": record.get("trial_id"),
                "metric": record.get("metric"),
                "status": record.get("status"),
                "stage": record.get("stage"),
                "promoted": bool(record.get("promoted")),
                "finding": record.get("finding"),
            }
        )
        if record.get("parent_id"):
            edges.append(
                {
                    "source": record["parent_id"],
                    "target": record["trial_id"],
                    "relation": "search_branch",
                }
            )
        if record.get("secondary_parent_id"):
            edges.append(
                {
                    "source": record["secondary_parent_id"],
                    "target": record["trial_id"],
                    "relation": "recombination",
                }
            )
    atomic_write_json(reports / "research_map.json", {"nodes": nodes, "edges": edges})

    findings = ["# Evidence memory", ""]
    for record in records:
        findings.extend(
            [
                f"## {record.get('trial_id')}: {record.get('status')}",
                "",
                str(record.get("finding") or "No finding recorded."),
                "",
            ]
        )
    atomic_write_text(reports / "FINDINGS.md", "\n".join(findings))


def status_payload(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"initialized": True, "baseline_complete": False, "trials": 0}
    best = champion(records)
    return {
        "initialized": True,
        "baseline_complete": any(record.get("trial_id") == "b0000" for record in records),
        "trials": len(records),
        "champion_id": best["trial_id"],
        "val_bpb": best["metric"],
        "promotions": sum(bool(record.get("promoted")) for record in records),
        "last": records[-1],
    }
