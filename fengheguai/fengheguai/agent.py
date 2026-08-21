from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import CampaignConfig
from .policy import SearchDecision
from .util import atomic_write_json, atomic_write_text, render_tokens


AGENT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "hypothesis": {"type": "string"},
        "change_summary": {"type": "string"},
        "expected_val_bpb_effect": {"type": "string"},
        "risk": {"type": "string"},
    },
    "required": ["hypothesis", "change_summary", "expected_val_bpb_effect", "risk"],
}


@dataclass
class AgentResult:
    success: bool
    return_code: int | None = None
    timed_out: bool = False
    hypothesis: str = ""
    change_summary: str = ""
    expected_val_bpb_effect: str = ""
    risk: str = ""
    errors: list[str] = field(default_factory=list)
    command: list[str] = field(default_factory=list)
    log_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_agent_schema(campaign_root: Path) -> Path:
    path = campaign_root / "control" / "agent_output.schema.json"
    atomic_write_json(path, AGENT_SCHEMA)
    return path


def _record_summary(record: dict[str, Any]) -> str:
    proposal = record.get("proposal") or {}
    hypothesis = str(proposal.get("hypothesis") or "").strip()
    change = str(proposal.get("change_summary") or "").strip()
    finding = str(record.get("finding") or "").strip()
    details = "; ".join(
        part
        for part in (
            f"hypothesis={hypothesis}" if hypothesis else "",
            f"change={change}" if change else "",
            f"evidence={finding}" if finding else "",
        )
        if part
    )
    metric = record.get("metric")
    status = record.get("status")
    return f"- {record.get('trial_id')}: val_bpb={metric}, status={status}; {details[:700]}"


def build_prompt(
    *,
    program: str,
    decision: SearchDecision,
    parent: dict[str, Any],
    secondary_parent: dict[str, Any] | None,
    records: list[dict[str, Any]],
    memory_items: int,
) -> str:
    champion = min(
        (record for record in records if record.get("metric") is not None),
        key=lambda record: float(record["metric"]),
    )
    recent = "\n".join(_record_summary(record) for record in records[-memory_items:])
    secondary = ""
    if secondary_parent:
        secondary = (
            "\nSecondary elite to synthesize conceptually:\n"
            + _record_summary(secondary_parent)
            + "\nDo not copy files from elsewhere; implement the useful combination in train.py.\n"
        )

    stage_guidance = {
        "explore": (
            "Try one coherent, materially different architecture, optimizer, schedule, capacity, "
            "batching, or kernel-aware recipe. Avoid a bundle of unrelated edits."
        ),
        "refine": (
            "Make one focused improvement to this elite branch, using the evidence history to avoid "
            "coordinates that already failed."
        ),
        "recombine": (
            "Synthesize compatible mechanisms from the two elite lines; preserve the causal identity "
            "of the combination so its result is interpretable."
        ),
        "debug": (
            "Repair the existing failed implementation without changing its underlying hypothesis. "
            "Use the failure evidence in the history."
        ),
    }[decision.stage]

    return f"""{program.rstrip()}

## Current search node

Search stage: {decision.stage}
Selection reason: {decision.reason}
Parent: {parent['trial_id']} with val_bpb={parent.get('metric')}
Current champion: {champion['trial_id']} with val_bpb={champion.get('metric')}
{secondary}
Stage instruction: {stage_guidance}

Evidence memory (newest last):
{recent or '- baseline only'}

Inspect the current train.py and the evidence above, form one falsifiable hypothesis, then implement it.

Return the required JSON summary after editing train.py. The controller will independently audit and score it.
"""


def _terminate(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


class AgentRunner:
    def __init__(self, campaign_root: Path, config: CampaignConfig) -> None:
        self.campaign_root = campaign_root.resolve()
        self.config = config
        self.schema_path = write_agent_schema(self.campaign_root)

    def run(self, trial_root: Path, trial_id: str, prompt: str) -> AgentResult:
        artifacts = self.campaign_root / "artifacts" / trial_id
        artifacts.mkdir(parents=True, exist_ok=True)
        prompt_path = artifacts / "prompt.md"
        log_path = artifacts / "agent.log"
        result_path = trial_root / ".fengheguai-agent-result.json"
        atomic_write_text(prompt_path, prompt)
        values = {
            "trial": str(trial_root.resolve()),
            "campaign": str(self.campaign_root),
            "agent_schema": str(self.schema_path),
            "agent_result": str(result_path),
            "python": sys.executable,
        }
        command = render_tokens(self.config.agent.command, values)
        timed_out = False
        return_code: int | None = None
        with log_path.open("w", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                command,
                cwd=trial_root,
                stdin=subprocess.PIPE,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            try:
                process.communicate(prompt, timeout=self.config.agent.timeout_seconds)
                return_code = process.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
                _terminate(process)
                return_code = process.returncode

        errors: list[str] = []
        payload: dict[str, Any] = {}
        if timed_out:
            errors.append("agent timed out")
        if return_code != 0:
            errors.append(f"agent exited with code {return_code}")
        if not result_path.is_file():
            errors.append("agent did not produce its structured result")
        else:
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                errors.append(f"agent result is invalid JSON: {exc}")
            finally:
                result_path.unlink(missing_ok=True)
        for key in AGENT_SCHEMA["required"]:
            if not isinstance(payload.get(key), str) or not payload[key].strip():
                errors.append(f"agent result is missing non-empty {key}")

        return AgentResult(
            success=not errors,
            return_code=return_code,
            timed_out=timed_out,
            hypothesis=str(payload.get("hypothesis") or ""),
            change_summary=str(payload.get("change_summary") or ""),
            expected_val_bpb_effect=str(payload.get("expected_val_bpb_effect") or ""),
            risk=str(payload.get("risk") or ""),
            errors=errors,
            command=command,
            log_path=str(log_path),
        )
