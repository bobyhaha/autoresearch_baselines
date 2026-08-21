"""Claude Code research-agent adapter for Fengheguai.

Fengheguai's agent contract is: receive the research prompt on stdin, edit only
``train.py`` inside the trial workspace, and leave a structured JSON summary at a
requested path. The Claude Code CLI has no ``--output-last-message`` flag, so this
adapter runs ``claude --print --output-format json`` and reconciles the structured
result itself. The agent is also given a tool allowlist without Bash, which makes
the program's "never launch training" rule mechanically true rather than merely
requested.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


REQUIRED_KEYS = ("hypothesis", "change_summary", "expected_val_bpb_effect", "risk")

DEFAULT_TOOLS = "Read,Edit,Write,Glob,Grep,TodoWrite"


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Return the last balanced JSON object in ``text`` that carries the schema keys."""

    candidates: list[dict[str, Any]] = []
    for match in re.finditer(r"\{", text):
        depth = 0
        in_string = False
        escape = False
        for index in range(match.start(), len(text)):
            char = text[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    blob = text[match.start() : index + 1]
                    try:
                        parsed = json.loads(blob)
                    except json.JSONDecodeError:
                        break
                    if isinstance(parsed, dict) and all(key in parsed for key in REQUIRED_KEYS):
                        candidates.append(parsed)
                    break
    return candidates[-1] if candidates else None


def _normalise(payload: dict[str, Any]) -> dict[str, str]:
    return {key: str(payload.get(key, "") or "").strip() for key in REQUIRED_KEYS}


def _valid(payload: dict[str, Any] | None) -> bool:
    return bool(payload) and all(str(payload.get(key, "") or "").strip() for key in REQUIRED_KEYS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", required=True, help="Trial workspace directory")
    parser.add_argument("--result", required=True, help="Path for the structured JSON result")
    parser.add_argument("--schema", default=None, help="Path to the agent output JSON schema")
    parser.add_argument("--model", default=os.environ.get("FENGHEGUAI_AGENT_MODEL", "claude-opus-5"))
    parser.add_argument("--claude", default=os.environ.get("FENGHEGUAI_CLAUDE_BIN", "claude"))
    parser.add_argument("--tools", default=os.environ.get("FENGHEGUAI_AGENT_TOOLS", DEFAULT_TOOLS))
    args = parser.parse_args()

    trial = Path(args.trial).resolve()
    result_path = Path(args.result)
    prompt = sys.stdin.read()

    schema_text = ""
    if args.schema and Path(args.schema).is_file():
        schema_text = Path(args.schema).read_text(encoding="utf-8").strip()

    instruction = f"""

## Output contract for this run

You have no shell access in this run, by design: you cannot and must not start
training or evaluation. Edit only `train.py` in {trial}.

When the edit is complete, write your summary as a single JSON object to:

{result_path}

with exactly these string keys: hypothesis, change_summary, expected_val_bpb_effect, risk.

Then end your turn with that same JSON object as your final message, and nothing else.
"""
    if schema_text:
        instruction += f"\nThe required schema is:\n\n```json\n{schema_text}\n```\n"

    command = [
        args.claude,
        "--print",
        "--output-format",
        "json",
        "--model",
        args.model,
        "--permission-mode",
        "acceptEdits",
        "--allowedTools",
        args.tools,
        "--add-dir",
        str(trial),
    ]

    print(f"[fengheguai] agent command: {' '.join(command)}", flush=True)
    completed = subprocess.run(
        command,
        input=prompt + instruction,
        cwd=str(trial),
        text=True,
        capture_output=True,
    )
    if completed.stdout:
        print(completed.stdout, flush=True)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, flush=True)

    # The agent may already have written the file itself; that copy wins when valid.
    existing: dict[str, Any] | None = None
    if result_path.is_file():
        try:
            loaded = json.loads(result_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing = loaded
        except (OSError, json.JSONDecodeError):
            existing = None
    if _valid(existing):
        result_path.write_text(json.dumps(_normalise(existing), indent=2), encoding="utf-8")
        return completed.returncode

    final_text = completed.stdout or ""
    try:
        envelope = json.loads(completed.stdout)
        if isinstance(envelope, dict) and isinstance(envelope.get("result"), str):
            final_text = envelope["result"]
    except json.JSONDecodeError:
        pass

    recovered = _extract_json_object(final_text)
    if _valid(recovered):
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(_normalise(recovered), indent=2), encoding="utf-8")
        return completed.returncode

    print("[fengheguai] agent produced no valid structured result", file=sys.stderr)
    return completed.returncode or 1


if __name__ == "__main__":
    raise SystemExit(main())
