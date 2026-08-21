"""Operator-in-the-loop research-agent adapter for Fengheguai.

Fengheguai expects to launch a subprocess that edits ``train.py`` and leaves a
structured JSON summary behind. This adapter is that subprocess when the
implementation researcher is an interactive Claude session rather than a headless
CLI: it publishes the generated prompt into a rendezvous inbox, blocks, and
returns as soon as a valid structured result appears at the requested path.

The protocol is deliberately one-directional and last-write-wins on a single
file: the researcher edits ``train.py`` first and writes the result JSON last, so
the appearance of a valid result means the candidate source is already in place.
Nothing here can influence scoring; the controller still audits and measures the
trial itself.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


REQUIRED_KEYS = ("hypothesis", "change_summary", "expected_val_bpb_effect", "risk")


def _valid(payload: Any) -> bool:
    return isinstance(payload, dict) and all(
        str(payload.get(key, "") or "").strip() for key in REQUIRED_KEYS
    )


def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--schema", default=None)
    parser.add_argument("--campaign", default=None)
    parser.add_argument(
        "--inbox",
        default=os.environ.get("FENGHEGUAI_RENDEZVOUS_INBOX"),
        help="Directory where pending prompts are published",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.environ.get("FENGHEGUAI_RENDEZVOUS_TIMEOUT", "840")),
    )
    parser.add_argument(
        "--poll",
        type=float,
        default=float(os.environ.get("FENGHEGUAI_RENDEZVOUS_POLL", "2")),
    )
    args = parser.parse_args()

    trial = Path(args.trial).resolve()
    result_path = Path(args.result)
    prompt = sys.stdin.read()

    inbox = Path(args.inbox) if args.inbox else trial.parent.parent / "rendezvous"
    inbox.mkdir(parents=True, exist_ok=True)

    # A stale result from a previous trial must never satisfy this request.
    if result_path.exists():
        result_path.unlink()

    request = {
        "trial_id": trial.name,
        "trial": str(trial),
        "train_py": str(trial / "train.py"),
        "result_path": str(result_path),
        "schema_path": args.schema,
        "campaign": args.campaign,
        "required_keys": list(REQUIRED_KEYS),
        "requested_at": time.time(),
        "deadline": time.time() + args.timeout,
    }
    (inbox / f"{trial.name}.prompt.md").write_text(prompt, encoding="utf-8")
    request_path = inbox / f"{trial.name}.request.json"
    request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")
    pending = inbox / "PENDING.json"
    pending.write_text(json.dumps(request, indent=2), encoding="utf-8")

    print(
        f"[fengheguai] awaiting researcher patch for {trial.name}\n"
        f"[fengheguai]   prompt:  {inbox / (trial.name + '.prompt.md')}\n"
        f"[fengheguai]   edit:    {trial / 'train.py'}\n"
        f"[fengheguai]   reply:   {result_path}\n"
        f"[fengheguai]   timeout: {args.timeout:.0f}s",
        flush=True,
    )

    deadline = time.time() + args.timeout
    last_note = 0.0
    while time.time() < deadline:
        if result_path.is_file():
            payload = _load(result_path)
            if _valid(payload):
                request["completed_at"] = time.time()
                request["status"] = "answered"
                request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")
                if pending.is_file():
                    pending.unlink()
                waited = request["completed_at"] - request["requested_at"]
                print(f"[fengheguai] researcher patch received after {waited:.0f}s", flush=True)
                return 0
        now = time.time()
        if now - last_note >= 60:
            last_note = now
            print(f"[fengheguai] still waiting ({deadline - now:.0f}s left)", flush=True)
        time.sleep(args.poll)

    request["status"] = "timeout"
    request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")
    if pending.is_file():
        pending.unlink()
    print("[fengheguai] researcher did not respond before the deadline", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
