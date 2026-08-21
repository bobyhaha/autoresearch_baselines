"""LLM backend: a rendezvous to the operator's Claude session.

AI-Scientist-v2 calls out to an API-backed model (`backend_anthropic` /
`backend_openai`) whenever it needs code written. In this deployment there is no
API key and no second interface — the single Claude session driving the campaign
*is* the coding model. So `query()` does not call an API; it publishes a request
into a rendezvous directory and blocks until a response file appears.

Two paths satisfy a request:

1. **Queue hit (non-blocking).** Candidates pre-authored against a specific parent
   node are popped immediately. This is what keeps the GPU busy: the agent answers
   the *next* request while the *current* 5-minute trial is still running.
2. **Blocking rendezvous.** No suitable queued candidate, so the request is written
   out and the harness waits for the agent to answer it.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Optional


class RendezvousTimeout(RuntimeError):
    """No response arrived within the allotted window."""


class RendezvousBackend:
    def __init__(
        self,
        rendezvous_dir: str | Path,
        timeout: float = 3600.0,
        poll: float = 3.0,
        logger=None,
    ) -> None:
        self.dir = Path(rendezvous_dir)
        self.queue_dir = self.dir / "queue"
        self.requests_dir = self.dir / "requests"
        self.responses_dir = self.dir / "responses"
        self.used_dir = self.dir / "used"
        for d in (self.queue_dir, self.requests_dir, self.responses_dir, self.used_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.poll = poll
        self.log = logger

    def _say(self, msg: str) -> None:
        if self.log:
            self.log.info(msg)
        else:
            print(msg, flush=True)

    # ---- path 1: pre-authored candidates ----

    def pop_queued(self, op: str, parent_id: Optional[str]) -> Optional[dict]:
        """Take the oldest queued candidate matching this (op, parent) if one exists."""
        candidates = sorted(self.queue_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
        for path in candidates:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if payload.get("op") != op:
                continue
            # `parent_id` must match exactly; a draft has no parent.
            if (payload.get("parent_id") or None) != (parent_id or None):
                continue
            if not str(payload.get("code", "")).strip():
                continue
            try:
                path.rename(self.used_dir / path.name)
            except OSError:
                continue  # lost a race with a concurrent pop; skip it
            payload["_source"] = f"queue:{path.name}"
            return payload
        return None

    def queue_depth(self, op: str | None = None, parent_id: str | None = None) -> int:
        n = 0
        for path in self.queue_dir.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if op is not None and payload.get("op") != op:
                continue
            if parent_id is not None and (payload.get("parent_id") or None) != (parent_id or None):
                continue
            n += 1
        return n

    # ---- path 2: blocking rendezvous ----

    def query(self, op: str, parent_id: Optional[str], context: dict) -> dict:
        """Return {plan, code} for the requested operation, blocking if necessary."""
        queued = self.pop_queued(op, parent_id)
        if queued is not None:
            self._say(f"[backend] queue hit for op={op} parent={parent_id} ({queued['_source']})")
            return queued

        req_id = f"{int(time.time())}_{uuid.uuid4().hex[:6]}"
        request = {
            "request_id": req_id,
            "op": op,
            "parent_id": parent_id,
            "created": time.time(),
            **context,
        }
        req_path = self.requests_dir / f"{req_id}.json"
        tmp = req_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(request, indent=2), encoding="utf-8")
        tmp.replace(req_path)  # atomic publish: the agent never sees a half-written request

        # A single pointer file so the agent can find the open request without globbing.
        (self.dir / "PENDING").write_text(req_id, encoding="utf-8")
        self._say(f"[backend] BLOCKING on rendezvous request {req_id} (op={op}, parent={parent_id})")

        resp_path = self.responses_dir / f"{req_id}.json"
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            if resp_path.exists():
                try:
                    payload = json.loads(resp_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    time.sleep(self.poll)  # still being written; try again
                    continue
                if str(payload.get("code", "")).strip():
                    (self.dir / "PENDING").unlink(missing_ok=True)
                    req_path.rename(self.used_dir / req_path.name)
                    payload["_source"] = f"rendezvous:{req_id}"
                    self._say(f"[backend] got response for {req_id}")
                    return payload
            # A queued candidate authored after the request was posted also satisfies it.
            queued = self.pop_queued(op, parent_id)
            if queued is not None:
                (self.dir / "PENDING").unlink(missing_ok=True)
                req_path.rename(self.used_dir / req_path.name)
                self._say(f"[backend] request {req_id} satisfied late by {queued['_source']}")
                return queued
            time.sleep(self.poll)

        (self.dir / "PENDING").unlink(missing_ok=True)
        raise RendezvousTimeout(f"no response for request {req_id} within {self.timeout}s")
