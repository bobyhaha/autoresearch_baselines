from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .util import canonical_json, sha256_bytes, utc_now


GENESIS_HASH = "0" * 64


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except ImportError:  # pragma: no cover - Windows fallback
            pass
        try:
            yield
        finally:
            try:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except ImportError:  # pragma: no cover
                pass


class Ledger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock_path = path.with_suffix(path.suffix + ".lock")

    def read(self, *, verify: bool = True) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        events: list[dict[str, Any]] = []
        for line_number, raw in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not raw.strip():
                continue
            try:
                events.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid ledger JSON at line {line_number}: {exc}") from exc
        if verify:
            self.verify(events)
        return events

    @staticmethod
    def verify(events: list[dict[str, Any]]) -> None:
        previous = GENESIS_HASH
        for index, event in enumerate(events):
            if event.get("seq") != index:
                raise ValueError(f"Ledger sequence mismatch at event {index}")
            if event.get("prev_hash") != previous:
                raise ValueError(f"Ledger chain broken at event {index}")
            material = {key: value for key, value in event.items() if key != "hash"}
            expected = sha256_bytes(canonical_json(material))
            if event.get("hash") != expected:
                raise ValueError(f"Ledger hash mismatch at event {index}")
            previous = expected

    def append(self, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        with _exclusive_lock(self.lock_path):
            events = self.read(verify=True)
            event: dict[str, Any] = {
                "seq": len(events),
                "timestamp": utc_now(),
                "kind": kind,
                "payload": payload,
                "prev_hash": events[-1]["hash"] if events else GENESIS_HASH,
            }
            event["hash"] = sha256_bytes(canonical_json(event))
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, sort_keys=True, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            return event


def trial_records(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [event["payload"] for event in events if event.get("kind") == "trial_completed"]


def started_trial_ids(events: list[dict[str, Any]]) -> set[str]:
    return {
        str(event["payload"]["trial_id"])
        for event in events
        if event.get("kind") == "trial_started"
    }

