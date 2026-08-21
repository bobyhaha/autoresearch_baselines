from __future__ import annotations

import ast
import json
import math
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ._audit_runner import SENTINEL
from .config import CampaignConfig
from .util import (
    atomic_write_json,
    environment_fingerprint,
    render_tokens,
    sha256_file,
    utc_now,
)


TRAINING_SECONDS_RE = re.compile(r"^training_seconds:\s*([0-9]+(?:\.[0-9]+)?)\s*$", re.MULTILINE)
PEAK_VRAM_RE = re.compile(r"^peak_vram_mb:\s*([0-9]+(?:\.[0-9]+)?)\s*$", re.MULTILINE)
STEPS_RE = re.compile(r"^num_steps:\s*([0-9]+)\s*$", re.MULTILINE)


@dataclass
class EvaluationResult:
    status: str
    metric: float | None = None
    training_seconds: float | None = None
    wall_seconds: float | None = None
    peak_vram_mb: float | None = None
    num_steps: int | None = None
    return_code: int | None = None
    timed_out: bool = False
    errors: list[str] = field(default_factory=list)
    log_path: str | None = None
    log_sha256: str | None = None
    train_sha256: str | None = None
    evidence_path: str | None = None
    command: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return self.status == "valid" and self.metric is not None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def file_manifest(root: Path, files: tuple[str, ...]) -> dict[str, str]:
    return {relative: sha256_file(root / relative) for relative in files}


def _target_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            names.add(child.id)
    return names


def _timing_nodes(tree: ast.AST) -> dict[str, list[str]]:
    protected: dict[str, list[str]] = {
        "initialize_total": [],
        "step_start": [],
        "step_end": [],
        "step_duration": [],
        "accumulate_time": [],
        "budget_break": [],
        "cuda_synchronize": [],
        "report_training_seconds": [],
        "timing_assignments": [],
    }
    for node in ast.walk(tree):
        dump = ast.dump(node, annotate_fields=True, include_attributes=False)
        if isinstance(node, ast.Assign):
            targets = set().union(*(_target_names(target) for target in node.targets))
            if targets.intersection({"total_training_time", "t0", "t1", "dt"}):
                protected["timing_assignments"].append(dump)
            if "total_training_time" in targets and isinstance(node.value, ast.Constant) and node.value.value == 0:
                protected["initialize_total"].append(dump)
            elif "t0" in targets:
                protected["step_start"].append(dump)
            elif "t1" in targets:
                protected["step_end"].append(dump)
            elif "dt" in targets and {"t0", "t1"}.issubset(_target_names(node.value)):
                protected["step_duration"].append(dump)
        elif isinstance(node, ast.AugAssign):
            if _target_names(node.target).intersection({"total_training_time", "t0", "t1", "dt"}):
                protected["timing_assignments"].append(dump)
        elif isinstance(node, ast.If):
            names = _target_names(node.test)
            if any(isinstance(child, ast.Break) for child in node.body) and {
                "step",
                "total_training_time",
                "TIME_BUDGET",
            }.issubset(names):
                protected["budget_break"].append(dump)
            if any(
                isinstance(child, ast.AugAssign)
                and "total_training_time" in _target_names(child.target)
                and "dt" in _target_names(child.value)
                for child in node.body
            ):
                protected["accumulate_time"].append(dump)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if (
                isinstance(call.func, ast.Attribute)
                and call.func.attr == "synchronize"
                and isinstance(call.func.value, ast.Attribute)
                and call.func.value.attr == "cuda"
            ):
                protected["cuda_synchronize"].append(dump)
            if isinstance(call.func, ast.Name) and call.func.id == "print":
                constants = [
                    str(child.value)
                    for child in ast.walk(call)
                    if isinstance(child, ast.Constant) and isinstance(child.value, str)
                ]
                if any("training_seconds:" in value for value in constants) and "total_training_time" in _target_names(call):
                    protected["report_training_seconds"].append(dump)
    return protected


def build_training_contract(train_path: Path) -> dict[str, Any]:
    tree = ast.parse(train_path.read_text(encoding="utf-8"), filename=str(train_path))
    nodes = _timing_nodes(tree)
    required = (
        "initialize_total",
        "step_start",
        "step_end",
        "step_duration",
        "accumulate_time",
        "budget_break",
        "report_training_seconds",
    )
    if not all(nodes[label] for label in required) or len(nodes["cuda_synchronize"]) < 2:
        return {"mode": "output_only", "reason": "seed does not expose the Karpathy timing structure"}
    return {"mode": "protected_ast", "nodes": nodes}


def _audit_training_contract(train_path: Path, contract: dict[str, Any] | None) -> list[str]:
    if not contract or contract.get("mode") != "protected_ast":
        return []
    try:
        tree = ast.parse(train_path.read_text(encoding="utf-8"), filename=str(train_path))
    except SyntaxError:
        return []
    actual = _timing_nodes(tree)
    errors: list[str] = []
    for label, expected_nodes in (contract.get("nodes") or {}).items():
        if sorted(actual.get(label, [])) != sorted(expected_nodes):
            errors.append(f"protected 300-second timing structure changed: {label}")
    return errors


def _assignment_names(node: ast.AST) -> list[str]:
    names: list[str] = []
    targets: list[ast.AST] = []
    if isinstance(node, ast.Assign):
        targets = list(node.targets)
    elif isinstance(node, ast.AnnAssign):
        targets = [node.target]
    elif isinstance(node, ast.AugAssign):
        targets = [node.target]
    for target in targets:
        for child in ast.walk(target):
            if isinstance(child, ast.Name):
                names.append(child.id)
            elif isinstance(child, ast.Attribute):
                names.append(child.attr)
    return names


def audit_candidate_source(
    trial_root: Path,
    immutable_manifest: dict[str, str],
    training_contract: dict[str, Any] | None = None,
) -> tuple[list[str], str | None]:
    errors: list[str] = []
    for relative, expected in immutable_manifest.items():
        path = trial_root / relative
        if not path.is_file():
            errors.append(f"immutable file missing: {relative}")
        elif sha256_file(path) != expected:
            errors.append(f"immutable file changed: {relative}")

    train_path = trial_root / "train.py"
    if not train_path.is_file():
        return errors + ["train.py is missing"], None
    source = train_path.read_text(encoding="utf-8")
    train_hash = sha256_file(train_path)
    if SENTINEL in source or "FENGHEGUAI" in source:
        errors.append("train.py contains a reserved evaluator marker")
    try:
        tree = ast.parse(source, filename=str(train_path))
    except SyntaxError as exc:
        return errors + [f"train.py syntax error: {exc}"], train_hash

    evaluate_calls = 0
    forbidden_imports = {"subprocess", "runpy", "importlib", "socket"}
    forbidden_calls = {"eval", "exec", "compile", "__import__"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in forbidden_imports:
                    errors.append(f"forbidden import in train.py: {alias.name}")
        elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in forbidden_imports:
            errors.append(f"forbidden import in train.py: {node.module}")
        elif isinstance(node, ast.Call):
            function_name = None
            if isinstance(node.func, ast.Name):
                function_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                function_name = node.func.attr
            if function_name == "evaluate_bpb":
                evaluate_calls += 1
            if isinstance(node.func, ast.Name) and function_name in forbidden_calls:
                errors.append(f"forbidden dynamic execution call: {function_name}")
            if function_name == "make_dataloader":
                split_node = node.args[3] if len(node.args) >= 4 else next(
                    (keyword.value for keyword in node.keywords if keyword.arg == "split"), None
                )
                if not (
                    isinstance(split_node, ast.Constant)
                    and isinstance(split_node.value, str)
                    and split_node.value == "train"
                ):
                    errors.append("all direct make_dataloader calls in train.py must use literal split='train'")
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            protected = {"TIME_BUDGET", "evaluate_bpb"}.intersection(_assignment_names(node))
            if protected:
                errors.append(f"assignment to protected name(s): {', '.join(sorted(protected))}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value == "val" or "shard_06542" in node.value:
                errors.append("train.py must not directly address validation data")

    if evaluate_calls != 1:
        errors.append(f"train.py must call evaluate_bpb exactly once; found {evaluate_calls}")
    errors.extend(_audit_training_contract(train_path, training_contract))
    return sorted(set(errors)), train_hash


def _parse_last(pattern: re.Pattern[str], text: str, cast: Any = float) -> Any | None:
    matches = pattern.findall(text)
    return cast(matches[-1]) if matches else None


def _gpu_fingerprint() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return {"available": False}
    try:
        result = subprocess.run(
            [
                executable,
                "--query-gpu=uuid,name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return {
            "available": result.returncode == 0,
            "rows": [line.strip() for line in result.stdout.splitlines() if line.strip()],
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": str(exc)}


def _terminate_process_group(process: subprocess.Popen[Any]) -> None:
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


class Evaluator:
    def __init__(
        self,
        *,
        campaign_root: Path,
        config: CampaignConfig,
        immutable_manifest: dict[str, str],
        training_contract: dict[str, Any] | None = None,
    ) -> None:
        self.campaign_root = campaign_root.resolve()
        self.config = config
        self.immutable_manifest = immutable_manifest
        self.training_contract = training_contract or {}
        self.audit_runner = Path(__file__).with_name("_audit_runner.py").resolve()

    def evaluate(self, trial_root: Path, *, run_label: str) -> EvaluationResult:
        trial_root = trial_root.resolve()
        errors, train_hash = audit_candidate_source(
            trial_root, self.immutable_manifest, self.training_contract
        )
        evidence_dir = self.campaign_root / "evidence" / run_label
        evidence_dir.mkdir(parents=True, exist_ok=True)
        log_path = evidence_dir / "run.log"
        evidence_path = evidence_dir / "evidence.json"
        if errors:
            result = EvaluationResult(
                status="rejected",
                errors=errors,
                train_sha256=train_hash,
                log_path=str(log_path),
                evidence_path=str(evidence_path),
            )
            self._write_evidence(evidence_path, result, trial_root, {})
            return result

        nonce = secrets.token_hex(24)
        values = {
            "trial": str(trial_root),
            "audit_runner": str(self.audit_runner),
            "campaign": str(self.campaign_root),
            "python": sys.executable,
        }
        command = render_tokens(self.config.train_command, values) + ["--nonce", nonce]
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        started = time.monotonic()
        timed_out = False
        return_code: int | None = None
        with log_path.open("w", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                command,
                cwd=trial_root,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=True,
                text=True,
            )
            try:
                return_code = process.wait(timeout=self.config.objective.process_timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                _terminate_process_group(process)
                return_code = process.returncode
        wall_seconds = time.monotonic() - started
        text = log_path.read_text(encoding="utf-8", errors="replace")

        training_seconds = _parse_last(TRAINING_SECONDS_RE, text)
        peak_vram = _parse_last(PEAK_VRAM_RE, text)
        num_steps = _parse_last(STEPS_RE, text, int)
        locked_records: list[dict[str, Any]] = []
        for line in text.splitlines():
            if line.startswith(SENTINEL):
                try:
                    record = json.loads(line[len(SENTINEL) :])
                except json.JSONDecodeError:
                    continue
                if record.get("nonce") == nonce:
                    locked_records.append(record)

        errors = []
        metric: float | None = None
        if timed_out:
            errors.append("process exceeded the wall-clock safety timeout")
        if return_code != 0:
            errors.append(f"training process exited with code {return_code}")
        if len(locked_records) != 1:
            errors.append(f"expected one nonce-bound locked evaluation, found {len(locked_records)}")
        else:
            try:
                metric = float(locked_records[0]["val_bpb"])
            except (KeyError, TypeError, ValueError):
                errors.append("locked evaluation did not contain numeric val_bpb")
            if metric is not None and (not math.isfinite(metric) or metric <= 0):
                errors.append(f"invalid val_bpb: {metric}")
                metric = None
        if training_seconds is None:
            errors.append("training_seconds is missing from output")
        elif training_seconds > (
            self.config.objective.training_seconds_limit
            + self.config.objective.training_seconds_tolerance
        ):
            errors.append(
                f"training_seconds {training_seconds} exceeds the 300-second contract plus tolerance"
            )
        post_errors, post_hash = audit_candidate_source(
            trial_root, self.immutable_manifest, self.training_contract
        )
        errors.extend(post_errors)
        if post_hash != train_hash:
            errors.append("train.py changed during its own evaluation")

        result = EvaluationResult(
            status="valid" if not errors else ("timeout" if timed_out else "failed"),
            metric=metric if not errors else None,
            training_seconds=training_seconds,
            wall_seconds=wall_seconds,
            peak_vram_mb=peak_vram,
            num_steps=num_steps,
            return_code=return_code,
            timed_out=timed_out,
            errors=sorted(set(errors)),
            log_path=str(log_path),
            log_sha256=sha256_file(log_path),
            train_sha256=train_hash,
            evidence_path=str(evidence_path),
            command=command,
        )
        self._write_evidence(
            evidence_path,
            result,
            trial_root,
            {"nonce_record_count": len(locked_records)},
        )
        return result

    def _write_evidence(
        self,
        path: Path,
        result: EvaluationResult,
        trial_root: Path,
        extra: dict[str, Any],
    ) -> None:
        payload = {
            "schema_version": 1,
            "recorded_at": utc_now(),
            "objective": {
                "metric": "val_bpb",
                "direction": "minimize",
                "training_seconds_limit": 300,
            },
            "trial_root": str(trial_root),
            "immutable_manifest": self.immutable_manifest,
            "result": result.to_dict(),
            "environment": environment_fingerprint(),
            "gpu": _gpu_fingerprint(),
            **extra,
        }
        atomic_write_json(path, payload)
