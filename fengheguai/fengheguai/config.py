from __future__ import annotations

import ast
import shlex
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .util import atomic_write_json, read_json


@dataclass(frozen=True)
class ObjectiveConfig:
    metric: str = "val_bpb"
    direction: str = "minimize"
    training_seconds_limit: float = 300.0
    training_seconds_tolerance: float = 2.0
    process_timeout_seconds: float = 660.0
    minimum_improvement: float = 0.0
    confirmation_runs: int = 1


@dataclass(frozen=True)
class SearchConfig:
    elite_size: int = 5
    exploration_fraction: float = 0.45
    failure_escape_after: int = 3
    recombine_every: int = 7
    max_debug_retries: int = 1
    ucb_weight: float = 0.15
    memory_items: int = 16
    seed: int = 1337


@dataclass(frozen=True)
class AgentConfig:
    command: tuple[str, ...] = (
        "codex",
        "exec",
        "--ephemeral",
        "--skip-git-repo-check",
        "--sandbox",
        "workspace-write",
        "--approve-for-me",
        "--cd",
        "{trial}",
        "--output-schema",
        "{agent_schema}",
        "--output-last-message",
        "{agent_result}",
        "-",
    )
    timeout_seconds: float = 900.0


@dataclass(frozen=True)
class CampaignConfig:
    name: str
    target: str
    source_files: tuple[str, ...] = ("train.py", "prepare.py", "pyproject.toml", "uv.lock")
    editable_files: tuple[str, ...] = ("train.py",)
    immutable_files: tuple[str, ...] = ("prepare.py", "pyproject.toml", "uv.lock")
    train_command: tuple[str, ...] = (
        "uv",
        "run",
        "--project",
        "{trial}",
        "python",
        "{audit_runner}",
        "--target",
        "{trial}",
    )
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in ("source_files", "editable_files", "immutable_files", "train_command"):
            value[key] = list(value[key])
        value["agent"]["command"] = list(value["agent"]["command"])
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CampaignConfig":
        return cls(
            name=str(value["name"]),
            target=str(value["target"]),
            source_files=tuple(value.get("source_files", cls.source_files)),
            editable_files=tuple(value.get("editable_files", cls.editable_files)),
            immutable_files=tuple(value.get("immutable_files", cls.immutable_files)),
            train_command=tuple(value.get("train_command", cls.train_command)),
            objective=ObjectiveConfig(**value.get("objective", {})),
            search=SearchConfig(**value.get("search", {})),
            agent=AgentConfig(
                command=tuple(value.get("agent", {}).get("command", AgentConfig.command)),
                timeout_seconds=float(value.get("agent", {}).get("timeout_seconds", 900.0)),
            ),
        )


def load_config(campaign_root: Path) -> CampaignConfig:
    return CampaignConfig.from_dict(read_json(campaign_root / "config.json"))


def save_config(campaign_root: Path, config: CampaignConfig) -> None:
    atomic_write_json(campaign_root / "config.json", config.to_dict())


def parse_command(command: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(shlex.split(command)) if command else default


def detect_train_command(target: Path) -> tuple[str, ...]:
    if (target / "uv.lock").is_file():
        return CampaignConfig.train_command
    return (
        "{python}",
        "{audit_runner}",
        "--target",
        "{trial}",
    )


def extract_time_budget(prepare_path: Path) -> float:
    tree = ast.parse(prepare_path.read_text(encoding="utf-8"), filename=str(prepare_path))
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == "TIME_BUDGET" for target in targets):
                value = node.value
                if isinstance(value, ast.Constant) and isinstance(value.value, (int, float)):
                    return float(value.value)
    raise ValueError(f"{prepare_path} must define TIME_BUDGET as a numeric constant")


def validate_config(config: CampaignConfig, source_root: Path) -> None:
    if config.objective.metric != "val_bpb" or config.objective.direction != "minimize":
        raise ValueError("Fengheguai has exactly one objective: minimize val_bpb")
    if config.objective.training_seconds_limit != 300.0:
        raise ValueError("The training budget is locked to exactly 300 seconds")
    if config.objective.confirmation_runs < 0:
        raise ValueError("confirmation_runs cannot be negative")
    if config.objective.minimum_improvement < 0:
        raise ValueError("minimum_improvement cannot be negative")
    if not 0 <= config.search.exploration_fraction <= 1:
        raise ValueError("exploration_fraction must be between 0 and 1")
    if config.search.elite_size < 1:
        raise ValueError("elite_size must be positive")
    if "train.py" not in config.editable_files or len(config.editable_files) != 1:
        raise ValueError("train.py must be the only editable source file")
    if "prepare.py" not in config.immutable_files:
        raise ValueError("prepare.py must be immutable")
    if "{audit_runner}" not in config.train_command:
        raise ValueError("train_command must invoke the trusted {audit_runner}")
    budget = extract_time_budget(source_root / "prepare.py")
    if budget != 300.0:
        raise ValueError(f"prepare.py TIME_BUDGET is {budget}, expected exactly 300")
    for relative in config.source_files:
        if not (source_root / relative).is_file():
            raise FileNotFoundError(source_root / relative)


def default_config(
    *,
    name: str,
    target: Path,
    agent_command: str | None = None,
    train_command: str | None = None,
    confirmation_runs: int = 1,
    minimum_improvement: float = 0.0,
) -> CampaignConfig:
    if confirmation_runs < 0:
        raise ValueError("confirmation_runs cannot be negative")
    if minimum_improvement < 0:
        raise ValueError("minimum_improvement cannot be negative")
    return CampaignConfig(
        name=name,
        target=str(target.resolve()),
        train_command=parse_command(train_command, detect_train_command(target)),
        objective=ObjectiveConfig(
            confirmation_runs=confirmation_runs,
            minimum_improvement=minimum_improvement,
        ),
        agent=AgentConfig(command=parse_command(agent_command, AgentConfig.command)),
    )


TOKEN_VALUES = ("trial", "audit_runner", "agent_schema", "agent_result", "python", "campaign")
PYTHON_EXECUTABLE = sys.executable
