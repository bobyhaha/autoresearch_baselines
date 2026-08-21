from __future__ import annotations

import difflib
import statistics
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .agent import AgentRunner, build_prompt, write_agent_schema
from .config import CampaignConfig, load_config, save_config, validate_config
from .evaluator import (
    Evaluator,
    EvaluationResult,
    audit_candidate_source,
    build_training_contract,
    file_manifest,
)
from .ledger import Ledger, started_trial_ids, trial_records
from .policy import SearchDecision, champion, choose_search_decision
from .reporting import render_reports, status_payload
from .util import (
    atomic_write_json,
    copy_snapshot,
    read_json,
    sha256_file,
    atomic_write_text,
    utc_now,
)


@contextmanager
def campaign_lock(campaign_root: Path) -> Iterator[None]:
    lock_path = campaign_root / ".campaign.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another Fengheguai process is already using this campaign") from exc
        except ImportError:  # pragma: no cover
            pass
        try:
            yield
        finally:
            try:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except ImportError:  # pragma: no cover
                pass


# A healthy campaign never fails this many agent invocations back to back. A run of
# them means the agent command itself is broken (wrong CLI flags, missing auth), which
# no amount of further searching can fix; each attempt would burn another trial id and
# append another empty record to the ledger.
MAX_CONSECUTIVE_AGENT_ERRORS = 5


def _make_immutable(root: Path, immutable_files: tuple[str, ...]) -> None:
    for relative in immutable_files:
        path = root / relative
        path.chmod(path.stat().st_mode & ~0o222)


def initialize_campaign(campaign_root: Path, config: CampaignConfig) -> "Campaign":
    campaign_root = campaign_root.resolve()
    if campaign_root.exists() and any(campaign_root.iterdir()):
        raise FileExistsError(f"Campaign directory is not empty: {campaign_root}")
    campaign_root.mkdir(parents=True, exist_ok=True)
    source = Path(config.target).resolve()
    validate_config(config, source)

    for directory in ("control", "nodes", "evidence", "artifacts", "reports"):
        (campaign_root / directory).mkdir(parents=True, exist_ok=True)
    seed = campaign_root / "seed"
    copy_snapshot(source, seed, config.source_files)
    _make_immutable(seed, config.immutable_files)
    manifest = file_manifest(seed, config.immutable_files)
    atomic_write_json(campaign_root / "control" / "immutable_manifest.json", manifest)
    training_contract = build_training_contract(seed / "train.py")
    training_contract_path = campaign_root / "control" / "training_contract.json"
    atomic_write_json(training_contract_path, training_contract)
    packaged_program = Path(__file__).resolve().parent / "program.md"
    if not packaged_program.is_file():
        packaged_program = Path(__file__).resolve().parent.parent / "program.md"
    program_path = campaign_root / "control" / "program.md"
    atomic_write_text(program_path, packaged_program.read_text(encoding="utf-8"))
    save_config(campaign_root, config)
    write_agent_schema(campaign_root)
    ledger = Ledger(campaign_root / "ledger.jsonl")
    ledger.append(
        "campaign_initialized",
        {
            "name": config.name,
            "created_at": utc_now(),
            "objective": "minimize val_bpb within 300 seconds",
            "source_target": str(source),
            "source_files": list(config.source_files),
            "immutable_manifest": manifest,
            "seed_train_sha256": sha256_file(seed / "train.py"),
            "program_sha256": sha256_file(program_path),
            "config_sha256": sha256_file(campaign_root / "config.json"),
            "training_contract_sha256": sha256_file(training_contract_path),
        },
    )
    render_reports(campaign_root, [])
    return Campaign(campaign_root)


class Campaign:
    def __init__(self, campaign_root: Path) -> None:
        self.root = campaign_root.resolve()
        if not (self.root / "config.json").is_file():
            raise FileNotFoundError(f"Not a Fengheguai campaign: {self.root}")
        self.config = load_config(self.root)
        validate_config(self.config, self.root / "seed")
        self.manifest = read_json(self.root / "control" / "immutable_manifest.json")
        self.training_contract = read_json(self.root / "control" / "training_contract.json")
        self.ledger = Ledger(self.root / "ledger.jsonl")
        self.evaluator = Evaluator(
            campaign_root=self.root,
            config=self.config,
            immutable_manifest=self.manifest,
            training_contract=self.training_contract,
        )

    def records(self) -> list[dict[str, Any]]:
        return trial_records(self.ledger.read())

    def status(self) -> dict[str, Any]:
        return status_payload(self.records())

    def _workspace_for(self, trial_id: str) -> Path:
        return self.root / "nodes" / trial_id

    def _record_by_id(self, records: list[dict[str, Any]], trial_id: str) -> dict[str, Any]:
        # A retried node appends a later record under the same id; the newest one wins.
        for record in reversed(records):
            if record.get("trial_id") == trial_id:
                return record
        raise KeyError(f"Unknown parent node: {trial_id}")

    def _next_run_offset(self, trial_id: str) -> int:
        evidence = self.root / "evidence"
        if not evidence.is_dir():
            return 0
        prefix = f"{trial_id}-r"
        offset = 0
        for path in evidence.iterdir():
            if path.name.startswith(prefix):
                suffix = path.name[len(prefix):]
                if suffix.isdigit():
                    offset = max(offset, int(suffix) + 1)
        return offset

    def _next_trial_id(self) -> str:
        used = started_trial_ids(self.ledger.read())
        index = 1
        while f"t{index:04d}" in used:
            index += 1
        return f"t{index:04d}"

    def _materialize(self, source: Path, destination: Path) -> None:
        copy_snapshot(source, destination, self.config.source_files)
        _make_immutable(destination, self.config.immutable_files)

    @staticmethod
    def _aggregate(results: list[EvaluationResult]) -> tuple[float, float]:
        metrics = [float(result.metric) for result in results if result.metric is not None]
        seconds = [
            float(result.training_seconds)
            for result in results
            if result.training_seconds is not None
        ]
        return statistics.median(metrics), statistics.median(seconds)

    def _measurements_payload(self, results: list[EvaluationResult]) -> list[dict[str, Any]]:
        payload = []
        for result in results:
            value = result.to_dict()
            evidence_path = Path(str(result.evidence_path))
            value["evidence_sha256"] = (
                sha256_file(evidence_path) if evidence_path.is_file() else None
            )
            payload.append(value)
        return payload

    def _write_patch(self, parent_workspace: Path, workspace: Path, trial_id: str) -> dict[str, str | None]:
        parent_lines = (parent_workspace / "train.py").read_text(encoding="utf-8").splitlines()
        candidate_lines = (workspace / "train.py").read_text(encoding="utf-8").splitlines()
        patch = "\n".join(
            difflib.unified_diff(
                parent_lines,
                candidate_lines,
                fromfile=f"{parent_workspace.name}/train.py",
                tofile=f"{trial_id}/train.py",
                lineterm="",
            )
        )
        if patch:
            patch += "\n"
        path = self.root / "artifacts" / trial_id / "change.patch"
        atomic_write_text(path, patch)
        return {"patch_path": str(path), "patch_sha256": sha256_file(path)}

    def baseline(self) -> dict[str, Any]:
        records = self.records()
        trial_id = "b0000"
        scored = next(
            (
                record
                for record in records
                if record.get("trial_id") == trial_id and record.get("metric") is not None
            ),
            None,
        )
        if scored:
            return scored
        workspace = self._workspace_for(trial_id)
        # A baseline that failed for an environmental reason is retried against the same
        # untouched seed snapshot. Earlier attempts keep their ledger events and evidence;
        # this attempt appends its own under labels that cannot collide with them.
        if not workspace.exists():
            self._materialize(self.root / "seed", workspace)
        self.ledger.append(
            "trial_started",
            {"trial_id": trial_id, "stage": "baseline", "parent_id": None},
        )
        runs = 1 + self.config.objective.confirmation_runs
        offset = self._next_run_offset(trial_id)
        results = [
            self.evaluator.evaluate(workspace, run_label=f"{trial_id}-r{offset + index}")
            for index in range(runs)
        ]
        if not all(result.valid for result in results):
            record = {
                "trial_id": trial_id,
                "parent_id": None,
                "secondary_parent_id": None,
                "stage": "baseline",
                "status": "failed",
                "promoted": False,
                "metric": None,
                "training_seconds": None,
                "workspace": str(workspace),
                "measurements": self._measurements_payload(results),
                "proposal": {"change_summary": "unaltered baseline"},
                "finding": "The locked baseline did not complete; candidate search is blocked.",
                "source_sha256": sha256_file(workspace / "train.py"),
                "debug_depth": 0,
            }
        else:
            metric, seconds = self._aggregate(results)
            record = {
                "trial_id": trial_id,
                "parent_id": None,
                "secondary_parent_id": None,
                "stage": "baseline",
                "status": "baseline",
                "promoted": True,
                "metric": metric,
                "training_seconds": seconds,
                "delta_vs_parent": None,
                "delta_vs_champion_before": None,
                "workspace": str(workspace),
                "measurements": self._measurements_payload(results),
                "proposal": {"change_summary": "unaltered baseline"},
                "finding": f"Locked baseline established at val_bpb={metric:.6f}.",
                "source_sha256": sha256_file(workspace / "train.py"),
                "debug_depth": 0,
            }
        self.ledger.append("trial_completed", record)
        render_reports(self.root, self.records())
        if record["metric"] is None:
            errors = [
                error
                for measurement in record["measurements"]
                for error in measurement.get("errors", [])
            ]
            raise RuntimeError("Baseline failed: " + "; ".join(errors))
        return record

    def plan_next(self) -> tuple[SearchDecision, dict[str, Any], dict[str, Any] | None]:
        records = self.records()
        if not any(record.get("trial_id") == "b0000" and record.get("metric") for record in records):
            raise RuntimeError("Run a valid baseline before planning candidates")
        decision = choose_search_decision(records, self.config.search)
        parent = self._record_by_id(records, decision.parent_id)
        secondary = (
            self._record_by_id(records, decision.secondary_parent_id)
            if decision.secondary_parent_id
            else None
        )
        return decision, parent, secondary

    def step(self) -> dict[str, Any]:
        records_before = self.records()
        if not records_before:
            self.baseline()
            records_before = self.records()
        decision, parent, secondary = self.plan_next()
        trial_id = self._next_trial_id()
        workspace = self._workspace_for(trial_id)
        self._materialize(Path(parent["workspace"]), workspace)
        debug_depth = int(parent.get("debug_depth", 0)) + 1 if decision.stage == "debug" else 0
        self.ledger.append(
            "trial_started",
            {
                "trial_id": trial_id,
                "stage": decision.stage,
                "parent_id": parent["trial_id"],
                "secondary_parent_id": decision.secondary_parent_id,
                "reason": decision.reason,
            },
        )

        prompt = build_prompt(
            program=(self.root / "control" / "program.md").read_text(encoding="utf-8"),
            decision=decision,
            parent=parent,
            secondary_parent=secondary,
            records=records_before,
            memory_items=self.config.search.memory_items,
        )
        agent_result = AgentRunner(self.root, self.config).run(workspace, trial_id, prompt)
        patch_meta = self._write_patch(Path(parent["workspace"]), workspace, trial_id)
        if not agent_result.success:
            record = self._failed_record(
                trial_id=trial_id,
                decision=decision,
                workspace=workspace,
                status="agent_error",
                proposal=agent_result.to_dict(),
                errors=agent_result.errors,
                debug_depth=debug_depth,
                patch_meta=patch_meta,
            )
            return self._complete(record)

        source_errors, source_hash = audit_candidate_source(
            workspace, self.manifest, self.training_contract
        )
        known_hashes = {record.get("source_sha256") for record in records_before}
        if source_errors:
            record = self._failed_record(
                trial_id=trial_id,
                decision=decision,
                workspace=workspace,
                status="rejected",
                proposal=agent_result.to_dict(),
                errors=source_errors,
                debug_depth=debug_depth,
                source_hash=source_hash,
                patch_meta=patch_meta,
            )
            return self._complete(record)
        if source_hash in known_hashes:
            record = self._failed_record(
                trial_id=trial_id,
                decision=decision,
                workspace=workspace,
                status="duplicate",
                proposal=agent_result.to_dict(),
                errors=["candidate train.py duplicates an already evaluated node"],
                debug_depth=debug_depth,
                source_hash=source_hash,
                patch_meta=patch_meta,
            )
            return self._complete(record)

        current_best = champion(records_before)
        results = [self.evaluator.evaluate(workspace, run_label=f"{trial_id}-r0")]
        first = results[0]
        promising = bool(
            first.valid
            and first.metric is not None
            and float(first.metric)
            < float(current_best["metric"]) - self.config.objective.minimum_improvement
        )
        if promising:
            for index in range(1, 1 + self.config.objective.confirmation_runs):
                results.append(
                    self.evaluator.evaluate(workspace, run_label=f"{trial_id}-r{index}")
                )

        all_valid = all(result.valid for result in results)
        metric: float | None = None
        seconds: float | None = None
        promoted = False
        if all_valid:
            metric, seconds = self._aggregate(results)
            promoted = metric < (
                float(current_best["metric"]) - self.config.objective.minimum_improvement
            )
            status = "keep" if promoted else "discard"
        else:
            status = "timeout" if any(result.timed_out for result in results) else "failed"

        parent_metric = parent.get("metric")
        delta_parent = (
            metric - float(parent_metric) if metric is not None and parent_metric is not None else None
        )
        delta_best = metric - float(current_best["metric"]) if metric is not None else None
        finding = self._finding(
            trial_id=trial_id,
            status=status,
            metric=metric,
            delta_best=delta_best,
            proposal=agent_result.to_dict(),
            results=results,
        )
        record = {
            "trial_id": trial_id,
            "parent_id": decision.parent_id,
            "secondary_parent_id": decision.secondary_parent_id,
            "stage": decision.stage,
            "selection_reason": decision.reason,
            "status": status,
            "promoted": promoted,
            "metric": metric,
            "training_seconds": seconds,
            "delta_vs_parent": delta_parent,
            "delta_vs_champion_before": delta_best,
            "champion_before": current_best["trial_id"],
            "workspace": str(workspace),
            "measurements": self._measurements_payload(results),
            "proposal": agent_result.to_dict(),
            "finding": finding,
            "source_sha256": source_hash,
            "debug_depth": debug_depth,
            **patch_meta,
        }
        return self._complete(record)

    def _failed_record(
        self,
        *,
        trial_id: str,
        decision: SearchDecision,
        workspace: Path,
        status: str,
        proposal: dict[str, Any],
        errors: list[str],
        debug_depth: int,
        source_hash: str | None = None,
        patch_meta: dict[str, str | None] | None = None,
    ) -> dict[str, Any]:
        return {
            "trial_id": trial_id,
            "parent_id": decision.parent_id,
            "secondary_parent_id": decision.secondary_parent_id,
            "stage": decision.stage,
            "selection_reason": decision.reason,
            "status": status,
            "promoted": False,
            "metric": None,
            "training_seconds": None,
            "delta_vs_parent": None,
            "delta_vs_champion_before": None,
            "workspace": str(workspace),
            "measurements": [],
            "proposal": proposal,
            "finding": f"{status}: " + "; ".join(errors),
            "source_sha256": source_hash,
            "debug_depth": debug_depth,
            **(patch_meta or {}),
        }

    @staticmethod
    def _finding(
        *,
        trial_id: str,
        status: str,
        metric: float | None,
        delta_best: float | None,
        proposal: dict[str, Any],
        results: list[EvaluationResult],
    ) -> str:
        hypothesis = str(proposal.get("hypothesis") or "unspecified hypothesis")
        if metric is not None and delta_best is not None:
            direction = "improved" if delta_best < 0 else "did not improve"
            return (
                f"{trial_id} {direction} the prior champion: val_bpb={metric:.6f}, "
                f"delta={delta_best:+.6f}. Hypothesis: {hypothesis}"
            )
        errors = [error for result in results for error in result.errors]
        return f"{trial_id} ended as {status}. Hypothesis: {hypothesis}. Evidence: {'; '.join(errors)}"

    def _complete(self, record: dict[str, Any]) -> dict[str, Any]:
        self.ledger.append("trial_completed", record)
        render_reports(self.root, self.records())
        return record

    def run(self, trials: int | None = None) -> list[dict[str, Any]]:
        completed: list[dict[str, Any]] = []
        if not any(
            record.get("trial_id") == "b0000" and record.get("metric") is not None
            for record in self.records()
        ):
            completed.append(self.baseline())
        remaining = trials
        consecutive_agent_errors = 0
        while remaining is None or remaining > 0:
            record = self.step()
            completed.append(record)
            if record.get("status") == "agent_error":
                consecutive_agent_errors += 1
                if consecutive_agent_errors >= MAX_CONSECUTIVE_AGENT_ERRORS:
                    raise RuntimeError(
                        f"The research agent failed {consecutive_agent_errors} times in a row; "
                        "the agent command is likely misconfigured. Last failure: "
                        f"{record.get('finding')}"
                    )
            else:
                consecutive_agent_errors = 0
            if remaining is not None:
                remaining -= 1
        return completed
