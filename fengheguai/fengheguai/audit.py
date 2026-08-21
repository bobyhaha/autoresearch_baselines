from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .campaign import Campaign
from .evaluator import audit_candidate_source
from .ledger import started_trial_ids
from .policy import champion
from .util import read_json, sha256_file


@dataclass
class AuditReport:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checked_records: int = 0
    champion_id: str | None = None
    champion_val_bpb: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "checked_records": self.checked_records,
            "champion_id": self.champion_id,
            "champion_val_bpb": self.champion_val_bpb,
        }


def _close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def audit_campaign(campaign_root: Path) -> AuditReport:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        campaign = Campaign(campaign_root)
        events = campaign.ledger.read(verify=True)
    except Exception as exc:
        return AuditReport(ok=False, errors=[str(exc)])

    records = campaign.records()
    initialized = [event for event in events if event.get("kind") == "campaign_initialized"]
    if len(initialized) != 1:
        errors.append(f"expected one campaign_initialized event, found {len(initialized)}")

    seed_errors, seed_hash = audit_candidate_source(
        campaign.root / "seed", campaign.manifest, campaign.training_contract
    )
    errors.extend(f"seed: {error}" for error in seed_errors)
    if initialized and seed_hash != initialized[0]["payload"].get("seed_train_sha256"):
        errors.append("seed train.py no longer matches the initialized hash")
    program_path = campaign.root / "control" / "program.md"
    if not program_path.is_file():
        errors.append("locked research program is missing")
    elif initialized and sha256_file(program_path) != initialized[0]["payload"].get("program_sha256"):
        errors.append("locked research program no longer matches the initialized hash")
    config_path = campaign.root / "config.json"
    if initialized and sha256_file(config_path) != initialized[0]["payload"].get("config_sha256"):
        errors.append("campaign config changed after initialization")
    training_contract_path = campaign.root / "control" / "training_contract.json"
    if not training_contract_path.is_file():
        errors.append("protected training-time contract is missing")
    elif initialized and sha256_file(training_contract_path) != initialized[0]["payload"].get(
        "training_contract_sha256"
    ):
        errors.append("protected training-time contract changed after initialization")

    completed_ids = {str(record.get("trial_id")) for record in records}
    dangling = started_trial_ids(events) - completed_ids
    if dangling:
        warnings.append("interrupted trials without completion records: " + ", ".join(sorted(dangling)))

    best_so_far: float | None = None
    for record in records:
        trial_id = str(record.get("trial_id"))
        workspace = Path(str(record.get("workspace") or "")).resolve()
        try:
            workspace.relative_to(campaign.root)
        except ValueError:
            errors.append(f"{trial_id}: workspace escapes campaign root")
            continue
        if not workspace.is_dir():
            errors.append(f"{trial_id}: workspace is missing")
            continue

        source_errors, source_hash = audit_candidate_source(
            workspace, campaign.manifest, campaign.training_contract
        )
        if record.get("status") not in {"rejected", "agent_error"}:
            errors.extend(f"{trial_id}: {error}" for error in source_errors)
        recorded_hash = record.get("source_sha256")
        if recorded_hash and recorded_hash != source_hash:
            errors.append(f"{trial_id}: train.py does not match its recorded hash")
        patch_path_value = record.get("patch_path")
        if patch_path_value:
            patch_path = Path(str(patch_path_value))
            if not patch_path.is_file():
                errors.append(f"{trial_id}: code patch evidence is missing")
            elif record.get("patch_sha256") != sha256_file(patch_path):
                errors.append(f"{trial_id}: code patch evidence hash mismatch")

        valid_metrics: list[float] = []
        valid_seconds: list[float] = []
        for index, measurement in enumerate(record.get("measurements") or []):
            label = f"{trial_id}/measurement-{index}"
            evidence_path = Path(str(measurement.get("evidence_path") or ""))
            log_path = Path(str(measurement.get("log_path") or ""))
            if not evidence_path.is_file():
                errors.append(f"{label}: evidence file is missing")
                continue
            if measurement.get("evidence_sha256") != sha256_file(evidence_path):
                errors.append(f"{label}: evidence hash mismatch")
            evidence = read_json(evidence_path)
            evidence_result = evidence.get("result") or {}
            if evidence_result.get("status") != measurement.get("status"):
                errors.append(f"{label}: evidence status differs from ledger")
            if measurement.get("log_sha256"):
                if not log_path.is_file() or sha256_file(log_path) != measurement.get("log_sha256"):
                    errors.append(f"{label}: run log hash mismatch")
            if measurement.get("status") == "valid":
                metric = measurement.get("metric")
                seconds = measurement.get("training_seconds")
                if metric is None or seconds is None:
                    errors.append(f"{label}: valid measurement lacks metric or training_seconds")
                else:
                    valid_metrics.append(float(metric))
                    valid_seconds.append(float(seconds))
                    if float(seconds) > (
                        campaign.config.objective.training_seconds_limit
                        + campaign.config.objective.training_seconds_tolerance
                    ):
                        errors.append(f"{label}: exceeds the 300-second contract")

        record_metric = record.get("metric")
        if record_metric is not None:
            if not valid_metrics:
                errors.append(f"{trial_id}: scored record has no valid measurement evidence")
            elif not _close(float(record_metric), float(statistics.median(valid_metrics))):
                errors.append(f"{trial_id}: aggregate metric differs from evidence median")
            if best_so_far is None:
                if trial_id != "b0000":
                    errors.append(f"{trial_id}: first scored node is not the baseline")
                best_so_far = float(record_metric)
            else:
                should_promote = float(record_metric) < (
                    best_so_far - campaign.config.objective.minimum_improvement
                )
                if bool(record.get("promoted")) != should_promote:
                    errors.append(f"{trial_id}: promotion decision is inconsistent with prior evidence")
                if should_promote:
                    best_so_far = float(record_metric)
        elif record.get("promoted"):
            errors.append(f"{trial_id}: unscored record cannot be promoted")

    best_record = None
    if records and any(record.get("metric") is not None for record in records):
        best_record = champion(records)
    return AuditReport(
        ok=not errors,
        errors=sorted(set(errors)),
        warnings=sorted(set(warnings)),
        checked_records=len(records),
        champion_id=str(best_record["trial_id"]) if best_record else None,
        champion_val_bpb=float(best_record["metric"]) if best_record else None,
    )
