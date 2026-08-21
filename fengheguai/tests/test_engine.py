from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from fengheguai.audit import audit_campaign
from fengheguai.campaign import (
    MAX_CONSECUTIVE_AGENT_ERRORS,
    Campaign,
    initialize_campaign,
)
from fengheguai.config import AgentConfig, CampaignConfig, ObjectiveConfig
from fengheguai.evaluator import (
    Evaluator,
    audit_candidate_source,
    build_training_contract,
    file_manifest,
)
from fengheguai.ledger import Ledger


PREPARE = """\
TIME_BUDGET = 300

def evaluate_bpb(model, tokenizer, batch_size):
    return model["score"]
"""

TRAIN = """\
from prepare import TIME_BUDGET, evaluate_bpb

SCORE = 2.0
model = {"score": SCORE}
tokenizer = None
DEVICE_BATCH_SIZE = 1
val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)
print("val_bpb:          0.000001")
print("training_seconds: 300.0")
print("peak_vram_mb:     1234.0")
print("num_steps:        42")
"""

MOCK_AGENT = """\
import json
import sys
from pathlib import Path

trial = Path(sys.argv[1])
result = Path(sys.argv[2])
train = trial / "train.py"
train.write_text(train.read_text().replace("SCORE = 2.0", "SCORE = 1.5"))
result.write_text(json.dumps({
    "hypothesis": "lower the synthetic score",
    "change_summary": "changed the synthetic model score",
    "expected_val_bpb_effect": "decrease",
    "risk": "none",
}))
"""

TIMED_TRAIN = """\
import time
import torch
from prepare import TIME_BUDGET, evaluate_bpb

total_training_time = 0
step = 0
while True:
    torch.cuda.synchronize()
    t0 = time.time()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0
    if step > 10:
        total_training_time += dt
    step += 1
    if step > 10 and total_training_time >= TIME_BUDGET:
        break
model = {"score": 2.0}
val_bpb = evaluate_bpb(model, None, 1)
print(f"training_seconds: {total_training_time:.1f}")
"""


STUB_TORCH = """\
_ACTIVE = {}


class _Autocast:
    def __init__(self, device_type, dtype=None, enabled=True):
        self.device_type = device_type
        self.dtype = dtype
        self.enabled = enabled

    def __enter__(self):
        if self.enabled:
            _ACTIVE[self.device_type] = self.dtype
        return self

    def __exit__(self, *exc):
        if self.enabled:
            _ACTIVE.pop(self.device_type, None)
        return False


def autocast(device_type, dtype=None, enabled=True):
    return _Autocast(device_type, dtype, enabled)


def is_autocast_enabled(device_type):
    return device_type in _ACTIVE


def get_autocast_dtype(device_type):
    return _ACTIVE.get(device_type, "float32")


class amp:
    autocast = staticmethod(autocast)


bfloat16 = "bfloat16"
"""

# The candidate calls evaluate_bpb inside an autocast block, exactly as the pristine
# nanoGPT target does. The locked evaluation is deferred until after the script exits,
# so it only agrees with the candidate's own call site if that context is replayed.
AUTOCAST_PREPARE = """\
import torch

TIME_BUDGET = 300


def evaluate_bpb(model, tokenizer, batch_size):
    if not torch.is_autocast_enabled("cuda"):
        raise RuntimeError("expected mat1 and mat2 to have the same dtype")
    return model["score"]
"""

AUTOCAST_TRAIN = """\
import torch
from prepare import TIME_BUDGET, evaluate_bpb

model = {"score": 2.0}
with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
    val_bpb = evaluate_bpb(model, None, 1)
print("training_seconds: 300.0")
"""


def make_target(root: Path) -> Path:
    target = root / "target"
    target.mkdir()
    (target / "prepare.py").write_text(PREPARE, encoding="utf-8")
    (target / "train.py").write_text(TRAIN, encoding="utf-8")
    return target


def make_config(target: Path, agent_script: Path | None = None) -> CampaignConfig:
    command = (
        sys.executable,
        str(agent_script),
        "{trial}",
        "{agent_result}",
    ) if agent_script else (sys.executable, "-c", "raise SystemExit(1)")
    return CampaignConfig(
        name="test",
        target=str(target),
        source_files=("train.py", "prepare.py"),
        editable_files=("train.py",),
        immutable_files=("prepare.py",),
        train_command=("{python}", "{audit_runner}", "--target", "{trial}"),
        objective=ObjectiveConfig(confirmation_runs=0),
        agent=AgentConfig(command=command, timeout_seconds=10),
    )


class EvaluatorTests(unittest.TestCase):
    def test_locked_evaluator_ignores_printed_fake_score(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = make_target(root)
            campaign_root = root / "campaign"
            campaign_root.mkdir()
            manifest = file_manifest(target, ("prepare.py",))
            evaluator = Evaluator(
                campaign_root=campaign_root,
                config=make_config(target),
                immutable_manifest=manifest,
            )
            result = evaluator.evaluate(target, run_label="locked")
            self.assertTrue(result.valid, result.errors)
            self.assertEqual(result.metric, 2.0)
            self.assertEqual(result.training_seconds, 300.0)

    def test_locked_evaluation_replays_candidate_autocast_context(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = make_target(root)
            (target / "torch.py").write_text(STUB_TORCH, encoding="utf-8")
            (target / "prepare.py").write_text(AUTOCAST_PREPARE, encoding="utf-8")
            (target / "train.py").write_text(AUTOCAST_TRAIN, encoding="utf-8")
            campaign_root = root / "campaign"
            campaign_root.mkdir()
            evaluator = Evaluator(
                campaign_root=campaign_root,
                config=make_config(target),
                immutable_manifest=file_manifest(target, ("prepare.py",)),
            )
            result = evaluator.evaluate(target, run_label="autocast")
            self.assertTrue(result.valid, result.errors)
            self.assertEqual(result.metric, 2.0)

    def test_source_audit_rejects_budget_assignment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = make_target(Path(temporary))
            manifest = file_manifest(target, ("prepare.py",))
            (target / "train.py").write_text(TRAIN + "\nTIME_BUDGET = 1\n", encoding="utf-8")
            errors, _ = audit_candidate_source(target, manifest)
            self.assertTrue(any("TIME_BUDGET" in error for error in errors))

    def test_source_audit_locks_seed_timing_structure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = make_target(Path(temporary))
            (target / "train.py").write_text(TIMED_TRAIN, encoding="utf-8")
            manifest = file_manifest(target, ("prepare.py",))
            contract = build_training_contract(target / "train.py")
            self.assertEqual(contract["mode"], "protected_ast")
            (target / "train.py").write_text(
                TIMED_TRAIN.replace("total_training_time += dt", "total_training_time += dt / 2"),
                encoding="utf-8",
            )
            errors, _ = audit_candidate_source(target, manifest, contract)
            self.assertTrue(any("timing structure" in error for error in errors))


    def test_run_aborts_on_systemic_agent_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = make_target(root)
            # make_config's default agent command exits non-zero, standing in for a
            # broken CLI invocation such as an unsupported flag.
            campaign = initialize_campaign(root / "campaign", make_config(target))
            campaign.baseline()
            with self.assertRaises(RuntimeError) as caught:
                campaign.run(50)
            self.assertIn("failed", str(caught.exception))
            statuses = [record.get("status") for record in campaign.records()]
            agent_errors = [status for status in statuses if status == "agent_error"]
            self.assertEqual(len(agent_errors), MAX_CONSECUTIVE_AGENT_ERRORS)


class LedgerTests(unittest.TestCase):
    def test_hash_chain_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ledger.jsonl"
            ledger = Ledger(path)
            ledger.append("one", {"value": 1})
            ledger.append("two", {"value": 2})
            lines = path.read_text(encoding="utf-8").splitlines()
            event = json.loads(lines[0])
            event["payload"]["value"] = 99
            lines[0] = json.dumps(event)
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                ledger.read()


class CampaignTests(unittest.TestCase):
    def test_end_to_end_promotes_verified_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = make_target(root)
            agent_script = root / "mock_agent.py"
            agent_script.write_text(MOCK_AGENT, encoding="utf-8")
            campaign_root = root / "campaign"
            campaign = initialize_campaign(campaign_root, make_config(target, agent_script))

            baseline = campaign.baseline()
            self.assertEqual(baseline["metric"], 2.0)
            candidate = campaign.step()
            self.assertEqual(candidate["status"], "keep")
            self.assertTrue(candidate["promoted"])
            self.assertEqual(candidate["metric"], 1.5)

            report = audit_campaign(campaign_root)
            self.assertTrue(report.ok, report.errors)
            self.assertEqual(report.champion_id, "t0001")
            self.assertEqual(Campaign(campaign_root).status()["val_bpb"], 1.5)


if __name__ == "__main__":
    unittest.main()


class ClaudeAgentAdapterTests(unittest.TestCase):
    """The Claude Code adapter must reconcile the structured result itself."""

    def _stub(self, root: Path, body: str) -> Path:
        stub = root / "stub-claude"
        stub.write_text("#!/bin/bash\ncat > /dev/null\n" + body + "\n", encoding="utf-8")
        stub.chmod(0o755)
        return stub

    def _run(self, root: Path, stub: Path) -> tuple[int, Path]:
        result_path = root / "result.json"
        trial = root / "trial"
        trial.mkdir(exist_ok=True)
        adapter = Path(__file__).resolve().parents[1] / "fengheguai" / "claude_agent.py"
        completed = subprocess.run(
            [
                sys.executable,
                str(adapter),
                "--trial",
                str(trial),
                "--result",
                str(result_path),
                "--claude",
                str(stub),
            ],
            input="prompt",
            text=True,
            capture_output=True,
        )
        return completed.returncode, result_path

    def test_structured_result_is_recovered_from_final_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = json.dumps(
                {
                    "hypothesis": "h",
                    "change_summary": "c",
                    "expected_val_bpb_effect": "e",
                    "risk": "r",
                }
            )
            envelope = json.dumps({"type": "result", "result": "done\n\n" + payload})
            stub = self._stub(root, "cat <<'JSON'\n" + envelope + "\nJSON")
            code, result_path = self._run(root, stub)
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(result_path.read_text())["hypothesis"], "h")

    def test_missing_structured_result_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stub = self._stub(root, "echo 'no json here'")
            code, result_path = self._run(root, stub)
            self.assertNotEqual(code, 0)
            self.assertFalse(result_path.exists())


class RendezvousAgentAdapterTests(unittest.TestCase):
    """The operator-in-the-loop adapter must block until a valid patch lands."""

    def _spawn(self, root: Path, timeout: float) -> tuple[subprocess.Popen[str], Path, Path]:
        trial = root / "nodes" / "t0001"
        trial.mkdir(parents=True, exist_ok=True)
        inbox = root / "rendezvous"
        result_path = trial / "result.json"
        adapter = Path(__file__).resolve().parents[1] / "fengheguai" / "rendezvous_agent.py"
        process = subprocess.Popen(
            [
                sys.executable,
                str(adapter),
                "--trial",
                str(trial),
                "--result",
                str(result_path),
                "--inbox",
                str(inbox),
                "--timeout",
                str(timeout),
                "--poll",
                "0.05",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        process.stdin.write("research prompt")
        process.stdin.close()
        return process, inbox, result_path

    def _await_request(self, inbox: Path) -> Path:
        request = inbox / "t0001.request.json"
        for _ in range(200):
            if request.is_file():
                return request
            time.sleep(0.05)
        self.fail("adapter never published its request")

    def test_publishes_prompt_and_returns_on_valid_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            process, inbox, result_path = self._spawn(root, timeout=30)
            request = self._await_request(inbox)
            self.assertEqual((inbox / "t0001.prompt.md").read_text(), "research prompt")
            self.assertTrue((inbox / "PENDING.json").is_file())
            self.assertEqual(json.loads(request.read_text())["trial_id"], "t0001")

            result_path.write_text(
                json.dumps(
                    {
                        "hypothesis": "h",
                        "change_summary": "c",
                        "expected_val_bpb_effect": "e",
                        "risk": "r",
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(process.wait(timeout=30), 0)
            self.assertFalse((inbox / "PENDING.json").is_file())
            self.assertEqual(json.loads(request.read_text())["status"], "answered")

    def test_incomplete_result_does_not_satisfy_the_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            process, inbox, result_path = self._spawn(root, timeout=3)
            self._await_request(inbox)
            result_path.write_text(json.dumps({"hypothesis": "h"}), encoding="utf-8")
            self.assertEqual(process.wait(timeout=30), 1)
