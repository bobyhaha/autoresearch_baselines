"""Trial execution: run one candidate `train.py` and parse its result.

This replaces AI-Scientist-v2's `interpreter.py`. AI-Scientist-v2 executes an
LLM-authored standalone script in a REPL-ish child process and then asks an LLM to
read the stdout and guess what the metric was. The autoresearch task has a much
stronger contract than that: `train.py` always ends with a fixed summary block, and
`val_bpb` from `prepare.evaluate_bpb` is the ground-truth metric. So the metric is
parsed deterministically here rather than by an LLM — no parse step can hallucinate
a score, and a run that fails to print `val_bpb` is unambiguously buggy.
"""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

# The keys train.py prints in its final summary block.
SUMMARY_KEYS = (
    "val_bpb",
    "training_seconds",
    "total_seconds",
    "peak_vram_mb",
    "mfu_percent",
    "total_tokens_M",
    "num_steps",
    "num_params_M",
    "depth",
)

# Files a trial workspace needs beside its own train.py. prepare.py is copied rather
# than symlinked so a trial can never mutate the read-only original.
SUPPORT_FILES = ("prepare.py", "pyproject.toml", "uv.lock")


@dataclass
class ExecutionResult:
    """The outcome of executing one candidate train.py."""

    term_out: str = ""
    exec_time: float = 0.0
    exit_code: int | None = None
    exc_type: str | None = None
    summary: dict = field(default_factory=dict)

    @property
    def val_bpb(self) -> float | None:
        return self.summary.get("val_bpb")


def parse_run_log(text: str) -> tuple[dict, int | None, str | None]:
    """Extract the summary block, exit code, and exception type from a run log."""
    summary: dict = {}
    for key in SUMMARY_KEYS:
        # Anchored to line start: the training progress line must never match.
        m = re.search(rf"^{re.escape(key)}:\s+([-\d.eE+]+)\s*$", text, re.MULTILINE)
        if m:
            try:
                summary[key] = float(m.group(1))
            except ValueError:
                pass

    exit_code = None
    m = re.search(r"^EXIT_CODE=(-?\d+)\s*$", text, re.MULTILINE)
    if m:
        exit_code = int(m.group(1))

    exc_type = None
    # Last exception line in the traceback is the one that actually killed the run.
    exc_matches = re.findall(r"^([A-Za-z_][\w.]*(?:Error|Exception|Interrupt)):", text, re.MULTILINE)
    if exc_matches:
        exc_type = exc_matches[-1]
    elif exit_code == 137:
        exc_type = "TimeoutOrKilled"
    elif exit_code not in (0, None):
        exc_type = f"NonZeroExit({exit_code})"

    return summary, exit_code, exc_type


def extract_traceback(text: str, max_chars: int = 4000) -> str:
    """Pull the last Python traceback out of a run log, for the debug prompt."""
    idx = text.rfind("Traceback (most recent call last)")
    if idx == -1:
        return text[-max_chars:]
    return text[idx : idx + max_chars]


class TrialRunner:
    """Materializes a candidate into its own workspace and runs it on the pinned GPU."""

    def __init__(
        self,
        base_dir: str | Path,
        task_dir: str | Path,
        gpu: int = 2,
        hard_timeout: int = 900,
        launcher: str | Path | None = None,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.task_dir = Path(task_dir)
        self.gpu = gpu
        self.hard_timeout = hard_timeout
        self.launcher = Path(launcher) if launcher else self.base_dir / "run_trial.sh"
        self.trials_dir = self.base_dir / "trials"
        self.trials_dir.mkdir(parents=True, exist_ok=True)

    def prepare_workspace(self, node_id: str, code: str) -> Path:
        workdir = self.trials_dir / node_id
        workdir.mkdir(parents=True, exist_ok=True)
        for name in SUPPORT_FILES:
            dst = workdir / name
            if not dst.exists():
                dst.write_bytes((self.task_dir / name).read_bytes())
        venv_link = workdir / ".venv"
        if not venv_link.exists():
            venv_link.symlink_to(self.task_dir / ".venv")
        (workdir / "train.py").write_text(code, encoding="utf-8")
        return workdir

    def run(self, node_id: str, code: str) -> tuple[ExecutionResult, Path]:
        workdir = self.prepare_workspace(node_id, code)
        log_path = workdir / "run.log"
        if log_path.exists():
            log_path.unlink()

        t0 = time.time()
        subprocess.run(
            [str(self.launcher), str(workdir), str(self.gpu), str(self.hard_timeout)],
            check=False,
            # The launcher already redirects the child's streams into run.log.
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        exec_time = time.time() - t0

        text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        summary, exit_code, exc_type = parse_run_log(text)
        return (
            ExecutionResult(
                term_out=text,
                exec_time=exec_time,
                exit_code=exit_code,
                exc_type=exc_type,
                summary=summary,
            ),
            workdir,
        )
