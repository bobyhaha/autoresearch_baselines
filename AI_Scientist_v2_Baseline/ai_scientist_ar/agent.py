"""The best-first tree-search agent, adapted to the autoresearch task.

Ported from AI-Scientist-v2's `parallel_agent.py`. Two things change:

* **One worker.** The campaign owns a single GPU, so `_select_node` returns one node
  instead of N. That collapses AI-Scientist-v2's `processed_trees` bookkeeping, whose
  only job was stopping parallel workers from piling onto the same tree in one step.
  The draft/debug/improve policy itself is unchanged.
* **Deterministic evaluation.** `parse_exec_result` reads the metric out of the run
  log instead of asking an LLM to interpret stdout, and additionally *audits* the run
  (see `_audit`) so a candidate cannot score by shortening training or by touching the
  read-only evaluation harness.
"""

from __future__ import annotations

import hashlib
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .backend import RendezvousBackend
from .interpreter import ExecutionResult, TrialRunner, extract_traceback
from .journal import Journal, Node
from .metric import MetricValue, WorstMetricValue

logger = logging.getLogger(__name__)

# train.py must consume essentially the whole fixed budget. prepare.TIME_BUDGET is 300s
# and the loop stops the first step at/after that, so a healthy run lands just over it.
MIN_TRAINING_SECONDS = 295.0


@dataclass
class SearchConfig:
    """Mirrors `agent.search` in AI-Scientist-v2's bfts_config.yaml."""

    max_debug_depth: int = 3
    debug_prob: float = 0.5
    num_drafts: int = 3


@dataclass
class AgentConfig:
    search: SearchConfig = field(default_factory=SearchConfig)
    gpu: int = 2
    hard_timeout: int = 900
    metric_name: str = "val_bpb"
    maximize: bool = False


class Agent:
    def __init__(
        self,
        cfg: AgentConfig,
        journal: Journal,
        backend: RendezvousBackend,
        runner: TrialRunner,
        task_memo: str,
        baseline_code: str,
        pristine_prepare_sha: str,
    ) -> None:
        self.cfg = cfg
        self.journal = journal
        self.backend = backend
        self.runner = runner
        self.task_memo = task_memo
        self.baseline_code = baseline_code
        self.pristine_prepare_sha = pristine_prepare_sha

    # ------------------------------------------------------------------
    # Search policy (ported from `_select_parallel_nodes`, single worker)
    # ------------------------------------------------------------------

    def _select_node(self) -> Optional[Node]:
        """Pick the node to work from. None means 'draft a fresh root'."""
        search_cfg = self.cfg.search

        # Drafting phase: build up `num_drafts` independent trees first.
        if len(self.journal.draft_nodes) < search_cfg.num_drafts:
            logger.info(
                "select: drafting (%d/%d roots)",
                len(self.journal.draft_nodes),
                search_cfg.num_drafts,
            )
            return None

        # Debugging phase, entered with probability debug_prob.
        if random.random() < search_cfg.debug_prob:
            debuggable = [
                n
                for n in self.journal.buggy_nodes
                if n.is_leaf and n.debug_depth <= search_cfg.max_debug_depth
            ]
            if debuggable:
                node = random.choice(debuggable)
                logger.info("select: debug node %s (depth %d)", node.id, node.debug_depth)
                return node

        # Improvement phase: best-first.
        if not self.journal.good_nodes:
            logger.info("select: no good nodes yet, drafting")
            return None
        best = self.journal.get_best_node()
        if best is None:
            return None
        logger.info("select: improve best node %s (%s)", best.id, best.metric)
        return best

    # ------------------------------------------------------------------
    # Context handed to the coding model
    # ------------------------------------------------------------------

    def _journal_summary(self, limit: int = 40) -> list[dict]:
        """Compact history so the coding model can avoid repeating dead ends."""
        rows = []
        for n in self.journal.nodes[-limit:]:
            rows.append(
                {
                    "step": n.step,
                    "id": n.id,
                    "parent_id": n.parent.id if n.parent else None,
                    "op": n.stage_name,
                    "plan": n.plan,
                    "val_bpb": n.metric.value if n.metric else None,
                    "is_buggy": n.is_buggy,
                    "analysis": n.analysis,
                    "steps": n.summary.get("num_steps"),
                    "mfu": n.summary.get("mfu_percent"),
                    "vram_mb": n.summary.get("peak_vram_mb"),
                }
            )
        return rows

    def _build_context(self, op: str, parent: Optional[Node]) -> dict:
        best = self.journal.get_best_node()
        ctx = {
            "task_memo": self.task_memo,
            "metric": self.cfg.metric_name,
            "lower_is_better": not self.cfg.maximize,
            "history": self._journal_summary(),
            "best_so_far": (
                {"id": best.id, "val_bpb": best.metric.value, "plan": best.plan}
                if best
                else None
            ),
            "num_nodes": len(self.journal),
            "num_drafts": len(self.journal.draft_nodes),
            "target_num_drafts": self.cfg.search.num_drafts,
        }
        if parent is None:
            # A draft starts from the pristine baseline train.py.
            ctx["base_code_path"] = str(self.runner.task_dir / "train.py")
            ctx["instruction"] = (
                "DRAFT: propose a new independent research direction, starting from the "
                "pristine baseline train.py. This becomes a new root of the search forest, "
                "so it should be meaningfully different from the existing roots."
            )
        else:
            ctx["parent"] = {
                "id": parent.id,
                "plan": parent.plan,
                "val_bpb": parent.metric.value if parent.metric else None,
                "is_buggy": parent.is_buggy,
                "analysis": parent.analysis,
                "summary": parent.summary,
                "code_path": str(Path(parent.trial_dir) / "train.py") if parent.trial_dir else None,
            }
            if op == "debug":
                ctx["instruction"] = (
                    "DEBUG: the parent run failed. Fix the bug in the parent's train.py. "
                    "Change as little as possible beyond what is needed to make it run."
                )
                ctx["traceback"] = extract_traceback(parent._term_out or "")
            else:
                ctx["instruction"] = (
                    "IMPROVE: make ONE well-motivated change to the parent's train.py to "
                    "lower val_bpb. Single atomic change so the result is attributable."
                )
        return ctx

    # ------------------------------------------------------------------
    # Node generation
    # ------------------------------------------------------------------

    def _new_node(self, op: str, parent: Optional[Node]) -> Node:
        ctx = self._build_context(op, parent)
        payload = self.backend.query(op, parent.id if parent else None, ctx)
        code = payload["code"]
        plan = payload.get("plan", "").strip() or "(no plan recorded)"
        node = Node(plan=plan, code=code, parent=parent)
        logger.info("generated %s node %s from %s", op, node.id, parent.id if parent else "baseline")
        return node

    def _draft(self) -> Node:
        return self._new_node("draft", None)

    def _debug(self, parent: Node) -> Node:
        return self._new_node("debug", parent)

    def _improve(self, parent: Node) -> Node:
        return self._new_node("improve", parent)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _audit(self, workdir: Path, result: ExecutionResult) -> Optional[str]:
        """Reject results that did not honour the task's fixed rules.

        The agent may only edit train.py; prepare.py holds the read-only evaluation
        harness and the fixed 300s budget. A candidate that trims training time or
        edits the harness would post a val_bpb that is not comparable to the others,
        so such runs are treated as buggy rather than scored.
        """
        prepare = workdir / "prepare.py"
        if not prepare.exists():
            return "prepare.py missing from trial workspace"
        sha = hashlib.sha256(prepare.read_bytes()).hexdigest()
        if sha != self.pristine_prepare_sha:
            return "prepare.py was modified (read-only evaluation harness)"

        train_secs = result.summary.get("training_seconds")
        if train_secs is None:
            return "no training_seconds in summary"
        if train_secs < MIN_TRAINING_SECONDS:
            return f"training_seconds={train_secs:.1f} below the fixed {MIN_TRAINING_SECONDS:.0f}s budget"

        val = result.summary.get("val_bpb")
        if val is None or val != val or val <= 0:  # NaN-safe
            return f"invalid val_bpb: {val}"
        return None

    def parse_exec_result(self, node: Node, result: ExecutionResult, workdir: Path) -> None:
        """Attach execution outcome, metric and bugginess to the node."""
        node._term_out = result.term_out
        node.exec_time = result.exec_time
        node.exc_type = result.exc_type
        node.exit_code = result.exit_code
        node.summary = result.summary
        node.trial_dir = str(workdir)

        audit_failure = self._audit(workdir, result)
        if audit_failure is not None:
            node.is_buggy = True
            node.metric = WorstMetricValue(maximize=self.cfg.maximize, name=self.cfg.metric_name)
            node.analysis = f"INVALID: {audit_failure}" + (
                f" (exc: {result.exc_type})" if result.exc_type else ""
            )
            return

        node.is_buggy = False
        node.metric = MetricValue(
            result.summary["val_bpb"], maximize=self.cfg.maximize, name=self.cfg.metric_name
        )
        node.analysis = (
            f"val_bpb={result.summary['val_bpb']:.6f} "
            f"steps={result.summary.get('num_steps')} "
            f"mfu={result.summary.get('mfu_percent')}% "
            f"vram={result.summary.get('peak_vram_mb', 0) / 1024:.1f}GB "
            f"params={result.summary.get('num_params_M')}M"
        )

    # ------------------------------------------------------------------
    # One search iteration
    # ------------------------------------------------------------------

    def step(self) -> Node:
        parent = self._select_node()
        if parent is None:
            node = self._draft()
        elif parent.is_buggy:
            node = self._debug(parent)
        else:
            node = self._improve(parent)

        logger.info("running trial for node %s (%s)", node.id, node.stage_name)
        result, workdir = self.runner.run(node.id, node.code)
        self.parse_exec_result(node, result, workdir)
        self.journal.append(node)
        logger.info(
            "node %s -> %s | %s",
            node.id,
            "BUGGY" if node.is_buggy else f"val_bpb={node.metric.value:.6f}",
            node.analysis,
        )
        return node
