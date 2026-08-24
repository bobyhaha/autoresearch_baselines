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
# Foreign memory on the device during a run means a co-tenant was computing alongside us.
# The budget is wall clock, so their SMs come out of our step count: two contended trials
# landed at 1109/1115 steps against a clean 1314, worth ~0.012 bpb — 96 sigma. Such a
# trial is not comparable to the rest of the campaign and must not be scored. The idle
# baseline is ~3GB of small display processes, so 10GB is an unambiguous signal.
MAX_FOREIGN_PEAK_MB = 4_000.0
# A repeat of identical source may lose at most this fraction of the established step
# count before the run is treated as environmentally compromised. Clean replicates vary by
# well under 1% (1301-1316 across 17 runs); the contaminated ones lost 24%.
MAX_STEP_DEFICIT = 0.05


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
    # A challenger must clear this to displace the incumbent. Set from the measured
    # pooled noise floor (2 sigma); 0 restores plain argmin.
    min_improvement: float = 0.0
    # Upstream runs num_seeds re-evaluations of the best node at each stage boundary.
    num_seeds: int = 0


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
        self.seed_values = [43, 137, 2024, 7, 1234]

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
        best = self.journal.get_best_node(min_improvement=self.cfg.min_improvement)
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
        best = self.journal.get_best_node(min_improvement=self.cfg.min_improvement)
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
        # A skipped trial is not a failed experiment. run_trial.sh exits 99 when no
        # clear GPU could be found, rather than running under contention and having
        # the result rejected eight minutes later; the distinction matters because a
        # skip says nothing about the candidate.
        if result.exit_code == 99:
            return ("no clear GPU available; trial skipped rather than run under "
                    "contention (not an experimental failure)")

        prepare = workdir / "prepare.py"
        if not prepare.exists():
            return "prepare.py missing from trial workspace"
        sha = hashlib.sha256(prepare.read_bytes()).hexdigest()
        if sha != self.pristine_prepare_sha:
            return "prepare.py was modified (read-only evaluation harness)"

        # Checked before the summary checks: an aborted-for-contention trial has no
        # training_seconds, and "no training_seconds" would hide the real cause.
        # Growth, not absolute level. A tenant already holding memory at launch and not
        # computing costs us nothing; one that ARRIVES mid-run takes SMs and depresses the
        # step count. Falls back to the absolute peak when growth was not recorded.
        growth = result.summary.get("foreign_growth_mb")
        foreign = growth if growth is not None else result.summary.get("foreign_peak_mb")
        if foreign is not None and foreign > MAX_FOREIGN_PEAK_MB:
            return (f"GPU contended: foreign memory grew {foreign/1024:.1f}GB during the run; "
                    f"step count is depressed by another tenant, score not comparable")

        # Repeated-configuration throughput check.
        #
        # Byte-identical source must produce the same step count on a healthy device. When
        # it does not, the environment is at fault, not the candidate. This catches what
        # the memory-based detector structurally cannot: a device that is slow for reasons
        # invisible in `nvidia-smi` memory — another tenant computing in a small footprint,
        # a lower clock, thermal throttling.
        #
        # Observed: after the gate began auto-switching devices, two replicates of the
        # incumbent landed on GPU 1 at ~1005 steps against ~1310 elsewhere — a 24% deficit
        # with foreign memory at the idle baseline and zero growth. They inflated the
        # incumbent's replicate sd from 0.00033 to 0.0044 and moved the reported
        # improvement by 0.11 points before being caught.
        code_path = workdir / "train.py"
        if code_path.exists():
            h = hashlib.sha256(code_path.read_bytes()).hexdigest()
            prior = [
                n.summary.get("num_steps")
                for n in self.journal.nodes
                if not n.is_buggy
                and n.summary.get("num_steps")
                and hashlib.sha256((n.code or "").encode()).hexdigest() == h
            ]
            steps = result.summary.get("num_steps")
            if len(prior) >= 3 and steps:
                prior_sorted = sorted(prior)
                median = prior_sorted[len(prior_sorted) // 2]
                if median and steps < median * (1 - MAX_STEP_DEFICIT):
                    return (
                        f"throughput anomaly: {int(steps)} steps against a median of "
                        f"{int(median)} for {len(prior)} runs of identical source "
                        f"({100*(1-steps/median):.0f}% deficit) — the device, not the candidate"
                    )

        train_secs = result.summary.get("training_seconds")
        if train_secs is None:
            return "no training_seconds in summary"
        if train_secs < MIN_TRAINING_SECONDS:
            return f"training_seconds={train_secs:.1f} below the fixed {MIN_TRAINING_SECONDS:.0f}s budget"

        val = result.summary.get("val_bpb")
        if val is None or val != val or val <= 0:  # NaN-safe
            return f"invalid val_bpb: {val}"

        # The evaluation path itself must be canonical. Hashing prepare.py proves the
        # harness is unmodified, but says nothing about how train.py *calls* it: a
        # candidate could train on the validation shard, or evaluate on a different
        # split, and still hash clean and consume the full budget. That would post a
        # spectacular val_bpb that means nothing. Reviewed 95 trials and none did this,
        # so this closes a hole rather than catching an offender.
        code = (workdir / "train.py").read_text(encoding="utf-8", errors="replace")
        if 'make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")' not in code:
            return "training dataloader is not the canonical train split"
        if code.count("make_dataloader(") != 1:
            return f"unexpected make_dataloader call count: {code.count('make_dataloader(')}"
        if "evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)" not in code:
            return "evaluate_bpb is not called canonically"
        if code.count("evaluate_bpb(") != 1:
            return f"unexpected evaluate_bpb call count: {code.count('evaluate_bpb(')}"
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
            # Tag infrastructure failures so the debug branch skips them.
            node.is_environmental = any(
                k in audit_failure
                for k in ("contended", "OutOfMemory", "skipped", "no clear GPU")
            ) or (result.exc_type or "").startswith("torch.OutOfMemory")
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

    def run_seed_eval(self, node: Node) -> list[Node]:
        """Re-evaluate a node at several seeds — the port of upstream multi-seed eval.

        AI-Scientist-v2 runs `multi_seed_eval.num_seeds` copies of the best node at each
        stage boundary (`_run_multi_seed_evaluation`, called from `agent_manager`). This
        port omitted it, and then spent the campaign rediscovering why it exists.

        Two adaptations were required rather than a straight copy:

        1. **Injection point.** Upstream *prepends* `torch.manual_seed(seed)` to the node
           source. `train.py` sets `torch.manual_seed(42)` partway down the file, which
           would override any prepended block — all "seeds" would train identically and
           report a spuriously tiny variance. The seed is therefore rewritten in place.
        2. **Interpretation.** A seed variant is a variance sample, never a candidate. It
           is recorded with `is_seed_eval` so it can never be adopted as the incumbent —
           the reference repo's own published run lists "random seed 42->137" among its
           kept improvements, which is precisely this mistake.
        """
        out: list[Node] = []
        base = node.code
        for seed in self.seed_values[: self.cfg.num_seeds]:
            code = base.replace("torch.manual_seed(42)", f"torch.manual_seed({seed})")
            code = code.replace("torch.cuda.manual_seed(42)", f"torch.cuda.manual_seed({seed})")
            if code == base:
                logger.warning("seed eval: seed line not found in %s; skipping", node.id)
                break
            child = Node(
                plan=(f"SEED EVAL of {node.id} at seed {seed} (variance sample, never a "
                      f"candidate). Measures initialisation variance, which fixed-seed "
                      f"replicates cannot see."),
                code=code,
                parent=node,
            )
            child.is_seed_eval = True
            logger.info("seed eval: running %s at seed %d", node.id, seed)
            result, workdir = self.runner.run(child.id, child.code)
            self.parse_exec_result(child, result, workdir)
            child.is_seed_eval = True
            self.journal.append(child)
            out.append(child)
            logger.info("seed eval %s seed=%d -> %s", node.id, seed,
                        "BUGGY" if child.is_buggy else f"{child.metric.value:.6f}")
        return out

    def step_replicate(self) -> Node:
        """Re-run the current best byte-identically, as a fallback trial.

        Used when the rendezvous times out. The previous version of this campaign
        *exited* on timeout and left a shared GPU idle for 55 hours. Exiting is the
        worst available response: a replicate is always scientifically useful — it
        tightens the run-to-run noise estimate that every other comparison is judged
        against — and it keeps the loop alive until the agent returns.
        """
        best = self.journal.get_best_node()
        if best is None:
            raise RuntimeError("no scored node to replicate")
        node = Node(
            plan=(
                f"AUTO-REPLICATE of {best.id} (rendezvous timeout fallback). Byte-identical "
                f"source; adds a sample to the run-to-run noise estimate."
            ),
            code=best.code,
            parent=best,
        )
        logger.info("timeout fallback: replicating best node %s", best.id)
        result, workdir = self.runner.run(node.id, node.code)
        self.parse_exec_result(node, result, workdir)
        self.journal.append(node)
        logger.info("node %s -> %s | %s", node.id,
                    "BUGGY" if node.is_buggy else f"val_bpb={node.metric.value:.6f}", node.analysis)
        return node

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
