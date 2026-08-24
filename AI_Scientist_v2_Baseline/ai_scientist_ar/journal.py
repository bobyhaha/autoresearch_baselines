"""Solution tree: Node + Journal.

Ported from AI-Scientist-v2 (`ai_scientist/treesearch/journal.py`). Trimmed of the
plotting / VLM / seed-aggregation machinery that the autoresearch task has no use
for, and given a `summary` field to carry the run's parsed metric block.

The tree semantics are unchanged and are what the best-first search depends on:
a node's `stage_name` is draft/debug/improve depending on its parent, `debug_depth`
counts consecutive debugging steps, and `good_nodes` are the non-buggy nodes that
the search is allowed to improve upon.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from .metric import MetricValue, WorstMetricValue


def trim_long_string(string: str, threshold: int = 5100, k: int = 2500) -> str:
    """Keep the head and tail of a long string, eliding the middle."""
    if len(string) <= threshold:
        return string
    first_k = string[:k]
    last_k = string[-k:]
    elided = len(string) - 2 * k
    return f"{first_k}\n ... [{elided} characters elided] ... \n{last_k}"


@dataclass(eq=False)
class Node:
    """A single node in the solution tree: one candidate `train.py` and its result."""

    # ---- code & plan ----
    plan: str = field(default="", kw_only=True)
    code: str = field(default="", kw_only=True)

    # ---- general attrs ----
    step: int = field(default=None, kw_only=True)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8], kw_only=True)
    ctime: float = field(default_factory=lambda: time.time(), kw_only=True)
    parent: Optional["Node"] = field(default=None, kw_only=True)
    children: set["Node"] = field(default_factory=set, kw_only=True)
    trial_dir: str = field(default=None, kw_only=True)

    # ---- execution info ----
    _term_out: str = field(default="", kw_only=True)
    exec_time: float = field(default=None, kw_only=True)
    exc_type: str | None = field(default=None, kw_only=True)
    exit_code: int | None = field(default=None, kw_only=True)

    # ---- evaluation ----
    analysis: str = field(default="", kw_only=True)
    metric: MetricValue = field(default=None, kw_only=True)
    is_buggy: bool = field(default=None, kw_only=True)
    summary: dict = field(default_factory=dict, kw_only=True)
    # Seed-eval nodes are variance samples, never candidates: excluded from
    # good_nodes so the search can never adopt a lucky seed as the incumbent.
    is_seed_eval: bool = field(default=False, kw_only=True)
    # Failed for an environmental reason (OOM, co-tenant contention, no clear GPU) rather
    # than because the candidate was wrong. Excluded from buggy_nodes so the debug branch
    # never selects one: there is nothing to fix, and the only edit that would "fix" an OOM
    # is shrinking the model, which would be attributed to the lever under test.
    is_environmental: bool = field(default=False, kw_only=True)

    def __post_init__(self) -> None:
        if isinstance(self.children, list):
            self.children = set(self.children)
        if self.parent is not None and not isinstance(self.parent, str):
            self.parent.children.add(self)

    # ---- tree semantics (unchanged from AI-Scientist-v2) ----

    @property
    def stage_name(self) -> Literal["draft", "debug", "improve"]:
        """draft if this is a root, debug if it descends from a buggy node, else improve."""
        if self.parent is None:
            return "draft"
        return "debug" if self.parent.is_buggy else "improve"

    @property
    def is_leaf(self) -> bool:
        return not self.children

    @property
    def debug_depth(self) -> int:
        """Length of the current consecutive-debugging path (0 if not a debug node)."""
        if self.stage_name != "debug":
            return 0
        return self.parent.debug_depth + 1

    @property
    def term_out(self) -> str:
        return trim_long_string(self._term_out or "")

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Node) and self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    # ---- serialization ----

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "step": self.step,
            "ctime": self.ctime,
            "plan": self.plan,
            "code": self.code,
            "trial_dir": self.trial_dir,
            "_term_out": trim_long_string(self._term_out or "", 20000, 10000),
            "exec_time": self.exec_time,
            "exc_type": self.exc_type,
            "exit_code": self.exit_code,
            "analysis": self.analysis,
            "metric": {
                "value": self.metric.value if self.metric else None,
                "maximize": self.metric.maximize if self.metric else None,
                "name": self.metric.name if self.metric else None,
            },
            "is_buggy": self.is_buggy,
            "summary": self.summary,
            "is_seed_eval": self.is_seed_eval,
            "is_environmental": self.is_environmental,
            "stage_name": self.stage_name,
            "parent_id": None if self.parent is None else self.parent.id,
            "children": sorted(c.id for c in self.children),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Node":
        data = dict(data)
        data.pop("parent_id", None)
        data.pop("children", None)
        data.pop("stage_name", None)
        m = data.pop("metric", None) or {}
        node = cls(
            plan=data.get("plan", ""),
            code=data.get("code", ""),
            step=data.get("step"),
            id=data.get("id") or uuid.uuid4().hex[:8],
            ctime=data.get("ctime") or time.time(),
            trial_dir=data.get("trial_dir"),
            _term_out=data.get("_term_out", ""),
            exec_time=data.get("exec_time"),
            exc_type=data.get("exc_type"),
            exit_code=data.get("exit_code"),
            analysis=data.get("analysis", ""),
            is_buggy=data.get("is_buggy"),
            summary=data.get("summary") or {},
            is_seed_eval=bool(data.get("is_seed_eval")),
            is_environmental=bool(data.get("is_environmental")),
        )
        if m.get("value") is None:
            node.metric = WorstMetricValue(maximize=m.get("maximize", False), name=m.get("name"))
        else:
            node.metric = MetricValue(
                m["value"], maximize=m.get("maximize", False), name=m.get("name")
            )
        return node


@dataclass
class Journal:
    """The full history of nodes, forming a forest of solution trees."""

    nodes: list[Node] = field(default_factory=list)

    def __getitem__(self, idx: int) -> Node:
        return self.nodes[idx]

    def __len__(self) -> int:
        return len(self.nodes)

    def append(self, node: Node) -> None:
        node.step = len(self.nodes)
        self.nodes.append(node)

    @property
    def draft_nodes(self) -> list[Node]:
        """Root nodes (no parent) — the independent trees of the forest."""
        return [n for n in self.nodes if n.parent is None]

    @property
    def buggy_nodes(self) -> list[Node]:
        # Environmental failures are excluded: they are the cluster's problem, not the
        # candidate's, and offering them to the debug branch wastes a rendezvous round trip
        # (observed: a 1800s block before the replicate fallback fired).
        return [n for n in self.nodes if n.is_buggy and not n.is_environmental]

    @property
    def good_nodes(self) -> list[Node]:
        # Seed-eval nodes are measurements, not candidates — excluding them here is what
        # stops a lucky seed becoming the incumbent.
        return [n for n in self.nodes if not n.is_buggy and not n.is_seed_eval]

    def get_node_by_id(self, node_id: str) -> Optional[Node]:
        return next((n for n in self.nodes if n.id == node_id), None)

    def get_metric_history(self) -> list[MetricValue]:
        return [n.metric for n in self.nodes]

    def get_best_node(
        self, only_good: bool = True, min_improvement: float = 0.0
    ) -> Optional[Node]:
        """Best node by metric quality, optionally requiring a meaningful margin.

        AI-Scientist-v2's default `get_best_node` asks an LLM to choose among candidates
        and explicitly warns against "relying too heavily on the validation loss alone";
        plain argmin is its non-default `use_val_metric_only` branch. This port used argmin
        unconditionally, which removed that guard and produced exactly the failure it
        exists to prevent: a byte-identical replicate was promoted as the new best on a
        0.04-sigma difference (see reasoning/02 Entry 35).

        An LLM judge is the wrong replacement here — `val_bpb` is exact and comparable, so
        a model's opinion adds irreproducibility to a number that is already ground truth.
        The guard is instead restored deterministically: a challenger must beat the
        incumbent by more than `min_improvement` to displace it. Within that band the
        *earliest* qualifying node wins, which also implements the task's simplicity
        criterion — an unchanged incumbent beats a tie.
        """
        pool = self.good_nodes if only_good else self.nodes
        pool = [n for n in pool if n.metric is not None and not n.metric.is_worst]
        if not pool:
            return None
        if min_improvement <= 0:
            return max(pool, key=lambda n: n.metric)

        best_metric = max(pool, key=lambda n: n.metric).metric
        maximize = bool(best_metric.maximize)
        if maximize:
            contenders = [n for n in pool if n.metric.value >= best_metric.value - min_improvement]
        else:
            contenders = [n for n in pool if n.metric.value <= best_metric.value + min_improvement]
        # Earliest qualifying node: ties go to the established incumbent, not the luckiest draw.
        return min(contenders, key=lambda n: (n.step if n.step is not None else 0, n.ctime))

    def get_leaves(self, node: Node) -> list[Node]:
        """All leaf nodes in the subtree rooted at `node`."""
        if not node.children:
            return [node]
        leaves: list[Node] = []
        for child in node.children:
            leaves.extend(self.get_leaves(child))
        return leaves

    # ---- persistence ----

    def to_dict(self) -> Dict:
        return {"nodes": [n.to_dict() for n in self.nodes]}

    def save(self, path: str | Path) -> None:
        """Persist the journal, honouring nodes removed on disk since the last save.

        The harness keeps the journal in memory and rewrites it after every trial. An
        external edit — archiving a contention-ruined trial with
        `scripts/purge_env_failures.py` — was therefore silently reverted on the next
        save, and four purged nodes reappeared in the journal while also sitting in
        `env_failures.json`. That is worse than the purge not happening: the record
        double-counts them, and the debug branch can select them again.

        A node this Journal has previously written, which is now absent from disk, was
        deliberately removed by an operator. External removal wins.
        """
        path = Path(path)
        if getattr(self, "_saved_ids", None) and path.exists():
            try:
                on_disk = {n["id"] for n in json.loads(path.read_text(encoding="utf-8"))["nodes"]}
            except (OSError, json.JSONDecodeError, KeyError):
                on_disk = None
            if on_disk is not None:
                removed = {i for i in self._saved_ids if i not in on_disk}
                if removed:
                    for n in self.nodes:
                        if n.parent is not None and n.parent.id in removed:
                            n.parent = None
                    self.nodes = [n for n in self.nodes if n.id not in removed]
                    for n in self.nodes:
                        n.children = {c for c in n.children if c.id not in removed}

        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        tmp.replace(path)  # atomic: a crash mid-write must not corrupt the journal
        self._saved_ids = {n.id for n in self.nodes}

    @classmethod
    def load(cls, path: str | Path) -> "Journal":
        path = Path(path)
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text(encoding="utf-8"))
        journal = cls()
        by_id: dict[str, Node] = {}
        for nd in raw.get("nodes", []):
            node = Node.from_dict(nd)
            by_id[node.id] = node
            journal.nodes.append(node)
        # Second pass: rebuild parent/child links now that every node exists.
        for nd in raw.get("nodes", []):
            parent_id = nd.get("parent_id")
            if parent_id and parent_id in by_id:
                child = by_id[nd["id"]]
                parent = by_id[parent_id]
                child.parent = parent
                parent.children.add(child)
        return journal
