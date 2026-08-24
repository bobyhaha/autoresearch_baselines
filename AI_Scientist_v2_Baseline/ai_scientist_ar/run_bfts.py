"""Campaign entrypoint — the analogue of AI-Scientist-v2's `launch_scientist_bfts.py`.

Runs best-first tree search over `train.py` variants on one pinned GPU, forever (or
for `--max-iters` iterations), persisting the journal after every trial so the
campaign survives a restart.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai_scientist_ar.agent import Agent, AgentConfig, SearchConfig
from ai_scientist_ar.backend import RendezvousBackend, RendezvousTimeout
from ai_scientist_ar.interpreter import TrialRunner
from ai_scientist_ar.journal import Journal


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    root.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)


def write_results_tsv(journal: Journal, path: Path, best_seen: list) -> None:
    """Task-format results log: one row per experiment, tab separated.

    `commit` is the node id rather than a git hash — the search keeps every candidate
    in its own trial directory, so the node id is the durable handle here.
    """
    lines = ["node_id\tval_bpb\tmemory_gb\tstatus\tdescription"]
    best = float("inf")
    for n in journal.nodes:
        val = n.metric.value if (n.metric and n.metric.value is not None) else 0.0
        mem = (n.summary.get("peak_vram_mb") or 0.0) / 1024
        if n.is_buggy:
            status = "crash"
        elif val < best:
            status = "keep"
            best = val
        else:
            status = "discard"
        desc = (n.plan or "").replace("\t", " ").replace("\n", " ")[:200]
        lines.append(f"{n.id}\t{val:.6f}\t{mem:.1f}\t{status}\t{desc}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=str(Path.home() / "ai_scientist_v2_baseline"))
    ap.add_argument("--task-dir", default=None, help="pristine autoresearch checkout")
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--max-iters", type=int, default=0, help="0 = run until stopped")
    ap.add_argument("--num-drafts", type=int, default=3)
    ap.add_argument("--debug-prob", type=float, default=0.5)
    ap.add_argument("--max-debug-depth", type=int, default=3)
    ap.add_argument("--hard-timeout", type=int, default=900)
    ap.add_argument("--rendezvous-timeout", type=float, default=7200.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-hours", type=float, default=0.0, help="0 = no deadline")
    ap.add_argument("--min-improvement", type=float, default=0.0,
                    help="challenger must beat the incumbent by this to displace it (2 sigma)")
    ap.add_argument("--num-seeds", type=int, default=0,
                    help="re-evaluate each newly confirmed best at N seeds (upstream multi_seed_eval)")
    args = ap.parse_args()

    base = Path(args.base_dir)
    task = Path(args.task_dir) if args.task_dir else base / "task"
    campaign = base / "campaign"
    campaign.mkdir(parents=True, exist_ok=True)

    setup_logging(campaign / "campaign.log")
    logging.info("=" * 70)
    logging.info("AI-Scientist-v2 (autoresearch port) starting on GPU %d", args.gpu)

    import random

    random.seed(args.seed)

    journal_path = campaign / "journal.json"
    journal = Journal.load(journal_path)
    logging.info("journal: %d existing nodes", len(journal))

    backend = RendezvousBackend(base / "rendezvous", timeout=args.rendezvous_timeout)
    runner = TrialRunner(base, task, gpu=args.gpu, hard_timeout=args.hard_timeout)

    memo_path = base / "TASK_MEMO.md"
    task_memo = memo_path.read_text(encoding="utf-8") if memo_path.exists() else ""
    baseline_code = (task / "train.py").read_text(encoding="utf-8")
    pristine_prepare_sha = hashlib.sha256((task / "prepare.py").read_bytes()).hexdigest()
    logging.info("pristine prepare.py sha256=%s", pristine_prepare_sha[:16])

    cfg = AgentConfig(
        search=SearchConfig(
            max_debug_depth=args.max_debug_depth,
            debug_prob=args.debug_prob,
            num_drafts=args.num_drafts,
        ),
        gpu=args.gpu,
        hard_timeout=args.hard_timeout,
        min_improvement=args.min_improvement,
        num_seeds=args.num_seeds,
    )
    agent = Agent(cfg, journal, backend, runner, task_memo, baseline_code, pristine_prepare_sha)

    it = 0
    last_best_id = None
    deadline = time.time() + args.max_hours * 3600 if args.max_hours > 0 else None
    while args.max_iters == 0 or it < args.max_iters:
        if deadline is not None and time.time() >= deadline:
            logging.info("reached --max-hours=%.1f deadline, stopping", args.max_hours)
            break
        it += 1
        logging.info("-" * 70)
        logging.info("iteration %d (journal size %d)", it, len(journal))
        t0 = time.time()
        try:
            node = agent.step()
        except RendezvousTimeout as exc:
            # Never exit on timeout. An idle GPU is strictly worse than a replicate,
            # which at least sharpens the noise floor and keeps the search alive.
            logging.warning("rendezvous timed out (%s) — falling back to a replicate", exc)
            try:
                node = agent.step_replicate()
            except Exception:
                logging.exception("replicate fallback failed; pausing 60s")
                time.sleep(60)
                continue
        except KeyboardInterrupt:
            logging.info("interrupted by operator")
            return 0
        except Exception:
            logging.exception("iteration %d failed; continuing", it)
            time.sleep(10)
            continue

        journal.save(journal_path)

        # Upstream runs multi-seed evaluation on the best node at each stage boundary.
        # This port has no stages, so the equivalent trigger is a confirmed change of
        # incumbent: that is when a claim is about to be made, and therefore when the
        # variance behind it needs measuring.
        best = journal.get_best_node(min_improvement=args.min_improvement)
        if args.num_seeds and best is not None and best.id != last_best_id:
            if last_best_id is not None:
                logging.info("incumbent changed %s -> %s; running %d-seed evaluation",
                             last_best_id, best.id, args.num_seeds)
                try:
                    agent.run_seed_eval(best)
                    journal.save(journal_path)
                except Exception:
                    logging.exception("seed evaluation failed; continuing")
            last_best_id = best.id

        write_results_tsv(journal, campaign / "results.tsv", [])
        best = journal.get_best_node(min_improvement=args.min_improvement)
        status = {
            "iteration": it,
            "journal_size": len(journal),
            "last_node": node.id,
            "last_val_bpb": node.metric.value if node.metric else None,
            "last_is_buggy": node.is_buggy,
            "best_node": best.id if best else None,
            "best_val_bpb": best.metric.value if best else None,
            "updated": time.time(),
            "iteration_seconds": time.time() - t0,
        }
        (campaign / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        logging.info(
            "iteration %d done in %.0fs | best so far: %s (%s)",
            it,
            time.time() - t0,
            best.id if best else "none",
            f"{best.metric.value:.6f}" if best else "n/a",
        )

    logging.info("reached max_iters=%d, stopping", args.max_iters)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
