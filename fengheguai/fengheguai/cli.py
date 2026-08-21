from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .audit import audit_campaign
from .campaign import Campaign, campaign_lock, initialize_campaign
from .config import default_config
from .reporting import status_payload


def _json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


def _campaign_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fengheguai",
        description="Autoresearch for one objective: minimize 300-second nanoGPT val_bpb.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="Create an isolated campaign from a nanoGPT target")
    init.add_argument("--target", required=True, help="Directory containing train.py and prepare.py")
    init.add_argument("--campaign", required=True, help="New campaign directory")
    init.add_argument("--name", default="fengheguai")
    init.add_argument(
        "--agent-command",
        help="Tokenized command template; placeholders include {trial}, {agent_schema}, and {agent_result}",
    )
    init.add_argument(
        "--train-command",
        help="Tokenized command template; it must contain the {audit_runner} token",
    )
    init.add_argument(
        "--confirmation-runs",
        type=int,
        default=1,
        help="Extra reruns for the baseline and prospective winners (default: 1)",
    )
    init.add_argument(
        "--minimum-improvement",
        type=float,
        default=0.0,
        help="Required absolute val_bpb decrease for promotion (default: 0)",
    )
    for name, help_text in (
        ("baseline", "Run and lock the unmodified baseline"),
        ("step", "Generate, audit, evaluate, and decide one candidate"),
        ("status", "Show the current champion and latest node"),
        ("audit", "Verify the evidence chain, source hashes, and promotion decisions"),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("--campaign", required=True)

    run = subparsers.add_parser("run", help="Run the baseline if needed, then search autonomously")
    run.add_argument("--campaign", required=True)
    limit = run.add_mutually_exclusive_group(required=True)
    limit.add_argument("--trials", type=int, help="Number of candidate trials")
    limit.add_argument("--forever", action="store_true", help="Continue until externally interrupted")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "init":
            target = Path(args.target).expanduser().resolve()
            root = _campaign_path(args.campaign)
            config = default_config(
                name=args.name,
                target=target,
                agent_command=args.agent_command,
                train_command=args.train_command,
                confirmation_runs=args.confirmation_runs,
                minimum_improvement=args.minimum_improvement,
            )
            campaign = initialize_campaign(root, config)
            _json({"ok": True, "campaign": str(campaign.root), "objective": "minimize val_bpb", "training_seconds": 300})
            return

        root = _campaign_path(args.campaign)
        if args.command == "audit":
            report = audit_campaign(root)
            _json(report.to_dict())
            raise SystemExit(0 if report.ok else 1)

        campaign = Campaign(root)
        if args.command == "status":
            _json(campaign.status())
            return

        with campaign_lock(root):
            if args.command == "baseline":
                _json(campaign.baseline())
            elif args.command == "step":
                _json(campaign.step())
            elif args.command == "run":
                if args.trials is not None and args.trials < 1:
                    raise ValueError("--trials must be positive")
                completed = campaign.run(None if args.forever else args.trials)
                _json(
                    {
                        "completed": len(completed),
                        "status": status_payload(campaign.records()),
                    }
                )
    except KeyboardInterrupt:
        print("Fengheguai stopped by user; completed evidence is durable.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"fengheguai: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
