"""Trusted wrapper that keeps metric evaluation outside agent-edited train.py.

The candidate still calls ``prepare.evaluate_bpb`` at its normal location. During the
training script that call captures the live model and returns NaN. After the script
finishes, this wrapper invokes the original locked evaluator itself and emits a
nonce-bound record for the parent process.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import math
import os
import runpy
import sys
from pathlib import Path
from typing import Any


SENTINEL = "__FENGHEGUAI_LOCKED_EVAL__"


def _coerce_metric(value: Any) -> float:
    if hasattr(value, "item"):
        value = value.item()
    metric = float(value)
    if not math.isfinite(metric):
        raise ValueError(f"locked evaluator produced non-finite metric: {metric}")
    return metric


_AUTOCAST_DEVICES = ("cuda", "cpu")


def _capture_autocast_state() -> list[tuple[str, Any]]:
    """Snapshot ambient autocast at the candidate's own ``evaluate_bpb`` call site.

    The locked evaluation is deliberately deferred until after the training script
    exits, so any ``with autocast(...)`` block wrapping the original call has already
    unwound. Replaying that context keeps the deferred evaluation numerically
    equivalent to the ordinary in-place call it stands in for.
    """
    try:
        import torch
    except ImportError:
        return []
    state: list[tuple[str, Any]] = []
    for device_type in _AUTOCAST_DEVICES:
        try:
            if torch.is_autocast_enabled(device_type):
                state.append((device_type, torch.get_autocast_dtype(device_type)))
        except (RuntimeError, TypeError):
            continue
    return state


@contextlib.contextmanager
def _restored_autocast(state: list[tuple[str, Any]]):
    if not state:
        yield
        return
    import torch

    with contextlib.ExitStack() as stack:
        for device_type, dtype in state:
            stack.enter_context(torch.autocast(device_type=device_type, dtype=dtype))
        yield


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--nonce", required=True)
    args = parser.parse_args()

    target = Path(args.target).resolve()
    train_path = target / "train.py"
    if not train_path.is_file():
        raise FileNotFoundError(train_path)

    os.chdir(target)
    sys.path.insert(0, str(target))
    prepare = importlib.import_module("prepare")
    locked_evaluate = prepare.evaluate_bpb
    calls: list[tuple[tuple[Any, ...], dict[str, Any], list[tuple[str, Any]]]] = []

    def capture_evaluation(*call_args: Any, **call_kwargs: Any) -> float:
        calls.append((call_args, call_kwargs, _capture_autocast_state()))
        return float("nan")

    prepare.evaluate_bpb = capture_evaluation
    old_argv = sys.argv
    sys.argv = [str(train_path)]
    exit_code = 0
    try:
        runpy.run_path(str(train_path), run_name="__main__")
    except SystemExit as exc:
        exit_code = int(exc.code or 0) if isinstance(exc.code, (int, type(None))) else 1
        if exit_code:
            raise
    finally:
        sys.argv = old_argv
        prepare.evaluate_bpb = locked_evaluate

    if exit_code != 0:
        return exit_code
    if len(calls) != 1:
        raise RuntimeError(f"expected exactly one evaluate_bpb call, observed {len(calls)}")

    call_args, call_kwargs, autocast_state = calls[0]
    with _restored_autocast(autocast_state):
        metric = _coerce_metric(locked_evaluate(*call_args, **call_kwargs))
    record = {"nonce": args.nonce, "val_bpb": metric, "evaluate_calls": len(calls)}
    print(SENTINEL + json.dumps(record, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

