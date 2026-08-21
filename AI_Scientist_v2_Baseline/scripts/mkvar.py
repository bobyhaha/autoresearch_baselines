#!/usr/bin/env python3
"""Derive a candidate train.py from a parent by exact-string substitution.

Candidates are almost always a small edit to an existing train.py, so the agent
never needs to move the whole 26KB file through its context. Each replacement must
match exactly once; anything else is an error rather than a silent no-op, which is
what stops a "change" from being enqueued that did not actually change anything.

  mkvar.py --from trials/<parent>/train.py --out /tmp/cand.py \
           --sub 'OLD TEXT' 'NEW TEXT' [--sub ...]
"""

from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from", dest="src", required=True, help="parent train.py")
    ap.add_argument("--out", required=True, help="candidate train.py to write")
    ap.add_argument(
        "--sub",
        nargs=2,
        action="append",
        metavar=("OLD", "NEW"),
        required=True,
        help="exact-string replacement; must match exactly once",
    )
    ap.add_argument("--allow-count", type=int, default=1, help="required match count per --sub")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    src = Path(args.src)
    original = src.read_text(encoding="utf-8")
    text = original

    for old, new in args.sub:
        count = text.count(old)
        if count != args.allow_count:
            print(
                f"ERROR: pattern matched {count} times (expected {args.allow_count}):\n"
                f"  {old!r}",
                file=sys.stderr,
            )
            return 1
        text = text.replace(old, new)

    if text == original:
        print("ERROR: substitutions produced no change", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")

    if not args.quiet:
        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            text.splitlines(keepends=True),
            fromfile=str(src),
            tofile=str(out),
            n=1,
        )
        sys.stdout.writelines(diff)
    # Syntax-check the candidate here so a typo fails now, not 6 minutes into a trial.
    try:
        compile(text, str(out), "exec")
    except SyntaxError as exc:
        print(f"ERROR: candidate has a syntax error: {exc}", file=sys.stderr)
        return 1
    print(f"\nOK wrote {out} ({len(text)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
